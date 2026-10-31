"""
GPT-2 style decoder-only transformer, sized for CPU training.

Deliberately close to nanoGPT: pre-norm blocks, learned positional embeddings,
GELU MLP with 4x expansion, and weight tying between the token embedding and
the output projection.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 24
    n_head: int = 24
    n_embd: int = 1536
    dropout: float = 0.1
    bias: bool = True

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # scaled_dot_product_attention has a fused CPU path; fall back to the
        # explicit implementation on older torch builds.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                    1, 1, cfg.block_size, cfg.block_size
                ),
                persistent=False,
            )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    @classmethod
    def from_config(
        cls, cfg: "GPTConfig", use_gradient_checkpointing: bool = False
    ) -> "GPT":
        """Construct a GPT from a GPTConfig (PyTorch convention)."""
        return cls(cfg, use_gradient_checkpointing=use_gradient_checkpointing)

    def __init__(self, cfg, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.config = cfg
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: saves ~32M parameters at this vocab size and is a
        # meaningful regulariser for a model this small.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2 paper, section 2.3).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
            n -= self.transformer.wte.weight.numel()
        return n

    def param_breakdown(self):
        """Parameter counts grouped by component, for the training report."""
        groups = {}
        for name, p in self.named_parameters():
            if name.startswith("transformer.wte"):
                key = "token_embedding (tied with lm_head)"
            elif name.startswith("transformer.wpe"):
                key = "positional_embedding"
            elif name.startswith("transformer.h"):
                key = "transformer_blocks"
            elif name.startswith("transformer.ln_f"):
                key = "final_layernorm"
            else:
                key = name
            groups[key] = groups.get(key, 0) + p.numel()
        return groups

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        if self.use_gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        else:
            # Inference: only the last position is needed.
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """Decay only tensors of dim >= 2 (matmuls/embeddings), not norms/biases."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
        return opt, sum(p.numel() for p in decay), sum(p.numel() for p in no_decay)

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_id=None,
    ):
        """Autoregressively sample `max_new_tokens` continuations of `idx`."""
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if finished.any():
                logits[finished, :] = float("-inf")

            if repetition_penalty > 1.0:
                for i in range(idx.size(0)):
                    if finished[i]:
                        continue
                    seen = torch.unique(idx[i])
                    logits[i, seen] /= repetition_penalty

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float("-inf")

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            if (logits == float("-inf")).all(dim=-1).any():
                # Fallback: if every token was masked, restore the original logits
                # for those rows so sampling can continue.
                logits = logits.clone()
                logits[(logits == float("-inf")).all(dim=-1), :] = 0.0

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                for i in range(idx.size(0)):
                    if idx_next[i].item() == eos_token_id:
                        finished[i] = True

            if finished.all():
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
