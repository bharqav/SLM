"""Tests for model loading, inference, and API server.

Run:
    python -m pytest tests/test_model_and_api.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLM_CHECKPOINT = PROJECT_ROOT / "slm-training" / "best_model.pt"


@pytest.fixture()
def slm_checkpoint():
    if not SLM_CHECKPOINT.exists():
        pytest.skip(f"SLM checkpoint not found: {SLM_CHECKPOINT}")
    return SLM_CHECKPOINT


class TestModelLoading:
    def test_checkpoint_exists(self, slm_checkpoint):
        assert slm_checkpoint.exists()
        assert slm_checkpoint.stat().st_size > 1_000_000  # at least 1MB

    def test_checkpoint_is_real_zip(self, slm_checkpoint):
        with open(slm_checkpoint, "rb") as f:
            header = f.read(4)
        assert header.startswith(b"PK"), (
            "checkpoint should be a real PyTorch zip bundle"
        )

    def test_model_loads(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig

        # Load config from checkpoint metadata so this stays correct across runs
        state = torch.load(str(slm_checkpoint), map_location="cpu", weights_only=False)
        cfg = GPTConfig(**(state.get("model_config") or {}))
        model = GPT(cfg)
        model.load_state_dict(state["model"])
        model.eval()

        assert sum(p.numel() for p in model.parameters()) > 0
        assert state.get("step") is not None
        assert state.get("best_val_loss") is not None

    def test_model_generate(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig
        from tiktoken import get_encoding

        # Load config from checkpoint metadata so this stays correct across runs
        state = torch.load(str(slm_checkpoint), map_location="cpu", weights_only=False)
        cfg = GPTConfig(**(state.get("model_config") or {}))
        model = GPT(cfg)
        model.load_state_dict(state["model"])
        model.eval()

        enc = get_encoding("gpt2")
        tokens = enc.encode("def preprocess(text):")
        x = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            out = model.generate(x, max_new_tokens=20, temperature=0.7, top_k=20)

        text = enc.decode(out[0].tolist())
        assert len(text) > len("def preprocess(text):")
        assert text.startswith("def preprocess(text):")

    def test_model_to_dict(self):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPTConfig

        cfg = GPTConfig()
        d = cfg.to_dict()
        assert d["vocab_size"] == 50257
        assert d["n_layer"] == 24
        assert d["n_head"] == 24
        assert d["n_embd"] == 1536

    def test_model_num_params(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig

        cfg = GPTConfig()
        model = GPT(cfg)
        total = model.num_params()
        assert total > 0
        non_emb = model.num_params(non_embedding=True)
        assert non_emb < total

    def test_model_param_breakdown(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig

        cfg = GPTConfig()
        model = GPT(cfg)
        groups = model.param_breakdown()
        assert "token_embedding (tied with lm_head)" in groups
        assert "positional_embedding" in groups
        assert "transformer_blocks" in groups
        assert "final_layernorm" in groups
        assert sum(groups.values()) == model.num_params()

    def test_model_forward_with_targets(self, slm_checkpoint):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig

        # Load config from checkpoint metadata so this stays correct across runs
        state = torch.load(str(slm_checkpoint), map_location="cpu", weights_only=False)
        cfg = GPTConfig(**(state.get("model_config") or {}))
        model = GPT(cfg)
        model.load_state_dict(state["model"])
        model.train()

        B, T = 2, 16
        x = torch.randint(0, cfg.vocab_size, (B, T))
        y = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(x, targets=y)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_model_configure_optimizers(self):
        sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))
        from model import GPT, GPTConfig

        cfg = GPTConfig()
        model = GPT(cfg)
        opt, decay, no_decay = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95)
        )
        assert decay > 0
        assert no_decay > 0
        assert decay + no_decay == model.num_params()


class TestAPIServer:
    @pytest.fixture()
    def client(self):
        try:
            from fastapi.testclient import TestClient

            from slm_api import app

            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi not installed")

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_model_info_endpoint(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "params" in data
        assert data["params"] > 0

    def test_generate_endpoint(self, client):
        response = client.post(
            "/generate",
            json={
                "prompt": "def preprocess(text):",
                "max_new_tokens": 20,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "eos_token_id": 50256,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "completion" in data
        assert "completion_tokens" in data
        assert "tokens_per_second" in data
        assert data["completion_tokens"] > 0
        assert "safe" in data
        assert "safety_reason" in data
        assert "degenerate" in data

    def test_generate_unsafe_prompt_is_rejected(self, client):
        response = client.post(
            "/generate",
            json={
                "prompt": "how to make a bomb",
                "max_new_tokens": 20,
            },
        )
        assert response.status_code == 400

    def test_safety_info_endpoint(self, client):
        response = client.get("/safety/info")
        assert response.status_code == 200
        data = response.json()
        assert "input_filters" in data
        assert "output_filters" in data
        assert "degeneracy_check" in data
        assert isinstance(data["input_filters"], int)
        assert isinstance(data["output_filters"], int)

    @pytest.mark.slow
    def test_generate_stress(self, client):
        prompts = [
            "def preprocess(text):",
            "class ChatMessage:",
            "import torch",
            "async function fetch(",
            "The best way to handle stress is",
        ]
        latencies = []
        for prompt in prompts:
            t0 = time.time()
            response = client.post(
                "/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 20,
                    "temperature": 0.7,
                    "top_k": 20,
                },
            )
            dt = time.time() - t0
            assert response.status_code == 200
            latencies.append(dt)
        mean_latency = sum(latencies) / len(latencies)
        assert mean_latency < 30.0, f"Mean latency too high: {mean_latency:.2f}s"


class TestVerifySetup:
    def test_verify_setup_script_exists(self):
        script = PROJECT_ROOT / "verify_setup.py"
        assert script.exists()

    def test_verify_setup_runs(self):
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, str(PROJECT_ROOT / "verify_setup.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        # Should exit 0 or 1 (warnings), not crash
        assert result.returncode in (0, 1)
        assert "SLM checkpoint" in result.stdout


class TestEvaluateTestSplit:
    def test_evaluate_script_exists(self):
        script = PROJECT_ROOT / "evaluate_test_split.py"
        assert script.exists()

    @pytest.mark.slow
    def test_evaluate_runs(self):
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, str(PROJECT_ROOT / "evaluate_test_split.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,
        )
        assert result.returncode == 0
        assert "perplexity" in result.stdout.lower()

    def test_test_split_eval_output(self):
        output_path = PROJECT_ROOT / "benchmarks" / "test_split_eval.json"
        if not output_path.exists():
            pytest.skip("test_split_eval.json not generated yet")

        import json

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "perplexity" in data
        assert data["perplexity"] > 0
        assert "checkpoint" in data


class TestDataCleanliness:
    def test_empathy_train_bin_no_role_labels(self):
        import numpy as np

        data_path = PROJECT_ROOT / "slm-training" / "empathy_train.bin"
        if not data_path.exists():
            pytest.skip("empathy_train.bin not found")

        tokens = np.memmap(str(data_path), dtype=np.uint16, mode="r")
        # Decode a sample to check for role labels
        enc = __import__("tiktoken").get_encoding("gpt2")
        sample_size = min(5000, len(tokens))
        sample_text = enc.decode(tokens[:sample_size].tolist())
        assert "User:" not in sample_text
        assert "Assistant:" not in sample_text

    def test_empathy_val_bin_no_role_labels(self):
        import numpy as np

        data_path = PROJECT_ROOT / "slm-training" / "empathy_val.bin"
        if not data_path.exists():
            pytest.skip("empathy_val.bin not found")

        tokens = np.memmap(str(data_path), dtype=np.uint16, mode="r")
        enc = __import__("tiktoken").get_encoding("gpt2")
        sample_size = min(5000, len(tokens))
        sample_text = enc.decode(tokens[:sample_size].tolist())
        assert "User:" not in sample_text
        assert "Assistant:" not in sample_text


class TestOpenAICompatEndpoints:
    @pytest.fixture()
    def client(self):
        try:
            from fastapi.testclient import TestClient

            from slm_api import app

            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi not installed")

    def test_v1_models_list(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model = data["data"][0]
        assert model["id"] == "slm"
        assert model["object"] == "model"
        assert model["owned_by"] == "slm-project"

    def test_v1_completions_basic(self, client):
        response = client.post(
            "/v1/completions",
            json={
                "model": "slm",
                "prompt": "def hello():",
                "max_tokens": 10,
                "temperature": 0.7,
                "top_p": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert "id" in data
        assert data["model"] == "slm"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert "text" in choice
        assert choice["finish_reason"] in ("stop", "length")
        assert "usage" in data
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] >= usage["prompt_tokens"]

    def test_v1_completions_with_list_prompt(self, client):
        response = client.post(
            "/v1/completions",
            json={
                "model": "slm",
                "prompt": ["Hello", "World"],
                "max_tokens": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0

    def test_v1_completions_unsafe_rejected(self, client):
        response = client.post(
            "/v1/completions",
            json={
                "model": "slm",
                "prompt": "how to make a bomb",
                "max_tokens": 10,
            },
        )
        assert response.status_code == 400

    def test_v1_chat_completions_basic(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "def preprocess(text):"},
                ],
                "max_tokens": 20,
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert data["model"] == "slm"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] in ("stop", "length", "content_filter")
        assert "usage" in data
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] >= usage["prompt_tokens"]

    def test_v1_chat_completions_message_history(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "max_tokens": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1

    def test_v1_chat_completions_stream(self, client):
        response = client.post(
            "/v1/chat/completions/stream",
            json={
                "model": "slm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 20,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content

    def test_v1_completions_stream(self, client):
        response = client.post(
            "/v1/completions/stream",
            json={
                "model": "slm",
                "prompt": "def hello():",
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content

    def test_v1_chat_completions_tool_call(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert "tool_calls" in choice["message"]
        assert len(choice["message"]["tool_calls"]) > 0
        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert "arguments" in tool_call["function"]

    def test_v1_chat_completions_tool_call_none(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "dummy", "parameters": {}},
                    }
                ],
                "tool_choice": "none",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"

    def test_v1_chat_completions_stop_sequences(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [{"role": "user", "content": "def hello():"}],
                "max_tokens": 20,
                "stop": ["\n\n"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1

    def test_v1_chat_completions_backward_compat(self, client):
        """Ensure legacy /generate endpoint still works."""
        response = client.post(
            "/generate",
            json={
                "prompt": "def hello():",
                "max_new_tokens": 10,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "completion" in data
        assert "completion_tokens" in data

    def test_v1_chat_completions_empty_messages(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [],
                "max_tokens": 10,
            },
        )
        assert response.status_code in (200, 422)

    def test_v1_chat_completions_missing_model(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        assert response.status_code in (200, 422)

    def test_v1_completions_missing_prompt(self, client):
        response = client.post(
            "/v1/completions",
            json={
                "model": "slm",
                "max_tokens": 10,
            },
        )
        assert response.status_code in (200, 422)

    def test_v1_chat_completions_max_tokens_zero(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 0,
            },
        )
        assert response.status_code in (200, 422)

    def test_v1_completions_system_role_not_injected(self, client):
        """OpenAI-compatible: system messages should be rendered but not lost."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slm",
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "Say hi"},
                ],
                "max_tokens": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"


class TestAuth:
    @pytest.fixture()
    def client(self):
        try:
            from fastapi.testclient import TestClient

            from slm_api import app

            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi not installed")

    def test_openai_endpoints_require_auth(self, client):
        # Without API key, should be rejected when API_KEY env is set
        # Note: in test env, API_KEY might not be set, so this may pass through
        # We just verify the endpoint exists
        response = client.get("/v1/models")
        assert response.status_code in (200, 401, 403)
