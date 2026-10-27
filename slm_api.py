#!/usr/bin/env python3
"""FastAPI server for the from-scratch SLM model.

Usage:
    python -m uvicorn slm_api:app --host 0.0.0.0 --port 8000

Or with docker-compose:
    docker-compose up slm-api
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from model import GPT, GPTConfig  # noqa: E402
from tiktoken import get_encoding  # noqa: E402

app = FastAPI(
    title="SLM Inference API",
    description="From-scratch GPT-style model serving endpoint",
    version="1.0.0",
)

# Globals for lazy loading
_model: GPT | None = None
_enc = None
_device = "cpu"

# Auth
security = HTTPBearer(auto_error=False)
_api_key = os.environ.get("API_KEY", "change-me-to-a-real-key")


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    if _api_key == "change-me-to-a-real-key":
        return "dev-mode"
    if credentials is None or credentials.credentials != _api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=200)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    completion_tokens: int
    prompt_tokens: int
    tokens_per_second: float
    latency_s: float


def load_model(checkpoint: str = "best_model.pt") -> tuple[GPT, any]:
    global _model, _enc, _device

    if _model is not None:
        return _model, _enc

    checkpoint_path = PROJECT_ROOT / "slm-training" / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = GPTConfig()
    model = GPT(cfg)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    _model = model
    _enc = get_encoding("gpt2")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device == "cuda":
        _model = _model.to("cuda")

    return _model, _enc


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    import time

    try:
        model, enc = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tokens = enc.encode(req.prompt)
    if not tokens:
        tokens = [enc.eot_token]

    x = torch.tensor([tokens], dtype=torch.long)
    if _device == "cuda":
        x = x.to("cuda")

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc

    dt = time.perf_counter() - t0
    n_new = max(out.numel() - len(tokens), 0)
    text = enc.decode(out[0].tolist())
    completion = text[len(req.prompt) :] if text.startswith(req.prompt) else text
    toks_per_sec = n_new / max(dt, 1e-6)

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        completion_tokens=n_new,
        prompt_tokens=len(tokens),
        tokens_per_second=round(toks_per_sec, 1),
        latency_s=round(dt, 4),
    )


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    try:
        model, _ = load_model()
        return {
            "params": sum(p.numel() for p in model.parameters()),
            "device": _device,
            "vocab_size": model.config.vocab_size,
            "n_layers": model.config.n_layer,
            "n_heads": model.config.n_head,
            "d_model": model.config.n_embd,
            "ctx_len": model.config.block_size,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
