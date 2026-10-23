"""Serve the merged empathetic model (or any merged HF folder) over HTTP.

Loads the merged fp16 model in 4-bit (NF4) so it fits an 8.5GB card
comfortably and exposes a tiny generation endpoint in-stream with the
backend gateway's `served_model` provider:

    POST /generate   {"prompt": "...", "max_new_tokens": 256}
    ->        {"text": "...", "completion_tokens": 12, "model": "..."}

Auth: set SERVE_API_TOKEN; backend must send it as `Authorization: Bearer ...`.
Bind: 127.0.0.1 only (same host as the backend container/dev box).

    .venv/Scripts/python serve_model.py --model merged_fp16 --port 8902
"""

from __future__ import annotations

import argparse

import torch
from fastapi import Depends, FastAPI, HTTPException, security
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = FastAPI(title="slm-ft model server", docs_url=None, redoc_url=None)
_state = {"model": None, "tok": None, "name": "", "token": ""}


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 256


class GenResp(BaseModel):
    text: str
    completion_tokens: int
    model: str


def bearer_auth(credentials: security.HTTPAuthorizationCredentials = Depends(
        security.HTTPBearer(auto_error=False))) -> None:
    if _state["token"] and credentials and credentials.credentials != _state["token"]:
        raise HTTPException(status_code=401, detail="invalid")


def load(model_dir: str = "merged_fp16", token: str = "") -> None:
    tok = AutoTokenizer.from_pretrained(model_dir)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    _state.update(model=model, tok=tok, name=model_dir, token=token)


@app.post("/generate", response_model=GenResp)
def generate(req: GenReq, _: None = Depends(bearer_auth)) -> GenResp:
    prompt = req.prompt[:48000]
    max_new = max(1, min(1024, req.max_new_tokens))
    with torch.inference_mode():
        ids = _state["tok"](prompt, return_tensors="pt").to("cuda")
        out = _state["model"].generate(
            **ids,
            max_new_tokens=max_new,
            pad_token_id=_state["tok"].eos_token_id,
        )
    text = _state["tok"].decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    return GenResp(
        text=text,
        completion_tokens=int(out.shape[1] - ids["input_ids"].shape[1]),
        model=_state["name"],
    )


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged_fp16")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--token", default="")
    args = ap.parse_args()
    load(args.model, args.token)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
