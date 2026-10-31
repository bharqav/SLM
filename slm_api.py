#!/usr/bin/env python3
"""FastAPI server for the from-scratch SLM model.

Usage:
    python -m uvicorn slm_api:app --host 0.0.0.0 --port 8000

Or with docker-compose:
    docker-compose up slm-api
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import (  # noqa: E402
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from guardrails import check_input, check_output, is_degenerate  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from quantize import QuantizationInfo, get_available_methods, quantize_model  # noqa: E402
from tiktoken import get_encoding  # noqa: E402

app = FastAPI(
    title="SLM Inference API",
    description="From-scratch GPT-style model serving endpoint",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            payload["data"] = record.extra_data
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("slm")
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
    return logger


logger = _configure_logging()


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "slm_requests_total",
    "Total API requests",
    ["endpoint", "http_status", "safe"],
)
REQUEST_LATENCY = Histogram(
    "slm_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
MODEL_LOADED = Gauge(
    "slm_model_loaded",
    "1 if model is loaded, 0 otherwise",
)
GENERATION_TOKENS = Counter(
    "slm_generation_tokens_total",
    "Total tokens generated",
    ["endpoint"],
)

# Globals for lazy loading
_model: GPT | None = None
_enc = None
_device = "cpu"
_quantization_info: QuantizationInfo | None = None

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


@app.middleware("http")
async def _metrics_and_logging(request: Request, call_next):  # type: ignore[type-arg]
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    log_extra = {"request_id": request_id}

    try:
        response: Response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception", extra={"request_id": request_id})
        raise

    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)

    logger.info(
        "%s %s %s %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        latency,
        extra=log_extra,
    )

    response.headers["X-Request-ID"] = request_id
    return response


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=200)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    eos_token_id: int | None = Field(
        default=50256, description="EOS token ID for early stopping"
    )


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    completion_tokens: int
    prompt_tokens: int
    tokens_per_second: float
    latency_s: float
    safe: bool = True
    safety_reason: str = ""
    degenerate: bool = False


def load_model(
    checkpoint: str = "best_model.pt",
    quantize: str | None = None,
) -> tuple[GPT, any, QuantizationInfo | None]:
    global _model, _enc, _device, _quantization_info

    if _model is not None:
        return _model, _enc, _quantization_info

    checkpoint_path = PROJECT_ROOT / "slm-training" / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_cfg = state.get("model_config")
    if model_cfg:
        cfg = GPTConfig(**model_cfg)
    else:
        cfg = GPTConfig()
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()

    _model = model
    _enc = get_encoding("gpt2")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device == "cuda":
        _model = _model.to("cuda")

    _quantization_info = None
    if quantize:
        QuantizationInfo(
            method=quantize,
            applied=False,
            device=_device,
            original_params=sum(p.numel() for p in _model.parameters()),
        )
        _model, _quantization_info = quantize_model(
            _model,
            method=quantize,
            device=_device,
            checkpoint_path=str(checkpoint_path),
        )

    return _model, _enc, _quantization_info


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    import time

    safety = check_input(req.prompt)
    if not safety.safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety.reason,
        )

    try:
        model, enc, _ = load_model()
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
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                eos_token_id=req.eos_token_id,
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

    out_safety = check_output(completion)
    degenerate = is_degenerate(completion)

    REQUEST_COUNT.labels(
        endpoint="/generate", http_status="200", safe=str(out_safety.safe).lower()
    ).inc()
    GENERATION_TOKENS.labels(endpoint="/generate").inc(n_new)

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        completion_tokens=n_new,
        prompt_tokens=len(tokens),
        tokens_per_second=round(toks_per_sec, 1),
        latency_s=round(dt, 4),
        safe=out_safety.safe,
        safety_reason=out_safety.reason,
        degenerate=degenerate,
    )


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    try:
        model, _, qinfo = load_model()
        resp = {
            "params": sum(p.numel() for p in model.parameters()),
            "device": _device,
            "vocab_size": model.config.vocab_size,
            "n_layers": model.config.n_layer,
            "n_heads": model.config.n_head,
            "d_model": model.config.n_embd,
            "ctx_len": model.config.block_size,
        }
        if qinfo:
            resp["quantization"] = {
                "method": qinfo.method,
                "applied": qinfo.applied,
                "compression_ratio": qinfo.compression_ratio,
                "warnings": qinfo.warnings,
            }
        return resp
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/quantize/info")
async def quantize_info(api_key: str = Depends(verify_api_key)):
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    available = get_available_methods()
    result: dict = {"model_loaded": _model is not None, "available_methods": available}
    if _quantization_info is not None:
        result["current"] = {
            "method": _quantization_info.method,
            "applied": _quantization_info.applied,
            "compression_ratio": _quantization_info.compression_ratio,
            "warnings": _quantization_info.warnings,
        }
    return result


@app.get("/safety/info")
async def safety_info(api_key: str = Depends(verify_api_key)):
    from guardrails import _INPUT_PATTERNS, _OUTPUT_PATTERNS

    return {
        "input_filters": len(_INPUT_PATTERNS),
        "output_filters": len(_OUTPUT_PATTERNS),
        "degeneracy_check": "repeated_4_grams > 50% or < 8 words",
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "slm"
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: list[str] | str | None = None
    tools: list[dict] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict | None = "auto"


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage
    system_fingerprint: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: str | None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


# Legacy /v1/completions


class CompletionRequest(BaseModel):
    model: str = "slm"
    prompt: str | list[str] = Field(..., description="Prompt text or list of prompts")
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = Field(default=False)
    stop: list[str] | str | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: dict | None = None
    finish_reason: str


class CompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: CompletionResponseUsage


# Models


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert OpenAI-style messages into the SLM prompt format."""
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}" if msg.content else "System:")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}" if msg.content else "User:")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}" if msg.content else "Assistant:")
        elif msg.role == "function":
            parts.append(f"Function: {msg.content}" if msg.content else "Function:")
    return "\n".join(parts)


def apply_stop(text: str, stop: list[str] | str | None) -> str:
    if stop is None:
        return text
    if isinstance(stop, str):
        stop = [stop]
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
    return text


def encode_and_generate(
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    eos_token_id: int | None = 50256,
) -> tuple[str, int, int, float]:
    """Run model generation and return (text, prompt_tokens, new_tokens, latency)."""
    model, enc, _ = load_model()
    tokens = enc.encode(prompt_text)
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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc

    dt = time.perf_counter() - t0
    n_new = max(out.numel() - len(tokens), 0)
    full_text = enc.decode(out[0].tolist())
    completion = (
        full_text[len(prompt_text) :]
        if full_text.startswith(prompt_text)
        else full_text
    )
    return completion, len(tokens), n_new, dt


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    return ModelListResponse(
        data=[
            ModelInfo(
                id="slm",
                created=int(time.time()),
                owned_by="slm-project",
            )
        ]
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest, api_key: str = Depends(verify_api_key)):
    prompt_text = req.prompt[0] if isinstance(req.prompt, list) else req.prompt

    safety = check_input(prompt_text)
    if not safety.safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety.reason,
        )

    try:
        completion, prompt_tokens, n_new, dt = encode_and_generate(
            prompt_text=prompt_text,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    completion = apply_stop(completion, req.stop)

    GENERATION_TOKENS.labels(endpoint="/v1/completions").inc(n_new)
    REQUEST_COUNT.labels(
        endpoint="/v1/completions", http_status="200", safe="true"
    ).inc()

    return CompletionResponse(
        model=req.model,
        choices=[
            CompletionChoice(
                text=completion,
                index=i,
                finish_reason="length" if n_new >= req.max_tokens else "stop",
            )
            for i in range(req.n)
        ],
        usage=CompletionResponseUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=n_new,
            total_tokens=prompt_tokens + n_new,
        ),
    )


@app.post("/v1/completions/stream")
async def completions_stream(
    req: CompletionRequest, api_key: str = Depends(verify_api_key)
):
    prompt_text = req.prompt[0] if isinstance(req.prompt, list) else req.prompt

    safety = check_input(prompt_text)
    if not safety.safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety.reason,
        )

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            completion, prompt_tokens, n_new, dt = encode_and_generate(
                prompt_text=prompt_text,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        except FileNotFoundError:
            yield f"data: {json.dumps({'error': 'model not found'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        completion = apply_stop(completion, req.stop)
        chunk_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        chunk_size = max(1, len(completion) // 8)
        for i in range(0, len(completion), chunk_size):
            chunk_text = completion[i : i + chunk_size]
            resp = ChatCompletionStreamResponse(
                id=chunk_id,
                created=created,
                model=req.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatMessage(role="assistant", content=chunk_text),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {resp.model_dump_json()}\n\n"

        resp = ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {resp.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest, api_key: str = Depends(verify_api_key)
):
    # Check if tools are present and tool_choice requests a call
    tool_call_result = _maybe_handle_tool_call(req)
    if tool_call_result is not None:
        return tool_call_result

    # Flatten messages into a single prompt
    prompt_text = format_messages_to_prompt(req.messages)

    safety = check_input(prompt_text)
    if not safety.safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety.reason,
        )

    try:
        completion, prompt_tokens, n_new, dt = encode_and_generate(
            prompt_text=prompt_text,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    completion = apply_stop(completion, req.stop)
    degenerate = is_degenerate(completion)

    if degenerate:
        finish_reason = "content_filter"
    elif n_new >= req.max_tokens:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    GENERATION_TOKENS.labels(endpoint="/v1/chat/completions").inc(n_new)
    out_safety = check_output(completion)
    REQUEST_COUNT.labels(
        endpoint="/v1/chat/completions",
        http_status="200",
        safe=str(out_safety.safe).lower(),
    ).inc()

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=completion),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionResponseUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=n_new,
            total_tokens=prompt_tokens + n_new,
        ),
    )


def _maybe_handle_tool_call(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | None:
    """If tools are present and tool_choice is not 'none', return a mock tool call."""
    if not req.tools:
        return None
    if req.tool_choice == "none":
        return None
    if req.tool_choice == "required":
        return _mock_tool_call(req)
    # tool_choice == "auto" or a specific tool dict
    # For mock purposes, always call the first available tool
    return _mock_tool_call(req)


def _mock_tool_call(req: ChatCompletionRequest) -> ChatCompletionResponse:
    tool = (
        req.tools[0]
        if req.tools
        else {"type": "function", "function": {"name": "mock_tool"}}
    )
    fn = tool.get("function", {})
    tool_name = fn.get("name", "mock_tool")
    tool_args = json.dumps({"query": "mock_result"})

    call_id = f"call_{uuid.uuid4().hex}"

    message = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args,
                },
            }
        ],
    )

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=message,
                finish_reason="tool_calls",
            )
        ],
        usage=ChatCompletionResponseUsage(
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
        ),
    )


@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(
    req: ChatCompletionRequest, api_key: str = Depends(verify_api_key)
):
    async def event_stream() -> AsyncGenerator[str, None]:
        # Check for tool call
        tool_call_result = _maybe_handle_tool_call(req)
        if tool_call_result is not None:
            # Stream the tool call as chunks
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            yield f"data: {ChatCompletionStreamResponse(id=chunk_id, created=created, model=req.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatMessage(role='assistant', content=None), finish_reason=None)]).model_dump_json()}\n\n"
            yield f"data: {ChatCompletionStreamResponse(id=chunk_id, created=created, model=req.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatMessage(role='assistant', content=None), finish_reason='tool_calls')]).model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        prompt_text = format_messages_to_prompt(req.messages)

        safety = check_input(prompt_text)
        if not safety.safe:
            yield f"data: {json.dumps({'error': 'unsafe prompt'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            completion, prompt_tokens, n_new, dt = encode_and_generate(
                prompt_text=prompt_text,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        except FileNotFoundError:
            yield f"data: {json.dumps({'error': 'model not found'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        completion = apply_stop(completion, req.stop)
        degenerate = is_degenerate(completion)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        finish_reason = (
            "content_filter"
            if degenerate
            else ("length" if n_new >= req.max_tokens else "stop")
        )

        yield f"data: {ChatCompletionStreamResponse(id=chunk_id, created=created, model=req.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatMessage(role='assistant', content=None), finish_reason=None)]).model_dump_json()}\n\n"

        chunk_size = max(1, len(completion) // 8)
        for i in range(0, len(completion), chunk_size):
            chunk_text = completion[i : i + chunk_size]
            yield f"data: {ChatCompletionStreamResponse(id=chunk_id, created=created, model=req.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatMessage(role='assistant', content=chunk_text), finish_reason=None)]).model_dump_json()}\n\n"

        yield f"data: {ChatCompletionStreamResponse(id=chunk_id, created=created, model=req.model, choices=[ChatCompletionStreamChoice(index=0, delta=ChatMessage(role='assistant', content=''), finish_reason=finish_reason)]).model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
