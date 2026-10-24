<div align="center">

# Emotional SLM

### A safety-first empathetic chat platform with a trained small language model

**FastAPI · Crisis triage (L0–L4) · Graceful-degradation model gateway · QLoRA fine-tuning · GDPR + observability**

[![CI](https://github.com/bharqav/emotional-slm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bharqav/emotional-slm/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-compose-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/pytest-63%20passing-brightgreen)](tests/)

</div>

---

Emotional SLM is a full-stack platform for empathetic conversational support: a FastAPI backend that wraps every model call in a staged safety pipeline, a tiered **model gateway** that degrades gracefully instead of crashing, and a **QLoRA fine-tuned Qwen2.5-3B** trained on empathetic dialogues with a behavioral gate — plus a from-scratch mini-GPT training track used to exercise in-process inference.

> **Why build this?** Chat apps that touch mental-health-adjacent topics fail in one of two ways: they let the model improvise on crisis messages, or they crash when a model provider goes down. This project treats the safety pipeline and the fallback chain as first-class, tested components — not as disclaimers in a README.

**What it is NOT:** this is not a deployed medical or crisis service. Trained weights are not committed to the repo, and the fine-tuned model is non-commercial (dataset licenses). Crisis-level turns receive fixed human-authored replies — the model never improvises them.

## Architecture

Every turn passes through a staged safety pipeline *before* and *after* inference. The gateway picks the best available tier and fails downward — it never forwards a request to a tier that has not passed its health check.

```mermaid
flowchart TB
    subgraph client [Client]
        W[Next.js UI - login + chat]
    end

    subgraph api [FastAPI backend]
        AUTH["Auth - JWT access + refresh rotation"]
        CHAT["Chat API - SSE / WebSocket / Idempotency-Key"]
    end

    subgraph pipeline [Safety pipeline - chat_pipeline.py]
        CRI["Crisis triage L0-L4"]
        INS["Input gate - injection + PII"]
        EMO["Emotion inference - cached"]
        MEM["Memory retrieval - opt-in"]
        OUT["Output gate pre-stream"]
    end

    subgraph gw [Model gateway - model_gateway/]
        TIER["Tier select - canary / primary / served / local"]
        CB["Circuit breaker + bounded retries"]
    end

    subgraph infra [Infrastructure]
        PG[(PostgreSQL - Alembic)]
        RD[(Redis - rate limits + emotion cache)]
        PR[Prometheus /metrics]
    end

    W --> CHAT
    CHAT --> AUTH
    CHAT --> pipeline
    pipeline --> TIER
    TIER --> CB
    CHAT --> PG
    pipeline --> RD
    api --> PR
```

### Request Flow — one chat turn

```mermaid
sequenceDiagram
    participant C as Client WS/SSE
    participant A as FastAPI
    participant P as Safety pipeline
    participant G as Model gateway
    participant D as PostgreSQL

    C->>A: message {content}
    A->>A: verify JWT, rate-limit, cap length
    A->>P: run_turn(conversation, user)
    P->>P: crisis triage (rules, L0-L4)
    alt crisis level L3/L4
        P-->>A: fixed human-written reply (model never invoked)
    else normal turn
        P->>P: input gate (injection/PII/backstops)
        P->>P: emotion tokens + memory context
        G->>G: pick tier, circuit-breaker guard
        G->>G: primary -> fallback -> served -> local -> stub
        G-->>P: completion (stub fallback on total failure)
        P->>P: output gate before any byte streams
        Note over P: violating output replaced with canned refusal
        P->>D: persist message + safety metadata
        P-->>A: reply + transparency field
    end
    A-->>C: response / SSE events / WS frame
```

## Features

| Capability | Implementation |
|-----------|----------------|
| **Crisis triage** | Rule-based levels L0–L4 (`backend/services/crisis_policy.py`); L3+ returns a fixed reply — the model is never invoked |
| **Multi-layer safety gates** | Prompt-injection regex + PII detection heuristics always run; optional HF classifiers fail **closed** (`SAFETY_FAIL_CLOSED=true`) |
| **Tiered model gateway** | Canary/fast/quality split → primary → fallback → opt-in served fine-tune → opt-in local artifact → canned stub; circuit breaker + bounded attempts/timeouts |
| **Auth hardening** | Single-use refresh rotation; replaying a rotated token revokes all sessions; HS256 with algorithm allow-list; `JWT_SECRET` enforced in production |
| **Rate limiting** | Per-IP budgets on auth endpoints, per-user chat budget, `Retry-After` on 429, fail-open (logged) when Redis is down |
| **Idempotency** | Atomic `Idempotency-Key` reservation bound to request-body hash; concurrent reuse → `409` |
| **Streaming** | WebSocket (JWT `token` query param) + SSE; completions gated **before the first byte** reaches the client |
| **Observability** | Prometheus `slm_*` counters/histograms, per-tier latency attribution, structured JSON logs with `X-Request-ID` |
| **GDPR** | `GET /api/v1/gdpr/export` (full account export) + `DELETE /api/v1/gdpr/account` |
| **Memory (opt-in)** | `MEMORY_ENABLED=true`: similarity/recency-weighted retrieval with inline PII redaction |
| **Emotion** | Redis-cached emotion inference, injected as prompt tokens only (`EMOTION_CACHE_*`) |
| **Trained empathetic model** | QLoRA on Qwen2.5-3B-Instruct (EmpatheticDialogues + ESConv), stage-2 behavioral steering, 5/5 behavioral gate, merged fp16 served via `serve_model.py` |
| **From-scratch GPT track** | 81.6M-param GPT trained on code prose (val ppl 19.7) + empathetic variant (val ppl 7.0), wired as the local fallback tier |

## Project Status

> **Beta** — feature-complete at v0.1.0 and fully covered by its test suite (63 backend tests, security audits) — but single-node by design and not production-hardened. The Roadmap below states exactly what that would take.

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/bharqav/emotional-slm.git
cd emotional-slm
cp .env.example .env         # add JWT_SECRET for anything non-local
docker compose up --build
```

Frontend: `http://localhost:3000` · API docs: `http://localhost:8000/docs` — migrations run automatically on API startup.

```bash
# Sign up (consent gates are enforced server-side)
curl -s -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"Str0ngPassw0rd!","accept_terms":true,"confirm_age_18":true}'

# Log in and chat (token from /auth/login response)
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":null,"content":"feeling drained today"}'
```

Without any external model configured, the gateway serves deterministic canned replies from the stub tier — the full safety pipeline exercises end to end.

### Local development (no Docker API container)

```bash
docker compose up -d postgres redis
pip install -r requirements.txt           # in a venv
alembic upgrade head                      # with DATABASE_URL set in .env
uvicorn backend.main:app --reload --port 8000

cd frontend
npm ci
npm run dev                               # Next.js on :3000
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health`, `/health/live`, `/health/ready` | Liveness / readiness probes |
| `POST` | `/api/v1/auth/signup` | Requires `accept_terms` + `confirm_age_18` |
| `POST` | `/api/v1/auth/login` · `/refresh` · `/logout` | Single-use refresh rotation |
| `GET/PATCH` | `/api/v1/auth/me` | Preferences: memory opt-in, tone, region |
| `POST` | `/api/v1/chat` | Full safety pipeline; optional `Idempotency-Key` header |
| `GET` | `/api/v1/chat/stream` | SSE stream (conversation_id + message params) |
| `WS` | `/api/v1/chat/ws/{conversation_id}?token=<jwt>` | Authenticated streaming |
| `GET` | `/api/v1/conversations` | Paginated conversation list |
| `POST` | `/api/v1/feedback` | Per-message user feedback |
| `GET` | `/metrics` | Prometheus scrape (optionally token-gated) |
| `GET/POST` | `/api/v1/admin/*` | Staff-only administration |
| `GET/DELETE` | `/api/v1/gdpr/*` | Data export / account deletion |
| `GET` | `/api/v1/auth/oauth/status` | OAuth roadmap probe (returns `501` until implemented) |

Interactive OpenAPI docs are served at `/docs` when the API runs.

## Package Structure

```
backend/
├── api/v1/                 # auth, chat, conversations, admin, gdpr, legal, meta
├── core/                   # config, security, logging, metrics, rate limiting
├── db/                     # async SQLAlchemy session
├── middleware/             # request-id, auth middleware
├── models/                 # SQLAlchemy models
├── schemas/                # Pydantic schemas
└── services/
    ├── chat_pipeline.py    # staged turn orchestration (the heart)
    ├── crisis_policy.py    # crisis levels + multilingual detection + fixed replies
    ├── safety_service.py   # input/output gates, PII + injection heuristics
    ├── memory_service.py   # opt-in retrieval + PII redaction
    └── model_gateway/
        ├── gateway.py      # tiered fallback chain + circuit breaker
        ├── providers.py    # HTTP inference providers
        └── local_provider.py
docs/                       # architecture, governance, incident response, runbooks
slm-ft/                     # QLoRA track: Qwen2.5-3B empathetic fine-tune
slm-training/               # from-scratch GPT track (code-prose + empathetic)
alembic/                    # migrations
scripts/                    # load smoke, demo seeding, training utilities
tests/                      # 63 tests: API, security hardening, governance, data
```

## Configuration Reference

Key variables (full list in [.env.example](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://slm:slm@localhost:5432/slm` | Postgres |
| `REDIS_URL` | `redis://localhost:6379/0` | Sessions, rate limits, emotion cache |
| `JWT_SECRET` | "" | Required in production; dev falls back to per-process random |
| `INFERENCE_PROVIDER` | `stub` | Canned replies end-to-end |
| `HF_API_TOKEN` / `HF_MODEL_URL` | "" | Primary cloud model tier |
| `HF_FALLBACK_MODEL_URL` | "" | Cloud fallback tier |
| `SERVED_MODEL_URL` | "" | Opt-in: locally served fine-tune (`slm-ft/serve_model.py`) |
| `LOCAL_TORCH_MODEL_ENABLED` | `false` | Opt-in: in-process local GPT artifact |
| `SAFETY_FAIL_CLOSED` | `true` | External classifier outage blocks instead of passing |
| `LOGIN_RATE_LIMIT_PER_MINUTE` | `10` | Per-IP auth-endpoint budget |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Per-user chat budget |
| `WS_MESSAGE_MAX_CHARS` | `32000` | WebSocket message cap |
| `MEMORY_ENABLED` | `false` | Long-term memory retrieval (PII redacted) |
| `METRICS_BEARER_TOKEN` | "" | Optional `/metrics` gating |
| `CANARY_TRAFFIC_PERCENT` | `0` | Gateway canary split |
| `TRUST_PROXY_X_FORWARDED_FOR` | `false` | Enable only behind a trusted proxy |

Never commit secrets — configure via environment only.

## Building & Testing

```bash
pip install -r requirements.txt
pytest -q                                        # 63 tests, SQLite-backed, no Docker needed

pip install -r requirements-dev.txt
bandit -q -ll -r backend -x tests                # static security scan
pip-audit -r requirements.txt                    # dependency audit
python scripts/check_links.py                    # documentation link check

cd frontend && npm ci
npm run lint && npm run build
npm audit --audit-level=critical --omit=dev
```

All of the above runs in CI (`.github/workflows/ci.yml`) on Python 3.11 + 3.12.

## Training

Two independent training tracks, both reproducible from the repo:

| Track | Model | Result | Artifacts |
|-------|-------|--------|-----------|
| [`slm-ft/`](slm-ft/README.md) | Qwen2.5-3B QLoRA | stage-1 val loss **2.39**; behavioral gate **5/5**; merged fp16 | weights (non-committed) |
| [`slm-training/`](slm-training/README.md) | 81.6M GPT, from scratch | val ppl **19.7** (code prose) / **7.0** (empathetic) | `best_model.pt` |

```bash
# Reproduce the QLoRA pipeline (GPU required)
cd slm-ft
.venv/Scripts/python prepare_data.py
.venv/Scripts/python train_qlora.py --data data/sft_train.jsonl ...
```

Full procedures, hardware notes, and stage-2 steering design: [docs/TRAINING_RUNBOOK.md](docs/TRAINING_RUNBOOK.md).

**Metrics note:** the numbers above are measured results from the documented runs (TensorBoard-equivalent logs in each track's README). No external benchmarks or production traffic numbers exist — none are claimed.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md) | Model tiers, fallback chain, training tracks |
| [docs/GOVERNANCE_AND_OPERATIONS.md](docs/GOVERNANCE_AND_OPERATIONS.md) | Safety policy, SLOs, compliance, threat model |
| [docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md) | Crisis-classification incident handling |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |
| [docs/TRAINING_RUNBOOK.md](docs/TRAINING_RUNBOOK.md) | Both training tracks end-to-end |
| [QUALITY_CHECKLIST.md](QUALITY_CHECKLIST.md) | Release quality gate |

## Roadmap

**Implemented** — safety pipeline, gateway fallback chain, auth hardening, GDPR, observability, both training tracks.

**Planned**
- OAuth provider login (routes currently return `501`)
- Multi-node production deployment guide (currently single-node)

**Potential**
- Feedback-driven preference tuning (feedback endpoint exists, no training loop yet)
- Streaming-first gateway tiers per-model

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) — the short version: fork, branch, `pytest -q` + lint locally, open a PR; safety-affecting changes need matching tests. See also [SECURITY.md](SECURITY.md) for vulnerability reporting and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE). Trained model weights inherit the **non-commercial** restrictions of their training datasets (EmpatheticDialogues: CC BY-NC 4.0; ESConv: CC BY-NC-SA 4.0).

## Acknowledgements

- Qwen team — [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (base model for the empathetic SLM)
- Rashkin et al. — [Towards Empathetic Open-domain Conversation Models](https://arxiv.org/abs/1811.00207) (EmpatheticDialogues) · Liu et al. — [Towards Emotional Support Dialog Systems](https://arxiv.org/abs/2106.01144) (ESConv)
- FastAPI & Next.js communities — the scaffolding this platform stands on
