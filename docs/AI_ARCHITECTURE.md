# Product-level AI architecture

This document maps **production** concerns to this repository: model gateway, fallbacks, memory/RAG, jobs, observability, release governance, security, and frontend resilience.

## 1. Managed model serving

**Current code path:** `backend/services/model_gateway/` — all text generation flows through **`generate_full`** / **`generate_stream`**.

| Provider style | Configuration |
|----------------|----------------|
| Hugging Face Inference API (HTTP JSON) | `HF_API_TOKEN` + `HF_MODEL_URL` |
| Primary + backup chain | `HF_FALLBACK_MODEL_URL` + `HF_FALLBACK_MODEL_NAME` |
| Canary model slice | `HF_CANARY_MODEL_URL` + `CANARY_TRAFFIC_PERCENT` (hash-stable per user) |
| Stub (dev) | No token/URL → empathetic canned reply |
| Degraded safe mode | `DEGRADED_MODE=true` → short safe message |

**To integrate** Replicate, Together, Azure OpenAI, or vLLM/TGI: add a small adapter class that implements the same contract as `hf_generate` (prompt in → text out) and register it in `gateway.py` behind a `MANAGED_PROVIDER` enum.

## 2. Fallback chain

Order: **primary** → (circuit breaker) → **fallback URL** → **stub**. Metrics: `slm_gateway_fallback_total{reason}`, `slm_gateway_circuit_breaker_open_total`.

## 3. Model gateway responsibilities

| Concern | Location |
|---------|----------|
| Prompt templating / versioning | `backend/data/prompt_templates/v*.json`, `PROMPT_VERSION` |
| Retries | `providers.hf_generate` (tenacity) |
| Circuit breaker | `model_gateway/circuit.py` (Redis) |
| Token accounting | `slm_inference_tokens_total` |
| Safety policy | Still in `safety_service` + crisis policy; gateway does **not** bypass safety |
| Local artifact tier | `backend/services/model_gateway/local_provider.py` — opt-in (`LOCAL_TORCH_MODEL_ENABLED=true`) in-process serving of `slm-training/best_model.pt`. Two artifacts exist: code-prose GPT (val ppl 19.7) and an empathetic-dialogue variant (`empathetic_run/`, val ppl 7.0, 2000 steps) — both toy-scale, keep disabled for real chat. **Empathetic SLM (main-tier ready)**: Qwen2.5-3B QLoRA in `slm-ft/` — stage-1 (2 epochs, val 2.39) + stage-2 policy steering, **behavioral gate 5/5**, merged to `slm-ft/merged_fp16/`; serve via `slm-ft/serve_model.py` and point the backend's served tier at it with `SERVED_MODEL_URL` |

## 4. RAG / memory (production hardening)

| Topic | Status / next step |
|-------|-------------------|
| Embeddings | `memory_service.embed_text`; optional `HF_EMBEDDING_URL` |
| PII before embed | `memory_pii.redact_for_embedding` when `MEMORY_PI_REDACT=true` |
| Background indexing | Stub `backend/jobs/tasks.py` + `jobs/README.md` — enqueue from API |
| Retrieval eval | Offline: precision@k, groundedness — add eval scripts under `scripts/eval/` |
| TTL / deletion | `MEMORY_CHUNK_TTL_DAYS` (enforce via periodic job); user delete cascades `memory_chunks` |

## 5. Async jobs

See **`backend/jobs/`**. Use **RQ**, **Celery**, or **Temporal** against Redis; keep chat synchronous except enqueue-after-commit.

## 6. AI observability

| Signal | Mechanism |
|--------|-----------|
| Per-turn model/prompt/safety version | `Message.moderation_flags["gateway"]`, `ChatResponse.transparency` |
| Tokens | Prometheus `slm_inference_tokens_total` |
| Latency | `slm_gateway_latency_seconds`, `slm_chat_turn_latency_seconds` |
| Fallback / safety | `slm_gateway_fallback_total`, `slm_safety_violation_flags_total` |

**Dashboards:** Grafana panels on fallback rate, timeout rate (derive from gateway logs), p95/p99 from histograms.

## 7. Release registry & canary

- File: `backend/data/release_registry.json`
- API: `GET /api/v1/meta/releases`
- Runtime: `CANARY_TRAFFIC_PERCENT` + `HF_CANARY_MODEL_URL` for hashed user cohorts
- **Full** canary controller (5% → 25% → 100% with auto-rollback) belongs in the deployment plane (flag service or mesh), not only in-app.

## 8. Security & compliance

| Item | Implementation notes |
|------|---------------------|
| Encrypt transcripts at rest | Add app-level encryption (KMS-wrapped DEK) or **database TDE**; optional `Message.content` ciphertext column + migration |
| Admin RBAC | Staff gate (`is_staff`); extend with roles table if needed |
| Audit trail | `audit_logs` table + `audit_service` on admin + GDPR delete |
| DSAR / delete | `DELETE /api/v1/gdpr/account` + cascade; purge **application logs** and **vector** stores in jobs |

## 9. Frontend resilience

| Item | Target |
|------|--------|
| Stream retry | Exponential backoff + user-visible “Retry” |
| Model unavailable | Clear empty state when API returns 503/502 or gateway fallback stub |
| Draft recovery | `localStorage` key per conversation |
| Crisis copy a11y / i18n | `aria-live` for alerts; JSON locale files for hotline strings |

---

See also: `docs/TRAINING_RUNBOOK.md`, `docs/GOVERNANCE_AND_OPERATIONS.md`.
