# Emotional SLM — Governance, Safety, and Operations

This document defines policies, targets, and processes for production readiness. It is **not legal or medical advice**; counsel and clinical advisors should review before public launch in your jurisdiction.

---

## 1. Safety and Crisis Escalation Policy

### 1.1 Red-line intents (taxonomy)

| Category | Examples (non-exhaustive) | Default severity |
|----------|---------------------------|------------------|
| **Self-harm / suicide** | Active plan, means, intent, goodbye messages | **Critical** |
| **Harm to others** | Credible threats, stalking, weapons, violence planning | **Critical** |
| **Abuse / exploitation** | CSA, trafficking, intimate partner violence with ongoing danger | **Critical–High** |
| **Psychosis / loss of reality testing** | Command hallucinations, dangerous delusions driving action | **High** |
| **Medical emergency** | Stroke, overdose symptoms, inability to breathe, loss of consciousness | **Critical** |

Severity feeds **routing**, not diagnosis. Models must not label users clinically.

### 1.2 Assistant behavior by level

| Level | Goals | Assistant behavior |
|-------|--------|-------------------|
| **L0 — Routine** | Supportive conversation | Normal empathy; no crisis script unless signals appear. |
| **L1 — Distress (no imminent risk)** | De-escalation, grounding | Validate feelings; short sentences; offer coping framing; **no** medical directives. |
| **L2 — Elevated risk (ambiguous)** | Safety-first, resources | Encourage reaching trusted humans; offer **region-appropriate** hotlines; avoid problem-solving that could increase risk. |
| **L3 — Imminent / critical** | Protect life, limit harm | **Do not** provide methods, locations, or encouragement for self-harm or violence. Provide **immediate** crisis numbers and emergency directive (e.g. call local emergency services). Refuse unsafe requests clearly. |
| **L4 — Human review (optional product mode)** | Oversight | Queue conversation (metadata + redacted excerpts) for **staff review** per internal policy; user notified only if policy requires. |

### 1.3 Regional crisis resources

- **Best-effort multilingual L3 detection:** crisis regexes additionally cover German and French imminent-risk phrasing (e.g. "Ich will sterben", "Je veux mourir", "Ich kann nicht atmen", "Je ne peux pas respirer"). Reliable non-English classification still requires the `SAFETY_INPUT_HF_URL` classifier (roadmap).

- **Primary signal:** user profile `region` / `country` (ISO 3166-1 alpha-2), browser locale (untrusted hint), or explicit selection during onboarding.
- **Mapping:** maintain a versioned **`crisis_resources.json`** (or DB table) keyed by region: name, phone, hours, URL, chat/SMS, minors-specific lines where applicable.
- **Unknown region:** show **global fallback**: link to international directories (e.g. findahelpline.com) + “contact local emergency services (e.g. 112/911/999)” + encourage a trusted person.
- **Localization:** hotline strings must be reviewed by native speakers; never fabricate numbers.

### 1.4 Escalation to human review (when enabled)

- **Triggers:** repeated L3 signals, L4 policy hits, abuse reports, admin flags, random sample for quality.
- **Data minimization:** store **ticket id**, timestamps, severity, **hashed user id**, and **redacted** snippets—not full transcripts unless legally required.
- **SLA:** internal triage within **24h** (business hours) for Sev1; backlog reviewed weekly.

---

## 2. Clinical and Legal Boundaries

### 2.1 Product scope

- **In scope:** emotional support, psychoeducation-style reflection, general wellness coping ideas (non-clinical), conversation memory (if user opts in).
- **Out of scope:** diagnosis, treatment planning, medication advice, therapy modality delivery as a licensed professional, crisis counseling as a substitute for emergency services.

### 2.2 Mandatory disclaimers

- **First session / persistent footer:** “This is not a substitute for professional medical or mental health care or emergency services.”
- **Crisis:** “If you’re in immediate danger, contact local emergency services or a crisis line now.”
- **Model limitations:** “Responses may be inaccurate; do not rely on this service for safety-critical decisions.”

### 2.3 Age and eligibility

- **Default:** **18+** unless a dedicated **minor flow** exists with parental consent where required by law.
- Block or gate sign-up for underage users per jurisdiction; do not collect unnecessary child data.

### 2.4 Informed consent

- At signup: purpose of data use, memory opt-in, analytics opt-out where applicable, link to full policy.
- **Re-consent** when enabling long-term memory, sharing data with new subprocessors, or material model changes.

### 2.5 Prohibited use cases

- Jailbreaks, illegal content generation, harassment, non-consensual intimate imagery, instructions for weapons or harm.
- **Clinical:** no triage that replaces emergency services; no “diagnosis” outputs in UI copy.

### 2.6 Clinician review policy (optional enterprise)

- **Sampling:** periodic review of flagged + random samples by qualified reviewers under contract.
- **Documentation:** reviewer credentials, rubric, escalation path; store audit logs **separately** from app DB if required.

---

## 3. SLOs, SLAs, and Error Budgets

### 3.1 Availability SLO (monthly)

| Surface | SLO | Measurement |
|---------|-----|-------------|
| API (excluding planned maintenance) | **99.9%** | Success on `/health/ready` + synthetic `POST /api/v1/auth/login` (staging token) |
| Chat write path (`POST /api/v1/chat`) | **99.5%** | 2xx/4xx not 5xx; auth failures excluded from error budget numerator |

### 3.2 Latency SLOs (chat turn, server-side)

Targets are **p50 / p95 / p99** in ms; tune per deployment.

| Endpoint | p50 | p95 | p99 |
|----------|-----|-----|-----|
| `POST /api/v1/chat` (end-to-end server) | ≤400 | ≤1200 | ≤2000 |
| `GET /api/v1/chat/stream` (TTFB) | ≤150 | ≤400 | ≤800 |
| `WS` first token | ≤200 | ≤500 | ≤1000 |
| `GET /health/ready` | ≤50 | ≤100 | ≤200 |

*Inference-heavy paths may exclude upstream HF variance in a separate “provider SLO” dashboard.*

### 3.3 Monthly error budget

- **Budget:** **0.1%** bad minutes (or 43.2 min/month at 99.9%) for API availability.
- **Exhausted budget actions (in order):** (1) freeze non-critical releases; (2) reliability sprint; (3) disable **non-essential** features (e.g. long-context memory, experimental models); (4) scale out / reduce traffic via queue.

### 3.4 Quality SLOs (offline + online)

| Metric | Target | Notes |
|--------|--------|--------|
| Safety **precision** (flagged → true positive) | ≥ **0.85** on eval set | Tune to avoid over-blocking |
| Safety **recall** (missed harm) | ≥ **0.95** on critical self-harm set | Higher bar than general |
| Hallucination rate (fact-sensitive probes) | ≤ **X%** | Define per domain |
| Emotion label drift vs. baseline | KS / ECE within thresholds | Alert on weekly eval |

---

## 4. Model Governance and Eval Framework

### 4.1 Offline eval sets

- **Safety:** self-harm, violence, CSAM-adjacent refusals, jailbreaks, injection strings.
- **Empathy:** rubric-scored turns (human or LLM-judge with calibration).
- **Refusal quality:** correct refusals without preachiness; no unsafe leakage.
- **Factuality:** curated non-clinical facts (hotlines, geography-agnostic).
- **Bias:** demographic slices for uneven harm or stereotyping.

### 4.2 Gates

| Gate | When | Pass criteria |
|------|------|----------------|
| **Pre-release** | Before promoting model/prompt | Safety recall ≥ threshold; no regression > δ on empathy; latency within SLO on canary |
| **Post-release** | First 24–72h | Error rate, safety escalations, user reports within bounds |

**Enforced in CI:** the deterministic policy layer (crisis regex + local rule-based safety backstops) is replayed in every CI run against `tests/fixtures/safety_probes.json` with the release thresholds from `scripts/training/thresholds.json` (`tests/test_eval_gate.py`). Input safety is **fail-closed**: a configured external safety classifier that errors blocks the turn (log at ERROR) instead of silently passing; local heuristic backstops (prompt-injection and PII detection) run even when no external provider is configured.

### 4.3 Prompt and version registry

- Store **`prompt_id`**, **`model_id`**, **`backend_version`**, **`safety_ruleset_version`** with each release artifact.
- **Rollback:** one-click revert to last known-good pair; feature flag to disable new prompt without redeploy.

---

## 5. Threat Model and Security Controls (STRIDE-oriented)

### 5.1 Threats and mitigations

| Threat | Example | Controls |
|--------|---------|----------|
| **Spoofing** | Stolen JWT | Short access TTL, refresh rotation, device binding (optional), anomaly detection |
| **Tampering** | WS message injection | TLS, auth on WS, validate JSON schema, rate limits |
| **Repudiation** | Abuse denied | Audit logs for admin actions; append-only security logs |
| **Information disclosure** | Token in URL leaked | Prefer `Sec-WebSocket-Protocol` or short-lived ticket for WS; never log tokens |
| **Denial of service** | Chat flood | Redis rate limits, IP + user quotas, body size caps |
| **Elevation** | Staff API abuse | RBAC (`is_staff`), MFA for admin (production) |

### 5.2 Additional controls

- **Secrets:** rotation runbook; KMS or vault; no secrets in repo; CI masked vars.
- **mTLS / internal auth:** service-to-service JWT or mesh identity for future split services.
- **CSP / headers:** strict CSP on frontend; `SameSite` cookies if cookie auth added.
- **CSRF:** primarily token APIs—use **SameSite** + avoid cookie session for API unless designed for it.
- **Dependency scanning:** Dependabot/Snyk; block critical CVEs on main.

### 5.3 Security test cadence

| Activity | Frequency |
|----------|-----------|
| SAST (CodeQL, bandit) | Every PR / nightly |
| Container scan | On build |
| DAST (staging) | Monthly |
| Third-party pentest | **Annual** or before major enterprise launch |
| Tabletop incident response | **Quarterly** |

### 5.4 Incident response (summary)

- **Sev1:** on-call page; comms template; customer notice if data at risk within **72h** where GDPR applies.
- **Post-incident:** blameless retro, action items with owners.

---

## 6. Data Governance and Compliance Mapping

### 6.1 Retention (indicative — legal must confirm)

| Store | Retention | Notes |
|-------|-----------|--------|
| `users` | Life of account + **30d** after delete | Hard-delete or anonymize PII |
| `conversations` / `messages` | User-configurable max **24 months** default | Shorter if policy requires |
| `sessions` (refresh) | Until expiry + **7d** grace | Rotating refresh tokens |
| `feedback` | Life of account or **2y** | Anonymize after account deletion |
| Application logs | **30–90d** | Redact PII; JSON logs with request id |
| Backups | **RPO** policy (see §7) | Encrypted, access-controlled |

### 6.2 GDPR mapping (examples)

| Topic | Control |
|-------|---------|
| Lawful basis | Consent + contract for service; legitimate interest for security |
| DSAR | Export (`/gdpr/export`), delete (`/gdpr/account`), **30-day** SLA |
| DPIA | Required before high-risk processing |
| Audit trails | Admin actions logged; access to prod DB logged |

### 6.3 HIPAA-style claims

- **Do not market as HIPAA-compliant** unless BAA, BA safeguards, and architecture review are complete. Map controls similarly: encryption, access, audit, minimum necessary.

### 6.4 Encryption and logging

- **In transit:** TLS 1.2+ everywhere.
- **At rest:** DB disk encryption (provider-managed); app-level secrets in vault.
- **Access logging:** who queried production PII (break-glass only).

### 6.5 Data residency and transfers

- Declare **primary region** for DB and inference.
- Cross-border: **SCCs** / IDTA + TIAs as required; document subprocessor list.

---

## 7. Reliability and Disaster Recovery

### 7.1 RPO / RTO (initial targets)

| Component | RPO | RTO |
|-----------|-----|-----|
| PostgreSQL | ≤ **15 min** (PITR) | ≤ **4h** |
| Redis (cache) | Best-effort | ≤ **1h** (cold acceptable) |
| Object storage (if any) | ≤ **1h** | ≤ **4h** |

### 7.2 Restore testing

- **Quarterly** restore from backup to staging; run smoke tests.

### 7.3 Failover strategy

- **DB:** managed HA + automatic failover; app retries with backoff.
- **Redis:** degrade to **no cache** (emotion cache miss, rate limit strict mode or IP-only).
- **HF / inference:** circuit breaker, stub/degraded short replies, queue (future).

### 7.4 Chaos / drills

| Scenario | Expected behavior |
|----------|-------------------|
| Redis down | App stays up; cache miss; rate limit fallback |
| HF timeouts spike | User-visible “try again”; metrics alert |
| DB failover | Brief errors; clients retry idempotent reads |

### 7.5 Backups and restore

- **Backups:** daily encrypted PostgreSQL backups plus WAL/PITR if the managed provider supports it.
- **Restore tests:** monthly restore into staging, followed by a smoke test of auth, chat, and export/delete flows.
- **RPO/RTO alignment:** keep the operational backup plan consistent with the targets in [DEPLOYMENT.md](DEPLOYMENT.md).
- **Secrets:** never back up plaintext secrets in application logs or ad hoc dumps.

---

## 8. Capacity and Cost Guardrails

### 8.1 Concurrency model (example tiers)

| Tier | Concurrent sessions (target) | Notes |
|------|------------------------------|--------|
| Dev | 10 | Single instance |
| Prod MVP | 500–1000 | Horizontal API replicas |

**Saturation signals:** CPU >70% sustained, p95 latency SLO burn, HF 429/5xx rate.

### 8.2 Autoscaling

- Scale API on **CPU + request queue depth**; scale workers on **queue lag** if async inference is added.

### 8.3 Token and cost budgets

- **Per user / day:** max tokens or messages; soft cap with UX.
- **Per session:** max context length (aligned with `CHAT_HISTORY_LIMIT`).
- **Org tier:** separate budgets for enterprise keys.

### 8.4 Degraded mode

- Reduce `max_new_tokens`, disable memory retrieval, fall back to smaller model or canned empathetic lines when budget or provider is constrained.

---

## 9. API Contract and Versioning Rules

### 9.1 OpenAPI governance

- **Single source:** FastAPI OpenAPI at `/openapi.json`; publish per release.
- **Breaking changes:** only on **`/api/v2`** or new path; **90-day** deprecation notice for removals.

### 9.2 Idempotency

- **POST** chat: optional **`Idempotency-Key`** header; dedupe **24h** in Redis for same user+key.

### 9.3 Pagination

- List endpoints: `limit` (default 20, max 100), `cursor` opaque.

### 9.4 Error envelope

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests",
    "request_id": "uuid",
    "details": {}
  }
}
```

*(Align implementation gradually; document actual responses until unified.)*

### 9.5 Retry semantics

- **429/503:** retry with exponential backoff + jitter; **do not** retry non-idempotent POST without idempotency key.

### 9.6 WebSocket / SSE contract

- **WS:** `type`: `user_message` | `token` | `done` | `error`; client **reconnect** with new token; server does not guarantee replay—client should sync via REST history (future).
- **SSE:** same event types in `data:` lines; **Last-Event-ID** optional for replay (if implemented).

---

## 10. Testing Strategy Expansion

### 10.1 Pyramid targets (coverage goals, not line % alone)

| Layer | Target | Examples |
|-------|--------|----------|
| Unit | Core logic | Safety thresholds, prompt builders |
| Integration | DB + API | Auth, chat, GDPR |
| E2E | Playwright | Login → chat → feedback |
| Load | k6/Locust | p95 under target at N RPS |
| Chaos | Gremlin/manual | Redis kill |
| Security | ZAP/Burp on staging | AuthZ, injection |

### 10.2 Synthetic conversation suites

- Fixed transcripts for **tone** and **safety** regression; run in CI nightly.
- Golden files for **refusal** strings (no exact match—semantic similarity threshold).

### 10.3 Release criteria

- **Must-pass:** unit + integration + critical E2E; **no open Sev1/2** security issues.
- **Flaky tests:** quarantine **<5** runs to fix; no more than **1%** flaky rate in main.

---

## 11. Observability Deepening

### 11.1 Canonical tracing IDs

- **`X-Request-ID`** (already): propagate to logs; add **`traceparent`** (W3C) when OpenTelemetry is enabled.
- **Frontend:** generate **`client_request_id`** on each mutation; send header to API.

### 11.2 Alerts and paging (example)

| Sev | Condition | Page |
|-----|-----------|------|
| 1 | Error rate >5% 5m OR data breach signal | Immediate |
| 2 | p95 latency >2× SLO 15m | Business hours + on-call |
| 3 | Elevated 4xx | Next day |

### 11.3 Dashboards (user impact)

- **Successful chat turns / session**, **time-to-first-token**, **abandonment after error**, **safety block rate** (with false-positive sampling).

---

## 12. Product and UX Guardrails

### 12.1 Accessibility

- Target **WCAG 2.2 Level AA** for web: focus order, contrast, labels, motion reduction.
- Automated: axe in CI on critical routes; manual screen reader spot checks each release.

### 12.2 Transparency when constrained

- When a reply is **blocked or softened**, show a neutral line: “We keep responses in a safe range. If you’re in crisis, consider contacting a crisis line.” (Copy reviewed by clinical advisor.)

### 12.3 User controls

| Control | Behavior |
|---------|----------|
| **Memory** | Opt-in; clear memory button; respect GDPR delete |
| **Export / delete** | Use `/api/v1/gdpr/*`; mirror in UI |
| **Tone / safety** | Preset “gentle / direct” within safe bounds; never disable safety entirely |

---

## Implementation mapping (codebase)

| Policy area | Code / data |
|---------------|-------------|
| Crisis levels + regional resources | `backend/services/crisis_policy.py`, `backend/data/crisis_resources.json`; L3 short-circuits LLM in `backend/services/chat_pipeline.py` |
| Legal copy & consent | `GET /api/v1/legal/*`; signup requires `accept_terms`, `confirm_age_18` in `backend/schemas/auth.py` |
| User prefs (memory, tone, region) | `PATCH /api/v1/auth/me`; `memory_service` respects `memory_opt_in` |
| SLO reference JSON | `GET /api/v1/meta/slo`, `backend/data/slo_targets.json` |
| API error envelope + validation | `backend/main.py` exception handlers |
| Idempotency | `POST /api/v1/chat` + `Idempotency-Key`, `backend/core/idempotency.py` |
| Pagination | `GET /api/v1/conversations?limit&offset` |
| Tracing IDs | `backend/middleware/tracing.py`, `X-Client-Request-Id` from frontend `lib/api.ts` |
| Transparency UX | `ChatResponse.transparency`; chat UI shows notes when constrained |
| Incident playbook | `docs/INCIDENT_RESPONSE.md` |
| Synthetic regression cases | `tests/fixtures/synthetic_conversations.json`, `tests/test_governance.py` |

---

## Document control

| Version | Date | Author |
|---------|------|--------|
| 1.0 | 2026-04-12 | Engineering + Policy template |
| 1.1 | 2026-04-12 | Implementation mapping added |

Review **quarterly** or after incidents, model changes, or new jurisdictions.
