# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is [SemVer](https://semver.org/).

## [0.1.0] - 2026-10-24

### Added

- FastAPI backend: JWT auth with single-use refresh-token rotation and replay detection, signup consent gating (`accept_terms`, `confirm_age_18`), admin/staff role bootstrap, GDPR export/delete, Prometheus `/metrics`, structured JSON logging with request IDs.
- Safety-first chat pipeline (`backend/services/chat_pipeline.py`): crisis triage levels (L0–L4) with fixed human-authored replies for crisis-critical turns, prompt-injection and PII heuristics, optional external safety classifiers that fail closed, output-safety gate that replaces violating completions, `transparency` annotations on every turn.
- Model gateway (`backend/services/model_gateway/`): tiered fallback chain — canary/fast/quality traffic shaping → primary → fallback → opt-in locally served fine-tune → opt-in in-process local artifact → canned stub — with per-tier latency metrics, bounded timeouts/attempts, and a circuit breaker.
- Training track `slm-ft/`: QLoRA fine-tune of Qwen2.5-3B-Instruct on EmpatheticDialogues + ESConv (stage-1 val loss 2.39), stage-2 behavioral-policy steering with a 5-probe behavioral gate, merge to fp16, local model server, and an unattended training/resume guard.
- Training track `slm-training/`: from-scratch 81.6M-param GPT trained on code prose (val ppl 19.7) and on empathetic dialogues (val ppl 7.0).
- Next.js frontend: login and chat UI with React Query + Zustand state layer.
- Governance, incident-response, AI architecture, deployment, and training runbook docs.
- CI: pytest matrix (3.11/3.12), pip-audit, bandit, frontend lint/build, npm audit.
