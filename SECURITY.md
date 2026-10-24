# Security Policy

## Reporting a vulnerability

This project processes sensitive conversation content by design, so reports are welcome and taken seriously.

**Please do not open a public GitHub issue for a security vulnerability.**

Report privately via [GitHub Security Advisories](https://github.com/bharqav/emotional-slm/security/advisories/new) ("Report a vulnerability").

Include:

- affected endpoint/component (e.g. `backend/services/chat_pipeline.py`)
- reproduction steps or a minimal proof of concept
- expected vs. actual behavior

## Response targets

- **Recognition:** within 7 days
- **Triage decision:** within 14 days
- **Patch or documented mitigation:** best effort within 30 days, guided by severity

This is a best-effort, volunteer-maintained project — targets are commitments to respond, not an SLA.

## Scope

In scope: authentication/session handling, the chat safety pipeline (`backend/services/`), model gateway, admin/staff authorization, GDPR endpoints, streaming (SSE/WebSocket) paths, and container configuration.

Out of scope: brute-force endpoints that are already rate-limited at the application layer (still reported on a best-effort basis), self-XSS in a single-user session, missing rate limits on endpoints behind Cloudflare-style upstream protections in hypothetical production deployments, and the *model's conversational output quality* (that is a product concern, not a security vulnerability).

## Supported versions

Only the latest `main` branch is supported. Trained model artifacts are never committed; if you believe a model weight file was exposed via a release asset, report it and treat the artifact as untrusted.
