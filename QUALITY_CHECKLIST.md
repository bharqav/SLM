# 10/10 Project Checklist

Use this checklist to keep the repository production-ready.

## Reliability

- [ ] Health endpoints are green (`/health`, `/health/live`, `/health/ready`).
- [ ] Critical workflows are covered by tests.
- [ ] Incidents and runbooks are updated in `docs/`.

## Code quality

- [ ] Changes are small, focused, and reversible.
- [ ] Public APIs have clear request/response contracts.
- [ ] Backward compatibility is considered for existing clients.

## Security and privacy

- [ ] No secrets committed.
- [ ] User input validation and auth checks were reviewed.
- [ ] GDPR flows still work (`/api/v1/gdpr/export`, `/api/v1/gdpr/account`).

## Operations

- [ ] Metrics remain intact (`/metrics`, `slm_*` telemetry).
- [ ] Logs remain structured and actionable.
- [ ] Rollout risk is assessed (canary, feature flags, or staged deploy).

## Delivery

- [ ] `pytest -q` passes locally.
- [ ] Frontend lint/build passes.
- [ ] CI is green.
