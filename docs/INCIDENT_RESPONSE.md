# Incident response (abbreviated playbook)

## Severity

| Sev | Examples | Initial response |
|-----|----------|------------------|
| 1 | Data breach, auth bypass in prod, widespread outage | Page on-call; preserve logs; begin containment |
| 2 | Partial outage, safety regression spike | Triage within 1h; rollback candidate |
| 3 | Single-tenant issue, non-critical bug | Next business day unless security-related |

## Steps

1. **Contain:** revoke keys if leaked; block abusive IPs; disable feature flags (e.g. `DEGRADED_MODE=true`).
2. **Communicate:** internal status channel; customer notice if personal data at risk (GDPR: without undue delay).
3. **Eradicate:** patch root cause; rotate secrets via KMS/vault runbook.
4. **Recover:** restore from backup if needed; verify `/health/ready`.
5. **Retro:** blameless postmortem; track actions in issue tracker.

## Contacts

Define `ONCALL_EMAIL` / PagerDuty in your deployment environment (not committed here).
