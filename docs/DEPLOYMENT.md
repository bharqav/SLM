# Deployment Guide

This repository is ready for a small production deployment, but it still needs a repeatable release process, protected secrets, and a rollback plan.

## Recommended deployment shape

- Frontend: Vercel or a static Node container behind a CDN.
- Backend API: managed container service or a VM behind a reverse proxy.
- Database: managed PostgreSQL with backups and PITR enabled.
- Redis: managed Redis or a self-hosted instance with persistence disabled if it is only used for cache and rate limits.

## Environment variables

Set production values in your secret manager or deployment environment, not in the repo.

- `DATABASE_URL` with a production PostgreSQL connection string.
- `REDIS_URL` for cache, sessions, and rate limiting.
- `JWT_SECRET` as a long random secret.
- `CORS_ORIGINS` for allowed frontend domains.
- `HF_API_TOKEN` and `HF_MODEL_URL` if using Hugging Face inference.
- `SENTRY_DSN` for error reporting.
- `PROMPT_VERSION` and `SAFETY_RULESET_VERSION` for release tracking.
- `DEGRADED_MODE=true` if you want a safe fallback response when the model provider is unavailable.

## Release steps

1. Apply migrations in staging first: `alembic upgrade head`.
2. Run the governance and API tests: `pytest`.
3. Build the frontend and backend images.
4. Deploy to staging and verify login, chat, and crisis routing.
5. Promote to production only after smoke tests pass.
6. Keep the previous release available for rollback.

## Reverse proxy

Put the API behind Nginx, Caddy, or a managed ingress that terminates TLS.

- Redirect HTTP to HTTPS.
- Forward `X-Request-ID` and `X-Forwarded-*` headers.
- Set request timeouts long enough for streaming chat responses.
- Add security headers such as `Strict-Transport-Security`, `X-Frame-Options`, and `Content-Security-Policy`.

## Database safety

- Use managed backups and point-in-time recovery.
- Test restore on staging at least monthly.
- Keep the migration workflow one-directional in production: backup first, migrate second.
- Track the schema version at deploy time.

## Rollback checklist

- Revert the application deployment.
- Confirm the previous image tag is still available.
- Verify the database migration is backward compatible before applying destructive changes.
- Switch feature flags off before cutting traffic back over.

## Demo bootstrap

For student demos, seed a sample user and conversation with `python scripts/seed_demo.py` after the database is ready.