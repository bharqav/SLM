## Contributing

Thanks for improving Emotional SLM.

## Development setup

1. Create env file:
   - Windows: `copy .env.example .env`
   - Unix: `cp .env.example .env`
2. Install backend dependencies:
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`
   - Optional tooling: `pip install -r requirements-dev.txt`
3. Install frontend dependencies:
   - `cd frontend && npm ci`
4. Install git hooks:
   - `pre-commit install`
   - `pre-commit install --hook-type pre-push`

## Local quality checks

Before opening a PR, run:

- `pytest -q`
- `cd frontend && npm run lint && npm run build`
- `pip-audit -r requirements.txt`
- `bandit -q -ll -r backend -x tests`
- `cd frontend && npm audit --audit-level=critical --omit=dev`

If Docker is preferred:

- `docker compose up --build`

## Pull request checklist

- Keep scope focused and explain the user impact.
- Add or update tests for behavior changes.
- Update docs (`README.md` or files in `docs/`) for API/ops changes.
- Ensure CI passes.
- Avoid committing secrets, credentials, or local env files.
