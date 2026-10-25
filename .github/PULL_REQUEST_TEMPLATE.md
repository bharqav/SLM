## Summary

<!-- What changed and why. Link the issue if there is one: Fixes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / refactor-adjacent cleanup
- [ ] Documentation
- [ ] Tests only
- [ ] CI / tooling

## Safety impact

Changes under `backend/services/` (safety pipeline, crisis policy, model
gateway) affect user-facing safety behavior.

- [ ] This PR does **not** change safety-relevant behavior
- [ ] This PR **does** change safety-relevant behavior, and I added or updated
      tests covering it

## Checklist

- [ ] `pytest -q` passes locally
- [ ] `cd frontend && npm run lint && npm run build` passes (if frontend changed)
- [ ] New/changed behavior is covered by tests
- [ ] Docs updated (`README.md` or `docs/`) if API or operations changed
- [ ] No secrets, credentials, `.env` files, model weights, or local paths added
