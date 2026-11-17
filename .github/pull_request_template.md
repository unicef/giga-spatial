<!--
Thanks for contributing to GigaSpatial!
Keep it short. Delete anything that doesn't apply.
-->

## Summary
<!-- What and why, in 2–4 sentences. Link the issue: Closes #123 -->

## Changes
<!-- Bullet the user-visible changes. For new or changed API, a short usage snippet helps. -->

## Test plan
<!-- Commands you ran and what you checked, e.g. `pytest tests/handlers -q`, or a script against real data. -->

## Impact
- [ ] Breaking change or rename (deprecation path / alias described above)
- [ ] Public API or lazy exports changed (`test_public_api.py` passes)
- [ ] New or changed dependency or extra (`pyproject.toml`, works on a clean install)
- [ ] None of the above

## Checklist
- [ ] `CHANGELOG.md` updated under Unreleased (or not user-facing)
- [ ] Docs updated (user guide / API reference / `mkdocs.yml` nav) if behavior changed
- [ ] New domain terms added to `GLOSSARY.md`; design decisions recorded as an ADR in `docs/adr/`
- [ ] No secrets in code, logs, errors, or `__repr__` (tokens masked, `.env_sample` updated)

### New handler? (delete if not)
- [ ] Config / Downloader / Reader follow the base handler pattern
- [ ] Data source license and attribution noted in the docstring and docs
- [ ] Credentials read from `Config` / `.env`, never hard-coded
- [ ] Tests or a reproducible example with a small area
