# Release Process

See the root `RELEASING.md` for the authoritative steps.

Summary:
- Bump version in `pyproject.toml` and `pyzaplineplus/_version.py`
- Update `CHANGELOG.md`
- Run tests + coverage and build docs
- Build distribution and check with Twine
- Tag a GitHub release (vX.Y.Z) to trigger publish

