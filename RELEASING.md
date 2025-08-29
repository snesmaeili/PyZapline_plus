Releasing PyZaplinePlus
=======================

This document describes how to cut a release and publish to PyPI using Trusted Publishing.

Prerequisites
- Permissions to create GitHub Releases on this repo
- PyPI project configured for Trusted Publishing (OIDC)

Versioning
- Follow SemVer (MAJOR.MINOR.PATCH)
- Update both:
  - pyproject.toml: [project].version
  - pyzaplineplus/_version.py: __version__
- Update CHANGELOG.md accordingly

Release Steps
1) Create a release branch
   git checkout -b release/vX.Y.Z

2) Bump version
   - Edit pyproject.toml and pyzaplineplus/_version.py
   - Update CHANGELOG.md

3) Run tests and coverage locally
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   pytest -q

4) Build docs locally (optional)
   mkdocs build

5) Build distribution
   python -m build
   twine check dist/*

6) Open PR, get approvals, merge to main

7) Create a GitHub Release tag
   - Tag format: vX.Y.Z
   - Title: X.Y.Z
   - Description: copy from CHANGELOG

8) Wait for GitHub Actions
   - The Publish workflow builds the package and uploads to PyPI

Post-Release
- Create a new "Unreleased" section in CHANGELOG.md
- Bump dev version if needed (e.g., X.Y.(Z+1)-dev)

Troubleshooting
- If PyPI publish fails, verify:
  - Release tag is correct (vX.Y.Z)
  - Trusted Publishing is enabled for the PyPI project
  - The publish workflow has id-token: write permissions

