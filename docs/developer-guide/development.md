# Development Setup

1) Create a virtual environment and install dev deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2) Lint and format:

```bash
ruff check .
black .
```

3) Run tests with coverage:

```bash
pytest --cov=pyzaplineplus --cov-report=term-missing
```

