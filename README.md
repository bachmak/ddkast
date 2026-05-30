# ddkast

Energy consumption forecasting for critical infrastructure.

Built for the *Data-Driven Optimization* course challenge (SoSe 2026, AIT).
The immediate target is electricity load forecasting for Germany using ENTSO-E data.
The long-term target is water consumption prediction for pump optimisation — the same pipeline applies because the underlying problem is identical: predict a time series 24 hours ahead so that an operator can schedule resources.

## Quick Start

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — the only tool you need to install manually. Everything else (Python, dependencies, virtual environment) is managed by uv.
- **VS Code** with the recommended extensions (you will be prompted on first open).

### One-time setup

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd ddkast

# 2. Install all dependencies and create the virtual environment
uv sync --group dev

# 3. Install pre-commit hooks (runs ruff + pyright before every commit)
uv run pre-commit install

# 4. Create your .env file
cp .env.example .env
```

If you have an **ENTSO-E API key**, open `.env` and replace the placeholder.
If you don't have one yet, set a dummy value for now:

```
ENTSOE_API_KEY=dummy
```

### Verify the setup

```bash
uv run pytest   # all tests should pass
```

### Run the pipeline

In VS Code, go to `Run and Debug` and run `pipeline (all stages)`