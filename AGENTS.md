# Repository Guidelines

## Project Structure & Modules
- Root: Python mono-repo managed by `uv` with workspace members.
- Core library: `livekit-agents/livekit/…`
- Plugins: `livekit-plugins/*` (provider integrations)
- Examples: `examples/…` (runnable agent samples)
- Tests: `tests/` (plus package-local tests where present)
- Utility/Docs: `utils/`, repo docs like `README.md`, guides in root.

## Build, Test, and Dev Commands
- Setup env: `uv sync` (installs workspace, dev deps from `pyproject.toml`)
- Lint: `uv run ruff check .` (static checks) • Format: `uv run ruff format .`
- Types: `uv run mypy .` (strict by default; add types for new code)
- Tests: `uv run pytest -q` (root tests; examples are ignored by default)
- Run example: `uv run python examples/voice_agents/basic_agent.py console`

## Coding Style & Naming
- Python 3.9+; 4-space indent; max line length 100.
- Use type hints everywhere; prefer explicit `TypedDict`/`Protocol` over `Any`.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`.
- Imports sorted by Ruff isort; group stdlib/third-party/first-party.
- Docstrings follow Google style (see `tool.ruff.lint.pydocstyle`).

## Testing Guidelines
- Framework: `pytest` (async via `pytest-asyncio`); place tests in `tests/` or nearest package `tests/`.
- Name tests `test_*.py`, functions `test_*`; use fixtures over globals.
- Write unit tests for new behavior; include negative paths and async variants where relevant.
- Run selectively: `uv run pytest tests/test_module.py -k some_case`.

## Commits & Pull Requests
- Commits: concise, present tense; use prefixes when helpful (`feat:`, `fix:`, `docs:`, `refactor:`) and reference issues/PRs (`#123`).
- PRs: clear description, scope narrowly; include tests, docs updates, and any required ENV details.
- CI expectations: Ruff clean, formatted, typed (`mypy`), and tests passing.

## Security & Configuration
- Keep secrets in environment or `.env` (loaded via `python-dotenv`); never commit keys.
- Common env for examples: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (plus model provider keys as needed).
- For production agents, prefer `dev`/`start` entrypoints as shown in `README.md`.

# docs: https://docs.livekit.io/llms.txt