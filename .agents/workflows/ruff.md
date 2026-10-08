---
description: Lint and format the codebase with Ruff
---

## Lint and format workflow

Run this after editing any Python files and before reporting a task as done:

1. Run the checks:
   ```bash
   uv run ruff check
   uv run ruff format --check
   ```

2. If any issues are found:
   - Lint errors: run `uv run ruff check --fix` for safe auto-fixes, then fix any remaining errors by hand.
   - Formatting differences: run `uv run ruff format`.
   - Re-run the checks to confirm everything is clean.

3. Report the status (e.g. "All checks passed!" or the fixes applied).
