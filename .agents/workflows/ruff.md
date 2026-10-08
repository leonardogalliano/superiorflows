---
description: # Ruff Workflow
---


The agent must automatically perform the following steps to lint and format the codebase:

1. Run Ruff lint check:
   ```bash
   uv run ruff check
   ```

2. Run Ruff format check:
   ```bash
   uv run ruff format --check
   ```

3. If any issues are found:
   - For lint errors: Run `uv run ruff check --fix` to auto-fix safe problems. For any remaining errors, locate and fix them in the source files.
   - For formatting differences: Run `uv run ruff format` to automatically reformat the codebase.
   - Re-run the checks to confirm everything is clean.

4. Provide a clear summary of the status (e.g. "All checks passed!" or details of the fixes applied).
