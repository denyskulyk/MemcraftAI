# Build rules for agents (2026-11-23)

Shared rules for everyone building MemcraftAI. Read the README, `src/memcraft/models.py`
and the code your task touches before you start. Run `git log --oneline` to see what exists.

- Python 3.15 venv in `.venv`. Postgres 18 runs in Podman. Run database tests only through
  `pwsh -NoProfile -ExecutionPolicy Bypass -File .\deploy\test.ps1 -Database <yourname>_test <files>`.
  Never print the password from `.env`.
- The app is LIVE now: Denys and friends use the real database `memcraft`, the web container
  (127.0.0.1:8000) and the worker (`deploy\worker.ps1`). Do not restart them, do not change
  their data, and do not write to `memcraft` except where your task says so.
- Touch only the files your task lists. `models.py`, `migrations/`, `pyproject.toml`, `web/app.py`
  and `config.py` belong to the lead: ask for exact lines instead of editing them.
- No commits, no pushes, no new Python dependencies without asking the lead.
- `e0-spike/` is read-only reference material. Denys's photos and outputs stay out of the repo,
  which is public. Use the AI stand-in photos in `e0-spike/photos-ai` for tests.
- Never queue work on Denys's own ComfyUI (127.0.0.1:8188) and never stop or restart it,
  llama-swap (127.0.0.1:8080) or Open WebUI. Do not change their configs, `.wslconfig`,
  scheduled tasks or Tailscale.
- GPU, one job at a time: (1) create `C:\Source\MemcraftAI\data\gpu.lock` exclusively (mode "x");
  if it exists, wait and retry every 30 s. (2) The live worker shares the GPU: only start when
  no job in `memcraft` is idea_running, rendering or releasing, then set `settings.paused` to
  true in `memcraft` (remember the old value), wait until no job runs, do your GPU work on the
  app's ComfyUI 127.0.0.1:8199, and put `paused` back to its old value. (3) Delete the lock,
  also on failure. Keep each GPU session under 20 minutes.
- RAM is tight. Run only what you need.
- `ruff check .` and `ruff format --check` must pass for your files. No mention of any AI
  assistant in code, comments or file names.
- Report: files, public API, verbatim test summary, what is untested, lines you need from the lead.
