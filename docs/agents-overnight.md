# Overnight build rules (2026-11-23)

Shared rules for everyone building MemcraftAI tonight. Read the README, `src/memcraft/models.py`
and the code your task touches before you start.

- Python 3.15 venv in `.venv`. Postgres 18 runs in Podman. Run database tests only through
  `pwsh -NoProfile -ExecutionPolicy Bypass -File .\deploy\test.ps1 -Database <yourname>_test <files>`.
  Never touch the real database `memcraft` and never print the password from `.env`.
- Touch only the files your task lists. `models.py`, `migrations/`, `pyproject.toml`, `web/app.py`
  and `config.py` belong to the lead: ask for exact lines instead of editing them.
- No commits, no pushes, no new Python dependencies without asking the lead.
- `e0-spike/` is read-only reference material: its scripts, prompts and results show what worked.
  Denys's photos and outputs stay out of the repo, which is public.
- Never queue work on Denys's own ComfyUI (127.0.0.1:8188) and never stop or restart it, llama-swap
  (127.0.0.1:8080) or Open WebUI. Do not change their configs, `.wslconfig`, scheduled tasks or
  Tailscale. The app's own ComfyUI is the second instance on 127.0.0.1:8199 (`deploy/comfyui.ps1`).
- GPU: one job at a time. Before any GPU work, create `C:\Source\MemcraftAI\data\gpu.lock`
  exclusively (open with mode "x"); if it exists, wait and retry every 30 s. Delete it when done,
  also on failure. Unload models after your run (`POST /free` on 8199).
- RAM is tight: about 25 GB free while one big model is loaded. Run only what you need.
- `ruff check .` and `ruff format --check` must pass for your files. No mention of any AI
  assistant in code, comments or file names.
- Report: files, public API, verbatim test summary, what is untested, lines you need from the lead.
