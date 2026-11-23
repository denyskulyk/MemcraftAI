# MemcraftAI

A photo and an idea become a short funny 9:16 video, fully local. The plan lives in the
architecture review and the
backlog.

| Folder | Holds |
| --- | --- |
| `src/memcraft` | Python code: settings, database, schema (`models.py`), file store (`storage.py`), web app |
| `frontend` | Plain HTML, CSS and JavaScript served by the web app. No Node, no build step |
| `migrations` | Alembic migrations; `0001` is schema v1 |
| `workflows` | ComfyUI graphs with a manifest that maps names to nodes |
| `deploy` | Postgres container, autostart, migrate, backup and restore scripts |
| `tests` | `pytest` tests; those marked `postgres` need `MEMCRAFT_TEST_DATABASE_URL` |
| `data`, `backups` | Local files and dumps, never in git |
| `e0-spike` | Hand tests of the E0 spike, never in git |

Set up the Python environment once:

```powershell
py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Lint and test:

```powershell
.\.venv\Scripts\ruff.exe check .; .\.venv\Scripts\ruff.exe format --check .; .\.venv\Scripts\python.exe -m pytest -q
```

Run the web app, then open http://127.0.0.1:8000/health:

```powershell
.\.venv\Scripts\python.exe -m uvicorn memcraft.web.app:app --host 127.0.0.1 --port 8000
```

Postgres: see [deploy/README.md](deploy/README.md).

Two rules for file handling:

- Files are written and deleted only through `memcraft.storage`, so a row and its file go together.
- The web container sees only `MEMCRAFT_SHARED_DIR`. Keyframes, latents and clips stay in
  `MEMCRAFT_PRIVATE_DIR`, which the worker cleans up after a delete made in the web app.
