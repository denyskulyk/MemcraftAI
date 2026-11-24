r"""Stores the app's workflows, engines, recipes and prompts in the database once.

Run after migrate.ps1: .venv\Scripts\python.exe deploy\seed.py
Running it again changes nothing. New prompt texts arrive as candidates.
"""

from pathlib import Path

from memcraft.db import new_session
from memcraft.engines import register_defaults
from memcraft.models import Setting
from memcraft.pipeline.choices import default_critic
from memcraft.roles.critic import SETTING, choose
from memcraft.roles.seed import seed_prompts

ROOT = Path(__file__).resolve().parents[1]

with new_session() as session:
    register_defaults(session, ROOT / "workflows", activate=True)
    seed_prompts(session)
    # E0: Gemma on the CPU is the default critic; the admin can switch it off later, and a
    # choice already made (also "off", Denys 2026-11-24) is kept when this runs again.
    if session.get(Setting, SETTING) is None:
        choose(session, default_critic(session))
    session.commit()
print("Workflows, engines, recipes and prompts are stored.")
