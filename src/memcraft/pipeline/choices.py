"""Which recipe and which critic a job gets when its render starts.

Recipe: the settings row "active_recipes" may name one recipe per regime, as
{"multishot": "<recipe id>", "keyframes": "<recipe id>"}; a recipe it names must be a candidate
or active and of that regime. Without it, the active recipe named like the regime is used
("multishot", "keyframes", as deploy/seed.py stores them). If no such recipe is active, the
engines' defaults are registered and activated (memcraft.engines.register_defaults), which is
what deploy/seed.py does too.

Critic: the settings row "active_critic_engine" (memcraft.roles.critic). A missing row means
the default, Gemma on the CPU; a row whose value is null or "off" turns the critic off. The
default Gemma critic engine is created on first use, as an active engine without a prompt
version of its own (it then reads the active Gemma critic prompt).

Planning only reads; render_recipe() and render_critic() may write, and run at render start,
right before versions.pin_job().
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from memcraft import versions
from memcraft.engines import RULES, RecipeRules, register_defaults, rules_for
from memcraft.engines.base import EngineError
from memcraft.models import (
    EngineSlot,
    EngineVersion,
    Job,
    JobRegime,
    Recipe,
    Setting,
    VersionStatus,
)
from memcraft.roles import critic as critic_role
from memcraft.roles.seed import GEMMA

log = logging.getLogger(__name__)

ACTIVE_RECIPES = "active_recipes"
DEFAULT_RECIPES = {JobRegime.MULTISHOT: "multishot", JobRegime.KEYFRAMES: "keyframes"}
OPEN = (VersionStatus.CANDIDATE, VersionStatus.ACTIVE)
GEMMA_CRITIC_LIMITS = {"runs_on": "cpu", "looks_at": ["keyframes", "last frames"]}


def chosen_recipe(session: Session, regime: JobRegime) -> Recipe | None:
    """The recipe a new render of this regime gets, or None if nothing fits. Reads only.
    Regime 3 renders with the keyframes recipes."""
    regime = regime.recipe_regime
    row = session.get(Setting, ACTIVE_RECIPES)
    named = row.value.get(regime.value) if row is not None and isinstance(row.value, dict) else None
    if named:
        try:
            recipe = session.get(Recipe, uuid.UUID(str(named)))
        except ValueError:
            recipe = None
        if recipe is not None and recipe.status in OPEN and recipe.regime is regime:
            return recipe
        log.warning("%s names no usable %s recipe; using the default", ACTIVE_RECIPES, regime)
    return versions.get_active(session, Recipe, name=DEFAULT_RECIPES[regime])


def rules_for_plan(session: Session, job: Job, regime: JobRegime) -> RecipeRules:
    """The rules that cut the job into keyframes and runs: of its pinned recipe, if any."""
    regime = regime.recipe_regime
    recipe = session.get(Recipe, job.recipe_id) if job.recipe_id else None
    if recipe is None:
        recipe = chosen_recipe(session, regime)
    if recipe is not None:
        try:
            return rules_for(recipe)
        except EngineError:
            log.warning("recipe %s has no rules; using the defaults", recipe.name)
    return RULES[DEFAULT_RECIPES[regime]]


def render_recipe(session: Session, job: Job, regime: JobRegime, workflows_dir: Path) -> Recipe:
    """The pinned recipe, or the one to pin now (registering the defaults if none is active)."""
    if job.recipe_id is not None:
        return session.get_one(Recipe, job.recipe_id)
    regime = regime.recipe_regime
    recipe = chosen_recipe(session, regime)
    if recipe is None:
        log.warning("no active %s recipe: registering the engines' defaults", regime.value)
        recipe = register_defaults(session, workflows_dir, activate=True).recipes[
            DEFAULT_RECIPES[regime]
        ]
    return recipe


def critic_planned(session: Session, job: Job) -> bool:
    """Whether the job's plan has a check: by its pin once pinned, else by the setting.
    Regime 3 never has one."""
    if job.regime is JobRegime.REGIME3:
        return False
    if job.recipe_id is not None:
        return job.critic_engine_id is not None
    row = session.get(Setting, critic_role.SETTING)
    return row is None or row.value not in (None, "", "off")


def render_critic(session: Session) -> EngineVersion | None:
    """The critic to pin now, or None when it is off."""
    if session.get(Setting, critic_role.SETTING) is None:
        return default_critic(session)
    return critic_role.chosen_engine(session)


def default_critic(session: Session) -> EngineVersion:
    """The active Gemma critic engine; created and activated on first use."""
    engine = versions.get_active(session, EngineVersion, slot=EngineSlot.CRITIC, name=GEMMA)
    if engine is None:
        engine = versions.create_engine_version(
            session, slot=EngineSlot.CRITIC, name=GEMMA, limits=dict(GEMMA_CRITIC_LIMITS)
        )
        versions.promote(session, engine, VersionStatus.CANDIDATE)
        versions.promote(session, engine, VersionStatus.ACTIVE)
        log.info("created the default Gemma critic engine")
    return engine
