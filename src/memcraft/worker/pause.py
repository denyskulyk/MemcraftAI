"""The admin's pause switch, settings.paused. The worker reads it before every step.

While paused, the worker finishes the step it is in, unloads its models and waits. Nothing is
lost: a resume goes on with the next step. Both functions are for the admin page too.
"""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from memcraft.models import Setting

PAUSED = "paused"


def is_paused(session: Session) -> bool:
    """True while the worker must not start a step. A missing setting means not paused."""
    return bool(session.scalar(select(Setting.value).where(Setting.key == PAUSED)))


def set_paused(session: Session, paused: bool) -> None:
    """Pause or resume. The caller commits."""
    changed = session.execute(
        update(Setting)
        .where(Setting.key == PAUSED)
        .values(value=bool(paused))
        .execution_options(synchronize_session=False)
    )
    if changed.rowcount == 0:
        session.add(Setting(key=PAUSED, value=bool(paused)))
    session.flush()
