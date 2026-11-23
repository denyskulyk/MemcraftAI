r"""Makes the newest candidate prompt of each role and model the active one, and commits.

Run after deploy\seed.py has stored changed prompt files as candidates:
    .venv\Scripts\python.exe deploy\promote_prompts.py            (promote and commit)
    .venv\Scripts\python.exe deploy\promote_prompts.py --dry-run  (only show what would change)

versions.promote() retires the version that was active until now; jobs that pinned it keep it.
A candidate older than the active version is left alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from memcraft.models import PromptVersion, VersionStatus
from memcraft.versions import promote


@dataclass(frozen=True)
class Change:
    role: str
    model: str
    before: int | None
    after: int

    def __str__(self) -> str:
        before = "no active version" if self.before is None else f"active v{self.before} retires"
        return f"{self.role} {self.model}: candidate v{self.after} -> active ({before})"


def promote_newest(session: Session, *, dry_run: bool = False) -> list[Change]:
    """Promote the newest candidate of every role and model. The caller commits."""
    rows = session.scalars(
        select(PromptVersion)
        .where(PromptVersion.status.in_((VersionStatus.CANDIDATE, VersionStatus.ACTIVE)))
        .order_by(PromptVersion.role, PromptVersion.model, PromptVersion.version)
    ).all()
    families: dict[tuple[str, str], list[PromptVersion]] = {}
    for row in rows:
        families.setdefault((row.role.value, row.model), []).append(row)
    changes = []
    for (role, model), family in sorted(families.items()):
        active = next((r for r in family if r.status is VersionStatus.ACTIVE), None)
        candidates = [r for r in family if r.status is VersionStatus.CANDIDATE]
        if not candidates:
            continue
        newest = max(candidates, key=lambda r: r.version)
        if active is not None and active.version > newest.version:
            continue
        if not dry_run:
            promote(session, newest, VersionStatus.ACTIVE)
        changes.append(
            Change(role, model, None if active is None else active.version, newest.version)
        )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="show the changes, commit nothing")
    args = parser.parse_args()

    from memcraft.db import new_session

    with new_session() as session:
        database = session.scalar(text("SELECT current_database()"))
        changes = promote_newest(session, dry_run=args.dry_run)
        for change in changes:
            print(change)
        if not changes:
            print(f"{database}: no candidate prompt is newer than the active one.")
        elif args.dry_run:
            session.rollback()
            print(f"{database}: dry run, nothing changed.")
        else:
            session.commit()
            print(f"{database}: {len(changes)} prompt(s) promoted and committed.")


if __name__ == "__main__":
    main()
