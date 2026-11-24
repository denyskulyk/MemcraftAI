"""Files on disk and their rows in `assets`, kept together.

save() leaves the commit to the caller. The delete functions, purge_deleted_files() and
sweep_orphans() commit the session themselves, including work the caller left open in it.
The root folders must already exist: a missing root means "not mounted", never "empty".
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta
from itertools import batched
from pathlib import Path
from typing import Any, BinaryIO, NamedTuple

from sqlalchemy import ColumnElement, Delete, Select, delete, event, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, SessionTransaction

from memcraft.models import Asset, AssetKind, AssetStore, FileDeletion, Job, User

log = logging.getLogger(__name__)

STORE_FOR_KIND = {
    AssetKind.PHOTO: AssetStore.SHARED,
    AssetKind.PREVIEW: AssetStore.SHARED,
    AssetKind.RELEASE: AssetStore.SHARED,
    AssetKind.GEMMA_COPY: AssetStore.PRIVATE,
    # Shared since E16, so the admin's debug drawer can show them; they come from the photos,
    # which are shared too. Keyframes saved before stay where they are.
    AssetKind.KEYFRAME: AssetStore.SHARED,
    AssetKind.LATENT: AssetStore.PRIVATE,
    AssetKind.CLIP: AssetStore.PRIVATE,
}

EXTENSION_FOR_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "video/mp4": "mp4",
    "application/x-safetensors": "safetensors",
}

_CHUNK = 1024 * 1024
_BATCH = 500
_PART = ".part"
_FOLDER = re.compile(r"[0-9a-f]{2}")
_NAME = re.compile(rf"[0-9a-f]{{32}}\.(?:{'|'.join(EXTENSION_FOR_MIME.values())})")


class StorageError(Exception):
    pass


class StoreUnavailableError(StorageError):
    """The root folder is not configured in this process, or it is not there."""


class UnsafePathError(StorageError):
    pass


class UnsupportedTypeError(StorageError):
    pass


class FileTooLargeError(StorageError):
    pass


class PurgeResult(NamedTuple):
    removed: int
    pending: int


class FileStore:
    def __init__(self, shared_dir: Path, private_dir: Path | None = None) -> None:
        self._roots = {AssetStore.SHARED: Path(shared_dir)}
        if private_dir is not None:
            # A sweep of one root must never see the other root's files.
            shared, private = Path(shared_dir).resolve(), Path(private_dir).resolve()
            if shared.is_relative_to(private) or private.is_relative_to(shared):
                raise ValueError("the shared and private folders must be separate")
            self._roots[AssetStore.PRIVATE] = Path(private_dir)

    def save(
        self,
        session: Session,
        *,
        owner_id: uuid.UUID,
        job_id: uuid.UUID,
        kind: AssetKind,
        mime: str,
        source: BinaryIO,
        ordinal: int = 0,
        step_id: uuid.UUID | None = None,
        expires_at: datetime | None = None,
        max_bytes: int | None = None,
    ) -> Asset:
        """Write the file and flush its row. The caller commits; a rollback removes the file."""
        extension = EXTENSION_FOR_MIME.get(mime)
        if extension is None:
            raise UnsupportedTypeError(f"type {mime!r} is not allowed")
        store = STORE_FOR_KIND[kind]
        name = uuid.uuid4().hex
        relative = f"{name[:2]}/{name}.{extension}"
        (self._root(store) / name[:2]).mkdir(exist_ok=True)
        final = self._resolve(store, relative)
        part = final.with_name(final.name + _PART)
        try:
            size, sha256 = _write(source, part, max_bytes)
            os.replace(part, final)
            _remove_on_rollback(session, final)
            # There are no relationships, so a job still pending would not sort before its asset.
            session.flush()
            asset = Asset(
                owner_id=owner_id,
                job_id=job_id,
                step_id=step_id,
                kind=kind,
                ordinal=ordinal,
                store=store,
                path=relative,
                mime=mime,
                size_bytes=size,
                sha256=sha256,
                expires_at=expires_at,
            )
            session.add(asset)
            session.flush()
        except BaseException:
            _unlink_quietly(part)
            _unlink_quietly(final)
            raise
        return asset

    def path_for(self, asset: Asset) -> Path:
        """Where the file is for this process. Does not check that the file exists."""
        return self._resolve(asset.store, asset.path)

    def delete_assets(
        self,
        session: Session,
        asset_ids: Iterable[uuid.UUID] | None = None,
        *,
        where: ColumnElement[bool] | None = None,
    ) -> int:
        """Delete rows and files together, by ids or by a condition. COMMITS the session.

        Rolls back and raises if the rows cannot go; then every file is still there. A file
        that cannot be removed now stays in file_deletions. Returns how many rows went.
        """
        if (asset_ids is None) == (where is None):
            raise ValueError("pass either asset_ids or where")
        return self._delete(session, where if where is not None else Asset.id.in_(list(asset_ids)))

    def delete_job(self, session: Session, job_id: uuid.UUID) -> int:
        """Delete a job with its files; steps and ideas cascade. COMMITS, as delete_assets."""
        return self._delete(
            session,
            Asset.job_id == job_id,
            lock=[select(Job.id).where(Job.id == job_id)],
            then=delete(Job).where(Job.id == job_id),
        )

    def delete_user(self, session: Session, user_id: uuid.UUID) -> int:
        """Hard-delete a user with all jobs and files. COMMITS, as delete_assets."""
        jobs = select(Job.id).where(Job.owner_id == user_id)
        return self._delete(
            session,
            or_(Asset.owner_id == user_id, Asset.job_id.in_(jobs)),
            lock=[select(User.id).where(User.id == user_id), jobs],
            then=delete(User).where(User.id == user_id),
        )

    def purge_deleted_files(self, session: Session) -> PurgeResult:
        """Remove every pending file this process can reach. COMMITS the session."""
        session.commit()  # so that every row seen below is a committed deletion
        removed = self._purge(session)
        pending = session.scalar(select(func.count()).select_from(FileDeletion)) or 0
        session.commit()
        return PurgeResult(removed, pending)

    def sweep_orphans(self, session: Session, older_than: timedelta) -> int:
        """Remove files older than `older_than` that have no row. COMMITS the session.

        Only touches names this store makes. `older_than` must be longer than any transaction
        that saves files, because an uncommitted row is not visible here.
        """
        if older_than <= timedelta(0):
            raise ValueError("older_than must be positive")
        session.commit()  # an uncommitted delete must not cost its file
        cutoff = time.time() - older_than.total_seconds()
        removed = 0
        for store in self._reachable():
            for chunk in batched(_old_files(self._root(store), cutoff), _BATCH, strict=False):
                relatives = {f"{path.parent.name}/{path.name}": path for path in chunk}
                known = set(
                    session.scalars(
                        select(Asset.path).where(Asset.store == store, Asset.path.in_(relatives))
                    )
                )
                for relative, path in relatives.items():
                    if relative not in known and _unlink_quietly(path):
                        removed += 1
        session.commit()
        return removed

    def _root(self, store: AssetStore) -> Path:
        root = self._roots.get(store)
        if root is None:
            raise StoreUnavailableError(f"the {store.value} folder is not configured here")
        if not root.is_dir():
            raise StoreUnavailableError(f"the {store.value} folder is missing")
        return root.resolve()

    def _reachable(self) -> list[AssetStore]:
        return [store for store, root in self._roots.items() if root.is_dir()]

    def _resolve(self, store: AssetStore, relative: str) -> Path:
        root = self._root(store)
        folder, _, name = relative.partition("/")
        # The strict shape leaves no room for "..", drive letters, streams or backslashes.
        if not (_FOLDER.fullmatch(folder) and _NAME.fullmatch(name) and name.startswith(folder)):
            raise UnsafePathError(f"not a storage path: {relative!r}")
        path = (root / folder / name).resolve()
        if not path.is_relative_to(root):  # a link or junction inside the root
            raise UnsafePathError(f"path leaves the {store.value} folder: {relative!r}")
        return path

    def _delete(
        self,
        session: Session,
        where: ColumnElement[bool],
        *,
        lock: Sequence[Select[Any]] = (),
        then: Delete | None = None,
    ) -> int:
        try:
            session.flush()
            # The lock makes a save() for the same job or user wait, then fail on its key.
            for statement in lock:
                session.execute(statement.with_for_update()).all()
            gone = session.execute(delete(Asset).where(where).returning(Asset.store, Asset.path))
            pending = [FileDeletion(store=row.store, path=row.path) for row in gone.all()]
            session.add_all(pending)
            session.flush()
            ids = [row.id for row in pending]
            if then is not None:
                session.execute(then)
            session.commit()
        except BaseException:
            session.rollback()
            raise
        try:
            for chunk in batched(ids, _BATCH, strict=False):
                self._purge(session, FileDeletion.id.in_(chunk))
        except SQLAlchemyError:
            # The rows are gone for good; the worker finishes the files.
            log.exception("files left for the next purge")
        return len(ids)

    def _purge(self, session: Session, *where: ColumnElement[bool]) -> int:
        removed = last = 0
        stores = self._reachable()
        try:
            while stores:
                rows = session.execute(
                    select(FileDeletion.id, FileDeletion.store, FileDeletion.path)
                    .where(FileDeletion.id > last, FileDeletion.store.in_(stores), *where)
                    .order_by(FileDeletion.id)
                    .limit(_BATCH)
                    .with_for_update(skip_locked=True)
                ).all()
                if not rows:
                    break
                last = rows[-1].id
                gone = [row.id for row in rows if self._unlink(row.store, row.path)]
                session.execute(delete(FileDeletion).where(FileDeletion.id.in_(gone)))
                session.commit()
                removed += len(gone)
            session.commit()
        except BaseException:
            session.rollback()
            raise
        return removed

    def _unlink(self, store: AssetStore, relative: str) -> bool:
        """True once the file is certainly gone."""
        try:
            self._resolve(store, relative).unlink()
        except FileNotFoundError:
            return store in self._reachable()  # the whole root may just have gone away
        except (StorageError, OSError) as error:
            log.warning("cannot delete %s/%s yet: %s", store.value, relative, error)
            return False
        return True


def _write(source: BinaryIO, part: Path, max_bytes: int | None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with part.open("xb") as target:
        while chunk := source.read(_CHUNK):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise FileTooLargeError(f"more than {max_bytes} bytes")
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    return size, digest.hexdigest()


def _unlink_quietly(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        log.warning("cannot delete %s, left for a later sweep: %s", path, error)
        return False
    return True


def _old_files(root: Path, cutoff: float) -> Iterator[Path]:
    with os.scandir(root) as entries:
        folders = [
            Path(entry.path)
            for entry in entries
            if _FOLDER.fullmatch(entry.name)
            and entry.is_dir(follow_symlinks=False)
            and not entry.is_junction()
        ]
    for folder in folders:
        with os.scandir(folder) as entries:
            files = [
                Path(entry.path)
                for entry in entries
                if _NAME.fullmatch(entry.name.removesuffix(_PART))
                and entry.is_file(follow_symlinks=False)
            ]
        for path in files:
            try:
                # Not the scandir time: Windows updates that one late for an open file.
                modified = path.lstat().st_mtime
            except OSError:
                continue
            if modified < cutoff:
                yield path


_NEW_FILES = "memcraft.storage.new_files"


class _NewFiles:
    """Files one session has saved but not committed, by the transaction that would undo them.

    Listens to the end of transactions, because Session.close() undoes work without any
    rollback event.
    """

    def __init__(self, session: Session) -> None:
        self.owners: dict[Path, SessionTransaction] = {}
        self.commit_sent = False
        event.listen(session, "after_commit", self.after_commit)
        event.listen(session, "after_transaction_end", self.after_transaction_end)

    def before_database_commit(self, _connection: object) -> None:
        self.commit_sent = True

    def after_commit(self, session: Session) -> None:
        nested = session.get_nested_transaction()
        if nested is None:
            self.owners.clear()
            return
        # A released savepoint: the enclosing transaction decides now.
        for path, owner in self.owners.items():
            if owner is nested:
                self.owners[path] = nested.parent

    def after_transaction_end(self, _session: Session, transaction: SessionTransaction) -> None:
        ended = [path for path, owner in self.owners.items() if owner is transaction]
        for path in ended:
            del self.owners[path]
            # A COMMIT that failed midway may still have reached the database, so keep the
            # file; sweep_orphans() removes it if there is no row.
            if not self.commit_sent:
                _unlink_quietly(path)
        if transaction.parent is None:
            self.commit_sent = False


def _remove_on_rollback(session: Session, path: Path) -> None:
    connection = session.connection(bind_arguments={"mapper": Asset})  # begins if needed
    new_files = session.info.get(_NEW_FILES)
    if new_files is None:
        new_files = session.info[_NEW_FILES] = _NewFiles(session)
    event.listen(connection, "commit", new_files.before_database_commit)
    new_files.owners[path] = session.get_nested_transaction() or session.get_transaction()
