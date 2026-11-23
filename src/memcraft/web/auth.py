"""Who is calling. Tailscale Serve proves it and the app only reads its headers.

This is safe only while nothing but Tailscale Serve can reach the app, so the app listens
on 127.0.0.1 and refuses unknown host names (a web page could otherwise point its own name
at 127.0.0.1 and send these headers itself).
"""

from __future__ import annotations

from collections.abc import Iterator
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from memcraft.config import Settings, get_settings
from memcraft.db import new_session
from memcraft.models import User, UserRole, UserStatus
from memcraft.storage import FileStore

LOGIN_HEADER = "Tailscale-User-Login"
NAME_HEADER = "Tailscale-User-Name"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def get_session() -> Iterator[Session]:
    with new_session() as session:
        yield session


def get_store(settings: Annotated[Settings, Depends(get_settings)]) -> FileStore:
    return FileStore(settings.shared_dir, settings.private_dir)


SessionDep = Annotated[Session, Depends(get_session)]
StoreDep = Annotated[FileStore, Depends(get_store)]


def same_site(request: Request) -> bool:
    """False for a request another web page made the friend's browser send."""
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site == "same-origin"
    origin = request.headers.get("origin")
    # No browser headers at all: not a browser, so no page can be behind it.
    return origin is None or origin.split("://", 1)[-1] == request.headers.get("host")


def _decode(value: str) -> str:
    # Serve sends names outside ASCII as RFC 2047 words.
    try:
        return str(make_header(decode_header(value)))
    except (HeaderParseError, LookupError, UnicodeError):
        return value


def current_login(request: Request) -> tuple[str, str | None]:
    logins = request.headers.getlist(LOGIN_HEADER)
    login = _decode(logins[0]).strip().lower() if len(logins) == 1 else ""
    if not login or len(login) > 320:
        raise HTTPException(401, "Open this page through Tailscale.")
    name = _decode(request.headers.get(NAME_HEADER, "")).strip()[:200]
    return login, name or None


def current_user(
    request: Request,
    session: SessionDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """The caller's row, created as pending on the first visit."""
    login, name = current_login(request)
    user = session.scalar(select(User).where(User.tailscale_login == login))
    if user is None:
        try:
            with session.begin_nested():
                user = User(tailscale_login=login, display_name=name)
                session.add(user)
        except IntegrityError:  # two first requests at once
            user = session.scalars(select(User).where(User.tailscale_login == login)).one()
    if login == settings.admin_login.strip().lower() and user.role != UserRole.ADMIN:
        user.role = UserRole.ADMIN
        user.status = UserStatus.APPROVED
        user.approved_at = func.now()
    session.commit()
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def approved_user(request: Request, user: CurrentUser) -> User:
    if user.status == UserStatus.PENDING:
        raise HTTPException(403, "waiting for approval")
    if user.status != UserStatus.APPROVED:
        raise HTTPException(403, "blocked")
    if request.method not in SAFE_METHODS and not same_site(request):
        raise HTTPException(403, "request came from another site")
    return user


ApprovedUser = Annotated[User, Depends(approved_user)]
