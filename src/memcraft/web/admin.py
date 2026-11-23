"""Admin pages: Tailscale login plus a passkey (E7.1), and who may use the app (E7.2).

The first passkey can be added with the admin's Tailscale login alone. Every later passkey
needs a signed-in admin, so somebody who only borrowed the Tailscale login cannot add one.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import webauthn
from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from webauthn.helpers import exceptions as webauthn_errors
from webauthn.helpers import structs

from memcraft.config import Settings, get_settings
from memcraft.models import (
    AdminSession,
    AuthChallenge,
    ChallengePurpose,
    Job,
    Passkey,
    User,
    UserRole,
    UserStatus,
)
from memcraft.web.auth import ApprovedUser, SessionDep, StoreDep

router = APIRouter(prefix="/api/admin")

COOKIE = "__Host-memcraft-admin"
SESSION_LIFETIME = timedelta(hours=12)
CHALLENGE_LIFETIME = timedelta(minutes=5)
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def admin_login(user: ApprovedUser) -> User:
    """The admin's Tailscale login, before the passkey. Others learn nothing about this page."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(404, "not found")
    return user


AdminLogin = Annotated[User, Depends(admin_login)]


def _signed_in(request: Request, session: Session, user: User) -> AdminSession | None:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    return session.scalar(
        select(AdminSession).where(
            AdminSession.token_sha256 == _hash(token),
            AdminSession.user_id == user.id,
            AdminSession.expires_at > _now(),
        )
    )


def admin_user(request: Request, user: AdminLogin, session: SessionDep) -> User:
    if _signed_in(request, session, user) is None:
        raise HTTPException(401, "sign in with your passkey")
    return user


AdminUser = Annotated[User, Depends(admin_user)]


class Status(BaseModel):
    has_passkey: bool
    signed_in: bool


@router.get("/status")
def status(request: Request, user: AdminLogin, session: SessionDep) -> Status:
    count = session.scalar(select(func.count()).where(Passkey.user_id == user.id)) or 0
    return Status(has_passkey=count > 0, signed_in=_signed_in(request, session, user) is not None)


def _new_challenge(session: Session, user: User, purpose: ChallengePurpose) -> bytes:
    session.execute(
        delete(AuthChallenge).where(
            (AuthChallenge.user_id == user.id) | (AuthChallenge.expires_at <= _now())
        )
    )
    challenge = secrets.token_bytes(32)
    session.add(
        AuthChallenge(
            user_id=user.id,
            purpose=purpose,
            challenge=challenge,
            expires_at=_now() + CHALLENGE_LIFETIME,
        )
    )
    session.commit()
    return challenge


def _take_challenge(session: Session, user: User, purpose: ChallengePurpose) -> bytes:
    """The open challenge, removed so that it works once."""
    row = session.execute(
        delete(AuthChallenge)
        .where(
            AuthChallenge.user_id == user.id,
            AuthChallenge.purpose == purpose,
            AuthChallenge.expires_at > _now(),
        )
        .returning(AuthChallenge.challenge)
    ).first()
    session.commit()
    if row is None:
        raise HTTPException(400, "this request has expired, please try again")
    return row.challenge


def _start_session(response: Response, session: Session, user: User, passkey: Passkey) -> None:
    token = secrets.token_urlsafe(32)
    session.add(
        AdminSession(
            token_sha256=_hash(token),
            user_id=user.id,
            passkey_id=passkey.id,
            expires_at=_now() + SESSION_LIFETIME,
        )
    )
    session.execute(delete(AdminSession).where(AdminSession.expires_at <= _now()))
    session.commit()
    response.set_cookie(
        COOKIE,
        token,
        max_age=int(SESSION_LIFETIME.total_seconds()),
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )


@router.post("/passkeys/options")
def passkey_options(
    request: Request, user: AdminLogin, session: SessionDep, settings: SettingsDep
) -> Response:
    existing = session.scalars(select(Passkey).where(Passkey.user_id == user.id)).all()
    if existing and _signed_in(request, session, user) is None:
        raise HTTPException(401, "sign in with your passkey to add another one")
    options = webauthn.generate_registration_options(
        rp_id=settings.passkey_rp_id,
        rp_name="MemcraftAI admin",
        user_id=user.id.bytes,
        user_name=user.tailscale_login,
        user_display_name=user.display_name or user.tailscale_login,
        challenge=_new_challenge(session, user, ChallengePurpose.REGISTER),
        authenticator_selection=structs.AuthenticatorSelectionCriteria(
            resident_key=structs.ResidentKeyRequirement.PREFERRED,
            user_verification=structs.UserVerificationRequirement.REQUIRED,
        ),
        exclude_credentials=[
            structs.PublicKeyCredentialDescriptor(id=key.credential_id) for key in existing
        ],
    )
    return Response(webauthn.options_to_json(options), media_type="application/json")


@router.post("/passkeys", status_code=201)
def add_passkey(
    request: Request,
    response: Response,
    user: AdminLogin,
    session: SessionDep,
    settings: SettingsDep,
    credential: Annotated[dict[str, Any], Body()],
) -> Status:
    challenge = _take_challenge(session, user, ChallengePurpose.REGISTER)
    existing = session.scalar(select(func.count()).where(Passkey.user_id == user.id)) or 0
    if existing and _signed_in(request, session, user) is None:
        raise HTTPException(401, "sign in with your passkey to add another one")
    try:
        verified = webauthn.verify_registration_response(
            credential=credential,
            expected_challenge=challenge,
            expected_rp_id=settings.passkey_rp_id,
            expected_origin=settings.public_origin,
            require_user_verification=True,
        )
    except (webauthn_errors.WebAuthnException, ValueError, KeyError, TypeError):
        raise HTTPException(400, "the passkey could not be checked") from None
    passkey = Passkey(
        user_id=user.id,
        credential_id=verified.credential_id,
        public_key=verified.credential_public_key,
        sign_count=verified.sign_count,
        name=f"passkey {existing + 1}",
    )
    session.add(passkey)
    session.flush()
    # Enrolling proves the passkey just as signing in does.
    _start_session(response, session, user, passkey)
    return Status(has_passkey=True, signed_in=True)


@router.post("/sign-in/options")
def sign_in_options(user: AdminLogin, session: SessionDep, settings: SettingsDep) -> Response:
    keys = session.scalars(select(Passkey).where(Passkey.user_id == user.id)).all()
    if not keys:
        raise HTTPException(409, "add a passkey first")
    options = webauthn.generate_authentication_options(
        rp_id=settings.passkey_rp_id,
        challenge=_new_challenge(session, user, ChallengePurpose.SIGN_IN),
        allow_credentials=[structs.PublicKeyCredentialDescriptor(id=k.credential_id) for k in keys],
        user_verification=structs.UserVerificationRequirement.REQUIRED,
    )
    return Response(webauthn.options_to_json(options), media_type="application/json")


@router.post("/sign-in")
def sign_in(
    response: Response,
    user: AdminLogin,
    session: SessionDep,
    settings: SettingsDep,
    credential: Annotated[dict[str, Any], Body()],
) -> Status:
    challenge = _take_challenge(session, user, ChallengePurpose.SIGN_IN)
    try:
        credential_id = webauthn.base64url_to_bytes(credential["rawId"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "the passkey could not be checked") from None
    # Only this admin's own passkeys count.
    passkey = session.scalar(
        select(Passkey)
        .where(Passkey.user_id == user.id, Passkey.credential_id == credential_id)
        .with_for_update()
    )
    if passkey is None:
        raise HTTPException(400, "this passkey is not registered")
    try:
        verified = webauthn.verify_authentication_response(
            credential=credential,
            expected_challenge=challenge,
            expected_rp_id=settings.passkey_rp_id,
            expected_origin=settings.public_origin,
            credential_public_key=passkey.public_key,
            credential_current_sign_count=passkey.sign_count,
            require_user_verification=True,
        )
    except (webauthn_errors.WebAuthnException, ValueError, KeyError, TypeError):
        raise HTTPException(400, "the passkey could not be checked") from None
    passkey.sign_count = verified.new_sign_count
    passkey.last_used_at = _now()
    _start_session(response, session, user, passkey)
    return Status(has_passkey=True, signed_in=True)


@router.post("/sign-out", status_code=204)
def sign_out(request: Request, response: Response, user: AdminLogin, session: SessionDep) -> None:
    token = request.cookies.get(COOKIE)
    if token:
        session.execute(delete(AdminSession).where(AdminSession.token_sha256 == _hash(token)))
        session.commit()
    response.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="strict")


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tailscale_login: str
    display_name: str | None
    role: UserRole
    status: UserStatus
    created_at: datetime
    approved_at: datetime | None
    jobs: int = 0


@router.get("/users")
def list_users(admin: AdminUser, session: SessionDep) -> list[UserOut]:
    jobs = select(Job.owner_id, func.count().label("jobs")).group_by(Job.owner_id).subquery("jobs")
    rows = session.execute(
        select(User, func.coalesce(jobs.c.jobs, 0))
        .outerjoin(jobs, jobs.c.owner_id == User.id)
        .order_by(User.status != UserStatus.PENDING, User.created_at.desc(), User.id)
    ).all()
    return [UserOut.model_validate(user).model_copy(update={"jobs": count}) for user, count in rows]


def _other_user(session: Session, admin: User, user_id: uuid.UUID) -> User:
    user = session.get(User, user_id, with_for_update=True)
    if user is None:
        raise HTTPException(404, "not found")
    if user.id == admin.id:
        raise HTTPException(409, "you cannot change your own access here")
    return user


@router.post("/users/{user_id}/approve")
def approve(user_id: uuid.UUID, admin: AdminUser, session: SessionDep) -> UserOut:
    user = _other_user(session, admin, user_id)
    user.status = UserStatus.APPROVED
    user.approved_at = func.now()
    session.commit()
    return UserOut.model_validate(user)


@router.post("/users/{user_id}/block")
def block(user_id: uuid.UUID, admin: AdminUser, session: SessionDep) -> UserOut:
    user = _other_user(session, admin, user_id)
    user.status = UserStatus.BLOCKED
    session.execute(delete(AdminSession).where(AdminSession.user_id == user.id))
    session.commit()
    return UserOut.model_validate(user)


@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: uuid.UUID, admin: AdminUser, session: SessionDep, store: StoreDep) -> None:
    """Hard delete: the person, their jobs and every file, at once."""
    user = _other_user(session, admin, user_id)
    if user.role == UserRole.ADMIN:
        raise HTTPException(409, "another admin cannot be deleted here")
    # Files the web app cannot see are left in file_deletions for the worker.
    store.delete_user(session, user.id)
