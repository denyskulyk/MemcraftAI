import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from alembic import command
from conftest import _postgres_engine
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_schema import alembic_config

from memcraft.models import (
    Base,
    EngineSlot,
    EngineVersion,
    Idea,
    IdeaSource,
    Job,
    JobLength,
    JobQuality,
    JobRegime,
    JobStep,
    PromptRole,
    PromptVersion,
    Recipe,
    StepName,
    User,
    VersionStatus,
    WorkflowVersion,
)
from memcraft.versions import (
    TRANSITIONS,
    AlreadyPinnedError,
    InvalidVersionError,
    LifecycleError,
    NotFoundError,
    PublishedError,
    TamperedError,
    bind_workflow,
    create_engine_version,
    create_prompt_version,
    create_recipe_version,
    create_workflow_version,
    edit_draft,
    get_active,
    pin_idea,
    pin_job,
    pin_step,
    pinned_workflow,
    promote,
    retire,
)
from memcraft.workflows import InvalidWorkflowError, load_workflow

S = VersionStatus
WORKFLOWS = Path(__file__).parents[1] / "workflows"
TABLES = ["workflow_versions", "prompt_versions", "engines", "recipes"]


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def migrated():
    """The schema as Alembic builds it, with the triggers. The models alone do not have them."""
    engine = _postgres_engine()
    config = alembic_config(os.environ["MEMCRAFT_TEST_DATABASE_URL"])

    def clean():
        Base.metadata.drop_all(engine)
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE IF EXISTS alembic_version"))

    clean()
    command.upgrade(config, "head")
    yield engine
    clean()
    with engine.begin() as connection:
        connection.execute(text("DROP FUNCTION IF EXISTS guard_version_row()"))
    engine.dispose()


def api(seed=0):
    return {
        "1": {"class_type": "Sampler", "inputs": {"seed": seed, "cfg": 1.0, "big": 1e16}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }


def manifest(name="demo", version=1):
    return {
        "format": 1,
        "name": name,
        "version": version,
        "inputs": {
            "seed": {
                "type": "int",
                "required": True,
                "targets": [{"node": "1", "class_type": "Sampler", "input": "seed"}],
            }
        },
        "outputs": {"picture": {"node": "2", "class_type": "SaveImage"}},
    }


def workflow(session, name="demo", version=1, to=S.DRAFT):
    return walk(session, create_workflow_version(session, api(), manifest(name, version)), to)


def prompt(session, role=PromptRole.CHAT, model="m", words="Be kind.", to=S.DRAFT):
    row = create_prompt_version(session, role=role, model=model, text=words)
    return walk(session, row, to)


def engine_version(session, slot=EngineSlot.CLIP, name="ltx", to=S.DRAFT, part=None):
    part = part or workflow(session, name=f"wf-{uuid.uuid4().hex[:8]}", to=S.ACTIVE)
    row = create_engine_version(session, slot=slot, name=name, workflow_version_id=part.id)
    return walk(session, row, to)


def recipe(session, name="multishot", to=S.DRAFT, clip=None):
    clip = clip or engine_version(session, name=f"e-{uuid.uuid4().hex[:8]}", to=S.ACTIVE)
    row = create_recipe_version(
        session, name=name, regime=JobRegime.MULTISHOT, clip_engine_id=clip.id
    )
    return walk(session, row, to)


MAKERS = {
    "workflow_versions": workflow,
    "prompt_versions": prompt,
    "engines": engine_version,
    "recipes": recipe,
}


def walk(session, row, to):
    path = {
        S.DRAFT: [],
        S.CANDIDATE: [S.CANDIDATE],
        S.ACTIVE: [S.CANDIDATE, S.ACTIVE],
        S.RETIRED: [S.CANDIDATE, S.ACTIVE, S.RETIRED],
    }[to]
    for step in path:
        retire(session, row) if step is S.RETIRED else promote(session, row, step)
    return row


def job(session, **fields):
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    row = Job(
        owner_id=user.id,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        **fields,
    )
    session.add(row)
    session.flush()
    return row


def step(session, parent):
    row = JobStep(job_id=parent.id, step=StepName.CLIP)
    session.add(row)
    session.flush()
    return row


# Creating


def test_prompt_numbers_count_per_role_and_model(session):
    rows = [prompt(session), prompt(session), prompt(session, model="other")]
    rows.append(prompt(session, role=PromptRole.CRITIC))
    assert [row.version for row in rows] == [1, 2, 1, 1]
    assert {row.status for row in rows} == {S.DRAFT}


def test_a_workflow_is_stored_once_per_version(session):
    row = workflow(session)
    assert (row.name, row.version) == ("demo", 1)
    assert workflow(session) is row
    with pytest.raises(LifecycleError, match="already exists with other content"):
        create_workflow_version(session, api(seed=1), manifest())
    with pytest.raises(InvalidWorkflowError):
        create_workflow_version(session, {"nodes": [], "links": []}, manifest(version=2))


@pytest.mark.parametrize("folder", sorted(WORKFLOWS.iterdir()), ids=lambda path: path.name)
def test_the_exported_workflows_survive_the_database(session, folder):
    package = load_workflow(folder)
    row = create_workflow_version(session, package.api_json, package.manifest_json)
    walk(session, row, S.ACTIVE)
    session.commit()
    session.expire_all()
    found = get_active(session, WorkflowVersion, name=package.manifest.name)
    assert found.sha256 == package.sha256
    values = {"positive_prompt": "a cat", "prompt": "a cat", "system": "s", "seed": 1}
    values = {k: v for k, v in values.items() if k in package.manifest.inputs}
    values |= {name: "a.png" for name, x in package.manifest.inputs.items() if x.type == "image"}
    assert bind_workflow(found, values)


def test_parts_must_fit(session):
    clip = engine_version(session)
    keyframe = engine_version(session, slot=EngineSlot.KEYFRAME, name="qwen")
    chat = prompt(session)
    with pytest.raises(InvalidVersionError, match="needs a workflow"):
        create_engine_version(session, slot=EngineSlot.CLIP, name="x")
    with pytest.raises(InvalidVersionError, match="critic prompt"):
        create_engine_version(session, slot=EngineSlot.CRITIC, name="x", prompt_version_id=chat.id)
    with pytest.raises(NotFoundError):
        create_engine_version(
            session, slot=EngineSlot.CLIP, name="x", workflow_version_id=uuid.uuid4()
        )
    with pytest.raises(InvalidVersionError, match="needs a clip engine"):
        create_recipe_version(
            session, name="r", regime=JobRegime.MULTISHOT, clip_engine_id=keyframe.id
        )
    with pytest.raises(InvalidVersionError, match="needs a keyframe engine"):
        create_recipe_version(session, name="r", regime=JobRegime.KEYFRAMES, clip_engine_id=clip.id)
    with pytest.raises(InvalidVersionError, match="another slot"):
        create_recipe_version(
            session,
            name="r",
            regime=JobRegime.KEYFRAMES,
            clip_engine_id=clip.id,
            keyframe_engine_id=clip.id,
        )


# Lifecycle


@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS)
@pytest.mark.parametrize("start", list(S))
@pytest.mark.parametrize("to", list(S))
def test_only_the_allowed_transitions(session, make, start, to):
    row = make(session, to=start)
    move = retire if to is S.RETIRED else lambda s, r: promote(s, r, to)
    if to in TRANSITIONS[start]:
        move(session, row)
        assert row.status is to
    else:
        with pytest.raises(LifecycleError):
            move(session, row)
        assert row.status is start


@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS)
def test_activating_retires_the_previous_one(session, make):
    first = make(session, to=S.ACTIVE)
    if isinstance(first, WorkflowVersion):
        second = make(session, version=2, to=S.CANDIDATE)
    else:
        second = make(session, to=S.CANDIDATE)
    assert (second.version, first.status) == (2, S.ACTIVE)
    promote(session, second, S.ACTIVE)
    assert (first.status, second.status) == (S.RETIRED, S.ACTIVE)


def test_active_is_per_role_and_model(session):
    rows = [
        prompt(session, to=S.ACTIVE),
        prompt(session, model="other", to=S.ACTIVE),
        prompt(session, role=PromptRole.DIRECTOR_MULTISHOT, to=S.ACTIVE),
    ]
    assert {row.status for row in rows} == {S.ACTIVE}
    assert get_active(session, PromptVersion, role=PromptRole.CHAT, model="other") is rows[1]
    assert get_active(session, PromptVersion, role=PromptRole.CRITIC, model="m") is None
    with pytest.raises(TypeError, match="role, model"):
        get_active(session, PromptVersion, role=PromptRole.CHAT)


def test_engine_names_are_per_slot(session):
    clip = engine_version(session, name="same", to=S.ACTIVE)
    keyframe = engine_version(session, slot=EngineSlot.KEYFRAME, name="same", to=S.ACTIVE)
    assert (clip.version, keyframe.version, clip.status) == (1, 1, S.ACTIVE)
    assert get_active(session, EngineVersion, slot=EngineSlot.CLIP, name="same") is clip


def test_parts_are_published_first(session):
    part = workflow(session)
    row = engine_version(session, part=part)
    with pytest.raises(LifecycleError, match="workflow_versions demo version 1 is draft"):
        promote(session, row, S.CANDIDATE)
    promote(session, part, S.CANDIDATE)
    promote(session, row, S.CANDIDATE)
    with pytest.raises(LifecycleError, match="is candidate"):
        promote(session, row, S.ACTIVE)
    assert row.status is S.CANDIDATE
    promote(session, part, S.ACTIVE)
    promote(session, row, S.ACTIVE)


def test_a_draft_can_be_edited(session):
    row = workflow(session)
    before = row.sha256
    edit_draft(session, row, api_json=api(seed=9))
    assert row.sha256 != before
    assert bind_workflow(row, {"seed": 3})["1"]["inputs"]["seed"] == 3
    with pytest.raises(InvalidVersionError, match="cannot change sha256"):
        edit_draft(session, row, sha256="0")
    with pytest.raises(InvalidWorkflowError, match="manifest of demo version 2"):
        edit_draft(session, row, manifest=manifest(version=2))
    words = prompt(session)
    edit_draft(session, words, text="Be brief.")
    assert words.text == "Be brief."
    plan = recipe(session)
    with pytest.raises(InvalidVersionError, match="keyframe engine"):
        edit_draft(session, plan, regime=JobRegime.KEYFRAMES)
    assert plan.regime is JobRegime.MULTISHOT


@pytest.mark.parametrize("status", [S.CANDIDATE, S.ACTIVE, S.RETIRED])
def test_a_published_version_cannot_be_edited(session, status):
    row = workflow(session, to=status)
    with pytest.raises(PublishedError):
        edit_draft(session, row, api_json=api(seed=9))
    words = prompt(session, to=status)
    with pytest.raises(PublishedError):
        edit_draft(session, words, text="Other.")
    assert words.text == "Be kind."


def test_tampered_content_is_refused(session):
    row = workflow(session, to=S.CANDIDATE)
    session.execute(update(WorkflowVersion).values(api_json=api(seed=666)))
    session.expire_all()
    with pytest.raises(TamperedError):
        bind_workflow(row, {"seed": 1})
    with pytest.raises(TamperedError):
        promote(session, row, S.ACTIVE)
    session.execute(update(WorkflowVersion).values(status=S.ACTIVE))
    with pytest.raises(TamperedError):
        get_active(session, WorkflowVersion, name="demo")


def test_content_of_another_version_is_refused(session):
    """The sha256 alone would not notice a row that carries another version's content."""
    row = workflow(session)
    other = workflow(session, version=2)
    session.execute(
        update(WorkflowVersion)
        .where(WorkflowVersion.id == row.id)
        .values(manifest=other.manifest, sha256=other.sha256)
    )
    session.expire_all()
    with pytest.raises(InvalidWorkflowError, match="manifest of demo version 2"):
        bind_workflow(row, {"seed": 1})


# Pinning


def test_a_job_is_pinned_once(session):
    row = job(session)
    plan = recipe(session, to=S.ACTIVE)
    critic = engine_version(session, slot=EngineSlot.CRITIC, name="c", to=S.ACTIVE)
    pin_job(session, row, recipe=plan, critic=critic)
    assert (row.recipe_id, row.critic_engine_id) == (plan.id, critic.id)
    pin_job(session, row, recipe=plan, critic=critic)
    other = recipe(session, name="other", to=S.ACTIVE)
    with pytest.raises(AlreadyPinnedError):
        pin_job(session, row, recipe=other, critic=critic)
    with pytest.raises(AlreadyPinnedError):
        pin_job(session, row, recipe=plan)  # "off" is a choice too
    assert (row.recipe_id, row.critic_engine_id) == (plan.id, critic.id)


def test_a_critic_that_was_off_stays_off(session):
    row = job(session)
    plan = recipe(session, to=S.ACTIVE)
    pin_job(session, row, recipe=plan)
    critic = engine_version(session, slot=EngineSlot.CRITIC, name="c", to=S.ACTIVE)
    with pytest.raises(AlreadyPinnedError):
        pin_job(session, row, recipe=plan, critic=critic)
    assert row.critic_engine_id is None


@pytest.mark.parametrize("status", [S.DRAFT, S.RETIRED])
def test_a_job_cannot_start_on_this(session, status):
    row = job(session)
    with pytest.raises(LifecycleError):
        pin_job(session, row, recipe=recipe(session, to=status))
    critic = engine_version(session, slot=EngineSlot.CRITIC, name="c", to=status)
    with pytest.raises(LifecycleError):
        pin_job(session, row, recipe=recipe(session, name="ok", to=S.ACTIVE), critic=critic)
    assert row.recipe_id is None and row.critic_engine_id is None


def test_a_job_pins_what_fits(session):
    plan = recipe(session, to=S.ACTIVE)
    clip = engine_version(session, to=S.ACTIVE)
    with pytest.raises(InvalidVersionError, match="not a critic"):
        pin_job(session, job(session), recipe=plan, critic=clip)
    with pytest.raises(InvalidVersionError, match="the job is keyframes"):
        pin_job(session, job(session, regime=JobRegime.KEYFRAMES), recipe=plan)
    pin_job(session, job(session, regime=JobRegime.MULTISHOT), recipe=plan)


def test_a_pinned_job_keeps_its_workflow(session):
    old = workflow(session, to=S.ACTIVE)
    clip = engine_version(session, part=old, to=S.ACTIVE)
    row = job(session)
    with pytest.raises(NotFoundError):
        pinned_workflow(session, row, EngineSlot.CLIP)
    pin_job(session, row, recipe=recipe(session, to=S.ACTIVE, clip=clip))
    new = workflow(session, version=2, to=S.ACTIVE)
    assert get_active(session, WorkflowVersion, name="demo") is new
    assert pinned_workflow(session, row, EngineSlot.CLIP) is old
    assert old.status is S.RETIRED
    pin_step(session, step(session, row), workflow=old)
    with pytest.raises(NotFoundError):
        pinned_workflow(session, row, EngineSlot.KEYFRAME)


def test_a_step_is_pinned_once(session):
    row = step(session, job(session))
    old, words = workflow(session, to=S.ACTIVE), prompt(session, to=S.ACTIVE)
    pin_step(session, row, prompt=words)
    pin_step(session, row, workflow=old, prompt=words)
    assert (row.workflow_version_id, row.prompt_version_id) == (old.id, words.id)
    with pytest.raises(AlreadyPinnedError):
        pin_step(session, row, workflow=workflow(session, version=2, to=S.ACTIVE))
    with pytest.raises(AlreadyPinnedError):
        pin_step(session, row, prompt=prompt(session, to=S.ACTIVE))
    assert (row.workflow_version_id, row.prompt_version_id) == (old.id, words.id)


def test_a_step_never_uses_a_draft(session):
    row = step(session, job(session))
    with pytest.raises(LifecycleError):
        pin_step(session, row, workflow=workflow(session))
    with pytest.raises(LifecycleError):
        pin_step(session, row, prompt=prompt(session))
    assert row.workflow_version_id is None and row.prompt_version_id is None


def test_an_idea_records_its_prompt(session):
    row = Idea(job_id=job(session).id, text="t", score=5, reason="r", proposed_by=IdeaSource.GEMMA)
    session.add(row)
    words = prompt(session, to=S.ACTIVE)
    pin_idea(session, row, words)
    assert row.prompt_version_id == words.id
    with pytest.raises(AlreadyPinnedError):
        pin_idea(session, row, prompt(session, to=S.ACTIVE))


# The triggers and indexes of migration 0002

CONTENT = {
    "workflow_versions": [
        "api_json = '{}'",
        "manifest = '{}'",
        "sha256 = 'x'",
        "name = 'x'",
        "version = 99",
        "created_at = now() + interval '1 day'",
        "id = gen_random_uuid()",
    ],
    "prompt_versions": ["text = 'x'", "role = 'critic'", "model = 'x'", "version = 99"],
    "engines": [
        "limits = '{\"a\": 1}'",
        "slot = 'critic'",
        "name = 'x'",
        "version = 99",
        "workflow_version_id = NULL",
        "prompt_version_id = (SELECT id FROM prompt_versions LIMIT 1)",
    ],
    "recipes": [
        "name = 'x'",
        "version = 99",
        "regime = 'keyframes'",
        "clip_engine_id = (SELECT id FROM engines WHERE slot = 'keyframe')",
        "keyframe_engine_id = (SELECT id FROM engines WHERE slot = 'keyframe')",
    ],
}


def refused(session, statement, match):
    with pytest.raises(IntegrityError, match=match), session.begin_nested():
        session.execute(text(statement))


@pytest.mark.postgres
@pytest.mark.parametrize("table", TABLES)
def test_triggers_freeze_published_rows(migrated, table):
    with Session(migrated) as session:
        prompt(session, role=PromptRole.CRITIC, model="spare")
        engine_version(session, slot=EngineSlot.KEYFRAME, name="spare")
        for status in (S.CANDIDATE, S.ACTIVE, S.RETIRED):
            row = MAKERS[table](session, **_fresh(table), to=status)
            for change in CONTENT[table]:
                statement = f"UPDATE {table} SET {change} WHERE id = '{row.id}'"  # noqa: S608
                refused(session, statement, "cannot be edited")
            refused(
                session,
                f"DELETE FROM {table} WHERE id = '{row.id}'",  # noqa: S608
                "cannot be deleted",
            )


def _fresh(table):
    name = uuid.uuid4().hex
    return {"model": name} if table == "prompt_versions" else {"name": name}


@pytest.mark.postgres
@pytest.mark.parametrize("table", TABLES)
def test_triggers_allow_drafts(migrated, table):
    with Session(migrated) as session:
        prompt(session, role=PromptRole.CRITIC, model="spare")
        engine_version(session, slot=EngineSlot.KEYFRAME, name="spare")
        for change in CONTENT[table]:
            row = MAKERS[table](session, **_fresh(table))
            statement = f"UPDATE {table} SET {change} WHERE id = '{row.id}' RETURNING id"  # noqa: S608
            changed = session.execute(text(statement)).scalar_one()
            session.execute(text(f"DELETE FROM {table} WHERE id = '{changed}'"))  # noqa: S608
            session.expunge(row)


@pytest.mark.postgres
@pytest.mark.parametrize("table", TABLES)
def test_triggers_only_allow_the_transitions(migrated, table):
    with Session(migrated) as session:
        for start in S:
            for to in S:
                row = MAKERS[table](session, **_fresh(table), to=start)
                statement = f"UPDATE {table} SET status = '{to}' WHERE id = '{row.id}'"  # noqa: S608
                if to is start or to in TRANSITIONS[start]:
                    session.execute(text(statement))
                else:
                    refused(session, statement, f"cannot go from {start} to {to}")


@pytest.mark.postgres
@pytest.mark.parametrize("table", TABLES)
def test_index_allows_one_active(migrated, table):
    with Session(migrated) as session:
        key = _fresh(table)
        MAKERS[table](session, **key, to=S.ACTIVE)
        if table == "workflow_versions":
            key["version"] = 2
        second = MAKERS[table](session, **key, to=S.CANDIDATE)
        refused(
            session,
            f"UPDATE {table} SET status = 'active' WHERE id = '{second.id}'",  # noqa: S608
            f"uq_{table}_active",
        )
        promote(session, second, S.ACTIVE)
        session.commit()


@pytest.mark.postgres
def test_two_activations_at_once(migrated):
    with Session(migrated) as session:
        ids = [prompt(session, to=S.CANDIDATE).id for _ in range(4)]
        session.commit()
    barrier = threading.Barrier(len(ids))

    def activate(prompt_id):
        with Session(migrated) as session:
            row = session.get(PromptVersion, prompt_id)
            barrier.wait(30)
            promote(session, row, S.ACTIVE)
            session.commit()

    with ThreadPoolExecutor(len(ids)) as pool:
        list(pool.map(activate, ids))
    with Session(migrated) as session:
        statuses = list(session.scalars(select(PromptVersion.status)))
    assert sorted(statuses) == [S.ACTIVE, S.RETIRED, S.RETIRED, S.RETIRED]


@pytest.mark.postgres
def test_two_pins_at_once(migrated):
    with Session(migrated) as session:
        job_id = job(session).id
        plans = [recipe(session, name=f"r{i}", to=S.ACTIVE).id for i in range(4)]
        session.commit()
    barrier = threading.Barrier(len(plans))

    def pin(recipe_id):
        with Session(migrated) as session:
            row, plan = session.get(Job, job_id), session.get(Recipe, recipe_id)
            barrier.wait(30)
            try:
                pin_job(session, row, recipe=plan)
            except AlreadyPinnedError:
                return None
            session.commit()
            return recipe_id

    with ThreadPoolExecutor(len(plans)) as pool:
        winners = [x for x in pool.map(pin, plans) if x]
    with Session(migrated) as session:
        assert winners == [session.get(Job, job_id).recipe_id]


@pytest.mark.postgres
def test_0002_goes_down_and_up_again(migrated):
    config = alembic_config(os.environ["MEMCRAFT_TEST_DATABASE_URL"])
    with Session(migrated) as session:
        row_id = workflow(session, to=S.ACTIVE).id
        session.commit()
    command.downgrade(config, "0001")
    with migrated.begin() as connection:
        connection.execute(text("UPDATE workflow_versions SET name = 'changed'"))
        left = "SELECT count(*) FROM pg_proc WHERE proname = 'guard_version_row'"
        assert connection.scalar(text(left)) == 0
    command.upgrade(config, "0002")
    with Session(migrated) as session:
        refused(
            session,
            f"UPDATE workflow_versions SET name = 'x' WHERE id = '{row_id}'",  # noqa: S608
            "cannot be edited",
        )
