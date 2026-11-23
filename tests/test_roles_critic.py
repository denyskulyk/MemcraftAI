import uuid

import pytest
from sqlalchemy.orm import Session
from test_roles_gemma import JPEG, FakeServer, reply

from memcraft.models import (
    EngineSlot,
    Job,
    JobLength,
    JobQuality,
    JobRegime,
    JobStep,
    PromptRole,
    PromptVersion,
    Setting,
    StepName,
    User,
    VersionStatus,
)
from memcraft.roles.critic import (
    SETTING,
    Critic,
    CriticError,
    check_with_gemma,
    check_with_qwen,
    choose,
    chosen_engine,
    loose_json,
    pinned_critic,
    read_verdict,
    schema,
)
from memcraft.roles.gemma import GemmaClient, GemmaError
from memcraft.roles.seed import read_prompt, seed_prompts
from memcraft.versions import (
    create_engine_version,
    create_recipe_version,
    create_workflow_version,
    pin_job,
    promote,
)

PASS = {
    "face_why": "Same round face and beard.",
    "face": "pass",
    "story_why": "He floats.",
    "story": "pass",
    "artefacts_why": "Clean.",
    "artefacts": "pass",
    "correction": "",
}
FAIL = PASS | {"face": "fail", "face_why": "Another\nperson.", "correction": "Keep his beard."}


def prompt(model="gemma"):
    return PromptVersion(
        role=PromptRole.CRITIC, model=model, text=read_prompt(PromptRole.CRITIC, model)
    )


def test_the_schema_puts_each_reason_before_its_verdict():
    assert schema()["required"] == [
        "face_why",
        "face",
        "story_why",
        "story",
        "artefacts_why",
        "artefacts",
        "correction",
    ]


def test_gemma_sees_photos_then_the_frame():
    server = FakeServer(reply(FAIL))
    client = GemmaClient("http://127.0.0.1:8098", post=server)
    frame = b"\x89PNG\r\n\x1a\nframe"
    verdict = check_with_gemma(client, prompt(), [JPEG, JPEG], frame, "He floats away.")
    assert not verdict.passed and not verdict.face and verdict.story
    assert verdict.face_why == "Another person."
    assert verdict.correction == "Keep his beard."
    body = server.calls[0][1]
    content = body["messages"][1]["content"]
    assert content[2]["image_url"]["url"].startswith("data:image/png")
    assert "Source photos: 2." in content[3]["text"] and "He floats away." in content[3]["text"]
    assert body["temperature"] == 0.2


def test_all_pass_drops_the_correction():
    verdict = read_verdict(PASS | {"correction": "something"}, "gemma", 1.0)
    assert verdict.passed and verdict.correction == ""
    assert verdict.as_output()["passed"] is True


@pytest.mark.parametrize(
    "data",
    [None, [], {k: v for k, v in PASS.items() if k != "story"}, PASS | {"story": "mostly"}],
)
def test_an_unclear_verdict_is_an_error(data):
    with pytest.raises(CriticError):
        read_verdict(data, "qwen3.8", 1.0)


def test_a_gemma_answer_outside_the_enum_is_a_gemma_error():
    client = GemmaClient("http://x:1", post=FakeServer(reply(PASS | {"face": "maybe"})))
    with pytest.raises(GemmaError):
        check_with_gemma(client, prompt(), [JPEG], JPEG, "s")


def test_loose_json():
    assert loose_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert loose_json('<think>x</think>Sure: {"a": "line\nbreak"} done') == {"a": "line\nbreak"}
    assert loose_json("no json") is None


def test_qwen_retries_once_when_a_check_is_missing():
    import json

    missing = {k: v for k, v in PASS.items() if k != "story"}  # as Qwen did in E0.14
    answers = [json.dumps(missing), "```" + json.dumps(PASS) + "```"]
    seeds = []

    def run_text(system, user, images, seed):
        seeds.append((seed, len(images)))
        return answers[len(seeds) - 1]

    verdict = check_with_qwen(run_text, prompt("qwen3.8"), [JPEG], JPEG, "s", seed=3)
    assert verdict.passed and verdict.attempts == 2 and verdict.model == "qwen3.8"
    assert seeds == [(3, 2), (4, 2)]


def test_a_critic_needs_its_runtime():
    engine = type("E", (), {"name": "gemma"})()
    with pytest.raises(CriticError):
        Critic(engine, prompt())
    engine.name = "other"
    with pytest.raises(CriticError):
        Critic(engine, prompt(), gemma=GemmaClient("http://x:1"))


# The setting and the pin


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def active(session, row):
    promote(session, row, VersionStatus.CANDIDATE)
    promote(session, row, VersionStatus.ACTIVE)
    return row


def critic_engine(session, name):
    return active(session, create_engine_version(session, slot=EngineSlot.CRITIC, name=name))


def started_job(session, critic):
    workflow = create_workflow_version(
        session,
        {"1": {"class_type": "SaveImage", "inputs": {"seed": 1}}},
        {
            "format": 1,
            "name": f"wf-{uuid.uuid4().hex[:6]}",
            "version": 1,
            "inputs": {},
            "outputs": {"video": {"node": "1", "class_type": "SaveImage"}},
        },
    )
    active(session, workflow)
    clip = active(
        session,
        create_engine_version(
            session, slot=EngineSlot.CLIP, name="ltx-2.5", workflow_version_id=workflow.id
        ),
    )
    recipe = active(
        session,
        create_recipe_version(
            session, name="multishot", regime=JobRegime.MULTISHOT, clip_engine_id=clip.id
        ),
    )
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(owner_id=user.id, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720)
    session.add(job)
    session.flush()
    pin_job(session, job, recipe=recipe, critic=critic)
    return job


def test_the_setting_chooses_gemma_qwen_or_off(session):
    seed_prompts(session)
    session.add(Setting(key=SETTING, value=None))
    session.flush()
    assert chosen_engine(session) is None
    gemma = critic_engine(session, "gemma")
    qwen = critic_engine(session, "qwen3.8")
    choose(session, gemma)
    assert chosen_engine(session) == gemma
    choose(session, qwen)
    assert chosen_engine(session) == qwen
    choose(session, None)
    assert chosen_engine(session) is None
    session.get(Setting, SETTING).value = "not an id"
    session.flush()
    with pytest.raises(CriticError):
        chosen_engine(session)


def test_a_job_keeps_the_critic_it_started_with(session):
    seed_prompts(session)
    gemma = critic_engine(session, "gemma")
    qwen = critic_engine(session, "qwen3.8")
    choose(session, gemma)
    job = started_job(session, chosen_engine(session))
    choose(session, qwen)  # changing the setting later does not move the job
    client = GemmaClient("http://x:1", post=FakeServer(reply(PASS)))
    critic = pinned_critic(session, job, gemma=client)
    assert critic.engine == gemma and not critic.needs_gpu
    assert critic.prompt.model == "gemma" and critic.prompt.role is PromptRole.CRITIC

    step = JobStep(job_id=job.id, step=StepName.CHECK)
    session.add(step)
    session.flush()
    verdict = critic.check_step(session, step, [JPEG], JPEG, "story")
    assert verdict.passed
    assert step.prompt_version_id == critic.prompt.id

    off = started_job(session, None)
    assert pinned_critic(session, off, gemma=client) is None
