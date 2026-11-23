import pytest

from memcraft.models import JobLength, PromptRole, PromptVersion
from memcraft.roles.director import (
    KEYFRAME_LABELS,
    DirectorError,
    FrameCountError,
    SectionsMissingError,
    check_frames,
    direct_keyframes,
    direct_multishot,
    for_ltx,
    image_tags,
    keyframe_frames,
    name_people,
    parse_sections,
    render_seeds,
    run_starts,
    split_runs,
)
from memcraft.roles.qwen import WORKFLOW, answer_text, text_graph
from memcraft.roles.seed import read_prompt
from memcraft.workflows import is_link, load_workflow

SHORT, LONG = JobLength.SHORT, JobLength.LONG
PHOTO = b"\x89PNG\r\n\x1a\n" + b"0" * 10

# Shaped like the E0.10 answers: blank lines between sections, "image1" without brackets.
KEYFRAMES_ONE = """TITLE: The Umbrella Lift

PEOPLE: <image1>: the man in the red jacket, short curly brown hair, full beard, red rain jacket

PLACE: A windy city park in autumn with a wide gravel path.

LIGHT: Soft overcast daylight.

CAMERA: Static eye-level medium shot, waist up.

KEYFRAME START: The man in the red jacket from image1 stands on the path and holds a closed
umbrella, smiling proudly.

KEYFRAME MIDDLE: The man in the red jacket from image1 opens the umbrella;
his heels leave the ground.

KEYFRAME END: The man in the red jacket from Image 1 floats above the path,
clinging to the umbrella with a shocked grin.

RUN 1: He opens the umbrella with a flourish. A gust catches it and lifts him slowly.
He says "Not again:
not today!" as leaves swirl around him. Wind howls.
"""

# Markdown labels, no blank lines, a quoted title, <image2> written as "image 2".
KEYFRAMES_TWO = """Here is the plan.
**TITLE:** "Ocean Crossing"
**PEOPLE:** <image1>: the woman in the yellow sweater, black bob with bangs, round glasses
| <image2>: the bald man, grey moustache, checked flannel shirt
**PLACE:** A tiny rowing boat in a round park fountain.
**LIGHT:** Bright noon sun.
**CAMERA:** Static medium shot from the fountain rim.
**KEYFRAME START:** The woman in the yellow sweater from <image1> grips an oar;
the bald man from image 2 points ahead.
**KEYFRAME MIDDLE:** The woman from <image1> rows hard while the bald man from <image2> salutes.
**KEYFRAME END:** The woman from <image1> and the bald man from <image2> plant a tiny flag
on the rim.
**RUN 1:** They row with grim faces. Water laps.
**RUN 2:** They reach the rim and plant the flag. A duck quacks.
"""

MULTISHOT_ONE = """TITLE: Toast Launch
PEOPLE: the woman in the yellow sweater: black bob with bangs, round glasses, yellow knitted sweater
RUN 1: The woman in the yellow sweater waits by the toaster.
The toast shoots up and sticks to the ceiling.
She sips her tea and says "Fine." Kitchen hum, a loud pop.
"""


def runner(*answers):
    calls = []

    def run_text(system, prompt, images, seed):
        calls.append({"system": system, "prompt": prompt, "images": list(images), "seed": seed})
        return answers[len(calls) - 1]

    return run_text, calls


def clock():
    ticks = iter(range(0, 1000, 60))
    return lambda: next(ticks)


def kf_prompt():
    return PromptVersion(
        role=PromptRole.DIRECTOR_KEYFRAMES,
        model="qwen3.8",
        text=read_prompt(PromptRole.DIRECTOR_KEYFRAMES, "qwen3.8"),
    )


def ms_prompt():
    return PromptVersion(
        role=PromptRole.DIRECTOR_MULTISHOT,
        model="qwen3.8",
        text=read_prompt(PromptRole.DIRECTOR_MULTISHOT, "qwen3.8"),
    )


# Frames


@pytest.mark.parametrize(
    ("frames", "length", "ok"),
    [
        (153, SHORT, True),
        (241, SHORT, True),
        (297, SHORT, True),
        (145, SHORT, False),
        (305, SHORT, False),
        (240, SHORT, False),
        (305, LONG, True),
        (481, LONG, True),
        (593, LONG, True),
        (297, LONG, False),
        (601, LONG, False),
        (482, LONG, False),
    ],
)
def test_frame_counts_are_8n_plus_1_in_range(frames, length, ok):
    if ok:
        assert check_frames(frames, length) == frames
    else:
        with pytest.raises(FrameCountError):
            check_frames(frames, length)


def test_frame_counts_must_be_integers():
    with pytest.raises(FrameCountError):
        check_frames(241.0, SHORT)


@pytest.mark.parametrize("frames", range(305, 594, 8))
def test_a_long_video_splits_into_two_runs_that_share_a_frame(frames):
    first, second = split_runs(frames, LONG)
    assert first + second - 1 == frames
    for run in (first, second):
        assert run % 8 == 1 and 153 <= run <= 297
    assert run_starts([first, second]) == [0, first - 1]


def test_keyframe_positions():
    assert keyframe_frames([241]) == {"start": 0, "middle": 120, "end": 240}
    assert keyframe_frames([249]) == {"start": 0, "middle": 128, "end": 248}
    # Long: the middle guides the first run half way, the end is the last frame of the second.
    assert keyframe_frames([241, 241]) == {"start": 0, "middle": 120, "end": 480}
    assert keyframe_frames([161, 153]) == {"start": 0, "middle": 80, "end": 312}
    for frames in range(153, 298, 8):
        assert keyframe_frames([frames])["middle"] % 8 == 0


def test_positions_match_the_engines_plan():
    from memcraft.engines.recipes import LAST, RULES, plan

    for length, frames in ((SHORT, 241), (SHORT, 249), (LONG, 481), (LONG, 313), (LONG, 593)):
        runs = split_runs(frames, length)
        engines = plan(RULES["keyframes"], length, frames=runs if length is LONG else runs[0])
        assert [run.frames for run in engines.runs] == runs
        ours = keyframe_frames(runs)
        names = engines.keyframes
        guided = {names[k]: f for run in engines.runs for k, f in run.guides if f != LAST}
        assert guided == {"middle": ours["middle"]}
        assert engines.frames == frames and ours["end"] == frames - 1


def test_render_seeds_are_repeatable():
    assert render_seeds(7, 4) == render_seeds(7, 4)
    assert render_seeds(7, 4) != render_seeds(8, 4)
    assert all(0 < seed < 2**31 for seed in render_seeds(1, 10))


# Sections


def test_sections_from_the_e0_shape():
    labels = [*KEYFRAME_LABELS, "RUN 1"]
    found, problems = parse_sections(KEYFRAMES_ONE, labels)
    assert problems == []
    assert found["TITLE"] == "The Umbrella Lift"
    assert found["CAMERA"] == "Static eye-level medium shot, waist up."
    # Quoted speech keeps its colon and line break joins into one paragraph.
    assert '"Not again: not today!"' in found["RUN 1"]
    assert found["RUN 1"].endswith("Wind howls.")


def test_sections_with_markdown_and_quotes():
    found, _ = parse_sections(KEYFRAMES_TWO, [*KEYFRAME_LABELS, "RUN 1", "RUN 2"])
    assert found["TITLE"] == "Ocean Crossing"
    assert found["PLACE"] == "A tiny rowing boat in a round park fountain."
    assert found["RUN 2"].startswith("They reach the rim")


def test_a_missing_or_empty_section_is_named():
    text = KEYFRAMES_ONE.replace("LIGHT: Soft overcast daylight.", "LIGHT:")
    with pytest.raises(SectionsMissingError) as caught:
        parse_sections(text, [*KEYFRAME_LABELS, "RUN 1", "RUN 2"])
    assert caught.value.missing == ["LIGHT", "RUN 2"]
    assert caught.value.raw == text


def test_an_invented_section_ends_the_one_before():
    text = MULTISHOT_ONE + "RUN 2: an extra run nobody asked for\n"
    found, problems = parse_sections(text, ["TITLE", "PEOPLE", "RUN 1"])
    assert "extra run" not in found["RUN 1"]
    assert problems == ["unexpected section RUN 2"]


def test_thinking_and_fences_are_ignored_and_duplicates_keep_the_first():
    text = "<think>TITLE: wrong</think>\n```\nTITLE: Right\nTITLE: Again\nPEOPLE: x\nRUN 1: y\n```"
    found, problems = parse_sections(text, ["TITLE", "PEOPLE", "RUN 1"])
    assert found["TITLE"] == "Right"
    assert problems == ["TITLE appears twice; the first is kept"]


def test_lowercase_labels_and_a_wide_colon_count():
    text = "Title\N{FULLWIDTH COLON} A\npeople: B\nRun 1 - x\nrun  1: C"
    found, _ = parse_sections(text, ["TITLE", "PEOPLE", "RUN 1"])
    # A line without a colon is text, not a label.
    assert found == {"TITLE": "A", "PEOPLE": "B Run 1 - x", "RUN 1": "C"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("from image1", "from <image1>"),
        ("from <image1>", "from <image1>"),
        ("from Image 2,", "from <image2>,"),
        ("image3's hat", "<image3>'s hat"),
        ("the image is sharp", "the image is sharp"),
        ("images 1", "images 1"),
    ],
)
def test_image_tags(text, expected):
    assert image_tags(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The man from image1 waves.", "The man waves."),
        ("The woman from <image2>, smiling, rows.", "The woman, smiling, rows."),
        ("He looks at image 1 again", "He looks at again"),
        ("No tags here.", "No tags here."),
    ],
)
def test_run_prompts_lose_picture_tags(text, expected):
    assert for_ltx(text) == expected


def test_an_answer_written_twice_keeps_the_first_copy():
    # Seen live: Qwen 3.8 repeated its whole answer once, without blank lines.
    found, problems = parse_sections(
        MULTISHOT_ONE + "\n" + MULTISHOT_ONE, ["TITLE", "PEOPLE", "RUN 1"]
    )
    assert found["RUN 1"].endswith("a loud pop.")
    assert len(problems) == 3


# Keyframes


def test_a_short_keyframe_plan():
    run_text, calls = runner(KEYFRAMES_ONE)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "He flies away", seed=7, clock=clock())
    assert plan.title == "The Umbrella Lift" and plan.attempts == 1 and plan.seconds == 60
    assert calls[0]["seed"] == 7 and calls[0]["images"] == [PHOTO]
    assert "8 seconds long, made as one run" in calls[0]["system"]
    assert "RUN 1: the whole video" in calls[0]["system"] and "RUN 2" not in calls[0]["system"]
    assert calls[0]["prompt"] == "Idea: He flies away"
    assert plan.bible["people"] == [
        {
            "image": 1,
            "tag": "the man in the red jacket",
            "look": "short curly brown hair, full beard, red rain jacket",
        }
    ]
    start, middle, end = plan.keyframes
    assert start.action.startswith("Static eye-level medium shot, waist up. The man")
    assert start.references == [1]
    assert middle.references == [1, "start"]
    assert "from <image1>" in end.action and "Image 1" not in end.action
    assert all(
        k.action.endswith("Keep every face and identity exactly as in the photos.")
        for k in plan.keyframes
    )
    # The full prompt: the new place first for the start, the same scene for the others.
    assert start.prompt(plan.bible).startswith("New place, replace every photo background")
    assert "A windy city park in autumn" in start.prompt(plan.bible)
    assert middle.prompt(plan.bible).startswith("Same scene as <image2>:")
    assert [k.frame for k in plan.keyframes] == [0, 120, 240]
    assert plan.frames == 241
    (run,) = plan.runs
    assert (run.frames, run.start, run.first) == (241, 0, "keyframe")
    assert run.keyframes == [("start", 0), ("middle", 120), ("end", 240)]
    # Picture tags mean nothing to LTX.
    assert "image" not in run.prompt.lower()
    assert len({k.seed for k in plan.keyframes} | {run.seed}) == 4
    assert plan.problems == []
    output = plan.as_output()
    assert output["keyframes"][0]["at"] == "start"
    assert output["keyframes"][1]["prompt"] == middle.prompt(plan.bible)


def test_a_long_keyframe_plan_with_two_photos():
    run_text, calls = runner(KEYFRAMES_TWO)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO, PHOTO], "Ocean", length=LONG)
    assert "made as two runs" in calls[0]["system"]
    assert "RUN 2: continues from the last frame of RUN 1" in calls[0]["system"]
    assert [k.frame for k in plan.keyframes] == [0, 120, 480]
    first, second = plan.runs
    assert (first.frames, first.start, first.first) == (241, 0, "keyframe")
    assert first.keyframes == [("start", 0), ("middle", 120)]
    assert (second.frames, second.start, second.first) == (241, 240, "previous_last_frame")
    assert second.keyframes == [("end", 240)]
    assert plan.keyframes[0].action.count("<image2>") == 1
    assert plan.keyframes[1].prompt(plan.bible).startswith("Same scene as <image3>:")
    assert [p.get("image") for p in plan.bible["people"]] == [1, 2]
    assert plan.problems == []


def test_people_not_named_or_too_many_images_are_problems():
    text = KEYFRAMES_TWO.replace("the bald man from image 2 points", "someone points").replace(
        "plant a tiny flag", "meet the man from image3 and plant a tiny flag"
    )
    run_text, _ = runner(text)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO, PHOTO], "Ocean", length=LONG)
    assert "KEYFRAME START: <image2> not named; a sentence was added" in plan.problems
    assert "KEYFRAME END: names <image3> but there are 2 photos" in plan.problems
    assert plan.keyframes[0].action.endswith(
        "The bald man from <image2>: grey moustache, checked flannel shirt. "
        "Keep every face and identity exactly as in the photos."
    )


def test_a_tag_without_its_picture_gets_it():
    # Seen live: Qwen 3.8 wrote "The red-jacket man sits ..." without "from image1".
    people = [{"image": 1, "tag": "the red-jacket man", "look": "beard"}]
    body, notes = name_people("The red-jacket man sits. The red-jacket man waves.", people, 1)
    assert body == "The red-jacket man from <image1> sits. The red-jacket man waves."
    assert notes == ["<image1> added after 'the red-jacket man'"]
    assert name_people("the man from <image1>", people, 1) == ("the man from <image1>", [])


def test_a_missing_section_costs_one_retry_with_a_new_seed():
    broken = KEYFRAMES_ONE.split("KEYFRAME END")[0]
    run_text, calls = runner(broken, KEYFRAMES_ONE)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "idea", seed=11)
    assert [call["seed"] for call in calls] == [11, 12]
    assert plan.attempts == 2 and plan.seed == 12
    assert plan.problems[0].startswith("first answer: missing KEYFRAME END, RUN 1")


def test_two_misses_give_up_with_the_raw_text():
    run_text, calls = runner("TITLE: only", "nothing")
    with pytest.raises(SectionsMissingError) as caught:
        direct_keyframes(run_text, kf_prompt(), [PHOTO], "idea")
    assert len(calls) == 2 and caught.value.raw == "nothing"


def test_keyframe_arguments():
    run_text, _ = runner()
    with pytest.raises(DirectorError):
        direct_keyframes(run_text, kf_prompt(), [], "idea")
    with pytest.raises(DirectorError):
        direct_keyframes(run_text, kf_prompt(), [PHOTO], " ")
    with pytest.raises(FrameCountError):
        direct_keyframes(run_text, kf_prompt(), [PHOTO], "idea", frames=300)


# Multishot


def test_a_single_take():
    run_text, calls = runner(MULTISHOT_ONE)
    plan = direct_multishot(run_text, ms_prompt(), PHOTO, "Toast flies")
    system = calls[0]["system"]
    assert "about 8 seconds" in system and "exactly 1 shot(s)" in system
    assert "One continuous take, no cuts." in system and "hard cut" not in system
    assert "made in one run" in system and "RUN 2" not in system
    assert plan.people == [
        {
            "tag": "the woman in the yellow sweater",
            "look": "black bob with bangs, round glasses, yellow knitted sweater",
        }
    ]
    (run,) = plan.runs
    assert run.frames == 241 and run.start == 0 and run.prompt.endswith("a loud pop.")
    assert '"Fine."' in run.prompt
    assert plan.problems == []


def test_two_shots_in_a_long_video_continue_from_the_last_frame():
    answer = (
        "TITLE: T\nPEOPLE: the man in red: beard\n"
        "RUN 1: He waits. A hard cut transitions to a close-up of the man in red.\n"
        "RUN 2: He laughs, no cut at all, and then a cut to the sky.\n"
    )
    run_text, calls = runner(answer)
    plan = direct_multishot(run_text, ms_prompt(), PHOTO, "idea", length=LONG, shots=2, frames=593)
    system = calls[0]["system"]
    assert "two runs of about 10 seconds" in system and "hard cut" in system
    assert "RUN 2: the second half, starting from the last frame of RUN 1" in system
    assert [(r.frames, r.start, r.first) for r in plan.runs] == [
        (297, 0, "photo"),
        (297, 296, "previous_last_frame"),
    ]
    assert plan.problems == []


def test_cut_words_in_a_single_take_are_a_problem():
    answer = MULTISHOT_ONE.replace("She sips", "A hard cut transitions to her. She sips")
    run_text, _ = runner(answer)
    plan = direct_multishot(run_text, ms_prompt(), PHOTO, "idea")
    assert plan.problems == ["RUN 1: 2 cut words in a single take"]


def test_multishot_arguments():
    run_text, _ = runner()
    with pytest.raises(DirectorError):
        direct_multishot(run_text, ms_prompt(), PHOTO, "idea", shots=3)
    with pytest.raises(FrameCountError):
        direct_multishot(run_text, ms_prompt(), PHOTO, "idea", length=LONG, frames=241)


# The Qwen text graph


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_the_text_graph_sees_each_picture_once(count):
    package = load_workflow(WORKFLOW)
    names = [f"memcraft/p{i}.png" for i in range(count)]
    graph = text_graph(package, "sys", "user", names, 5, max_length=900)
    loaded = [
        node["inputs"]["image"] for node in graph.values() if node["class_type"] == "LoadImage"
    ]
    assert loaded == names
    (generate,) = [node for node in graph.values() if node["class_type"] == "TextGenerateV2"]
    links = sorted(key for key, value in generate["inputs"].items() if key.startswith("images."))
    assert links == [f"images.image{i + 1}" for i in range(count)]
    assert generate["inputs"]["prompt"] == "user"
    assert generate["inputs"]["sampling_mode.seed"] == 5
    assert generate["inputs"]["max_length"] == 900
    system = generate["inputs"]["system_prompt"]
    assert graph[system[0]]["inputs"]["value"] == "sys"
    # Every link points at a node that is still there.
    for node in graph.values():
        for value in node["inputs"].values():
            if is_link(value):
                assert value[0] in graph


def test_the_text_graph_needs_a_picture():
    with pytest.raises(Exception, match="1 to 4"):
        text_graph(load_workflow(WORKFLOW), "s", "u", [], 1)


def test_answer_text():
    assert answer_text({"12": {"text": ["TITLE: a", "\nPEOPLE: b"]}}, "12") == "TITLE: a\nPEOPLE: b"
    assert answer_text({"12": {"text": "x"}}, "12") == "x"
    with pytest.raises(Exception, match="no text"):
        answer_text({}, "12")


def test_the_runner_uploads_by_hash_and_runs_only_the_text_node():
    from memcraft.comfy import RunResult
    from memcraft.roles.qwen import ComfyTextRunner

    class FakeComfy:
        def __init__(self):
            self.uploads, self.runs = [], []

        def upload(self, data, name, *, subfolder=""):
            self.uploads.append((name, subfolder))
            return f"{subfolder}/{name}"

        def run(self, graph, *, timeout, outputs, should_cancel=None):
            self.runs.append((graph, outputs, timeout))

            return RunResult("id", {"12": {"text": ["TITLE: ok"]}}, 1.0)

    comfy = FakeComfy()
    run_text = ComfyTextRunner(comfy, timeout=99)
    assert run_text("sys", "user", [PHOTO, b"\xff\xd8\xff" + b"1"], 3) == "TITLE: ok"
    (first, _), (second, _) = comfy.uploads
    assert first.endswith(".png") and second.endswith(".jpg") and len(first) == 36
    _, outputs, timeout = comfy.runs[0]
    assert outputs == ["12"] and timeout == 99
