from pathlib import Path

import pytest

from memcraft.models import JobLength, PromptRole, PromptVersion
from memcraft.roles import director
from memcraft.roles.director import (
    KEYFRAME_LABELS,
    SAME_FRAMING,
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
    no_music,
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


def kf_prompt(compose=False):
    """The shipped prompt; without ``compose`` its background part is cut off, which is how
    versions before E16's background-first layout read."""
    text = read_prompt(PromptRole.DIRECTOR_KEYFRAMES, "qwen3.8")
    if not compose:
        text = text.split("=== background ===")[0]
    return PromptVersion(role=PromptRole.DIRECTOR_KEYFRAMES, model="qwen3.8", text=text)


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
        # E16: a short video has no middle keyframe.
        assert guided == ({"middle": ours["middle"]} if length is LONG else {})
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
    assert "No music." in calls[0]["system"] and "no cranes" in calls[0]["system"]
    assert "RUN 1: the whole video" in calls[0]["system"] and "RUN 2" not in calls[0]["system"]
    assert calls[0]["prompt"] == "Idea: He flies away"
    assert plan.bible["people"] == [
        {
            "image": 1,
            "tag": "the man in the red jacket",
            "look": "short curly brown hair, full beard, red rain jacket",
        }
    ]
    # E16: a short video draws start and end; the director's middle is not drawn.
    start, end = plan.keyframes
    assert (start.at, end.at) == ("start", "end")
    assert end.frame == 240 and end.seed == director.render_seeds(7, 4)[2]
    assert start.action.startswith("Static eye-level medium shot, waist up. The man")
    assert start.references == [1]
    assert end.references == [1, "start"]
    assert plan.runs[0].keyframes == [("start", 0), ("end", 240)]
    assert "from <image1>" in end.action and "Image 1" not in end.action
    assert all(
        k.action.endswith("Keep every face and identity exactly as in the photos.")
        for k in plan.keyframes
    )
    # The full prompt: the new place first for the start, the same scene for the others.
    assert start.prompt(plan.bible).startswith("New place, replace every photo background")
    assert "A windy city park in autumn" in start.prompt(plan.bible)
    assert end.prompt(plan.bible).startswith("Same scene as <image2>:")
    assert [k.frame for k in plan.keyframes] == [0, 240]
    assert plan.frames == 241
    (run,) = plan.runs
    assert (run.frames, run.start, run.first) == (241, 0, "keyframe")
    assert run.keyframes == [("start", 0), ("end", 240)]
    # Picture tags mean nothing to LTX, and there is never music.
    assert "image" not in run.prompt.lower()
    assert run.prompt.endswith("Wind howls. No music.")
    assert len({k.seed for k in plan.keyframes} | {run.seed}) == 3
    assert plan.problems == []
    output = plan.as_output()
    assert output["keyframes"][0]["at"] == "start"
    assert output["keyframes"][1]["prompt"] == end.prompt(plan.bible)


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


THREE_PEOPLE = KEYFRAMES_TWO.replace(
    "| <image2>: the bald man", "| <image3>: the girl in red, braids | <image2>: the bald man"
)
NO_PEOPLE = KEYFRAMES_TWO.replace(KEYFRAMES_TWO.split("**PEOPLE:** ")[1].split("\n**PLACE")[0], "|")


def test_three_people_draw_middle_and_end_from_the_photos_only():
    # E0: with 3 people the start keyframe as a reference lost a face and added noise.
    run_text, _ = runner(THREE_PEOPLE)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO] * 3, "Ocean", length=LONG)
    assert len(plan.bible["people"]) == 3
    start, middle, end = plan.keyframes
    assert start.references == [1, 2, 3]
    assert middle.references == end.references == [1, 2, 3]
    for keyframe in (middle, end):
        assert keyframe.action.startswith(
            f"Static medium shot from the fountain rim. {SAME_FRAMING}"
        )
        prompt = keyframe.prompt(plan.bible)
        assert prompt.startswith("New place, replace every photo background completely:")
        assert "A tiny rowing boat in a round park fountain." in prompt
        assert "same place, light, camera position and framing" in prompt.lower()
        assert "<image4>" not in prompt


def test_two_people_keep_the_start_keyframe_as_a_reference():
    run_text, _ = runner(KEYFRAMES_TWO)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO] * 2, "Ocean", length=LONG)
    _, middle, end = plan.keyframes
    assert middle.references == end.references == [1, 2, "start"]
    assert SAME_FRAMING not in middle.action


@pytest.mark.parametrize(("photos", "chained"), [(2, True), (3, False)])
def test_without_people_in_the_bible_the_photos_count(photos, chained):
    run_text, _ = runner(NO_PEOPLE)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO] * photos, "Ocean", length=LONG)
    assert plan.bible["people"] == []
    assert ("start" in plan.keyframes[1].references) is chained
    assert ("start" in plan.keyframes[2].references) is chained


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
    assert "about 8 seconds" in system and "one continuous take with no cuts" in system
    assert "hard cut" not in system and "$" not in system
    assert "No music." in system and "no cranes" in system
    assert "made in one run" in system and "RUN 2" not in system
    assert plan.people == [
        {
            "tag": "the woman in the yellow sweater",
            "look": "black bob with bangs, round glasses, yellow knitted sweater",
        }
    ]
    (run,) = plan.runs
    assert run.frames == 241 and run.start == 0 and run.prompt.endswith("a loud pop. No music.")
    assert '"Fine."' in run.prompt
    assert plan.problems == []


def test_a_long_single_take_continues_from_the_last_frame():
    answer = (
        "TITLE: T\nPEOPLE: the man in red: beard\n"
        'RUN 1: He waits. Soft piano music plays. He shouts "Now!"\n'
        "RUN 2: He laughs out loud. No music.\n"
    )
    run_text, calls = runner(answer)
    plan = direct_multishot(run_text, ms_prompt(), PHOTO, "idea", length=LONG, frames=593)
    system = calls[0]["system"]
    assert "two runs of about 10 seconds" in system and "hard cut" not in system
    assert "RUN 2: the second half, starting from the last frame of RUN 1" in system
    assert [(r.frames, r.start, r.first) for r in plan.runs] == [
        (297, 0, "photo"),
        (297, 296, "previous_last_frame"),
    ]
    # The music sentence goes, "No music." comes last.
    assert [r.prompt for r in plan.runs] == [
        'He waits. He shouts "Now!" No music.',
        "He laughs out loud. No music.",
    ]
    assert plan.shots == 1 and plan.problems == []


def test_cut_words_in_a_single_take_are_a_problem():
    answer = MULTISHOT_ONE.replace("She sips", "A hard cut transitions to her. She sips")
    run_text, _ = runner(answer)
    plan = direct_multishot(run_text, ms_prompt(), PHOTO, "idea")
    assert plan.problems == ["RUN 1: 2 cut words in a single take"]


def test_multishot_arguments():
    run_text, _ = runner()
    # One photo makes a single take only (E0: faces held in 4 of 5, with 2 shots in 2 of 4).
    with pytest.raises(DirectorError, match="single take"):
        direct_multishot(run_text, ms_prompt(), PHOTO, "idea", shots=2)
    with pytest.raises(FrameCountError):
        direct_multishot(run_text, ms_prompt(), PHOTO, "idea", length=LONG, frames=241)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("He jumps. Upbeat music swells. Glass shatters.", "He jumps. Glass shatters. No music."),
        ('She yells "Stop!" A musical sting plays.', 'She yells "Stop!" No music.'),
        ("A soundtrack hums under it all", "No music."),
        ("Loud crash and laughter.", "Loud crash and laughter. No music."),
        ("They cheer. No music.", "They cheer. No music."),
    ],
)
def test_run_prompts_never_ask_for_music(text, expected):
    assert no_music(text) == expected
    assert no_music(no_music(text)) == expected


@pytest.mark.parametrize("role", [PromptRole.DIRECTOR_MULTISHOT, PromptRole.DIRECTOR_KEYFRAMES])
def test_director_prompts_carry_the_e0_rules(role):
    text = read_prompt(role, "qwen3.8")
    # Props LTX draws badly.
    assert "no cranes (birds or machines)" in text
    assert "no pigeons or other birds in flight" in text
    assert "no knives or blades" in text
    # Loud sound with emotion, never music.
    assert "loud, clear sound effects" in text and "voices with emotion" in text
    assert "laughing, shouting" in text and "end every" in text and '"No music."' in text


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


@pytest.mark.parametrize(
    "name", ["director-keyframes-qwen3.8.txt", "director-multishot-qwen3.8.txt"]
)
def test_russian_speech_uses_acute_stress_marks(name):
    text = (Path(director.__file__).parent / "prompts" / name).read_text(encoding="utf-8")
    assert "U+0301" in text and "кре́сло" in text


def test_stress_marks_survive_the_run_prompt():
    said = 'He shouts "Где моё кре́сло?"'
    assert "кре́сло" in director.run_prompt(said)


# Framing: the camera and the props word for word in every keyframe


WITH_PROPS = KEYFRAMES_ONE.replace(
    "CAMERA: Static eye-level medium shot, waist up.",
    "CAMERA: Static eye-level full-body shot.\n\n"
    "PROPS: a black umbrella with a curved wooden handle | a small brown paper bag",
)


def test_the_camera_and_the_props_go_into_every_keyframe():
    # Seen live: a waist-up start, a full-body middle, and a black umbrella that turned red.
    run_text, calls = runner(WITH_PROPS)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "He flies away")
    assert "PROPS:" in calls[0]["system"] and "word for word" in calls[0]["system"]
    assert plan.bible["props"] == [
        "a black umbrella with a curved wooden handle",
        "a small brown paper bag",
    ]
    props = (
        "Each of these objects is there only once and keeps exactly this look in every picture: "
        "a black umbrella with a curved "
        "wooden handle; a small brown paper bag."
    )
    for keyframe in plan.keyframes:
        assert keyframe.action.startswith("Static eye-level full-body shot. The man")
        assert keyframe.action.endswith(
            f"{props} Keep every face and identity exactly as in the photos."
        )
        assert keyframe.action.count("Static eye-level full-body shot.") == 1
    assert plan.keyframes[1].references == [1, "start"]
    assert "shot size" in plan.keyframes[1].prompt(plan.bible)
    assert plan.problems == []


def test_three_people_get_camera_props_and_same_framing():
    text = THREE_PEOPLE.replace(
        "**KEYFRAME START:**", "**PROPS:** a tiny red flag on a stick\n**KEYFRAME START:**"
    )
    run_text, _ = runner(text)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO] * 3, "Ocean", length=LONG)
    start, middle, _ = plan.keyframes
    assert start.action.startswith("Static medium shot from the fountain rim. The woman")
    assert middle.action.startswith(f"Static medium shot from the fountain rim. {SAME_FRAMING}")
    assert "a tiny red flag on a stick." in middle.action


@pytest.mark.parametrize("value", ["none", "None.", "n/a", " | "])
def test_no_props(value):
    assert director.props_of(value) == []


def test_an_answer_without_props_still_parses():
    # Jobs directed with an older prompt version, which did not ask for PROPS.
    run_text, _ = runner(KEYFRAMES_ONE)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "He flies away")
    assert plan.bible["props"] == []
    assert all("these objects" not in k.action for k in plan.keyframes)
    assert plan.keyframes[1].action.startswith("Static eye-level medium shot, waist up. The man")


def test_props_do_not_run_into_the_camera():
    found, problems = parse_sections(
        WITH_PROPS, [*KEYFRAME_LABELS, "RUN 1"], director.OPTIONAL_KEYFRAME_LABELS
    )
    assert found["CAMERA"] == "Static eye-level full-body shot."
    assert found["PROPS"].startswith("a black umbrella") and problems == []


# The cast: people the idea names.

CAST = [
    {"tag": "denys", "photo": 1, "position": 1},
    {"tag": "maksym", "photo": 1, "position": 2},
    {"tag": "olga", "photo": 2, "position": 1},
]
CAST_IDEA = '<denys> says "Hi!", <maksym> answers "Hi to you too", and <olga> laughs.'

# Two photos, three people; the director left two name tags in its answer.
KEYFRAMES_CAST = """TITLE: Hello Chain
PEOPLE: <image1>: the man in the red jacket, short dark hair
| <image1>: the man in the grey hoodie, buzz cut
| <image2>: the woman in the green coat, long blond hair
PLACE: A sunny street corner.
LIGHT: Midday sun.
CAMERA: Static full-body shot at eye level.
KEYFRAME START: The man in the red jacket from <image1> waves at the man in the grey hoodie
from <image1>; the woman in the green coat from <image2> watches.
KEYFRAME MIDDLE: The man in the grey hoodie from <image1> waves back, grinning, while the woman
in the green coat from <image2> covers her mouth.
KEYFRAME END: The woman in the green coat from <image2> laughs, the man in the red jacket
from <image1> and <maksym> shrug.
RUN 1: The man in the red jacket waves and says "Hi!". <maksym> answers "Hi to you too".
The woman in the green coat bursts out laughing. Street noise.
"""


def test_a_full_length_idea_reaches_the_director_whole():
    from memcraft.web.upload import MAX_IDEA

    idea = ("<denys> greets <maksym>. " * 60)[
        : MAX_IDEA - 40
    ] + " the very last detail: a red kite."
    idea = idea + "x" * (MAX_IDEA - len(idea))
    assert len(idea) == MAX_IDEA
    run_text, calls = runner(KEYFRAMES_CAST)
    direct_keyframes(run_text, kf_prompt(), [PHOTO, PHOTO], idea, cast=CAST, clock=clock())
    user = calls[0]["prompt"]
    assert user.startswith(f"Idea: {idea}\n\nCAST:") and "a red kite" in user
    run_text, calls = runner(MULTISHOT_ONE)
    direct_multishot(run_text, ms_prompt(), PHOTO, idea, cast=CAST[:2])
    assert calls[0]["prompt"].startswith(f"Idea: {idea}\n\nCAST:")
    # And the text graph puts it in ComfyUI's prompt input as it is.
    graph = text_graph(load_workflow(WORKFLOW), "sys", calls[0]["prompt"], ["p.png"], 1)
    (generate,) = [node for node in graph.values() if node["class_type"] == "TextGenerateV2"]
    assert generate["inputs"]["prompt"] == calls[0]["prompt"]


def test_the_director_gets_a_cast_and_writes_visual_tags():
    run_text, calls = runner(KEYFRAMES_CAST)
    plan = direct_keyframes(
        run_text, kf_prompt(), [PHOTO, PHOTO], CAST_IDEA, cast=CAST, clock=clock()
    )
    user = calls[0]["prompt"]
    assert user.startswith(f"Idea: {CAST_IDEA}\n\nCAST: ")
    assert "<denys> = <image1>, the 1st of 2 people from the left" in user
    assert "<maksym> = <image1>, the 2nd of 2 people from the left" in user
    assert "<olga> = <image2>, the person in it" in user
    assert "the man in the red jacket says" in user and "Keep the spoken words" in user
    assert "acute accent" in calls[0]["system"]  # the Russian speech rule stays
    assert [(p["image"], p["tag"]) for p in plan.bible["people"]] == [
        (1, "the man in the red jacket"),
        (1, "the man in the grey hoodie"),
        (2, "the woman in the green coat"),
    ]
    end = plan.keyframes[1].action
    assert "and the man in the grey hoodie from <image1> shrug" in end
    assert "<maksym>" not in end
    (run,) = plan.runs
    assert "<" not in run.prompt and "maksym" not in run.prompt.lower()
    assert '. The man in the grey hoodie answers "Hi to you too".' in run.prompt
    assert plan.problems == [
        "KEYFRAME END: 1 name tags replaced",
        "RUN 1: 1 name tags replaced by visual tags",
    ]
    # Three people over two photos: middle and end come from the photos only.
    assert plan.keyframes[1].references == [1, 2]


def test_a_person_in_the_wrong_photo_or_missing_is_a_problem():
    answer = KEYFRAMES_CAST.replace(
        "\n| <image2>: the woman in the green coat, long blond hair", ""
    ).replace("<image1>: the man in the grey hoodie", "<image2>: the man in the grey hoodie")
    run_text, _ = runner(answer)
    plan = direct_keyframes(
        run_text, kf_prompt(), [PHOTO, PHOTO], CAST_IDEA, cast=CAST, clock=clock()
    )
    assert "PEOPLE does not list <olga>" in plan.problems
    assert "PEOPLE puts <maksym> in the wrong photo" in plan.problems


def test_without_tags_the_director_call_is_as_before():
    run_text, calls = runner(KEYFRAMES_ONE)
    plan = direct_keyframes(
        run_text, kf_prompt(), [PHOTO], "He flies away", cast=CAST, clock=clock()
    )
    assert calls[0]["prompt"] == "Idea: He flies away"
    assert plan.problems == []
    run_text, calls = runner(MULTISHOT_ONE)
    direct_multishot(run_text, ms_prompt(), PHOTO, "Toast flies", cast=CAST)
    assert calls[0]["prompt"] == "Idea: Toast flies"


def test_a_multishot_cast():
    answer = MULTISHOT_ONE.replace(
        "PEOPLE: the woman in the yellow sweater: black bob",
        "PEOPLE: the man in the red jacket: short hair"
        " | the woman in the yellow sweater: black bob",
    ).replace('She sips her tea and says "Fine."', '<maksym> sips her tea and says "Fine."')
    cast = [
        {"tag": "denys", "photo": 1, "position": 1},
        {"tag": "maksym", "photo": 1, "position": 2},
    ]
    run_text, calls = runner(answer)
    plan = direct_multishot(run_text, ms_prompt(), PHOTO, "<maksym> and <denys> wait", cast=cast)
    assert "<image1> is the photo" in calls[0]["prompt"]
    assert "<denys> = <image1>, the 1st of 2 people from the left" in calls[0]["prompt"]
    (run,) = plan.runs
    assert 'The woman in the yellow sweater sips her tea and says "Fine."' in run.prompt


@pytest.mark.parametrize("role", [PromptRole.DIRECTOR_MULTISHOT, PromptRole.DIRECTOR_KEYFRAMES])
def test_an_older_director_prompt_takes_the_shipped_cast_part(role):
    text = read_prompt(role, "qwen3.8")
    older = PromptVersion(role=role, model="qwen3.8", text=text.split("=== cast ===")[0])
    run_text, calls = runner(
        KEYFRAMES_CAST if role is PromptRole.DIRECTOR_KEYFRAMES else MULTISHOT_ONE
    )
    if role is PromptRole.DIRECTOR_KEYFRAMES:
        direct_keyframes(run_text, older, [PHOTO, PHOTO], CAST_IDEA, cast=CAST, clock=clock())
    else:
        direct_multishot(run_text, older, PHOTO, CAST_IDEA, cast=CAST[:2])
    assert "\n\nCAST: " in calls[0]["prompt"]


def test_keyframe_prompt_keeps_actions_before_and_after():
    text = (
        Path(director.__file__).parent / "prompts" / "director-keyframes-qwen3.8.txt"
    ).read_text(encoding="utf-8")
    assert "wind-up" in text and "never a pose that undoes it" in text


WITH_ASSETS = KEYFRAMES_ONE.replace(
    "KEYFRAME START:",
    "PROPS: a black umbrella with a curved wooden handle | a red kite\n"
    "CLEAN: <image1>\n"
    "PROP PICTURES: a black umbrella with a curved wooden handle from <image1> | a red kite"
    " | a flying saucer\n"
    "KEYFRAME START:",
)


def test_asset_pictures_replace_the_photo_and_join_the_start_keyframe():
    run_text, _ = runner(WITH_ASSETS)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "He flies away")
    clean, umbrella, kite = plan.assets
    assert (clean.kind, clean.photo) == ("clean", 1)
    assert clean.prompt.startswith("Edit <image1>: exactly the same the man in the red jacket")
    assert "the man in the red jacket: short curly brown hair" in clean.prompt
    assert (umbrella.kind, umbrella.photo) == ("prop", 1)
    assert umbrella.prompt.startswith("Show only the object from <image1>: a black umbrella")
    assert (kite.kind, kite.photo) == ("prop", None) and kite.prompt.startswith("On the plain")
    assert "PROP PICTURES: 'a flying saucer' is not in PROPS; left out" in plan.problems
    start, end = plan.keyframes
    assert start.references == ["asset:0", "asset:1", "asset:2"]
    prompt = start.prompt(plan.bible)
    assert "from <image1>." in prompt and "Take a red kite from <image3>." in prompt
    assert "Each person appears exactly once." in prompt
    assert end.references == ["start", "asset:0"]
    later = end.prompt(plan.bible)
    assert later.startswith("Edit <image1>: keep the place")
    assert "Faces come from <image2>." in later
    assert "from <image2>" in later.split("as follows.")[1]
    assert len({a.seed for a in plan.assets}) == 3
    assert plan.as_output()["assets"][0]["kind"] == "clean"


def test_prop_pictures_fit_the_four_references():
    answer = THREE_PEOPLE.replace(
        "**KEYFRAME START:**",
        "**PROPS:** a wooden oar | a tiny flag\n**PROP PICTURES:** a wooden oar | a tiny flag\n"
        "**KEYFRAME START:**",
    )
    run_text, _ = runner(answer)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO, PHOTO, PHOTO], "Rowing")
    assert [a.words for a in plan.assets] == ["a wooden oar"]
    assert "PROP PICTURES: 1 left out (at most 1)" in plan.problems
    assert plan.keyframes[0].references == [1, 2, 3, "asset:0"]


def test_a_prop_from_a_photo_cleans_that_photo():
    answer = WITH_ASSETS.replace("CLEAN: <image1>", "CLEAN: none")
    run_text, _ = runner(answer)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "He flies away")
    assert [(a.kind, a.photo) for a in plan.assets] == [("clean", 1), ("prop", 1), ("prop", None)]
    assert "CLEAN: <image1> added, a prop picture comes from it" in plan.problems


def test_without_asset_sections_the_references_stay_as_before():
    run_text, _ = runner(KEYFRAMES_ONE)
    plan = direct_keyframes(run_text, kf_prompt(), [PHOTO], "He flies away")
    assert plan.assets == [] and plan.keyframes[1].references == [1, "start"]


# E16, Denys's layout: background, people and props first; keyframes put together from them.

COMPOSED = KEYFRAMES_ONE.replace(
    "KEYFRAME START:",
    "CAMERA MIDDLE: same\n"
    "CAMERA END: Close-up on the man's face from the front.\n"
    "PROPS: a black umbrella with a curved wooden handle | a red kite\n"
    "PROP PICTURES: a black umbrella with a curved wooden handle from <image1> | a red kite\n"
    "KEYFRAME START:",
)


def test_the_compose_layout_draws_background_people_and_props_first():
    run_text, calls = runner(COMPOSED)
    plan = direct_keyframes(run_text, kf_prompt(compose=True), [PHOTO], "He flies away")
    assert "CAMERA END:" in calls[0]["system"] and "CLEAN:" not in calls[0]["system"]
    background, man, umbrella = plan.assets  # the kite is in no photo: words only
    assert background.kind == "background" and not background.cutout
    assert background.prompt.startswith("The plain grey <image1> is only the canvas")
    assert "leave out every person" in background.prompt
    assert plan.bible["place"] in background.prompt and plan.bible["camera"] in background.prompt
    assert (man.kind, man.photo, man.cutout) == ("clean", 1, True)
    assert umbrella.cutout and umbrella.photo == 1
    assert "PROP PICTURES: 'a red kite' is in no photo; drawn from its words" in plan.problems
    start, end = plan.keyframes
    for keyframe in (start, end):
        assert keyframe.references == ["asset:0", "asset:1", "asset:2"]
        assert keyframe.layout == "compose" and "start" not in keyframe.references
    assert not start.camera_moves and end.camera_moves
    assert end.action.startswith("Close-up on the man's face from the front.")
    first = start.prompt(plan.bible)
    assert first.startswith("Put the people and objects into the empty place of <image1>")
    assert "the camera position and the framing" in first
    assert (
        "<image3> shows only the look of a black umbrella with a curved wooden handle: draw it "
        "at the size and angle, and in the grip, that the pose needs." in first
    )
    assert "from <image2>" in first.split("Each person appears exactly once.")[1]
    assert "the camera position and the framing" not in end.prompt(plan.bible)
    output = plan.as_output()
    assert output["assets"][1]["cutout"] is True and output["keyframes"][0]["layout"] == "compose"


def test_compose_keeps_props_within_six_pictures():
    answer = THREE_PEOPLE.replace(
        "**KEYFRAME START:**",
        "**PROPS:** a wooden oar | a tiny flag | a duck\n"
        "**PROP PICTURES:** a wooden oar from <image1> | a tiny flag from <image2> | a duck"
        " from <image3>\n**KEYFRAME START:**",
    )
    run_text, _ = runner(answer)
    plan = direct_keyframes(run_text, kf_prompt(compose=True), [PHOTO, PHOTO, PHOTO], "Rowing")
    kinds = [a.kind for a in plan.assets]
    # The two props kept come from photos 1 and 2, so those are redrawn empty-handed.
    assert kinds == ["background", "clean", "clean", "cut", "prop", "prop"]
    assert "PROP PICTURES: 1 left out (at most 2)" in plan.problems
    assert len(plan.keyframes[0].references) == 6


def test_camera_for():
    start = "Static full-body shot."
    assert director.camera_for("same", start) == (start, False)
    assert director.camera_for("Same as before.", start) == (start, False)
    assert director.camera_for("", start) == (start, False)
    assert director.camera_for("Close-up.", start) == ("Close-up.", True)


def test_parse_inventory():
    answer = (
        "<think>x</think>\nImage 1: a man in blue | holds: a knife and a cup\n"
        "<image2>: a troll | holds: nothing\n"
    )
    entries = director.parse_inventory(answer, 3)
    assert entries[0] == {"photo": 1, "who": "a man in blue", "holds": ["knife", "cup"]}
    assert entries[1]["holds"] == [] and entries[2] == {"photo": 3, "who": "", "holds": []}


def test_the_checked_inventory_reaches_the_director_and_decides_the_props():
    run_text, calls = runner(COMPOSED)
    inventory = [{"photo": 1, "who": "a man", "holds": ["kite"], "checked": {"kite": False}}]
    plan = direct_keyframes(
        run_text, kf_prompt(compose=True), [PHOTO], "He flies away", inventory=inventory
    )
    assert "<image1>: holds nothing" in calls[0]["prompt"]
    # The umbrella is claimed "from <image1>", but the man holds no umbrella there.
    assert (
        "PROP PICTURES: 'a black umbrella with a curved wooden handle' is not held in <image1>; "
        "from its words" in plan.problems
    )
    assert [a.kind for a in plan.assets] == ["background", "cut"]
