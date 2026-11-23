import subprocess

import pytest
from PIL import Image

from memcraft.pipeline import media

try:
    FFMPEG = media.find_ffmpeg()
    media.find_tool("ffprobe")
except media.MediaError:
    FFMPEG = None

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")


def make_clip(path, frames, *, audio=True, size=(64, 112), color="red"):
    """A clip of exactly `frames` frames at 30 fps, with a tone as long as the video."""
    seconds = frames / 30
    args = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-f", "lavfi"]
    args += ["-i", f"color=c={color}:s={size[0]}x{size[1]}:r=30:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:d={seconds}"]
    args += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    args.append(str(path))
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603 - ffmpeg
    return path


def test_the_filter_drops_first_frames_and_crossfades():
    graph = media.join_filter(3, 30, True, 0.05)
    assert graph.count("trim=start_frame=1") == 2
    assert graph.count("acrossfade=d=0.05") == 2
    assert "concat=n=3:v=1:a=0" in graph
    assert "[a]" in graph
    assert "acrossfade" not in media.join_filter(2, 30, False)
    with pytest.raises(ValueError):
        media.join_filter(1, 30, True)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("runs", "total"),
    [([241, 241], 481), ([153, 297], 449), ([297, 297], 593), ([153, 153], 305)],
)
def test_join_frame_counts(tmp_path, runs, total):
    clips = [make_clip(tmp_path / f"run{i}.mp4", n) for i, n in enumerate(runs)]
    info = media.join(clips, tmp_path / "joined.mp4")
    assert info.frames == total
    assert info.has_audio
    assert 305 <= info.frames <= 593
    assert (info.width, info.height) == (64, 112)


@needs_ffmpeg
def test_join_without_sound_and_three_runs(tmp_path):
    clips = [make_clip(tmp_path / f"r{i}.mp4", 97, audio=False) for i in range(3)]
    info = media.join(clips, tmp_path / "joined.mp4")
    assert info.frames == 97 * 3 - 2
    assert not info.has_audio


@needs_ffmpeg
def test_one_clip_is_copied(tmp_path):
    clip = make_clip(tmp_path / "one.mp4", 241)
    info = media.join([clip], tmp_path / "out.mp4")
    assert info.frames == 241


@needs_ffmpeg
def test_clips_of_different_sizes_are_refused(tmp_path):
    a = make_clip(tmp_path / "a.mp4", 9)
    b = make_clip(tmp_path / "b.mp4", 9, size=(96, 160))
    with pytest.raises(media.MediaError, match="size"):
        media.join([a, b], tmp_path / "out.mp4")


@needs_ffmpeg
def test_join_keeps_run_two_from_its_second_frame(tmp_path):
    a = make_clip(tmp_path / "a.mp4", 17, color="red")
    b = make_clip(tmp_path / "b.mp4", 17, color="blue")
    media.join([a, b], tmp_path / "j.mp4")
    seam = media.frame_at(tmp_path / "j.mp4", 16, tmp_path / "seam.png")
    after = media.frame_at(tmp_path / "j.mp4", 17, tmp_path / "after.png")
    with Image.open(seam) as image:
        red, _, blue = image.convert("RGB").getpixel((32, 56))
        assert red > blue
    with Image.open(after) as image:
        red, _, blue = image.convert("RGB").getpixel((32, 56))
        assert blue > red


@needs_ffmpeg
def test_contact_sheet(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", 61)
    sheet = media.contact_sheet(clip, tmp_path / "sheet.jpg", count=4, height=112)
    with Image.open(sheet) as image:
        assert image.height == 112
        assert image.width == 4 * 64 + 3 * 4
    assert not list(tmp_path.glob(".sheet-frames"))
