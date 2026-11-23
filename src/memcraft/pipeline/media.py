"""Video files with ffmpeg: the join of a long video's runs, frame counts, frames and sheets.

The join follows E0.5: every run after the first starts on the real last frame of the run
before, so it drops its first frame; the sound gets a short crossfade at each seam. It is one
encode pass (libx264), never a stream copy, so the seams are exact frames.

ffmpeg comes from winget (Gyan.FFmpeg.Essentials). find_ffmpeg() looks on PATH and in the two
folders where winget puts its links, because a shell older than the install lacks the PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Seconds of sound that overlap at a seam; E0.5 used one frame, a little more hides the click.
CROSSFADE = 0.05
CRF = 17
TIMEOUT = 900.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class MediaError(Exception):
    pass


def find_tool(name: str) -> str:
    """The path of ffmpeg or ffprobe."""
    found = shutil.which(name)
    if found:
        return found
    folders = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "WinGet" / "Links",
    ]
    for folder in folders:
        for candidate in (folder / f"{name}.exe", folder / name):
            if candidate.is_file():
                return str(candidate)
    raise MediaError(f"{name} is not installed (winget install Gyan.FFmpeg.Essentials)")


def find_ffmpeg() -> str:
    return find_tool("ffmpeg")


def _run(args: Sequence[str], timeout: float = TIMEOUT) -> str:
    try:
        done = subprocess.run(  # noqa: S603 - a fixed tool with arguments built here
            list(args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaError(f"{Path(args[0]).name} failed: {error}") from error
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-5:]
        raise MediaError(f"{Path(args[0]).name} exited {done.returncode}: {' | '.join(tail)}")
    return done.stdout


@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int
    seconds: float
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    """Counts the frames by decoding, so the number is exact."""
    out = _run(
        [
            find_tool("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,nb_read_frames,r_frame_rate,width,height:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(out)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"{path.name} has no video stream")
    num, _, den = str(video.get("r_frame_rate", "30/1")).partition("/")
    fps = float(num) / float(den or 1)
    return VideoInfo(
        frames=int(video.get("nb_read_frames") or 0),
        fps=round(fps, 3),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        seconds=float((data.get("format") or {}).get("duration") or 0.0),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def join_filter(count: int, fps: int, with_audio: bool, crossfade: float = CROSSFADE) -> str:
    """The filter graph of join(): later runs lose their first frame, sound crossfades."""
    if count < 2:
        raise ValueError("a join needs two clips or more")
    parts = []
    video_labels = ["[0:v]"]
    for i in range(1, count):
        parts.append(f"[{i}:v]trim=start_frame=1,setpts=PTS-STARTPTS[v{i}]")
        video_labels.append(f"[v{i}]")
    parts.append(f"{''.join(video_labels)}concat=n={count}:v=1:a=0,format=yuv420p[v]")
    if with_audio:
        frame = 1 / fps
        previous = "[0:a]"
        for i in range(1, count):
            parts.append(f"[{i}:a]atrim=start={frame:.6f},asetpts=PTS-STARTPTS[a{i}]")
            out = "[a]" if i == count - 1 else f"[x{i}]"
            parts.append(f"{previous}[a{i}]acrossfade=d={crossfade}:c1=tri:c2=tri{out}")
            previous = out
    return ";".join(parts)


def join(
    clips: Sequence[Path],
    dest: Path,
    *,
    fps: int = 30,
    crossfade: float = CROSSFADE,
    ffmpeg: str | None = None,
) -> VideoInfo:
    """Join the runs into one mp4 in one encode pass. Returns what the file holds."""
    clips = [Path(clip) for clip in clips]
    if len(clips) == 1:
        shutil.copyfile(clips[0], dest)
        return probe(dest)
    infos = [probe(clip) for clip in clips]
    sizes = {(info.width, info.height) for info in infos}
    if len(sizes) != 1:
        raise MediaError(f"the clips differ in size: {sorted(sizes)}")
    with_audio = all(info.has_audio for info in infos)
    args = [ffmpeg or find_ffmpeg(), "-hide_banner", "-nostdin", "-y"]
    for clip in clips:
        args += ["-i", str(clip)]
    args += [
        "-filter_complex",
        join_filter(len(clips), fps, with_audio, crossfade),
        "-map",
        "[v]",
    ]
    if with_audio:
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    args += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(CRF),
        "-r",
        str(fps),
        "-movflags",
        "+faststart",
        str(dest),
    ]
    _run(args)
    info = probe(dest)
    expected = sum(i.frames for i in infos) - (len(infos) - 1)
    if info.frames != expected:
        raise MediaError(f"the join has {info.frames} frames, not {expected}")
    return info


def frame_at(video: Path, index: int, dest: Path, *, ffmpeg: str | None = None) -> Path:
    """One frame of the video as a PNG; a negative index counts from the end."""
    if index < 0:
        index = probe(video).frames + index
    _run(
        [
            ffmpeg or find_ffmpeg(),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            str(dest),
        ]
    )
    return dest


def contact_sheet(
    video: Path, dest: Path, *, count: int = 6, height: int = 480, ffmpeg: str | None = None
) -> Path:
    """A JPEG of `count` frames side by side, evenly spread from the first to the last."""
    info = probe(video)
    last = max(info.frames - 1, 0)
    picks = sorted({round(last * i / max(count - 1, 1)) for i in range(count)})
    tiles = []
    folder = dest.parent / f".{dest.stem}-frames"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for index in picks:
            path = frame_at(video, index, folder / f"{index:05d}.png", ffmpeg=ffmpeg)
            with Image.open(path) as image:
                width = round(image.width * height / image.height)
                tiles.append(image.convert("RGB").resize((width, height)))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), height), "white")
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + 4
    sheet.save(dest, "JPEG", quality=88)
    return dest
