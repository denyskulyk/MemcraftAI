"use strict";

const LABELS = {
  idea_queued: "Waiting for the idea check",
  idea_running: "Checking the idea",
  awaiting_approval: "Here is an idea for your photo",
  closed: "Closed",
  render_queued: "Waiting to be made",
  rendering: "Your video is being made",
  preview_ready: "Your preview is ready",
  expired: "The preview has expired",
  release_queued: "Waiting for the final video",
  releasing: "Making the final video",
  released: "Your video is ready",
  failed: "Something went wrong",
  cancelled: "This video was cancelled",
};

// The server's queued and running states: only then does the progress stream run.
const QUEUED = new Set(["idea_queued", "render_queued", "release_queued"]);
const RUNNING = new Set(["idea_running", "rendering", "releasing"]);
const MOVING = new Set([...QUEUED, ...RUNNING]);
const DELETE_NOTE = "Delete removes the video, its photos and everything made from them at once. It cannot be undone.";

const $ = (id) => document.getElementById(id);
const jobId = new URLSearchParams(location.search).get("id");
const jobPath = "/api/jobs/" + encodeURIComponent(jobId || "");
let shown = null;
let stream = null;
let retry = null;

async function call(method, path, body) {
  const answer = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = answer.status === 204 ? null : await answer.json().catch(() => null);
  if (!answer.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : "";
    const error = new Error(detail || "The server answered " + answer.status + ".");
    error.status = answer.status;
    throw error;
  }
  return data;
}

function say(text) {
  $("status").textContent = text;
}

function showError(text) {
  $("error").textContent = text || "";
  $("error").hidden = !text;
}

function fileUrl(asset, download) {
  return "/api/files/" + encodeURIComponent(asset.id) + (download ? "?download=1" : "");
}

function setVideo(element, asset) {
  const url = fileUrl(asset, false);
  if (element.getAttribute("src") !== url) element.src = url;
}

function showProgress(state) {
  const box = $("progress-box");
  box.hidden = !MOVING.has(state.status);
  if (box.hidden) return;
  const percent = Math.round((state.progress || 0) * 100);
  $("progress").value = RUNNING.has(state.status) ? percent : 0;
  let text = percent + "% done.";
  if (QUEUED.has(state.status)) {
    const ahead = state.queue_position;
    if (ahead === null || ahead === undefined) text = "Waiting in line.";
    else if (ahead === 0) text = "Next in line.";
    else if (ahead === 1) text = "1 video ahead of yours.";
    else text = ahead + " videos ahead of yours.";
  }
  $("progress-text").textContent = text;
}

function showIdea(job) {
  const idea = job.idea;
  $("idea-box").hidden = !idea && !job.idea_text;
  if (idea) {
    const mine = idea.proposed_by === "user";
    $("idea-title").textContent = mine ? "Your idea, as the app understood it" : "The app's idea";
    $("idea-text").textContent = idea.text;
    $("idea-score").textContent = "Fun guess: " + idea.score + " / 10";
    $("idea-reason").textContent = idea.understood
      ? "How the app understood it: " + idea.understood + " Weak spot: " + idea.reason
      : idea.reason;
  } else if (job.idea_text) {
    $("idea-title").textContent = "Your idea";
    $("idea-text").textContent = job.idea_text;
    $("idea-score").textContent = "";
    $("idea-reason").textContent = "The app is reading your idea.";
  }
  $("decide").hidden = job.status !== "awaiting_approval" || !idea;
}

function showPreview(job) {
  const preview = job.assets.find((a) => a.kind === "preview");
  const ready = job.status === "preview_ready" && Boolean(preview);
  $("preview-box").hidden = !ready;
  if (!ready) return;
  setVideo($("preview"), preview);
  for (const radio of document.querySelectorAll('input[name="quality"]')) {
    if (!radio.dataset.touched) radio.checked = radio.value === job.quality;
  }
  if (preview.expires_at) {
    const until = new Date(preview.expires_at).toLocaleString();
    $("preview-hint").textContent = "The preview is kept until " + until + ". Release it to keep the video.";
  }
}

function render(job) {
  shown = job;
  $("title").textContent = LABELS[job.status] || job.status;
  const quality = job.quality === "fhd1080" ? "1080p" : "720p";
  say(new Date(job.created_at).toLocaleString() + " · " + quality + " · " + job.length);
  showError(job.status === "failed" ? job.error || "The video could not be made." : "");
  showProgress(job);
  showIdea(job);
  showPreview(job);

  const video = job.assets.find((a) => a.kind === "release");
  $("video-box").hidden = !video;
  if (video) {
    setVideo($("video"), video);
    $("download").href = fileUrl(video, true);
  }

  const busy = RUNNING.has(job.status);
  $("delete-box").hidden = false;
  $("delete").disabled = busy;
  $("delete-hint").textContent = busy
    ? "This video is being worked on right now. You can delete it when this step finishes."
    : DELETE_NOTE;

  follow(job.status);
}

async function load() {
  try {
    render(await call("GET", jobPath));
  } catch (error) {
    if (error.status === 404) return gone("This video does not exist, or it was deleted.");
    if (error.status === 401) return say("Open this page through Tailscale.");
    if (error.status === 403) {
      return say(error.message === "blocked" ? "Your access is blocked." : "You are waiting for approval.");
    }
    say("Something went wrong. Please reload.");
  }
}

function gone(text) {
  stopFollowing();
  for (const id of ["progress-box", "idea-box", "preview-box", "video-box", "delete-box"]) {
    $(id).hidden = true;
  }
  $("title").textContent = "Not found";
  say(text);
}

function stopFollowing() {
  if (stream) stream.close();
  stream = null;
  clearTimeout(retry);
}

// The stream reports changes only; when the status moves, the page loads the job again.
function follow(status) {
  if (!MOVING.has(status)) return stopFollowing();
  if (stream) return;
  stream = new EventSource(jobPath + "/events");
  stream.onmessage = (event) => {
    const state = JSON.parse(event.data);
    if (shown && state.status !== shown.status) {
      shown.status = state.status;
      load();
    } else {
      showProgress(state);
    }
  };
  stream.addEventListener("end", () => {
    stopFollowing();
    load();
  });
  stream.addEventListener("gone", () => gone("This video was deleted."));
  stream.onerror = () => {
    // EventSource reconnects by itself, unless the server refused it: then try again later.
    if (stream && stream.readyState === EventSource.CLOSED) {
      stream = null;
      retry = setTimeout(load, 5000);
    }
  };
}

async function act(control, action) {
  control.disabled = true;
  showError("");
  try {
    const job = await action();
    if (job) render(job);
  } catch (error) {
    await load();
    showError(error.message);
  } finally {
    control.disabled = shown ? RUNNING.has(shown.status) && control.id === "delete" : false;
  }
}

// Share: the mp4 itself goes to the phone's share sheet (TikTok, Instagram, YouTube,
// WhatsApp). It is fetched from the owner-checked /api/files and never gets a public link.
const SHARES = {
  preview: { button: "share-preview", hint: "share-preview-hint", label: "Share preview",
    title: "My MemcraftAI preview", name: "memcraft-preview.mp4" },
  release: { button: "share", hint: "share-hint", label: "Share",
    title: "My MemcraftAI video", name: "memcraft-video.mp4" },
};
const sharedFiles = new Map(); // asset id -> File, so that a second tap shares at once

function canShareFiles(files) {
  if (!navigator.share || !navigator.canShare) return false;
  try {
    return navigator.canShare({ files });
  } catch {
    return false;
  }
}

function downloadInstead(asset, hint) {
  const link = document.createElement("a");
  link.href = fileUrl(asset, true);
  document.body.append(link);
  link.click();
  link.remove();
  hint.textContent = "This browser cannot share videos, so the mp4 is downloading instead.";
  hint.hidden = false;
}

async function shareVideo(kind) {
  const share = SHARES[kind];
  const button = $(share.button);
  const hint = $(share.hint);
  const asset = shown && shown.assets.find((a) => a.kind === kind);
  if (!asset || button.disabled) return;
  hint.hidden = true;
  const probe = new File([""], share.name, { type: "video/mp4" });
  if (!canShareFiles([probe])) return downloadInstead(asset, hint);

  let file = sharedFiles.get(asset.id);
  if (!file) {
    button.disabled = true;
    button.textContent = "Getting the video…";
    try {
      const answer = await fetch(fileUrl(asset, false));
      if (!answer.ok) throw new Error("The server answered " + answer.status + ".");
      const blob = await answer.blob();
      file = new File([blob], share.name, { type: blob.type || asset.mime || "video/mp4" });
      sharedFiles.set(asset.id, file);
    } catch {
      hint.textContent = "The video cannot be loaded right now. Please try again.";
      hint.hidden = false;
      return;
    } finally {
      button.disabled = false;
      button.textContent = share.label;
    }
    if (!canShareFiles([file])) return downloadInstead(asset, hint);
  }
  try {
    await navigator.share({ files: [file], title: share.title });
  } catch (error) {
    if (error.name === "AbortError") return; // the friend closed the share sheet
    if (error.name === "NotAllowedError") {
      // The tap was too long ago by the time the video arrived; now it is here.
      hint.textContent = "The video is ready: tap " + share.label + " again.";
      hint.hidden = false;
      return;
    }
    downloadInstead(asset, hint);
  }
}

$("share-preview").addEventListener("click", () => shareVideo("preview"));
$("share").addEventListener("click", () => shareVideo("release"));
$("approve").addEventListener("click", (e) => act(e.target, () => call("POST", jobPath + "/approve")));
$("reject").addEventListener("click", (e) => act(e.target, () => call("POST", jobPath + "/reject")));
$("release").addEventListener("click", (e) => {
  const picked = document.querySelector('input[name="quality"]:checked');
  act(e.target, () => call("POST", jobPath + "/release", { quality: picked ? picked.value : null }));
});
for (const radio of document.querySelectorAll('input[name="quality"]')) {
  radio.addEventListener("change", () => {
    for (const other of document.querySelectorAll('input[name="quality"]')) other.dataset.touched = "yes";
  });
}
$("delete").addEventListener("click", (e) => {
  const control = e.target;
  if (control.dataset.armed !== "yes") {
    control.dataset.armed = "yes";
    control.textContent = "Delete for good?";
    return;
  }
  act(control, async () => {
    await call("DELETE", jobPath);
    stopFollowing();
    location.href = "/";
    return null;
  });
});

if (!jobId) gone("No video was chosen.");
else load();
