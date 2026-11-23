"use strict";

const MESSAGES = {
  pending: "Hello! You are waiting for approval. Ask the owner to let you in.",
  blocked: "Your access is blocked.",
};

const LABELS = {
  idea_queued: "Waiting for the idea check",
  idea_running: "Checking the idea",
  awaiting_approval: "An idea is waiting for you",
  closed: "Closed",
  render_queued: "Waiting to be made",
  rendering: "Being made",
  preview_ready: "Preview ready",
  expired: "Preview expired",
  release_queued: "Waiting for the final video",
  releasing: "Making the final video",
  released: "Ready",
  failed: "Something went wrong",
  cancelled: "Cancelled",
};

const ENDED = new Set(["closed", "expired", "failed", "cancelled"]);

function jobItem(job) {
  const item = document.createElement("li");
  const link = document.createElement("a");
  link.href = "/job.html?id=" + encodeURIComponent(job.id);
  link.textContent = LABELS[job.status] || job.status;
  const when = document.createElement("span");
  when.className = "hint";
  when.textContent = new Date(job.created_at).toLocaleString();
  item.append(link, when);
  if (job.idea_text) {
    const idea = document.createElement("span");
    idea.className = "idea-line";
    idea.textContent = withNames(job.idea_text, job.cast);
    item.append(idea);
  }
  if (job.video_id) {
    const download = document.createElement("a");
    download.className = "button";
    download.href = "/api/files/" + encodeURIComponent(job.video_id) + "?download=1";
    download.textContent = "Download mp4";
    item.append(download);
  }
  return item;
}

function fill(name, jobs) {
  document.getElementById(name).replaceChildren(...jobs.map(jobItem));
  document.getElementById(name + "-box").hidden = jobs.length === 0;
}

async function start() {
  const status = document.getElementById("status");
  const answer = await fetch("/api/me");
  if (answer.status === 401) {
    status.textContent = "Open this page through Tailscale.";
    return;
  }
  if (!answer.ok) throw new Error(`/api/me answered ${answer.status}`);
  const me = await answer.json();
  if (me.status !== "approved") {
    status.textContent = MESSAGES[me.status];
    return;
  }
  status.textContent = `Hello, ${me.display_name || me.tailscale_login}.`;
  document.getElementById("nav").hidden = false;
  const listed = await fetch("/api/jobs");
  if (!listed.ok) throw new Error(`/api/jobs answered ${listed.status}`);
  const jobs = await listed.json();
  fill("videos", jobs.filter((job) => job.status === "released"));
  fill("earlier", jobs.filter((job) => ENDED.has(job.status)));
  fill("current", jobs.filter((job) => job.status !== "released" && !ENDED.has(job.status)));
  document.getElementById("empty").hidden = jobs.length > 0;
}

start().catch(() => {
  document.getElementById("status").textContent = "Something went wrong. Please reload.";
});
