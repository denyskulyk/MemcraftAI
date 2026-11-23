"use strict";

const MESSAGES = {
  pending: "Hello! You are waiting for approval. Ask the owner to let you in.",
  blocked: "Your access is blocked.",
};

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
  const jobs = await (await fetch("/api/jobs")).json();
  for (const job of jobs) {
    const item = document.createElement("li");
    item.textContent = `${new Date(job.created_at).toLocaleString()}: ${job.status}`;
    document.getElementById("jobs").append(item);
  }
}

start().catch(() => {
  document.getElementById("status").textContent = "Something went wrong. Please reload.";
});
