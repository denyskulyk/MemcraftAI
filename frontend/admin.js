"use strict";

// Base64url and bytes, so passkeys work in browsers without the newer JSON helpers.
function toBytes(text) {
  const plain = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(plain, (c) => c.charCodeAt(0));
}

function toText(buffer) {
  let plain = "";
  for (const byte of new Uint8Array(buffer)) plain += String.fromCharCode(byte);
  return btoa(plain).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

const $ = (id) => document.getElementById(id);

async function call(method, path, body) {
  const answer = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = answer.status === 204 ? null : await answer.json().catch(() => null);
  if (!answer.ok) throw new Error((data && data.detail) || "The server answered " + answer.status + ".");
  return data;
}

function say(text) {
  $("status").textContent = text;
}

async function createPasskey() {
  const options = await call("POST", "/api/admin/passkeys/options");
  options.challenge = toBytes(options.challenge);
  options.user.id = toBytes(options.user.id);
  options.excludeCredentials = (options.excludeCredentials || []).map((c) => ({ ...c, id: toBytes(c.id) }));
  const created = await navigator.credentials.create({ publicKey: options });
  await call("POST", "/api/admin/passkeys", {
    id: created.id,
    rawId: toText(created.rawId),
    type: created.type,
    response: {
      clientDataJSON: toText(created.response.clientDataJSON),
      attestationObject: toText(created.response.attestationObject),
      transports: created.response.getTransports ? created.response.getTransports() : [],
    },
  });
}

async function signIn() {
  const options = await call("POST", "/api/admin/sign-in/options");
  options.challenge = toBytes(options.challenge);
  options.allowCredentials = (options.allowCredentials || []).map((c) => ({ ...c, id: toBytes(c.id) }));
  const got = await navigator.credentials.get({ publicKey: options });
  await call("POST", "/api/admin/sign-in", {
    id: got.id,
    rawId: toText(got.rawId),
    type: got.type,
    response: {
      clientDataJSON: toText(got.response.clientDataJSON),
      authenticatorData: toText(got.response.authenticatorData),
      signature: toText(got.response.signature),
      userHandle: got.response.userHandle ? toText(got.response.userHandle) : null,
    },
  });
}

function button(text, className, action) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = text;
  if (className) element.className = className;
  element.addEventListener("click", () => run(action, element));
  return element;
}

function userRow(user) {
  const row = document.createElement("tr");
  const cells = [
    user.tailscale_login,
    user.display_name || "",
    user.role === "admin" ? "admin" : user.status,
    String(user.jobs),
    new Date(user.created_at).toLocaleDateString(),
  ];
  for (const text of cells) {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.append(cell);
  }
  row.children[2].className = user.status;
  const actions = document.createElement("td");
  actions.className = "actions";
  if (user.role !== "admin") {
    if (user.status !== "approved") {
      const label = user.status === "blocked" ? "Unblock" : "Approve";
      actions.append(button(label, "", () => call("POST", "/api/admin/users/" + user.id + "/approve")));
    }
    if (user.status !== "blocked") {
      actions.append(button("Block", "quiet", () => call("POST", "/api/admin/users/" + user.id + "/block")));
    }
    const remove = button("Delete", "danger", async () => {
      if (remove.dataset.armed !== "yes") {
        remove.dataset.armed = "yes";
        remove.textContent = "Delete for good?";
        return false;
      }
      return call("DELETE", "/api/admin/users/" + user.id);
    });
    actions.append(remove);
  }
  row.append(actions);
  return row;
}

async function showPeople() {
  const users = await call("GET", "/api/admin/users");
  $("users").replaceChildren(...users.map(userRow));
  const waiting = users.filter((u) => u.status === "pending").length;
  say(waiting ? waiting + " waiting for approval." : "Nobody is waiting for approval.");
  $("people").hidden = false;
}

const FINAL = new Set(["closed", "expired", "released", "failed", "cancelled"]);
let pauseTimer = null;

function cell(text, className) {
  const element = document.createElement("td");
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function jobRow(job) {
  const row = document.createElement("tr");
  const steps = job.steps.total
    ? job.steps.done + " / " + job.steps.total + (job.steps.current ? ", now " + job.steps.current : "")
    : "";
  const progress = job.claimed_by ? " " + Math.round(job.progress * 100) + "%" : "";
  row.append(
    cell(job.owner_login),
    cell(job.status.replace(/_/g, " ") + progress, "job-" + job.status),
    cell(job.regime || ""),
    cell(job.quality === "fhd1080" ? "1080p" : "720p"),
    cell(steps),
    cell(job.error || job.steps.last_error || "", "error-cell"),
    cell(new Date(job.created_at).toLocaleString()),
  );
  const actions = document.createElement("td");
  actions.className = "actions";
  if (!FINAL.has(job.status)) {
    const cancel = button("Cancel", "danger", async () => {
      if (cancel.dataset.armed !== "yes") {
        cancel.dataset.armed = "yes";
        cancel.textContent = "Cancel for good?";
        return false;
      }
      return call("POST", "/api/admin/jobs/" + job.id + "/cancel", { expected: job.status });
    });
    actions.append(cancel);
  }
  row.append(actions);
  return row;
}

function pauseText(state) {
  if (!state.paused) {
    return state.running ? "Running: " + state.running + " job(s) in hand." : "Running: the worker takes the next job when there is one.";
  }
  if (state.stopped) return "Stopped: paused, and no job is running.";
  return "Pausing: waiting for " + state.running + " job(s) to finish the step they are in.";
}

async function showPause() {
  const state = await call("GET", "/api/admin/pause");
  $("pause-state").textContent = pauseText(state);
  $("pause-state").className = "pause-state " + (state.stopped ? "stopped" : state.paused ? "pausing" : "");
  $("pause").textContent = state.paused ? "Resume" : "Pause";
  $("pause").dataset.paused = state.paused ? "yes" : "no";
  return state;
}

async function showQueue() {
  clearTimeout(pauseTimer);
  const jobs = await call("GET", "/api/admin/jobs");
  $("jobs").replaceChildren(...jobs.map(jobRow));
  const state = await showPause();
  $("queue").hidden = false;
  // Watch until the worker has let go.
  if (state.paused && !state.stopped) pauseTimer = setTimeout(() => showQueue().catch(() => {}), 5000);
}

async function refresh() {
  const answer = await fetch("/api/admin/status");
  if (answer.status === 404 || answer.status === 403) return say("This page is only for the admin.");
  if (answer.status === 401) return say("Open this page through Tailscale.");
  const status = await answer.json();
  $("passkey").hidden = status.signed_in;
  $("enrol").hidden = status.has_passkey;
  $("sign-in").hidden = !status.has_passkey;
  $("people").hidden = true;
  $("queue").hidden = true;
  clearTimeout(pauseTimer);
  if (!status.signed_in) {
    say(status.has_passkey ? "Sign in with your passkey." : "Create a passkey for this admin page first.");
    return;
  }
  await showPeople();
  await showQueue();
}

async function run(action, control) {
  if (control) control.disabled = true;
  try {
    const done = await action();
    if (done !== false) await refresh();
  } catch (error) {
    say(error.name === "NotAllowedError" ? "The passkey request was cancelled." : error.message);
  } finally {
    if (control) control.disabled = false;
  }
}

$("enrol").addEventListener("click", (e) => run(createPasskey, e.target));
$("sign-in").addEventListener("click", (e) => run(signIn, e.target));
$("add-passkey").addEventListener("click", (e) => run(createPasskey, e.target));
$("sign-out").addEventListener("click", (e) => run(() => call("POST", "/api/admin/sign-out"), e.target));
$("pause").addEventListener("click", (e) =>
  run(() => call("POST", "/api/admin/pause", { paused: e.target.dataset.paused !== "yes" }), e.target));
$("reload-jobs").addEventListener("click", (e) => run(showQueue, e.target));
run(refresh);
