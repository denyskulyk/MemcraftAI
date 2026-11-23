"use strict";

// The "New video" page: photos with a crop frame each, length, quality, idea, upload.

const MAX_PHOTOS = 3;
const MAX_MB = 20;
const TYPES = ["image/png", "image/jpeg"];
const ACCESS = {
  pending: "Hello! You are waiting for approval. Ask the owner to let you in.",
  blocked: "Your access is blocked.",
};

const photos = []; // {file, url, card, title, message, warning, error, frame, faces, gone}
const notes = new Map();
let uploading = false;

const byId = (id) => document.getElementById(id);
const DETECT_SIDE = 1024; // the browser's own face check sees the photo at most this large

function choice(name) {
  return document.querySelector(`input[name="${name}"]:checked`).value;
}

function isPhotoFile(file) {
  if (file.type) return TYPES.includes(file.type);
  return /\.(png|jpe?g)$/i.test(file.name); // some systems send no type
}

function addFiles(files) {
  const problems = [];
  for (const file of files) {
    if (photos.length >= MAX_PHOTOS) {
      problems.push(`You can add at most ${MAX_PHOTOS} photos.`);
      break;
    }
    if (!isPhotoFile(file)) {
      problems.push(`"${file.name}" is not a PNG or JPEG photo.`);
    } else if (file.size > MAX_MB * 1024 * 1024) {
      problems.push(`"${file.name}" is larger than ${MAX_MB} MB.`);
    } else {
      addPhoto(file);
    }
  }
  byId("photo-error").textContent = problems.join(" ");
  changed();
}

function addPhoto(file) {
  const photo = { file, url: URL.createObjectURL(file), error: null, frame: null, faces: null };
  const card = document.createElement("div");
  card.className = "photo";
  const head = document.createElement("div");
  head.className = "photo-head";
  const title = document.createElement("strong");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => removePhoto(photo));
  head.append(title, remove);
  const message = document.createElement("p");
  message.className = "error";
  // Only a warning: the friend may want someone out of the picture.
  const warning = document.createElement("p");
  warning.className = "warning";
  warning.setAttribute("role", "status");
  warning.hidden = true;
  card.append(head, message);
  Object.assign(photo, { card, title, message, warning });
  photos.push(photo);
  byId("photos").append(card);

  const img = new Image();
  img.alt = "";
  img.draggable = false;
  img.addEventListener("load", () => {
    photo.frame = new CropFrame(img, () => showFaceWarning(photo));
    if (!photo.frame.usable) {
      photo.error = "This photo is too small: it needs at least 288 x 512 pixels.";
      message.textContent = photo.error;
    }
    card.append(photo.frame.element, warning);
    if (photo.frame.usable) checkFaces(photo, img);
    renumber();
    changed();
  });
  img.addEventListener("error", () => {
    photo.error = "This photo cannot be opened. Is it really a PNG or JPEG?";
    message.textContent = photo.error;
    changed();
  });
  img.src = photo.url;
  renumber();
}

function removePhoto(photo) {
  const index = photos.indexOf(photo);
  if (index < 0 || uploading) return;
  photos.splice(index, 1);
  photo.gone = true;
  photo.card.remove();
  URL.revokeObjectURL(photo.url);
  byId("photo-error").textContent = "";
  renumber();
  changed();
}

// Once per photo: where the faces are. Then every move of the frame checks them in JS.
async function checkFaces(photo, img) {
  let faces = null;
  if ("FaceDetector" in window) {
    try {
      faces = await facesInBrowser(img);
    } catch {
      faces = null; // the API is there but cannot run here: ask the server
    }
  }
  if (faces === null) faces = await facesOnServer(photo.file);
  if (photo.gone || faces === null) return; // no answer: no warning, the friend can go on
  photo.faces = faces;
  showFaceWarning(photo);
}

async function facesInBrowser(img) {
  const scale = Math.min(1, DETECT_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  // drawImage paints the photo upright, the way the page shows it.
  canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
  const found = await new FaceDetector({ maxDetectedFaces: 20 }).detect(canvas);
  return found.map(({ boundingBox: box }) => ({
    x: box.x / canvas.width,
    y: box.y / canvas.height,
    w: box.width / canvas.width,
    h: box.height / canvas.height,
  }));
}

async function facesOnServer(file) {
  let type = file.type;
  if (!TYPES.includes(type)) type = /\.png$/i.test(file.name) ? "image/png" : "image/jpeg";
  try {
    const answer = await fetch("/api/faces", {
      method: "POST",
      headers: { "Content-Type": type },
      body: file,
    });
    if (!answer.ok) return null;
    const body = await answer.json();
    return Array.isArray(body.faces) ? body.faces : null;
  } catch {
    return null;
  }
}

function showFaceWarning(photo) {
  if (!photo.faces || !photo.frame) return;
  const text = faceWarning(photo.faces, photo.frame.crop());
  // Set only on a change, so that a screen reader is not told the same thing on every move.
  if (photo.warning.textContent !== text) photo.warning.textContent = text;
  photo.warning.hidden = !text;
}

function renumber() {
  photos.forEach((photo, index) => {
    photo.title.textContent = `Photo ${index + 1}`;
    if (photo.frame && photo.frame.usable) {
      photo.frame.label(`Crop frame for photo ${index + 1}`);
    }
  });
}

function ready() {
  return (
    !uploading &&
    photos.length >= 1 &&
    photos.length <= MAX_PHOTOS &&
    photos.every((photo) => photo.frame && photo.frame.usable && !photo.error)
  );
}

function changed() {
  byId("confirm").disabled = !ready();
  byId("files").disabled = uploading || photos.length >= MAX_PHOTOS;
  updateNote().catch(() => {
    byId("note").textContent = "The note cannot be loaded right now.";
  });
}

// The rule lives on the server, so the page asks it and remembers the answers.
async function updateNote() {
  const note = byId("note");
  if (photos.length === 0) {
    note.textContent = "Add a photo to see how your video will be made.";
    return;
  }
  const key = `photos=${photos.length}&length=${encodeURIComponent(choice("length"))}`;
  if (!notes.has(key)) {
    const answer = await fetch(`/api/regime-note?${key}`);
    if (!answer.ok) throw new Error(`regime-note answered ${answer.status}`);
    notes.set(key, await answer.json());
  }
  const current = `photos=${photos.length}&length=${encodeURIComponent(choice("length"))}`;
  if (current === key) note.textContent = notes.get(key).note;
}

function problemText(status, body) {
  if (body && typeof body.detail === "string") return body.detail;
  if (status === 413) return `The upload is too large. Each photo can be up to ${MAX_MB} MB.`;
  if (status === 0) return "The upload failed. Check your connection and try again.";
  return `The upload failed (code ${status}). Please try again.`;
}

function submit(event) {
  event.preventDefault();
  if (!ready()) return;
  const form = new FormData();
  for (const photo of photos) form.append("photos", photo.file, photo.file.name);
  form.append("crops", JSON.stringify(photos.map((photo) => photo.frame.crop())));
  form.append("length", choice("length"));
  form.append("quality", choice("quality"));
  form.append("idea", byId("idea").value);

  uploading = true;
  changed();
  byId("submit-error").textContent = "";
  byId("progress-box").hidden = false;
  const progress = byId("progress");
  const progressText = byId("progress-text");
  progress.value = 0;
  progressText.textContent = "Uploading…";

  const request = new XMLHttpRequest();
  request.open("POST", "/api/jobs");
  request.upload.addEventListener("progress", (step) => {
    if (!step.lengthComputable) return;
    const percent = Math.round((step.loaded / step.total) * 100);
    progress.value = percent;
    progressText.textContent = `Uploading… ${percent}%`;
  });
  request.upload.addEventListener("load", () => {
    progress.value = 100;
    progressText.textContent = "Uploaded. Checking the photos…";
  });
  request.addEventListener("load", () => {
    let body = null;
    try {
      body = JSON.parse(request.responseText);
    } catch {
      body = null;
    }
    if (request.status === 201 && body) {
      finished(body);
    } else {
      failed(problemText(request.status, body));
    }
  });
  request.addEventListener("error", () => failed(problemText(0, null)));
  request.addEventListener("abort", () => failed(problemText(0, null)));
  request.send(form);
}

function failed(text) {
  uploading = false;
  byId("progress-box").hidden = true;
  byId("submit-error").textContent = text;
  changed();
}

function finished(job) {
  const ahead = job.queue_position;
  let text;
  if (ahead === null || ahead === undefined) text = "Work on it has already started.";
  else if (ahead === 0) text = "It is next in line.";
  else if (ahead === 1) text = "1 video is ahead of yours.";
  else text = `${ahead} videos are ahead of yours.`;
  byId("done-text").textContent = text;
  byId("form").hidden = true;
  byId("done").hidden = false;
  for (const photo of photos) URL.revokeObjectURL(photo.url);
  byId("done").scrollIntoView();
  showToast(job.id, text);
}

// A short toast, then the job page opens by itself; tapping the toast opens it at once.
const TOAST_SECONDS = 3;

function showToast(id, text) {
  const page = "/job.html?id=" + encodeURIComponent(id);
  const toast = document.createElement("a");
  toast.className = "toast";
  toast.href = page;
  toast.setAttribute("role", "status");
  toast.textContent = "Your video is on its way. " + text + " Opening it now…";
  document.body.append(toast);
  setTimeout(() => location.assign(page), TOAST_SECONDS * 1000);
}

async function start() {
  const status = byId("status");
  const answer = await fetch("/api/me");
  if (answer.status === 401) {
    status.textContent = "Open this page through Tailscale.";
    return;
  }
  if (!answer.ok) throw new Error(`/api/me answered ${answer.status}`);
  const me = await answer.json();
  if (me.status !== "approved") {
    status.textContent = ACCESS[me.status];
    return;
  }
  status.hidden = true;
  byId("form").hidden = false;
  byId("files").addEventListener("change", (event) => {
    addFiles(Array.from(event.target.files));
    event.target.value = ""; // so that the same photo can be chosen again
  });
  for (const radio of document.querySelectorAll('input[name="length"]')) {
    radio.addEventListener("change", changed);
  }
  byId("form").addEventListener("submit", submit);
  changed();
}

start().catch(() => {
  byId("status").hidden = false;
  byId("status").textContent = "Something went wrong. Please reload.";
});
