"use strict";

// The "New video" page: photos with a crop frame each, length, quality, idea, upload.

const MAX_PHOTOS = 3;
const MAX_MB = 20;
const TYPES = ["image/png", "image/jpeg"];
const ACCESS = {
  pending: "Hello! You are waiting for approval. Ask the owner to let you in.",
  blocked: "Your access is blocked.",
};

// {file, url, card, title, message, warning, error, frame, faces, gone,
//  namesInput, namesError, namesHint, names, nameError, markers}
const photos = [];
const notes = new Map();
let uploading = false;
let isAdmin = false;

const byId = (id) => document.getElementById(id);
const DETECT_SIDE = 1024; // the browser's own face check sees the photo at most this large

// Names of the people in a photo, and the <tags> of the idea: the same rules as
// memcraft/cast.py on the server, which has the last word.
const MAX_NAME = 20;
const MAX_PER_PHOTO = 5;
const MAX_PEOPLE = 15;
const NAME = /^[\p{L}\p{N}][\p{L}\p{N}_-]*$/u;
const RESERVED = /^(person|image)\d+$/i;
const TAG = /<([\p{L}\p{N}][\p{L}\p{N}_-]{0,40})>/gu;
const GENERIC = /^person(\d+)$/i;
const IN_FRAME = 0.5; // a face shown at least this much counts as one of the people in the frame

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
  const people = namesField(photo);
  card.append(head, message);
  Object.assign(photo, { card, title, message, warning, people, names: [], nameError: "" });
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
    if (photo.frame.usable) {
      card.append(photo.people);
      checkFaces(photo, img);
    }
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
  namesChanged();
}

// "Who's in this photo?": names left to right, which the idea can use as <name>.
function namesField(photo) {
  const box = document.createElement("div");
  box.className = "people";
  const id = `names-${Math.random().toString(36).slice(2)}`;
  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = "Who's in this photo? (left to right, optional)";
  const input = document.createElement("input");
  input.id = id;
  input.type = "text";
  input.autocomplete = "off";
  input.autocapitalize = "none";
  input.spellcheck = false;
  input.placeholder = "for example: denys, maksym";
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent =
    "Short names, separated by commas: the first name is the person furthest left.";
  const error = document.createElement("p");
  error.className = "error";
  error.setAttribute("role", "alert");
  const mismatch = document.createElement("p");
  mismatch.className = "warning";
  mismatch.setAttribute("role", "status");
  mismatch.hidden = true;
  input.setAttribute("aria-describedby", `${id}-hint`);
  hint.id = `${id}-hint`;
  input.addEventListener("input", namesChanged);
  box.append(label, input, hint, error, mismatch);
  Object.assign(photo, { namesInput: input, namesError: error, namesHint: mismatch });
  return box;
}

function splitNames(text) {
  return text
    .split(",")
    .map((name) => name.trim().replace(/^<+|>+$/g, "").trim())
    .filter((name) => name);
}

// Reads every names field, and says what is wrong with them.
function readNames() {
  const seen = new Set();
  for (const photo of photos) {
    if (!photo.namesInput) continue;
    const names = splitNames(photo.namesInput.value);
    let problem = "";
    if (names.length > MAX_PER_PHOTO) {
      problem = `Name at most ${MAX_PER_PHOTO} people per photo.`;
    }
    for (const name of names) {
      if (problem) break;
      if (name.length > MAX_NAME || !NAME.test(name)) {
        problem =
          `"${name}" cannot be a name. Use up to ${MAX_NAME} letters or digits, ` +
          "and - or _ (not first), without spaces.";
      } else if (RESERVED.test(name)) {
        problem = `"${name}" is kept for the app. Please choose another name.`;
      } else if (seen.has(name.toLowerCase())) {
        problem = `The name "${name}" is used twice. Each name must be different.`;
      }
      seen.add(name.toLowerCase());
    }
    photo.names = problem ? [] : names;
    photo.nameError = problem;
    photo.namesError.textContent = problem;
  }
}

function namesChanged() {
  readNames();
  showAllPeople();
  showChips();
  ideaChanged();
  changed();
}

// The faces inside the frame, left to right.
function facesInFrame(photo) {
  if (!photo.faces || !photo.frame) return null;
  const crop = photo.frame.crop();
  return photo.faces
    .filter((face) => faceShown(face, crop) >= IN_FRAME)
    .sort((a, b) => a.x + a.w / 2 - (b.x + b.w / 2));
}

// Markers numbered in reading order, as <person1>, <person2>, ... count: photo 1 from left
// to right, then photo 2.
function showAllPeople() {
  let before = 0;
  for (const photo of photos) before += showPeople(photo, before);
}

// A numbered marker on every face in the frame, and a hint when names and faces differ.
// Returns how many faces are in the frame.
function showPeople(photo, before) {
  if (!photo.frame || !photo.frame.usable) return photo.names.length;
  for (const marker of photo.markers || []) marker.remove();
  photo.markers = [];
  const faces = facesInFrame(photo);
  if (faces === null) {
    photo.namesHint.hidden = true;
    return photo.names.length;
  }
  faces.forEach((face, index) => {
    const marker = document.createElement("span");
    marker.className = "face-marker";
    const name = photo.names[index];
    marker.textContent = name ? `${before + index + 1} ${name}` : String(before + index + 1);
    marker.style.left = `${(face.x + face.w / 2) * 100}%`;
    marker.style.top = `${face.y * 100}%`;
    // Above the face, or on its forehead when there is no room above it.
    marker.classList.toggle("on-face", face.y < 0.08);
    marker.setAttribute("aria-hidden", "true");
    photo.frame.element.append(marker);
    photo.markers.push(marker);
  });
  let text = "";
  const named = photo.names.length;
  if (named && named !== faces.length) {
    const sees =
      faces.length === 0
        ? "finds no face"
        : faces.length === 1
          ? "finds 1 face"
          : `finds ${faces.length} faces`;
    text =
      `You typed ${named === 1 ? "1 name" : `${named} names`} and the app ${sees} in the ` +
      "frame. Names go to the people from left to right; the face check can miss someone, " +
      "so you can go on.";
  }
  if (photo.namesHint.textContent !== text) photo.namesHint.textContent = text;
  photo.namesHint.hidden = !text;
  return faces.length;
}

// Every name in reading order, as a button that puts <name> into the idea.
function showChips() {
  const box = byId("cast-chips");
  for (const chip of box.querySelectorAll("button")) chip.remove();
  const names = photos.flatMap((photo) => photo.names);
  for (const name of names) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = `<${name}>`;
    chip.setAttribute("aria-label", `Put ${name} into the idea`);
    chip.addEventListener("click", () => insertTag(name));
    box.append(chip);
  }
  box.hidden = names.length === 0;
}

function insertTag(name) {
  const box = byId("idea");
  const text = `<${name}>`;
  const start = box.selectionStart ?? box.value.length;
  const end = box.selectionEnd ?? start;
  if (box.value.length - (end - start) + text.length > box.maxLength) {
    showCount(true);
    box.focus();
    return;
  }
  box.focus();
  box.setRangeText(text, start, end, "end");
  ideaChanged();
}

// The idea box grows with its text, so even the longest idea needs no scrolling inside it.
function fitIdea() {
  const box = byId("idea");
  box.style.height = "auto";
  box.style.height = `${box.scrollHeight + box.offsetHeight - box.clientHeight}px`;
}

function showCount(blocked = false) {
  const box = byId("idea");
  const used = box.value.length;
  const count = byId("idea-count");
  const full = used >= box.maxLength || blocked;
  const text = full
    ? `That's the limit: ${box.maxLength} characters.`
    : `${used} / ${box.maxLength}`;
  if (count.textContent !== text) count.textContent = text;
  count.classList.toggle("at-limit", full);
}

// Tags in the idea that name no one: a warning here, the server refuses them.
function unknownTags() {
  const named = photos.flatMap((photo) => photo.names);
  const known = new Set(named.map((name) => name.toLowerCase()));
  const everyPhotoNamed = photos.length > 0 && photos.every((photo) => photo.names.length > 0);
  const most = everyPhotoNamed ? named.length : MAX_PEOPLE;
  const unknown = [];
  for (const [, tag] of byId("idea").value.matchAll(TAG)) {
    if (known.has(tag.toLowerCase())) continue;
    const generic = GENERIC.exec(tag);
    const number = generic ? Number(generic[1]) : 0;
    if (number >= 1 && number <= most) continue;
    if (!unknown.includes(tag)) unknown.push(tag);
  }
  return unknown;
}

function ideaChanged() {
  fitIdea();
  showCount();
  const unknown = unknownTags();
  const box = byId("idea-tags");
  let text = "";
  if (unknown.length) {
    const list = unknown.map((tag) => `<${tag}>`).join(", ");
    text =
      `No one in the photos is called ${list}. Type the name under a photo, ` +
      "or write <person1>, <person2>, … for the people from left to right.";
  }
  if (box.textContent !== text) box.textContent = text;
  box.hidden = !text;
  byId("idea-needed").hidden = !(regime3() && !byId("idea").value.trim());
  byId("confirm").disabled = !ready();
}

// Admins only: regime 3 takes 1 to 3 photos and needs an idea.
function regime3() {
  return isAdmin && byId("regime3").checked;
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
  showAllPeople();
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
    (!regime3() || byId("idea").value.trim().length > 0) &&
    photos.every((photo) => photo.frame && photo.frame.usable && !photo.error && !photo.nameError)
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
  const noteKey = () =>
    `photos=${photos.length}&length=${encodeURIComponent(choice("length"))}` +
    (regime3() ? "&regime=regime3" : "");
  const key = noteKey();
  if (!notes.has(key)) {
    const answer = await fetch(`/api/regime-note?${key}`);
    if (!answer.ok) throw new Error(`regime-note answered ${answer.status}`);
    notes.set(key, await answer.json());
  }
  if (noteKey() === key) note.textContent = notes.get(key).note;
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
  if (regime3()) form.append("regime", "regime3");
  if (photos.some((photo) => photo.names.length)) {
    form.append("names", JSON.stringify(photos.map((photo) => photo.names)));
  }

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
  isAdmin = me.role === "admin";
  byId("admin-box").hidden = !isAdmin;
  byId("regime3").addEventListener("change", () => {
    ideaChanged();
    changed();
  });
  byId("form").hidden = false;
  byId("files").addEventListener("change", (event) => {
    addFiles(Array.from(event.target.files));
    event.target.value = ""; // so that the same photo can be chosen again
  });
  for (const radio of document.querySelectorAll('input[name="length"]')) {
    radio.addEventListener("change", changed);
  }
  byId("form").addEventListener("submit", submit);
  byId("idea").addEventListener("input", ideaChanged);
  window.addEventListener("resize", fitIdea);
  ideaChanged();
  changed();
}

start().catch(() => {
  byId("status").hidden = false;
  byId("status").textContent = "Something went wrong. Please reload.";
});
