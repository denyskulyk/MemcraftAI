"use strict";

// A 9:16 frame over one photo, moved and resized with a mouse, a finger or the keyboard.
// Positions are kept in the photo's own pixels, the way the browser shows it (after the
// EXIF orientation), and drawn as percentages, so the frame follows any change of layout.

const CROP_MIN_WIDTH = 288; // with 9:16 that is 288 x 512, the smallest crop the server takes
const CROP_TALL = 16 / 9;

class CropFrame {
  constructor(img, onChange) {
    this.img = img;
    this.onChange = onChange || (() => {});
    this.width = img.naturalWidth;
    this.height = img.naturalHeight;
    this.maxW = Math.min(this.width, this.height / CROP_TALL);
    this.usable = this.maxW >= CROP_MIN_WIDTH;
    this.drag = null;
    // The largest 9:16 frame there is, in the middle.
    const tallest = this.maxW * CROP_TALL;
    this.clamp((this.width - this.maxW) / 2, (this.height - tallest) / 2, this.maxW);

    this.element = document.createElement("div");
    this.element.className = "crop";
    // As wide as it can be without the photo getting taller than the screen allows.
    this.element.style.aspectRatio = `${this.width} / ${this.height}`;
    this.element.style.width = `min(100%, calc(var(--crop-height) * ${this.width / this.height}))`;
    this.frame = document.createElement("div");
    this.frame.className = "crop-frame";
    this.element.append(img);
    if (!this.usable) return;

    this.frame.tabIndex = 0;
    this.frame.setAttribute("role", "group");
    for (const corner of ["nw", "ne", "sw", "se"]) {
      const handle = document.createElement("div");
      handle.className = `crop-handle ${corner}`;
      handle.dataset.corner = corner;
      this.frame.append(handle);
    }
    this.element.append(this.frame);
    this.frame.addEventListener("pointerdown", (event) => this.start(event));
    this.frame.addEventListener("pointermove", (event) => this.move(event));
    this.frame.addEventListener("pointerup", (event) => this.end(event));
    this.frame.addEventListener("pointercancel", (event) => this.end(event));
    this.frame.addEventListener("keydown", (event) => this.key(event));
    this.render();
  }

  get h() {
    return this.w * CROP_TALL;
  }

  // The crop as fractions of the photo, at full precision.
  crop() {
    return {
      x: this.x / this.width,
      y: this.y / this.height,
      w: this.w / this.width,
      h: this.h / this.height,
    };
  }

  label(text) {
    this.frame.setAttribute("aria-label", text);
  }

  // Inside the photo whatever the rounding: the server refuses a crop that starts below 0.
  clamp(x, y, w) {
    this.w = Math.min(Math.max(w, CROP_MIN_WIDTH), this.maxW);
    this.x = Math.max(0, Math.min(x, this.width - this.w));
    this.y = Math.max(0, Math.min(y, this.height - this.h));
  }

  place(x, y, w) {
    this.clamp(x, y, w);
    this.render();
    this.onChange(this);
  }

  render() {
    const style = this.frame.style;
    style.left = `${(this.x / this.width) * 100}%`;
    style.top = `${(this.y / this.height) * 100}%`;
    style.width = `${(this.w / this.width) * 100}%`;
    style.height = `${(this.h / this.height) * 100}%`;
  }

  // Where the pointer is, in the photo's own pixels.
  point(event) {
    const box = this.img.getBoundingClientRect();
    const scale = this.width / box.width;
    return { x: (event.clientX - box.left) * scale, y: (event.clientY - box.top) * scale };
  }

  start(event) {
    if (event.button !== 0 || this.drag) return;
    event.preventDefault();
    const corner = event.target.dataset.corner || null;
    const right = this.x + this.w;
    const bottom = this.y + this.h;
    this.drag = {
      id: event.pointerId,
      corner,
      from: this.point(event),
      x: this.x,
      y: this.y,
      // The corner that stays where it is while another one is dragged.
      anchorX: corner && corner.includes("w") ? right : this.x,
      anchorY: corner && corner.includes("n") ? bottom : this.y,
    };
    this.frame.setPointerCapture(event.pointerId);
    this.frame.focus({ preventScroll: true });
  }

  move(event) {
    const drag = this.drag;
    if (!drag || drag.id !== event.pointerId) return;
    const at = this.point(event);
    if (!drag.corner) {
      this.place(drag.x + at.x - drag.from.x, drag.y + at.y - drag.from.y, this.w);
      return;
    }
    const sideX = drag.corner.includes("e") ? 1 : -1;
    const sideY = drag.corner.includes("s") ? 1 : -1;
    const roomX = sideX > 0 ? this.width - drag.anchorX : drag.anchorX;
    const roomY = (sideY > 0 ? this.height - drag.anchorY : drag.anchorY) / CROP_TALL;
    let w = Math.max((at.x - drag.anchorX) * sideX, ((at.y - drag.anchorY) * sideY) / CROP_TALL);
    w = Math.min(Math.max(w, CROP_MIN_WIDTH), roomX, roomY, this.maxW);
    const x = sideX > 0 ? drag.anchorX : drag.anchorX - w;
    const y = sideY > 0 ? drag.anchorY : drag.anchorY - w * CROP_TALL;
    this.place(x, y, w);
  }

  end(event) {
    if (this.drag && this.drag.id === event.pointerId) this.drag = null;
  }

  key(event) {
    const step = event.shiftKey ? this.w / 4 : this.w / 40;
    const moves = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const sizes = { "+": 1.1, "=": 1.1, "-": 1 / 1.1, _: 1 / 1.1 };
    if (event.key in moves) {
      const [dx, dy] = moves[event.key];
      this.place(this.x + dx, this.y + dy, this.w);
    } else if (event.key in sizes) {
      const w = Math.min(Math.max(this.w * sizes[event.key], CROP_MIN_WIDTH), this.maxW);
      const centerX = this.x + this.w / 2;
      const centerY = this.y + this.h / 2;
      this.place(centerX - w / 2, centerY - (w * CROP_TALL) / 2, w);
    } else {
      return;
    }
    event.preventDefault();
  }
}

// How much of a face the frame shows: from 0 (none of it) to 1 (all of it). Both boxes are
// fractions of the upright photo, {x, y, w, h}.
function faceShown(face, crop) {
  const width = Math.min(face.x + face.w, crop.x + crop.w) - Math.max(face.x, crop.x);
  const height = Math.min(face.y + face.h, crop.y + crop.h) - Math.max(face.y, crop.y);
  if (width <= 0 || height <= 0 || face.w <= 0 || face.h <= 0) return 0;
  return (width * height) / (face.w * face.h);
}

const FACE_INSIDE = 0.9; // a face shown less than this is cut off
const FACE_OUTSIDE = 0.02; // and shown at most this much, it is outside the frame

// The warning for the faces against the frame, or "" when everybody is in it.
function faceWarning(faces, crop) {
  let cut = 0;
  let outside = 0;
  for (const face of faces || []) {
    const shown = faceShown(face, crop);
    if (shown <= FACE_OUTSIDE) outside += 1;
    else if (shown < FACE_INSIDE) cut += 1;
  }
  const parts = [];
  if (cut === 1) parts.push("Someone is cut off at the edge.");
  else if (cut > 1) parts.push(`${cut} people are cut off at the edge.`);
  if (outside === 1) parts.push("1 person is outside the frame.");
  else if (outside > 1) parts.push(`${outside} people are outside the frame.`);
  return parts.join(" ");
}
