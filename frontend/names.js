"use strict";

// Tags in an idea (<olena>, <person2>) shown as names. Used by the video list and job page.
const TAG = /<([\p{L}\p{N}][\p{L}\p{N}_-]{0,40})>/gu;

function withNames(text, cast) {
  const named = new Map();
  for (const entry of cast || []) {
    if (entry.photo !== null && !/^person\d+$/i.test(entry.tag)) {
      named.set(`${entry.photo}/${entry.position}`, entry.tag);
    }
  }
  return String(text).replace(TAG, (whole, tag) => {
    const generic = /^person(\d+)$/i.exec(tag);
    if (!generic) return tag;
    const entry = (cast || []).find((e) => e.tag.toLowerCase() === tag.toLowerCase());
    const name = entry && named.get(`${entry.photo}/${entry.position}`);
    return name || `person ${generic[1]}`;
  });
}
