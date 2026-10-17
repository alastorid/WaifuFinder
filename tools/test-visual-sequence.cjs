const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "site/index.html"), "utf8");
assert.equal((html.match(/data-scene="\d"/g) || []).length, 8);
assert.ok(html.includes("No personal photos or video samples."));
assert.ok(html.includes("116.296 ms") && html.includes("29.068 ms"));
assert.ok(html.includes("Grouping is off by default"));

const tiles = Array.from({ length: 8 }, (_, id) => ({ id }));
const strip = { children: tiles.slice(), append(tile) {
  this.children = this.children.filter((item) => item !== tile);
  this.children.push(tile);
} };
const route = { setAttribute(name, value) { this[name] = value; } };
const caption = { textContent: "" };
const controls = ["mixed", "sorted", "reverse"].map((mode) => ({
  dataset: { sequenceMode: mode }, attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
  addEventListener(name, callback) { this[name] = callback; },
}));
const demo = {
  querySelector(selector) { return ({ "[data-sequence-strip]": strip, "[data-sequence-route]": route, "[data-sequence-caption]": caption })[selector]; },
  querySelectorAll() { return controls; },
};
const document = {
  querySelector(selector) { return selector === "[data-sequence-demo]" ? demo : null; },
  querySelectorAll() { return []; },
};
vm.runInNewContext(fs.readFileSync(path.join(root, "site/site.js"), "utf8"), {
  document, fetch: () => Promise.reject(new Error("offline test")),
});
for (const mode of ["mixed", "sorted", "reverse", "sorted"]) {
  const control = controls.find((item) => item.dataset.sequenceMode === mode);
  control.click();
  const expected = ({ mixed: [2,7,0,5,3,1,6,4], sorted: [0,1,2,3,4,5,6,7], reverse: [7,6,5,4,3,2,1,0] })[mode];
  assert.deepEqual(strip.children.map((tile) => tile.id), expected);
  assert.equal(new Set(strip.children).size, 8);
  assert.equal(controls.filter((item) => item.attributes["aria-pressed"] === "true").length, 1);
  assert.equal(control.attributes["aria-pressed"], "true");
  assert.ok(route.d.startsWith(mode === "reverse" ? "M765 45" : mode === "mixed" ? "M255 140" : "M55 150"));
  assert.ok(caption.textContent.length > 0);
}
console.log("Synthetic visual-sequence page checks passed: eight original drawings, reversible sequence, path updates, accessible controls and measured benchmark copy.");
