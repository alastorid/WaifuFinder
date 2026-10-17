(() => {
  const sequence = document.querySelector("[data-sequence-demo]");
  if (sequence) {
    const strip = sequence.querySelector("[data-sequence-strip]");
    const route = sequence.querySelector("[data-sequence-route]");
    const caption = sequence.querySelector("[data-sequence-caption]");
    const tiles = Array.from(strip.children);
    const points = [[55,150],[155,130],[255,140],[355,95],[455,115],[555,65],[655,80],[765,45]];
    const modes = {
      mixed: [2,7,0,5,3,1,6,4],
      sorted: [0,1,2,3,4,5,6,7],
      reverse: [7,6,5,4,3,2,1,0],
    };
    const captions = {
      mixed: "The same drawings in a mixed order: many jumps between appearances.",
      sorted: "A continuous visual sequence, from amber to twilight.",
      reverse: "The identical visual route, followed backward.",
    };
    sequence.querySelectorAll("[data-sequence-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.dataset.sequenceMode;
        modes[mode].forEach((index) => strip.append(tiles[index]));
        route.setAttribute("d", modes[mode].map((index, position) => `${position ? "L" : "M"}${points[index].join(" ")}`).join(" "));
        caption.textContent = captions[mode];
        sequence.querySelectorAll("[data-sequence-mode]").forEach((control) => control.setAttribute("aria-pressed", String(control === button)));
      });
    });
  }
  const releasePage = "https://github.com/alastorid/WaifuFinder/releases";
  const releaseAPI =
    "https://api.github.com/repos/alastorid/WaifuFinder/releases/latest";
  const links = document.querySelectorAll("[data-release-link]");
  const labels = document.querySelectorAll("[data-download-label]");
  const versionLabel = document.querySelector("[data-release-version]");

  fetch(releaseAPI, { headers: { Accept: "application/vnd.github+json" } })
    .then((response) => {
      if (!response.ok) throw new Error("No published release yet");
      return response.json();
    })
    .then((release) => {
      const dmg = release.assets.find(
        (asset) => asset.name === "WaifuFinder-macos-arm64.dmg",
      );
      if (!dmg) return;
      const nonCommercial = /non-commercial edition/i.test(release.name || "");
      const edition = nonCommercial ? " · non-commercial" : "";
      links.forEach((link) => {
        link.href = dmg.browser_download_url;
      });
      labels.forEach((label) => {
        label.textContent = `Download ${release.tag_name}${edition} (.dmg)`;
      });
      if (versionLabel)
        versionLabel.textContent = `${release.tag_name}${edition} · disk image`;
    })
    .catch(() => {
      links.forEach((link) => {
        link.href = releasePage;
      });
      labels.forEach((label) => {
        label.textContent = "Browse releases";
      });
      if (versionLabel) versionLabel.textContent = "See releases on GitHub";
    });
})();

// A small, explicitly labeled illustration of the app's all-required / any-excluded logic.
(() => {
  const controls = [...document.querySelectorAll("[data-trait]")];
  const exclusion = document.querySelector("#exclude-text");
  const frames = [...document.querySelectorAll("[data-tags]")];
  const count = document.querySelector("#filter-count");
  if (!exclusion || !count) return;
  function update() {
    const required = controls
      .filter((button) => button.getAttribute("aria-pressed") === "true")
      .map((button) => button.dataset.trait);
    const excludeText = exclusion.getAttribute("aria-pressed") === "true";
    let matches = 0;
    frames.forEach((frame) => {
      const tags = frame.dataset.tags.split(" ");
      const match =
        required.every((tag) => tags.includes(tag)) &&
        !(excludeText && tags.includes("text"));
      frame.classList.toggle("is-filtered", !match);
      frame.setAttribute(
        "aria-label",
        `${frame.querySelector("strong").textContent}: ${match ? "matches" : "filtered out"}`,
      );
      if (match) matches++;
    });
    count.textContent = `${matches} of ${frames.length} sample frames match`;
  }
  [...controls, exclusion].forEach((button) =>
    button.addEventListener("click", () => {
      button.setAttribute(
        "aria-pressed",
        String(button.getAttribute("aria-pressed") !== "true"),
      );
      update();
    }),
  );
})();
