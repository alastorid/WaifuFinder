async function refresh() {
  let bridge;
  try { bridge = await browser.runtime.sendNativeMessage("com.alastorid.waifufinder", { command: "status" }); }
  catch (_) { bridge = { enabled: false }; }
  document.getElementById("status").textContent = bridge.enabled
    ? `Connected · ${bridge.suffixes.map(s => "." + s).join(", ")}`
    : "Not connected. Set up Firefox in WaifuFinder’s settings.";
}
refresh();
