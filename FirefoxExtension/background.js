/* Only explicit top-level media navigations are candidates, never page subresources. */
const HOST = "com.alastorid.waifufinder";
let cachedSettings;
let settingsExpireAt = 0;
async function settings() {
  if (cachedSettings && Date.now() < settingsExpireAt) return cachedSettings;
  try { cachedSettings = await browser.runtime.sendNativeMessage(HOST, { command: "status" }); }
  catch (_) { cachedSettings = { enabled: false, suffixes: [] }; }
  settingsExpireAt = Date.now() + 5000;
  return cachedSettings;
}
async function heartbeat() {
  try {
    await browser.runtime.sendNativeMessage(HOST, { command: "health" });
    cachedSettings = undefined;
    settingsExpireAt = 0;
  }
  catch (_) { /* The app bridge may not be registered yet. */ }
}
function matches(text, suffixes) {
  try {
    const url = new URL(text);
    return ["http:", "https:"].includes(url.protocol) &&
      suffixes.includes(decodeURIComponent(url.pathname).split("/").pop().split(".").pop().toLowerCase());
  } catch (_) { return false; }
}
async function forward(url) {
  const state = await settings();
  if (!state.enabled || !matches(url, state.suffixes)) return false;
  try {
    const result = await browser.runtime.sendNativeMessage(HOST, { command: "open", url });
    return result.opened === true;
  } catch (_) { return false; }
}
browser.webRequest.onBeforeRequest.addListener(async details => {
  // Failure leaves the original navigation intact.
  return await forward(details.url) ? { cancel: true } : {};
}, { urls: ["http://*/*", "https://*/*"], types: ["main_frame"] }, ["blocking"]);
browser.menus.create({ id: "waifu-open", title: "Send media link to WaifuFinder", contexts: ["link"] });
browser.menus.onClicked.addListener(async (info) => {
  if (info.menuItemId === "waifu-open") await forward(info.linkUrl);
});

// A lightweight native-messaging ping lets WaifuFinder show a live connection
// state in Settings. It contains no browsing history or page data.
browser.alarms.create("waifufinder-health", { periodInMinutes: 1 });
browser.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === "waifufinder-health") heartbeat();
});
browser.runtime.onStartup.addListener(heartbeat);
browser.runtime.onInstalled.addListener(heartbeat);
heartbeat();
