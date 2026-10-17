// Run with node tools/test-firefox-extension.cjs; no browser/profile changes.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
let alive = true, opened = true, calls = [], listener, filter, alarmListener;
const browser = {
  runtime: {
    sendNativeMessage: async (_, request) => {
      calls.push(request);
      if (!alive) throw Error('Missing native bridge');
      if (request.command === 'status') return { enabled: true, suffixes: ['mp4', 'dms', 'ts', 'm3u8'] };
      if (request.command === 'open') return { opened };
      return { enabled: true };
    },
    onStartup: { addListener() {} },
    onInstalled: { addListener() {} }
  },
  alarms: {
    create(name, options) { assert.equal(name, 'waifufinder-health'); assert.equal(options.periodInMinutes, 1); },
    onAlarm: { addListener(fn) { alarmListener = fn; } }
  },
  webRequest: { onBeforeRequest: { addListener: (fn, suppliedFilter) => { listener = fn; filter = suppliedFilter; } } },
  menus: { create() {}, onClicked: { addListener() {} } }
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../FirefoxExtension/background.js'), 'utf8'), { browser, URL });
const result = async url => JSON.parse(JSON.stringify(await listener({ url })));
(async () => {
  assert.deepEqual(Array.from(filter.types), ['main_frame']);
  assert(calls.some(request => request.command === 'health'), 'the add-on announces that it is running');
  alarmListener({ name: 'waifufinder-health' });
  assert.equal(calls.at(-1).command, 'health', 'the scheduled check refreshes app health');

  assert.deepEqual(await result('https://example.com/page.html?filename=x.mp4'), {});
  assert.equal((await result('https://example.com/VIDEO.MP4?token=abc')).cancel, true);
  assert.equal(calls.at(-1).url, 'https://example.com/VIDEO.MP4?token=abc');
  for (const url of ['https://example.com/clip.dms', 'https://example.com/clip.ts', 'https://example.com/clip.m3u8']) {
    assert.equal((await result(url)).cancel, true);
  }
  assert.deepEqual(await result('https://example.com/video.mp4/page'), {});

  opened = false;
  assert.deepEqual(await result('https://example.com/video.ts'), {});
  alive = false;
  assert.deepEqual(await result('https://example.com/video.m3u8'), {});
  console.log('Firefox forwarding and health checks passed.');
})().catch(error => { console.error(error); process.exit(1); });
