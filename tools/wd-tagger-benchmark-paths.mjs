// setup-wd-tagger-benchmarks.sh creates this independent Core ML installation.
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const coremlDirectory = path.resolve(
  process.env.WAIFU_WD_COREML_DIRECTORY ?? path.join(root, '.build/wd-optimization/coreml'),
);
