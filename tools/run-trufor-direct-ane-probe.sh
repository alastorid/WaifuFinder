#!/bin/bash
set -euo pipefail

TRUFOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  echo "Usage: $0 MODEL_DIR IMAGE.png REFERENCE.tfm1 COREML_FP32.mlpackage|mlmodelc [samples] [directOutputSide]" >&2
  exit 64
fi

TRUFOR_MODEL_DIR="$1"
TRUFOR_IMAGE="$2"
TRUFOR_REFERENCE="$3"
TRUFOR_COREML="$4"
TRUFOR_SAMPLES="${5:-7}"
TRUFOR_OUTPUT_SIDE="${6:-512}"
TRUFOR_WORKER="$TRUFOR_ROOT/.native-runtime/waifu-trufor-direct-ane-probe"
TRUFOR_TIMEOUT="${WAIFU_TRUFOR_DIRECT_ANE_TIMEOUT_SECONDS:-120}"

if [ ! -x "$TRUFOR_WORKER" ]; then "$TRUFOR_ROOT/tools/build-native-trufor-direct-ane-probe.sh"; fi
case "$TRUFOR_TIMEOUT" in
  ''|*[!0-9]*) echo 'The direct-ANE timeout must be a positive number of seconds.' >&2; exit 64 ;;
esac
if [ "$TRUFOR_TIMEOUT" -lt 1 ]; then echo 'The direct-ANE timeout must be positive.' >&2; exit 64; fi

TRUFOR_WATCH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/trufor-direct-ane-watch.XXXXXX")"
TRUFOR_TIMEOUT_MARKER="$TRUFOR_WATCH_DIR/timed-out"
trap 'rmdir "$TRUFOR_WATCH_DIR" 2>/dev/null || true' EXIT

"$TRUFOR_WORKER" "$TRUFOR_MODEL_DIR" "$TRUFOR_IMAGE" "$TRUFOR_REFERENCE" \
  "$TRUFOR_COREML" "$TRUFOR_SAMPLES" "$TRUFOR_OUTPUT_SIDE" &
TRUFOR_WORKER_PID=$!
(
  TRUFOR_SLEEP_PID=""
  trap 'if [ -n "$TRUFOR_SLEEP_PID" ]; then kill "$TRUFOR_SLEEP_PID" 2>/dev/null || true; fi; exit 0' TERM INT
  sleep "$TRUFOR_TIMEOUT" &
  TRUFOR_SLEEP_PID=$!
  wait "$TRUFOR_SLEEP_PID"
  if kill -0 "$TRUFOR_WORKER_PID" 2>/dev/null; then
    : > "$TRUFOR_TIMEOUT_MARKER"
    kill -TERM "$TRUFOR_WORKER_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$TRUFOR_WORKER_PID" 2>/dev/null || true
  fi
) &
TRUFOR_WATCHDOG_PID=$!

set +e
wait "$TRUFOR_WORKER_PID"
TRUFOR_EXIT_CODE=$?
set -e
kill "$TRUFOR_WATCHDOG_PID" 2>/dev/null || true
wait "$TRUFOR_WATCHDOG_PID" 2>/dev/null || true

if [ -f "$TRUFOR_TIMEOUT_MARKER" ]; then
  echo "Direct ANE worker exceeded ${TRUFOR_TIMEOUT}s; running the validated public Core ML FP32 fallback." >&2
elif [ "$TRUFOR_EXIT_CODE" -ne 0 ]; then
  echo "Direct ANE worker exited with status $TRUFOR_EXIT_CODE; running the validated public Core ML FP32 fallback." >&2
else
  exit 0
fi

exec "$TRUFOR_WORKER" --coreml-only "$TRUFOR_MODEL_DIR" "$TRUFOR_IMAGE" \
  "$TRUFOR_REFERENCE" "$TRUFOR_COREML" "$TRUFOR_SAMPLES"
