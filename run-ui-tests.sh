#!/bin/bash
# Approval belongs to this exact test invocation, not to an agent/session.
set -euo pipefail

usage() {
    echo "Usage: ./run-ui-tests.sh EXECUTABLE_TEST_SCRIPT [ARGUMENTS...]"
    echo "Shows a native approval dialog, then notifies you when the UI test ends."
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi
if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi
if [ "$(uname -s)" != Darwin ]; then
    echo "UI tests require macOS; no test was started." >&2
    exit 2
fi
if [ ! -f "$1" ] || [ ! -x "$1" ]; then
    echo "Specify an existing, executable UI test script; no test was started." >&2
    exit 2
fi

script_directory=$(cd -- "$(dirname -- "$1")" && pwd -P)
script_path="$script_directory/$(basename -- "$1")"
shift
printf -v invocation '%q ' "$script_path" "$@"

# Pass the invocation as data, never interpolate it into AppleScript source.
# Cancel, a closed dialog, or any dialog error exits before the test can run.
if ! approval=$(/usr/bin/osascript - "$invocation" <<'APPLESCRIPT'
on run argv
    activate
    set invocation to item 1 of argv
    set promptText to "Allow this UI test script to run now?" & linefeed & linefeed & invocation & linefeed & linefeed & "This script may control the mouse, keyboard and application windows. Please pause your use of the computer while it runs." & linefeed & linefeed & "Approval applies only to this invocation and ends when the script exits."
    set response to display dialog promptText with title "WaifuFinder — Approve UI test script" buttons {"Cancel", "OK"} default button "Cancel" cancel button "Cancel" with icon note
    return button returned of response
end run
APPLESCRIPT
); then
    echo "UI test not approved; no test was started." >&2
    exit 1
fi
if [ "$approval" != OK ]; then
    echo "UI test not approved; no test was started." >&2
    exit 1
fi

# There is no persisted approval, timeout approval, or bypass flag. Keep the
# launcher alive so it can report when the UI steps end and control is returned.
notify_control_return() {
    local test_status=$?
    if ! /usr/bin/osascript - "$test_status" >/dev/null 2>&1 <<'APPLESCRIPT'
on run argv
    set testStatus to item 1 of argv
    if testStatus is "0" then
        set notificationText to "UI test finished. Control is back with you."
    else
        set notificationText to "UI test ended with exit code " & testStatus & ". Control is back with you."
    end if
    display notification notificationText with title "WaifuFinder"
end run
APPLESCRIPT
    then
        echo "Could not display the UI test completion notification." >&2
    fi
}
trap notify_control_return EXIT

"$script_path" "$@"
