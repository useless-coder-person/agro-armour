#!/usr/bin/env bash
# Push the Llama .pte + tokenizer to the app's external files dir (not in the APK).
# Launch the app once first: it creates the target folder itself.
# Usage: scripts/push_models.sh [model.pte] [tokenizer.model]
set -euo pipefail
ADB=${ADB:-adb}
PTE=${1:-$(dirname "$0")/../models/llama/model.pte}
TOK=${2:-$(dirname "$0")/../models/llama/tokenizer.model}
# adb.exe (Windows, called from WSL) needs Windows paths
win() { case "$ADB" in *.exe) wslpath -w "$1";; *) echo "$1";; esac; }
DST=/sdcard/Android/data/org.agroarmour/files/models/llama
"$ADB" shell am start -n org.agroarmour/.MainActivity >/dev/null; sleep 3
"$ADB" push "$(win "$PTE")" $DST/model.pte
"$ADB" push "$(win "$TOK")" $DST/tokenizer.model
"$ADB" shell sha256sum $DST/model.pte $DST/tokenizer.model
