#!/usr/bin/env bash
# Offline proof: airplane mode ON -> full pipeline on a bundled-style test image in the chosen language
# (vision -> card -> Llama -> ML Kit translation -> TTS) -> assert LLM advice and translations -> airplane mode OFF.
# Usage: scripts/airplane_check.sh <image.jpg> [crop] [lang]
set -euo pipefail
ADB=${ADB:-$(command -v adb || echo /mnt/c/Users/gpran/AppData/Local/Android/Sdk/platform-tools/adb.exe)}
win() { case "$ADB" in *.exe) wslpath -w "$1";; *) echo "$1";; esac; }
IMG=$1; CROP=${2:-rice}; LANG_=${3:-hi}
APPDIR=/sdcard/Android/data/org.agroarmour/files
restore() { "$ADB" shell cmd connectivity airplane-mode disable >/dev/null 2>&1 || true; }
trap restore EXIT
"$ADB" push "$(win "$IMG")" $APPDIR/bench/$(basename "$IMG") >/dev/null
"$ADB" shell am force-stop org.agroarmour
"$ADB" shell cmd connectivity airplane-mode enable
sleep 3
echo "airplane mode: $("$ADB" shell settings get global airplane_mode_on | tr -d '\r')"
"$ADB" logcat -c
"$ADB" shell am start -n org.agroarmour/.MainActivity --es lang "$LANG_" --es bench_image "$(basename "$IMG")" --es crop "$CROP" >/dev/null
for i in $(seq 1 120); do "$ADB" logcat -d -s AgroArmourBench | grep -q "done $(basename "$IMG")" && break; sleep 2; done
line=$("$ADB" logcat -d -s AgroArmourBench | grep -o '{"stages_ms.*' | tail -1 | tr -d '\\')  # JSON escapes "/" as "\/"
echo "$line" > docs/bench/airplane_check_$(date +%F_%H%M).json
echo "$line" | grep -q '"advice_mode":"llm"' || { echo "FAIL: advice was not produced by the LLM"; exit 1; }
if [ "$LANG_" != en ]; then
  t=$(echo "$line" | grep -o '"translated_sentences":"[0-9]*/[0-9]*"' | grep -o '[0-9]*/[0-9]*')
  [ -n "$t" ] && [ "${t%/*}" = "${t#*/}" ] && [ "${t%/*}" -gt 0 ] || { echo "FAIL: translations $t"; exit 1; }
fi
echo "PASS: offline pipeline in $LANG_ (translated ${t:-n/a}); log in docs/bench/airplane_check_$(date +%F_%H%M).json"
