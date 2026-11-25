#!/usr/bin/env bash
# Per-stage latency, TTFT, tok/s, peak PSS on the connected phone. Each image runs the full pipeline
# (vision -> fusion -> card -> Llama advice). Results: logcat tag AgroArmourBench + files/metrics.jsonl, pulled to
# docs/bench/<device>_<date>.jsonl. Usage: scripts/bench_device.sh img1.jpg[:crop] img2.jpg[:crop] ...
set -euo pipefail
ADB=${ADB:-$(command -v adb || echo /mnt/c/Users/gpran/AppData/Local/Android/Sdk/platform-tools/adb.exe)}
win() { case "$ADB" in *.exe) wslpath -w "$1";; *) echo "$1";; esac; }
APPDIR=/sdcard/Android/data/org.agroarmour/files
"$ADB" shell am force-stop org.agroarmour  # cold start: the first image overlaps the app-start model load
"$ADB" shell am start -n org.agroarmour/.MainActivity >/dev/null; sleep 2
"$ADB" shell rm -f $APPDIR/metrics.jsonl
"$ADB" logcat -c
for spec in "$@"; do
  img=${spec%%:*}; crop=${spec#*:}; [ "$crop" = "$spec" ] || [ -z "$crop" ] && crop=none  # "none" = farmer said "don't know"
  "$ADB" push "$(win "$img")" $APPDIR/bench/$(basename "$img") >/dev/null
  "$ADB" shell am start -n org.agroarmour/.MainActivity --es bench_image "$(basename "$img")" --es crop "$crop" >/dev/null
  for i in $(seq 1 120); do  # wait for "done" (LLM advice can take a minute on a mid-range phone)
    "$ADB" logcat -d -s AgroArmourBench | grep -q "done $(basename "$img")" && break; sleep 2
  done
done
mkdir -p docs/bench
out=docs/bench/$("$ADB" shell getprop ro.product.model | tr -d '\r')_$(date +%F_%H%M).jsonl
"$ADB" shell cat $APPDIR/metrics.jsonl | tr -d '\r' > "$out"
echo "wrote $out ($(wc -l < "$out") runs)"
