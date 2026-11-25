#!/usr/bin/env bash
# Thread-count sweep. For each N: cold start with --ei threads N (all models share one
# ExecuTorch threadpool of N), wait for the app-start load, then one photo (vision + advisory) and two fixed advisories
# (G_PLANTHOPPER HIGH). Records -> docs/bench/thread_sweep_<date>.jsonl; table in docs/report/figures/bench_device.md.
# Usage: scripts/thread_sweep.sh <image.jpg> [crop] [N ...]     (REST=seconds between N, default 60)
set -euo pipefail
ADB=${ADB:-$(command -v adb || echo /mnt/c/Users/gpran/AppData/Local/Android/Sdk/platform-tools/adb.exe)}
win() { case "$ADB" in *.exe) wslpath -w "$1";; *) echo "$1";; esac; }
IMG=$1; CROP=${2:-rice}; shift $(( $# < 2 ? $# : 2 )); NS=${*:-1 2 4 6 8}
APPDIR=/sdcard/Android/data/org.agroarmour/files
OUT=docs/bench/thread_sweep_$(date +%F_%H%M).jsonl
wait_for() { for _ in $(seq 1 150); do "$ADB" logcat -d -s AgroArmourBench < /dev/null | grep -q "$1" && return 0; sleep 2; done; echo "TIMEOUT $1"; }
"$ADB" push "$(win "$IMG")" $APPDIR/bench/$(basename "$IMG") > /dev/null
"$ADB" shell rm -f $APPDIR/metrics.jsonl < /dev/null
for n in $NS; do
  "$ADB" shell am force-stop org.agroarmour < /dev/null; sleep 2; "$ADB" logcat -c < /dev/null
  "$ADB" shell am start -n org.agroarmour/.MainActivity --ei threads "$n" > /dev/null < /dev/null
  wait_for '"preload":true'
  echo "N=$n: $("$ADB" logcat -d < /dev/null | grep -o 'Resetting threadpool to [0-9]* threads' | tr '\n' ';')"
  "$ADB" logcat -c < /dev/null
  "$ADB" shell am start -n org.agroarmour/.MainActivity --es bench_image "$(basename "$IMG")" --es crop "$CROP" > /dev/null < /dev/null
  wait_for "done $(basename "$IMG")"
  for k in 1 2; do
    "$ADB" logcat -c < /dev/null
    "$ADB" shell am start -n org.agroarmour/.MainActivity --es eval_id "s$n$k" --es eval_key G_PLANTHOPPER --es eval_sev HIGH > /dev/null < /dev/null
    wait_for "done s$n$k"
  done
  sleep "${REST:-60}"
done
"$ADB" shell am force-stop org.agroarmour < /dev/null
"$ADB" shell cat $APPDIR/metrics.jsonl < /dev/null | tr -d '\r' > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") records)"
