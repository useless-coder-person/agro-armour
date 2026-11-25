#!/usr/bin/env bash
# On-device LLM eval: every case in ml/llm/cases/cases.jsonl through the real app path (KB card -> FACTS ->
# Llama via ExecuTorch -> OutputValidator -> retry/template). English. Then ml/llm/prompt_eval.py summarises.
set -euo pipefail
ADB=${ADB:-$(command -v adb || echo /mnt/c/Users/gpran/AppData/Local/Android/Sdk/platform-tools/adb.exe)}
APPDIR=/sdcard/Android/data/org.agroarmour/files
LANG_=${1:-en}; FILTER=${2:-.}   # usage: llm_eval_device.sh [en|hi|ta] [case-id regex]
OUT=docs/bench/llm_eval_${LANG_}_$(date +%F_%H%M).jsonl
"$ADB" shell am start -n org.agroarmour/.MainActivity --es lang "$LANG_" >/dev/null; sleep 3
"$ADB" shell rm -f $APPDIR/metrics.jsonl
while IFS= read -r c <&3; do  # fd 3: adb shell would swallow stdin
  id=$(echo "$c" | sed 's/.*"id": "\([^"]*\)".*/\1/'); key=$(echo "$c" | sed 's/.*"key": "\([^"]*\)".*/\1/')
  sev=$(echo "$c" | sed 's/.*"sev": "\([^"]*\)".*/\1/'); q=$(echo "$c" | grep -o '"q": "[^"]*"' | sed 's/"q": "\(.*\)"/\1/' || true)
  "$ADB" logcat -c
  if [ -n "$q" ]; then "$ADB" shell "am start -n org.agroarmour/.MainActivity --es eval_id $id --es eval_key $key --es eval_sev $sev --es eval_q '$q'" >/dev/null
  else "$ADB" shell am start -n org.agroarmour/.MainActivity --es eval_id $id --es eval_key $key --es eval_sev $sev >/dev/null; fi
  for i in $(seq 1 90); do "$ADB" logcat -d -s AgroArmourBench | grep -q "done $id" && break; sleep 2; done; [ "$i" = 90 ] && echo "TIMEOUT $id"
  echo "$id done"
done 3< <(grep -E "\"id\": \"($FILTER)" ml/llm/cases/cases.jsonl)
"$ADB" shell cat $APPDIR/metrics.jsonl | tr -d '\r' > "$OUT"
[ "$FILTER" = . ] && ${PY:-$HOME/cropdoc/venv/bin/python} ml/llm/prompt_eval.py "$OUT" || echo "subset run: $OUT"
