#!/usr/bin/env bash
# Wait for a Kaggle kernel to finish, then download its output. Usage: scripts/kaggle_wait.sh <owner/kernel> <out dir>
set -uo pipefail
K=~/.local/bin/kaggle
until st=$("$K" kernels status "$1" 2>&1 | tail -1); echo "$st" | grep -qiE "complete|error|cancel"; do sleep 30; done
echo "$st"
mkdir -p "$2" && "$K" kernels output "$1" -p "$2" 2>&1 | tail -2
