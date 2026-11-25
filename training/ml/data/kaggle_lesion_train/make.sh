#!/usr/bin/env bash
# Builds train_kernel.py (one file, as Kaggle script kernels need) from the repo sources, then pushes it.
set -euo pipefail; cd "$(dirname "$0")"; R=../../..
{ cat $R/ml/train/lesion_net.py; sed '/^if __name__/,$d' $R/ml/data/lesion_qc.py;sed "/^if __name__/,$$d" $R/ml/data/lesion_qc.py; grep -v "^from lesion_net import\|^sys.path.insert" $R/ml/train/train_lesion_seg.py | sed "s/^if __name__ == \"__main__\":/if __name__ == \"__main__\" and not Path(\"\/kaggle\/input\").exists():/";
  echo "CFG = \"\"\"$(cat $R/ml/configs/train_lesion_512.yaml)\"\"\""; cat runner.py; } > train_kernel.py
~/.local/bin/kaggle kernels push -p . | tail -1
