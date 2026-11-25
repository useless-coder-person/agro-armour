# Training, evaluation and export for Agro ARMour. Run from this folder on Linux or WSL2.
# Datasets, runs and exported models live under $(WORK); nothing large is written here.
SHELL := /bin/bash
PY ?= $(HOME)/cropdoc/venv/bin/python
WORK ?= $(HOME)/cropdoc
ASSETS = ../android/app/src/main/assets/models

.PHONY: env data-pest data-disease data-lesion train-pest train-disease train-lesion eval eval-lesion \
        export-vision bundle-models kb-sources kb-lint llm-eval bench thread-sweep

# Python env (torch, ultralytics, executorch 1.5.1; see VERSIONS.lock)
env:
	mkdir -p $(WORK)
	test -x $(PY) || ~/.local/bin/uv venv --python 3.11 $(WORK)/venv
	VIRTUAL_ENV=$(WORK)/venv ~/.local/bin/uv pip install -r ml/requirements.txt --index-strategy unsafe-best-match

# ---- data
data-pest:          # IP102 audit, YOLO conversion, official-list splits
	$(PY) ml/data/audit_ip102.py > $(WORK)/datasets/pest_audit.txt; tail -14 $(WORK)/datasets/pest_audit.txt
	$(PY) ml/data/ip102_to_yolo.py
	$(PY) ml/data/make_splits.py

data-disease:       # Paddy Doctor + PlantDoc + PlantVillage + not-a-leaf, pHash de-duplication
	$(PY) ml/data/build_disease_dataset.py

data-lesion:        # Kaggle GPU: RiceSeg + PlantSeg + SAM 2.1 leaf masks -> $(WORK)/datasets/lesion_ds
	cd ml/data/kaggle_lesion_build && kaggle kernels push -p . | tail -1
	bash scripts/kaggle_wait.sh pranavg77/cropdoc-lesion-build $(WORK)/kaggle/lesion_build
	mkdir -p $(WORK)/datasets && unzip -qo $(WORK)/kaggle/lesion_build/lesion_ds.zip -d $(WORK)/datasets
	$(PY) ml/data/lesion_qc.py $(WORK)/datasets/lesion_ds

# ---- training (resume: make train-pest RESUME=--resume)
train-pest:
	$(PY) ml/train/train_pest.py ml/configs/train_pest_n640.yaml --name $(or $(RUN),P1) $(RESUME)

train-disease:
	$(PY) ml/train/train_disease.py ml/configs/train_disease_n224.yaml --name $(or $(RUN),D2) $(RESUME)

train-lesion:       # Kaggle T4
	bash ml/data/kaggle_lesion_train/make.sh
	bash scripts/kaggle_wait.sh pranavg77/cropdoc-lesion-train $(WORK)/kaggle/lesion_train
	mkdir -p runs/lesion && cp -r $(WORK)/kaggle/lesion_train/L1 runs/lesion/

# ---- evaluation
eval:
	$(PY) ml/eval/eval_disease.py
	$(PY) ml/eval/eval_pest.py --name pest

eval-lesion:
	$(PY) ml/eval/eval_lesion.py --data $(WORK)/datasets/lesion_ds

# ---- export to ExecuTorch (.pte, XNNPACK) and copy into the app
export-vision:
	$(PY) ml/export/export_vision.py runs/pest/P1/weights/best.pt --name pest --imgsz 640
	$(PY) ml/export/export_vision.py runs/disease/D2/weights/best.pt --name disease --imgsz 224
	$(PY) ml/export/verify_pte.py --name disease --weights runs/disease/D2/weights/best.pt --data $(WORK)/datasets/disease --split val
	$(PY) ml/export/verify_pte.py --name pest --weights runs/pest/P1/weights/best.pt --data $(WORK)/datasets/pest/pest.yaml --split val
	$(PY) ml/export/export_lesion_seg.py --weights runs/lesion/L1/best.pt

bundle-models:
	for m in pest disease lesion; do mkdir -p $(ASSETS)/$$m; cp $(WORK)/models/$$m/{model.pte,io_spec.json,labels.json} $(ASSETS)/$$m/; done
	cp $(WORK)/models/pest/thresholds.json $(ASSETS)/pest/

# ---- knowledge base (cards live in ../kb)
kb-sources:         # NIPHM/DPPQS IPM package texts used to verify every dose
	bash ml/llm/fetch_sources.sh

kb-lint:
	$(PY) ml/llm/kb_lint.py

# ---- on-device measurement (phone over adb, app installed, Llama pushed)
llm-eval:
	bash scripts/llm_eval_device.sh

bench:
	bash scripts/bench_device.sh

thread-sweep:       # make thread-sweep IMG=photo.jpg
	bash scripts/thread_sweep.sh $(IMG) rice
