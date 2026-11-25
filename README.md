# Agro ARMour

**Offline crop pest and disease advisor for Android.** Take a photo of a leaf or an insect. Agro ARMour names the pest or disease, says how serious it is, and gives treatment advice from Government of India IPM packages in English, Hindi or Tamil, spoken aloud. Everything runs on the phone's Arm CPU, with no internet.

Built for the Bharat AI SoC 2026-27 challenge, Problem Statement 1. The full technical report is in [report/report.pdf](report/report.pdf); the demo video is [AgroARMour_demo.mp4](AgroARMour_demo.mp4), also on [Google Drive](https://drive.google.com/file/d/17GcxcBSrBRmKhqIAPmYjh3DNcH7IPdel/view?usp=sharing).

---

## What is in this folder

```
AgroARMour.apk          ready-to-install app (debug build of the code below)
android/             the Android app (Kotlin, Jetpack Compose); vision models are inside app/src/main/assets/models
kb/                  knowledge base: 47 advisory cards, Hindi/Tamil glossaries, states
models/llama/        Llama 3.2 1B Instruct SpinQuant INT4 (model.pte, tokenizer.model, Meta LICENSE.txt, USE_POLICY.md)
scripts/             push_models.sh: copies the Llama files to the phone
training/            everything used to build the models: data prep, training, evaluation, export
AgroARMour_demo.mp4  demo video (1:57, with voice-over)
report/              final project report: report.tex (LaTeX source) and report.pdf (12 pages)
LICENSE              AGPL-3.0; third-party licences are listed at the end of this README
```

## Run it

**Phone:** Android 8.0 or newer, arm64, 6 GB RAM recommended (tested on a Samsung Galaxy M34). USB debugging on.

1. Install the app: `adb install AgroARMour.apk`
2. Open Agro ARMour once, so it creates its model folder.
3. Copy the language model to the phone (about 1.1 GB): `bash scripts/push_models.sh`. In the GitHub repository, `model.pte` is attached to the `v1.0` release (it is over GitHub's 100 MB file limit); download it into `models/llama/` first.
4. In the app's **Setup** screen, pick a language and your state. For Hindi or Tamil, tap **Download language pack** and install the offline voice (needs internet once).
5. When the readiness checklist is all green, turn on airplane mode and take a photo.

**Build the app yourself:** Android Studio (or JDK 17 and the Android SDK), then

```
cd android
./gradlew assembleDebug        # APK in app/build/outputs/apk/debug/
```

## Models

All four models run on the phone through **ExecuTorch 1.5.1** with the XNNPACK CPU backend.

| Model | What it does | Base model | How it was trained | Result (held-out test) |
|---|---|---|---|---|
| **Pest detector** | Finds insects and names them (42 classes in 27 advisory groups) | YOLO26n, pre-trained on COCO | Fine-tuned on 10,513 IP102 images at 640 px: 87 epochs (early stop), cosine learning rate, mosaic and flips. RTX 4050 laptop, 2.45 h | mAP50 0.648; right advisory group on 86.4 % of photos |
| **Disease classifier** | Names the leaf disease (26 classes over rice, tomato, potato, maize) or "not a leaf" | YOLO26n-cls, pre-trained on ImageNet | Fine-tuned on 14,250 images (Paddy Doctor, PlantDoc, PlantVillage, COCO for not-a-leaf) at 224 px: 53 epochs (early stop), random erasing. RTX 4050 laptop, 51 min | field-photo macro-F1 0.730; not-a-leaf recall 1.00 |
| **Lesion segmenter** | Marks diseased area on the leaf to grade severity | LR-ASPP MobileNetV3-Large, ImageNet backbone | Fine-tuned on 7,813 photos with human-drawn lesion masks (RiceSeg, PlantSeg) and SAM 2.1 leaf masks at 512 px: 40 epochs, cross-entropy + Dice loss, strong colour augmentation. Kaggle T4, 71 min | rice: severity level correct on 93 % of photos |
| **Advisor** | Explains the problem and answers follow-up questions | Llama 3.2 1B Instruct, Meta's SpinQuant INT4 | Used as released, no fine-tuning. Every fact it may use comes from the knowledge-base card, and a validator removes any sentence with a dose, number or pesticide that is not on the card | first token 3.6 s after the photo is analysed; 6.8 tokens/s |

Translation uses Google ML Kit's on-device models; speech uses Android's built-in text-to-speech.

## How it works

1. **Photo → vision.** The pest detector, disease classifier and lesion segmenter run on the photo. The result and the advice card appear within one to two seconds.
2. **Severity.** LOW / MEDIUM / HIGH from the insect count or the share of the leaf that is diseased, raised by the farmer's answer to "How many of 10 nearby plants show this?".
3. **Knowledge base.** A card from the NIPHM/DPPQS IPM packages gives what to do now, a chemical option with its dose, safety advice and state-specific notes. Every dose is checked word for word against the government source, and banned pesticides are rejected.
4. **Advice.** Llama turns the card into 4-6 plain sentences, streamed one by one. Each sentence is checked, translated and spoken as soon as it is ready.
5. **Questions.** The farmer can type follow-up questions. Anything unrelated to the crop problem gets "I cannot help with that."

## Performance (Samsung Galaxy M34, Exynos 1280, CPU only)

| Step | Time |
|---|---|
| Photo → result on screen | 1.2 s (median) |
| Photo → first advice sentence | 8.7 s |
| Follow-up question → first word of the answer | 3.6 s |
| Peak memory | 1.4 GB |

## Training

Everything used to build the models is in [training/](training/). Run from that folder on Linux or WSL2:

```
make env                                   # Python env: torch, ultralytics, executorch 1.5.1
make data-pest data-disease data-lesion    # build the datasets
make train-pest train-disease train-lesion # train (lesion runs on Kaggle)
make eval eval-lesion                      # test-set evaluation
make export-vision bundle-models           # export to ExecuTorch .pte and copy into the app
make kb-lint                               # check every dose in the knowledge base against its source
```

Layout: `ml/data` (dataset builders), `ml/train`, `ml/eval`, `ml/export`, `ml/configs` (training configs and class lists), `ml/llm` (knowledge-base tools and LLM evaluation), `scripts/` (on-phone benchmarks and the airplane-mode check).

## Licences and attribution

The code in this folder is AGPL-3.0 (see `LICENSE`), because the Ultralytics YOLO code and trained weights are AGPL-3.0.

**Built with Llama.** Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The Llama files in `models/llama/` are redistributed under that licence; its full text (`LICENSE.txt`) and Meta's Acceptable Use Policy (`USE_POLICY.md`) are included next to them.

| Component | Used for | Licence |
|---|---|---|
| Llama 3.2 1B Instruct, SpinQuant INT4 (Meta) | advisory text, shipped in `models/llama/` | Llama 3.2 Community License |
| Ultralytics YOLO26 (code and weights) | pest detector, disease classifier | AGPL-3.0 |
| ExecuTorch, PyTorch, torchvision, torchao | on-device runtime, training, export | BSD-3-Clause |
| Jetpack Compose, AndroidX, kotlinx-coroutines | app | Apache-2.0 |
| Google ML Kit Translation | on-device translation | ML Kit terms of service |
| SAM 2.1 (Meta) | main-leaf masks for the lesion dataset, offline only (not shipped) | Apache-2.0 |
| IP102 (Wu et al., CVPR 2019) | pest training data | academic use only; not redistributed |
| PlantDoc (Singh et al., 2020) | disease training and test data | CC BY 4.0 |
| Paddy Doctor (Petchiammal et al., 2022; arXiv:2205.11108) | rice disease data | CC BY |
| PlantVillage (Hughes and Salathé, 2015) | disease training data | CC0 1.0 |
| RiceSeg-5932 lesion masks (doi:10.17632/92jc6w6mcy.1) | lesion training and test data | CC BY 4.0 |
| Rice leaf photos for RiceSeg (Sethy et al., Comput. Electron. Agric. 175, 2020) | lesion training and test data | CC0 1.0 (Kaggle copy) |
| PlantSeg v7 (Zenodo 17719108) | lesion training and test data | CC BY-NC 4.0, non-commercial; not redistributed |
| COCO 2017, 900 images | "not a leaf" class | per-image Flickr licences; not redistributed |
| NIPHM / DPPQS IPM packages and banned-pesticide list (Government of India) | knowledge-base content, cited on every card | government publications; paraphrased and cited |

No dataset images are included in this folder. Full citations are in the references of `report/report.pdf`.
