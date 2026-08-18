# UCI HAR Model Compression — Pruning, Knowledge Distillation & Quantization

**EE 446: Tiny Machine Learning for Ultra Low-Power Edge Computing | University of Washington, Spring 2026**

A model-compression case study on the UCI Human Activity Recognition dataset: the same DNN classifier is compressed through three independent techniques — magnitude pruning, knowledge distillation, and post-training/QAT quantization — then compared on size and accuracy.

---

## What it does

Trains a baseline DNN to classify human activities (walking, sitting, standing, etc.) from the UCI HAR dataset's engineered feature vectors, then applies and compares compression techniques:

- **Quantization** (`EE446_Lab3_UCI_HAR_DNN_Quantization_Student_TODO.ipynb`) — dynamic-range, float16, and full int8 post-training quantization, plus quantization-aware training (QAT)
- **Pruning + Quantization** (`EE446_UCI_HAR_DNN_Pruning_Quantization_Student_TODO.ipynb`) — magnitude-based weight pruning combined with quantization
- **Knowledge Distillation + Pruning + Quantization** (`EE446_UCI_HAR_DNN_Knowledge_Distillation_Pruning_Quantization_Student_TODO.ipynb` / `..._COMPLETE.ipynb`) — a smaller student model distilled from the baseline teacher, then pruned and quantized

The `_COMPLETE.ipynb` notebook is the fully worked version of the KD+Pruning+Quantization notebook; the three `_Student_TODO.ipynb` notebooks are the lab's starting templates.

---

## Results (model variants)

13 exported `.tflite` variants in `models/` span the full compression space:

| Variant | Technique |
|---|---|
| `baseline_fp32` | Uncompressed baseline |
| `baseline_fp16` | Float16 post-training quantization |
| `baseline_dynamic_range` | Dynamic-range post-training quantization |
| `baseline_ptq_int8` | Full int8 post-training quantization |
| `qat_int8` | Quantization-aware training, int8 |
| `stripped_sparse_fp32` / `fp16` | Pruned (mask stripped), float |
| `pruned_with_mask_fp32` / `fp16` | Pruned, mask retained, float |
| `distilled_student_fp32` | Knowledge-distilled student model, float |
| `distilled_stripped_sparse_fp32` / `int8` | Distilled + pruned + quantized |
| `pruned_distilled_with_mask_fp32` | Distilled + pruned (mask retained) |

> Note: "mask" in these filenames refers to TensorFlow Model Optimization's pruning weight masks, not an image/face mask.

---

## Repository contents

```
EE446_Lab3_UCI_HAR_DNN_Quantization_Student_TODO.ipynb
EE446_UCI_HAR_DNN_Pruning_Quantization_Student_TODO.ipynb
EE446_UCI_HAR_DNN_Knowledge_Distillation_Pruning_Quantization_Student_TODO.ipynb
EE446_UCI_HAR_DNN_KD_Pruning_Quantization_COMPLETE.ipynb   ← Fully worked version of the KD notebook
Lab3_EE_446.pdf                                             ← Written report (answers for all three notebooks)
models/                                                      ← 13 .tflite variants (see table above)
setup/
  setup_tinyml_arduino_env.sh                ← Default environment setup
  setup_tinyml_arduino_env_linux.sh
  setup_tinyml_arduino_env_macos_intel.sh
  setup_tinyml_arduino_env_windows_wsl2.sh
```

---

## Dataset

Trained on the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (~270 MB, not included in this repo due to size). Download it from the UCI Machine Learning Repository and place it in a `UCI HAR Dataset/` folder at the repo root to run the notebooks.

---

## Quick start

```bash
# Set up the environment (pick the script for your OS)
bash setup/setup_tinyml_arduino_env_linux.sh      # or _macos_intel.sh / _windows_wsl2.sh

# Download and extract the UCI HAR dataset into "UCI HAR Dataset/" at the repo root, then:
jupyter notebook EE446_UCI_HAR_DNN_KD_Pruning_Quantization_COMPLETE.ipynb
```

**Dependencies:** `numpy`, `pandas`, `tensorflow`, `tensorflow-model-optimization`, `scikit-learn`, `matplotlib`

---

## Authors

Sparsh Dadhich — University of Washington, ECE / Neuroscience
