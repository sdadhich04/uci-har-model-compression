# UCI HAR Model Compression

EE 446 lab work on compressing a dense neural-network classifier for the UCI Human Activity Recognition (HAR) dataset. The main runnable workflow is [`EE446_UCI_HAR_DNN_KD_Pruning_Quantization_COMPLETE.ipynb`](EE446_UCI_HAR_DNN_KD_Pruning_Quantization_COMPLETE.ipynb).

## What it does

The notebook downloads `UCI HAR Dataset.zip` from the UCI URL embedded in the notebook when the archive is absent, extracts it at the repository root when `UCI HAR Dataset/` is absent, and loads:

- `train/X_train.txt` and `test/X_test.txt` as `float32` feature arrays;
- `train/y_train.txt` and `test/y_test.txt` as `int32` labels, shifted from one-based to zero-based indices.

It uses the dataset's 561 input features and six activity labels. The dataset and downloaded ZIP are not tracked; `UCI HAR Dataset/` is ignored.

The complete notebook trains a 512-256-128 teacher DNN and a 128-64 student DNN. It distills the student using sparse categorical cross-entropy and KL-divergence losses (`alpha=0.3`, `temperature=4.0`), then applies TensorFlow Model Optimization magnitude pruning. The pruning schedule progresses from 20% to 85% sparsity over 10 epochs. After stripping pruning wrappers, the notebook exports sparse FP32 and sparse INT8 TensorFlow Lite models; INT8 conversion uses up to 200 training samples as a representative dataset. Pre-exported `.tflite` files are in [`models/`](models/).

## Hardware and tools

The notebook identifies Arduino as the intended deployment target. This checkout does not include a board-specific Arduino sketch, board configuration, or on-device inference program. The setup scripts create a `tinyml-arduino` Jupyter environment and can optionally clone the Arduino TensorFlow Lite Micro examples library.

Tools used by the notebook and setup scripts include Python 3.11, TensorFlow 2.14.1, TensorFlow Model Optimization 0.8.0, NumPy, Pandas, Matplotlib, scikit-learn, JupyterLab, and TensorFlow Lite.

## Run

From the repository root, run the setup script appropriate to your system. For Windows, use WSL2:

```bash
bash setup/setup_tinyml_arduino_env_windows_wsl2.sh
```

The setup scripts accept `RUN_SMOKE_TEST=0` to skip their generated smoke test. After setup, activate the environment or use its `launch_jupyter.sh` helper, select the `Python (tinyml-arduino)` kernel, and open:

```text
EE446_UCI_HAR_DNN_KD_Pruning_Quantization_COMPLETE.ipynb
```

The notebook handles the UCI HAR download and extraction when it runs. It requires network access to the UCI URL in the notebook.

## Credits

Sparsh Dadhich. EE 446: Tiny Machine Learning for Ultra Low-Power Edge Computing, University of Washington.
