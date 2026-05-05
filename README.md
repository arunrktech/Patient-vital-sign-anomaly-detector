# Patient Vital Sign Anomaly Detector

## Overview

This project builds a binary classification model entirely from scratch using only NumPy, with no machine learning libraries. The goal is to develop a genuine bottom-up understanding of how machine learning works, starting from raw data generation all the way to a trained model that identifies abnormal patients and writes results back to Excel.

The problem domain is clinical. Given four patient vital signs (heart rate, blood pressure, blood oxygen saturation, and body temperature), the model learns to classify whether each patient's vitals are normal or abnormal. The model discovers this entirely on its own from the data, with no prior knowledge of what the features mean.

---

## How the data flows

```
Cell 1: np.random generates 1000 patient records
              ↓
        written to patient_vitals.xlsx (no label column)
              ↓
        read back from Excel into X and df
              ↓
Cells 2-8: model trains on X and learns weights
              ↓
Cell 9: predictions written to results_output.xlsx (3 sheets)
```

In a real project, the np.random generation in Cell 1 would be removed entirely and replaced with a direct `pd.read_excel()` call pointing at your actual hospital data file. The generation block exists only because we are working with synthetic data.

---

## Requirements

No installation needed. Run in Google Colab at colab.research.google.com which has all dependencies pre-installed.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from google.colab import files
```

---

## Dataset

Synthetically generated inside Cell 1 using the same random seed for reproducibility.

| Feature | Normal range | Abnormal range |
|---|---|---|
| Heart rate (bpm) | mean 72, std 8 | mean 110, std 15 |
| Systolic BP (mmHg) | mean 120, std 10 | mean 160, std 20 |
| SpO2 (%) | mean 98, std 1 | mean 93, std 3 |
| Temperature (C) | mean 37, std 0.3 | mean 38.5, std 0.5 |

500 normal and 500 abnormal patients are generated, giving 1000 total records. The Excel input file contains raw vitals only with no label column.

---

## Notebook structure

### Cell 1: Generate data and load from Excel

Generates 1000 synthetic patient records using `np.random.normal`, writes them to `patient_vitals.xlsx` at `/content/patient_vitals.xlsx` with no label column, then reads the file back into a pandas DataFrame and extracts the feature matrix `X` and label array `y`.

`X` has shape `(1000, 4)`. `y` is derived from row position: first 500 rows are labeled 0 (normal), last 500 are labeled 1 (abnormal). Labels are only used during training to compute loss. They are never used during prediction.

### Cell 2: Normalize the data

Computes the mean and standard deviation of each column using `X.mean(axis=0)` and `X.std(axis=0)`, then applies the formula `z = (x - mean) / std` to every value. This puts all 4 features on the same scale so the model treats them fairly.

After normalization all columns have a mean of 0.0 and a standard deviation of 1.0. The mean and std computed here are saved as `X_mean` and `X_std` and reused in Cells 8 and 9 when normalizing new patients for prediction.

### Cell 3: Sigmoid function

Defines `sigmoid(z) = 1 / (1 + e^(-z))`, which squashes any number into a probability between 0 and 1. This is the function that converts the model's raw output into a prediction.

`sigmoid(-6)` is near 0 (very likely normal), `sigmoid(0)` is exactly 0.5 (uncertain), `sigmoid(+6)` is near 1 (very likely abnormal).

### Cell 4: Forward pass

Initializes all weights `W` to zero and bias `b` to zero. Computes `z = X_norm @ W + b` for all 1000 patients simultaneously using NumPy's dot product operator `@`. Passes `z` through sigmoid to produce `y_hat`, the predicted probability for each patient.

With weights at zero, every patient gets a prediction of exactly 0.5. The model knows nothing yet. This flat line of 0.5 is the honest starting point before any learning happens.

### Cell 5: Loss function

Defines binary cross-entropy loss, which measures how wrong the current predictions are as a single number. Lower is better.

`0.6931` means the model is no better than a coin flip. This is the baseline that training must beat. `0.0` would be perfect. The tiny `1e-9` added inside the log prevents a divide-by-zero error if any prediction ever reaches exactly 0 or 1.

### Cell 6: Gradients and weight update

Defines `gradient_step()` which computes how wrong each prediction was (`error = y_hat - y`), uses that error to compute a gradient for each weight (`dW = X.T @ error / n`) and bias (`db = mean(error)`), then updates every weight by moving in the opposite direction of the gradient (`W = W - lr * dW`).

After just one step the heart rate and blood pressure weights go positive (higher values push toward abnormal) and the SpO2 weight goes negative (lower oxygen pushes toward abnormal). The model figured that out from the data alone in a single update, which already matches clinical reality.

### Cell 7: Training loop

Repeats the forward pass, loss computation, and gradient update 1000 times. Each repetition is called an epoch. Prints loss and accuracy every 100 epochs so you can watch the model improve.

Loss falls from `0.6931` down to around `0.10`. Accuracy climbs from `50%` up to around `96%`. The trained weights `W` and bias `b` produced by this cell are used in all subsequent cells.

### Cell 8: Evaluate and plot

Computes final predictions and accuracy on the full dataset. Prints the learned weight for each feature so you can see what the model discovered. Plots the loss curve across all 1000 epochs. Tests the model on two hand-crafted patients to confirm it predicts correctly on new data it has never seen.

### Cell 9: Write results to Excel

Runs predictions on all 1000 patients, adds `prediction` and `probability_pct` columns to the DataFrame, and writes `results_output.xlsx` to `/content/results_output.xlsx`. The file downloads automatically.

---

## Output file structure

`results_output.xlsx` contains three sheets:

| Sheet | Contents | Header color |
|---|---|---|
| `original_data` | Raw vitals only, exactly as loaded from the input file | Dark green |
| `all_patients` | All 1000 patients with `prediction` and `probability_pct` added | Purple |
| `abnormal_patients` | Flagged rows only | Dark red |

The `probability_pct` column shows the model's confidence as a percentage. A value of `94.3` means the model is 94.3% confident that patient is abnormal.

---

## Expected results

| Checkpoint | Expected value |
|---|---|
| Loss at epoch 0 | 0.6931 |
| Accuracy at epoch 0 | 50.0% |
| Loss at epoch 1000 | ~0.10 |
| Accuracy at epoch 1000 | ~96% |
| Heart rate weight direction | positive (toward abnormal) |
| Blood pressure weight direction | positive (toward abnormal) |
| SpO2 weight direction | negative (toward normal) |
| Temperature weight direction | positive (toward abnormal) |
| Flagged abnormal patients | ~496 out of 1000 |

---

## Key concepts covered

| Concept | Cell |
|---|---|
| Synthetic data generation with fixed seed | 1 |
| Reading and writing Excel files with openpyxl | 1, 9 |
| Feature extraction from pandas DataFrame | 1 |
| Feature normalization (z-score standardization) | 2 |
| Sigmoid activation function | 3 |
| Dot product and forward pass | 4 |
| Binary cross-entropy loss | 5 |
| Gradient computation and weight update | 6 |
| Training loop and gradient descent | 7 |
| Model evaluation and accuracy | 8 |
| Prediction on new unseen patients | 8 |
| Multi-sheet Excel output with styling | 9 |

---

## Replacing synthetic data with real data

To use a real patient dataset instead of the generated one, remove the data generation block from Cell 1 and replace the entire cell with:

```python
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

INPUT_PATH = '/content/your_real_file.xlsx'   # upload your file first
df = pd.read_excel(INPUT_PATH)

feature_cols = ['heart_rate_bpm', 'systolic_bp_mmhg', 'spo2_pct', 'temperature_c']
X = df[feature_cols].values
n = len(df) // 2
y = np.array([0]*n + [1]*n, dtype=float)

print('Dataset shape:', X.shape)
print(df.head(3))
```

All subsequent cells work unchanged because they only depend on `X`, `y`, `X_mean`, `X_std`, and `df`.

---

## What comes next

This is Stage 1 of a three-stage learning project.

**Stage 2** rebuilds the same model in sklearn in 5 lines, introduces Random Forest and Gradient Boosting, and shifts focus to model evaluation: confusion matrix, precision vs recall, ROC-AUC, and feature importance.

**Stage 3** reimplements the model as a 3-layer neural network in PyTorch, introducing `nn.Module`, autograd, and the Adam optimizer to show how PyTorch automates the gradient math written by hand in Cell 6.
