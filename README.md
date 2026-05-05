# ML from scratch: patient vital sign anomaly detector

## Overview

This project builds a binary classification model entirely from scratch using only NumPy, with no machine learning libraries. The goal is to develop a genuine bottom-up understanding of how machine learning works, from raw data all the way to a trained model that makes predictions.

The problem domain is clinical: given a patient's four vital signs (heart rate, blood pressure, blood oxygen saturation, and body temperature), the model learns to classify whether the patient's vitals are normal or abnormal. This is a deliberately simple and interpretable problem so that the focus stays on understanding the mechanics rather than the domain complexity.

By the end of this project you will have implemented, from scratch and without any ML framework, every component that sits inside every modern machine learning system: data normalization, the forward pass, a loss function, gradient computation, and a weight update loop.

---

## Why build from scratch?

Most ML tutorials start with sklearn or PyTorch and treat the underlying math as a black box. The problem is that when something goes wrong in a real project, you have no mental model of what is actually happening. Building logistic regression by hand means that when you later use a framework, you understand exactly what it is doing for you. The gap between "I can call fit()" and "I understand what fit() is doing" is the gap this project closes.

---

## Project structure

```
ml-from-scratch/
└── vitals_anomaly_detector.ipynb    # single Colab notebook, one cell per step
```

All work lives in one Google Colab notebook. Each step is a self-contained cell that builds on the previous one.

---

## Dataset

The dataset is synthetically generated inside the notebook. No downloads or external files are needed.

| Feature | Normal range | Abnormal range |
|---|---|---|
| Heart rate (bpm) | mean 72, std 8 | mean 110, std 15 |
| Systolic BP (mmHg) | mean 120, std 10 | mean 160, std 20 |
| SpO2 (%) | mean 98, std 1 | mean 93, std 3 |
| Temperature (C) | mean 37, std 0.3 | mean 38.5, std 0.5 |

500 normal patients and 500 abnormal patients are generated, giving 1000 total records with balanced classes.

---

## Requirements

No installation needed beyond Google Colab, which comes with NumPy and Matplotlib pre-installed. Open a new notebook at colab.research.google.com and run each cell in order.

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Stage 1: Logistic regression from scratch in NumPy

### Step 1: Create the dataset

Generate 500 normal and 500 abnormal synthetic patient records, each with 4 vital signs: heart rate, blood pressure, SpO2, and temperature. Stack them into a single dataset of 1000 rows and assign labels (0 for normal, 1 for abnormal).

```python
import numpy as np

np.random.seed(42)
n = 500

X_normal = np.column_stack([
    np.random.normal(72,  8,   n),
    np.random.normal(120, 10,  n),
    np.random.normal(98,  1,   n),
    np.random.normal(37,  0.3, n),
])
X_abnormal = np.column_stack([
    np.random.normal(110, 15,  n),
    np.random.normal(160, 20,  n),
    np.random.normal(93,  3,   n),
    np.random.normal(38.5, 0.5, n),
])

X = np.vstack([X_normal, X_abnormal])
y = np.array([0]*n + [1]*n, dtype=float)

print("Dataset shape:", X.shape)
print("First normal patient :", X[0].round(2))
print("First abnormal patient:", X[500].round(2))
```

### Step 2: Normalize the data

Compute the mean and standard deviation of each column from the full dataset. Apply the formula `z = (x - mean) / std` to every value. This puts all 4 features on the same scale so the model treats them fairly during training.

> Important: in a real project, always compute mean and std from training data only, then apply those same values to the test set. Never let test data influence normalization.

```python
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_norm = (X - X_mean) / X_std

print("Normalized column means:", X_norm.mean(axis=0).round(4))
print("Normalized column stds: ", X_norm.std(axis=0).round(4))
```

### Step 3: Define the sigmoid function

Implement the sigmoid function `1 / (1 + e^(-z))` which squashes any number into a probability between 0 and 1. This is the function that converts the model's raw output into a prediction.

- `sigmoid(-6)` is close to 0 (very likely normal)
- `sigmoid(0)` is exactly 0.5 (uncertain)
- `sigmoid(+6)` is close to 1 (very likely abnormal)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Step 4: Forward pass

Initialize weights and bias to zero. Compute `z = X @ W + b` to combine each patient's features with the current weights. Pass `z` through sigmoid to produce a predicted probability for each patient.

With all weights at zero, every patient gets a prediction of exactly 0.5. The model knows nothing yet.

```python
W = np.zeros(4)
b = 0.0

z = X_norm @ W + b
y_hat = sigmoid(z)

print("First 5 predictions:", y_hat[:5].round(4))
```

### Step 5: Compute the loss

Implement binary cross-entropy loss to measure how wrong the current predictions are. A loss of 0.6931 means the model is no better than random. Lower is better, and 0 is perfect.

The `1e-9` added inside the log prevents numerical errors when predictions are exactly 0 or 1.

```python
def compute_loss(y_true, y_pred):
    return -np.mean(
        y_true * np.log(y_pred + 1e-9) +
        (1 - y_true) * np.log(1 - y_pred + 1e-9)
    )

loss = compute_loss(y, y_hat)
print("Initial loss:", round(loss, 4))
```

### Step 6: Compute gradients and update weights

Compute the prediction error per patient as `y_hat - y`. Use that error to compute a gradient for each weight and for the bias. Update every weight by subtracting `learning_rate x gradient`, nudging the model in the direction that reduces loss.

```python
def gradient_step(X, y, W, b, lr=0.1):
    n = len(y)
    z = X @ W + b
    y_hat = sigmoid(z)
    loss = compute_loss(y, y_hat)
    error = y_hat - y
    dW = X.T @ error / n
    db = np.mean(error)
    W_new = W - lr * dW
    b_new = b - lr * db
    return W_new, b_new, loss
```

### Step 7: Training loop and evaluation

Repeat steps 4 through 6 for 1000 epochs. Print loss and accuracy every 100 epochs to watch the model improve. After training, inspect the learned weights to confirm they are clinically sensible, plot the loss curve, and test the model on two new invented patients.

```python
W = np.zeros(4)
b = 0.0
lr = 0.1
epochs = 1000
loss_history = []

for epoch in range(epochs):
    z = X_norm @ W + b
    y_hat = sigmoid(z)
    loss = compute_loss(y, y_hat)
    loss_history.append(loss)
    error = y_hat - y
    dW = X_norm.T @ error / len(y)
    db = np.mean(error)
    W = W - lr * dW
    b = b - lr * db
    if epoch % 100 == 0:
        preds = (y_hat >= 0.5).astype(int)
        accuracy = np.mean(preds == y) * 100
        print(f"Epoch {epoch:4d} | loss: {loss:.4f} | accuracy: {accuracy:.1f}%")
```

---

## Expected outcomes

| Checkpoint | Expected value |
|---|---|
| Loss at epoch 0 | 0.6931 |
| Accuracy at epoch 0 | ~50% |
| Loss at epoch 1000 | below 0.15 |
| Accuracy at epoch 1000 | 95% or higher |
| SpO2 weight direction | negative |
| Heart rate, BP, temp weight direction | positive |

---

## Key concepts covered

| Concept | Where it appears |
|---|---|
| NumPy array operations | all steps |
| Random data generation with seed | step 1 |
| Feature normalization | step 2 |
| Sigmoid activation function | step 3 |
| Dot product and forward pass | step 4 |
| Binary cross-entropy loss | step 5 |
| Gradient computation | step 6 |
| Gradient descent weight update | step 6 |
| Training loop | step 7 |
| Model evaluation and accuracy | step 7 |

---

## What comes next

This is Stage 1 of a three-stage project.

**Stage 2** rebuilds the same model in sklearn in 5 lines, compares results, then introduces Random Forest and Gradient Boosting. The focus shifts to evaluation: confusion matrix, precision vs recall, ROC-AUC, and feature importance.

**Stage 3** reimplements the model as a 3-layer neural network in PyTorch, introducing `nn.Module`, autograd, and the Adam optimizer. The goal is to see how PyTorch automates exactly the gradient math you wrote by hand here.
