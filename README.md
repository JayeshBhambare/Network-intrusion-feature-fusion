# Network Intrusion Detection Using Feature Fusion

## Overview

This project is a simplified implementation inspired by:

**Ayantayo et al., Journal of Big Data (2023), 10:167**

The project demonstrates three deep-learning approaches for Network Intrusion Detection:

1. **Early Fusion**
2. **Late Fusion**
3. **Late Ensemble**

The goal is to show how heterogeneous network features can be processed and combined using neural networks.

---

## Project Architecture

### 1. Early Fusion

Different feature types are processed and combined near the beginning of the network.

**Flow:**

`Float Features + Integer Features + Categorical Features → Concatenation → Neural Network → Classification`

This allows the model to learn relationships between all feature types together.

### 2. Late Fusion

Each feature type is processed by its own neural-network branch.

**Flow:**

`Float → Branch`

`Integer → Branch`

`Categorical → Branch`

The learned representations are then averaged and passed to the classifier.

### 3. Late Ensemble

The learned representations from the Early Fusion and Late Fusion models are extracted and given to another neural network.

**Flow:**

`Early Representation + Late Representation → Ensemble Network → Classification`

---

## Dataset

This simplified version generates **5,000 synthetic samples**.

Three heterogeneous feature groups are used:

- 10 floating-point features
- 5 integer features
- 3 categorical features:
  - Protocol
  - Service
  - Flag

There are five classes:

- Normal
- DoS
- Probe
- R2L
- U2R

### Important

The dataset is **synthetic**. It is used only to demonstrate the feature-fusion architecture.

It is **not equivalent to NSL-KDD or another real intrusion-detection dataset**.

The synthetic labels are generated from a simple relationship between selected features so that the neural networks have a learnable signal.

---

## Requirements

Python 3.x

Install the required libraries:

```bash
pip install numpy tensorflow scikit-learn pandas
```

Google Colab already provides most of these libraries.

---

## How to Run in Google Colab

1. Open Google Colab.
2. Create a new notebook.
3. Copy the Python code into a cell.
4. Run the cell.
5. Wait for the three models to train.
6. Check the accuracy comparison and classification report.

---

## How to Run on GitHub

Create a repository with:

```text
network-intrusion-feature-fusion/
│
├── feature_fusion_nids.py
├── README.md
└── requirements.txt
```

### requirements.txt

```text
numpy
tensorflow
scikit-learn
pandas
```

Run:

```bash
pip install -r requirements.txt
python feature_fusion_nids.py
```

---

## Expected Output

The program prints training progress followed by a table similar to:

```text
========== RESULTS ==========
         Model  Accuracy
  Early Fusion    0.xxxx
   Late Fusion    0.xxxx
 Late Ensemble    0.xxxx
```

The exact accuracy can change slightly depending on the TensorFlow environment.

It also prints a classification report containing:

- Precision
- Recall
- F1-score
- Support

for the five intrusion classes.

---

## Output Analysis

### Accuracy Comparison

The final table allows the three architectures to be compared on the same test set.

### Early Fusion

Early Fusion combines the different feature types before most of the deep learning layers.

**Advantage:** The model can directly learn interactions between heterogeneous features.

**Limitation:** The raw feature groups are mixed relatively early.

### Late Fusion

Late Fusion gives each feature group a dedicated processing branch.

**Advantage:** Each feature type can learn its own representation before fusion.

**Limitation:** The branches are combined only after separate processing, so some low-level cross-feature relationships may not be learned early.

### Late Ensemble

Late Ensemble combines representations produced by the Early Fusion and Late Fusion models.

**Advantage:** It attempts to use information learned by both architectures.

**Limitation:** It requires training additional models and therefore increases computational cost.

---

## How to Interpret the Results

Do **not** assume that one architecture will always have the highest accuracy.

The result depends on:

- Dataset
- Feature distribution
- Feature quality
- Class balance
- Network architecture
- Hyperparameters
- Random initialization

For this demonstration, the most useful result is the **comparison of the three fusion strategies**, rather than treating the synthetic-data accuracy as a real-world intrusion-detection benchmark.

---

## Relation to the Reference Paper

The implementation follows the general concept of feature fusion:

- heterogeneous feature processing
- early feature fusion
- late feature fusion
- ensemble of learned representations
- deep-learning-based classification

The neural-network sizes and dataset generation have been simplified so the project is easier to understand, execute, and explain in a student project.

This is **not an exact reproduction of every experiment or hyperparameter in the reference paper**.

---

## Possible Improvements

For a stronger final-year project, replace the synthetic dataset with a real intrusion-detection dataset such as:

- NSL-KDD
- CICIDS2017
- UNSW-NB15

Then perform:

1. Data cleaning
2. Missing-value handling
3. Categorical encoding
4. Feature normalization
5. Class balancing
6. Model training
7. Accuracy comparison
8. Precision/Recall/F1 comparison
9. Confusion matrices
10. Training/validation curves

---

## Conclusion

This project demonstrates how heterogeneous network features can be combined using different deep-learning fusion strategies.

The three approaches are:

**Early Fusion → Combine features early**

**Late Fusion → Learn separate representations and combine them later**

**Late Ensemble → Combine representations from multiple fusion strategies**

The simplified implementation is intended for learning, demonstration, and as a starting point for implementation with a real intrusion-detection dataset.
