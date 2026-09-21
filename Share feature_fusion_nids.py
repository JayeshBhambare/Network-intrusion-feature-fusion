# Network Intrusion Detection using Feature Fusion
# Early Fusion vs Late Fusion vs Late Ensemble
# Colab-ready simplified implementation

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------
# 1. Create a small synthetic heterogeneous dataset
# ---------------------------------------------------------
N = 5000
CLASSES = 5

X_float = np.random.randn(N, 10).astype("float32")
X_int = np.random.randint(0, 100, (N, 5)).astype("float32")

# Categorical features: protocol, service, flag
protocol = np.random.randint(0, 3, (N, 1))
service = np.random.randint(0, 5, (N, 1))
flag = np.random.randint(0, 4, (N, 1))
X_cat = np.hstack([protocol, service, flag]).astype("float32")

# Create learnable synthetic labels instead of completely random labels.
score = (
    X_float[:, 0]
    + 0.02 * X_int[:, 0]
    + 0.4 * X_cat[:, 0]
    + 0.2 * X_cat[:, 1]
    - 0.2 * X_cat[:, 2]
)
y = np.digitize(score, [-1.5, -0.4, 0.4, 1.5])

Xf_train, Xf_tmp, Xi_train, Xi_tmp, Xc_train, Xc_tmp, y_train, y_tmp = train_test_split(
    X_float, X_int, X_cat, y, test_size=0.30, random_state=42, stratify=y
)
Xf_val, Xf_test, Xi_val, Xi_test, Xc_val, Xc_test, y_val, y_test = train_test_split(
    Xf_tmp, Xi_tmp, Xc_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

train = {"float": Xf_train, "int": Xi_train, "cat": Xc_train}
val = {"float": Xf_val, "int": Xi_val, "cat": Xc_val}
test = {"float": Xf_test, "int": Xi_test, "cat": Xc_test}

# ---------------------------------------------------------
# 2. Common preprocessing
# ---------------------------------------------------------
def preprocess():
    f = layers.Input((10,), name="float")
    i = layers.Input((5,), name="integer")
    c = layers.Input((3,), name="categorical")

    fn = layers.Normalization()
    inn = layers.Normalization()
    fn.adapt(Xf_train)
    inn.adapt(Xi_train)

    f = fn(f)
    i = inn(i)

    # Categorical integer IDs -> one-hot encoding
    c1 = layers.CategoryEncoding(num_tokens=3, output_mode="one_hot")(c[:, 0])
    c2 = layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")(c[:, 1])
    c3 = layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")(c[:, 2])
    c = layers.Concatenate()([c1, c2, c3])

    return f, i, c

# ---------------------------------------------------------
# 3. Early Fusion
# ---------------------------------------------------------
def early_model():
    f, i, c = preprocess()
    x = layers.Concatenate()([f, i, c])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    rep = layers.Dense(32, activation="relu", name="early_rep")(x)
    out = layers.Dense(CLASSES, activation="softmax")(rep)
    return Model(
        [layers.Input((10,), name="float"),
         layers.Input((5,), name="integer"),
         layers.Input((3,), name="categorical")],
        out
    )

# The preprocessing layers need to be connected to the same inputs.
def build_early():
    fi = layers.Input((10,), name="float")
    ii = layers.Input((5,), name="integer")
    ci = layers.Input((3,), name="categorical")

    fn, inn = layers.Normalization(), layers.Normalization()
    fn.adapt(Xf_train)
    inn.adapt(Xi_train)

    f = fn(fi)
    i = inn(ii)
    c = layers.Concatenate()([
        layers.CategoryEncoding(3, output_mode="one_hot")(ci[:, 0]),
        layers.CategoryEncoding(5, output_mode="one_hot")(ci[:, 1]),
        layers.CategoryEncoding(4, output_mode="one_hot")(ci[:, 2])
    ])

    x = layers.Concatenate()([f, i, c])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    rep = layers.Dense(32, activation="relu", name="early_rep")(x)
    out = layers.Dense(CLASSES, activation="softmax")(rep)
    return Model([fi, ii, ci], out)

# ---------------------------------------------------------
# 4. Late Fusion
# ---------------------------------------------------------
def build_late():
    fi = layers.Input((10,), name="float")
    ii = layers.Input((5,), name="integer")
    ci = layers.Input((3,), name="categorical")

    fn, inn = layers.Normalization(), layers.Normalization()
    fn.adapt(Xf_train)
    inn.adapt(Xi_train)

    f = layers.Dense(32, activation="relu")(fn(fi))
    f = layers.Dense(16, activation="relu")(f)

    i = layers.Dense(32, activation="relu")(inn(ii))
    i = layers.Dense(16, activation="relu")(i)

    c = layers.Concatenate()([
        layers.CategoryEncoding(3, output_mode="one_hot")(ci[:, 0]),
        layers.CategoryEncoding(5, output_mode="one_hot")(ci[:, 1]),
        layers.CategoryEncoding(4, output_mode="one_hot")(ci[:, 2])
    ])
    c = layers.Dense(32, activation="relu")(c)
    c = layers.Dense(16, activation="relu")(c)

    fused = layers.Average()([f, i, c])
    rep = layers.Dense(16, activation="relu", name="late_rep")(fused)
    out = layers.Dense(CLASSES, activation="softmax")(rep)
    return Model([fi, ii, ci], out)

# ---------------------------------------------------------
# 5. Prepare data and train
# ---------------------------------------------------------
def inputs(d):
    return [d["float"], d["int"], d["cat"]]

def train_model(model, name):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"\nTraining {name}...")
    history = model.fit(
        inputs(train), y_train,
        validation_data=(inputs(val), y_val),
        epochs=8,
        batch_size=64,
        verbose=1
    )
    return history

early = build_early()
late = build_late()

h_early = train_model(early, "Early Fusion")
h_late = train_model(late, "Late Fusion")

# ---------------------------------------------------------
# 6. Late Ensemble
# ---------------------------------------------------------
early_extractor = Model(
    early.inputs, early.get_layer("early_rep").output
)
late_extractor = Model(
    late.inputs, late.get_layer("late_rep").output
)

early_train = early_extractor.predict(inputs(train), verbose=0)
late_train = late_extractor.predict(inputs(train), verbose=0)
early_val = early_extractor.predict(inputs(val), verbose=0)
late_val = late_extractor.predict(inputs(val), verbose=0)
early_test = early_extractor.predict(inputs(test), verbose=0)
late_test = late_extractor.predict(inputs(test), verbose=0)

a = layers.Input((early_train.shape[1],))
b = layers.Input((late_train.shape[1],))
x = layers.Concatenate()([a, b])
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(CLASSES, activation="softmax")(x)
ensemble = Model([a, b], out)

ensemble.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Late Ensemble...")
ensemble.fit(
    [early_train, late_train], y_train,
    validation_data=([early_val, late_val], y_val),
    epochs=8,
    batch_size=64,
    verbose=1
)

# ---------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------
def evaluate(model, x, y):
    pred = np.argmax(model.predict(x, verbose=0), axis=1)
    return accuracy_score(y, pred), pred

acc_e, pred_e = evaluate(early, inputs(test), y_test)
acc_l, pred_l = evaluate(late, inputs(test), y_test)
acc_en, pred_en = evaluate(ensemble, [early_test, late_test], y_test)

results = pd.DataFrame({
    "Model": ["Early Fusion", "Late Fusion", "Late Ensemble"],
    "Accuracy": [acc_e, acc_l, acc_en]
})

print("\n========== RESULTS ==========")
print(results.to_string(index=False))

print("\n========== LATE ENSEMBLE CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test, pred_en,
    target_names=["Normal", "DoS", "Probe", "R2L", "U2R"],
    zero_division=0
))

print("\n========== ANALYSIS ==========")
best_name = results.loc[results["Accuracy"].idxmax(), "Model"]
best_acc = results["Accuracy"].max()

print(f"Highest test accuracy in this run: {best_name} ({best_acc:.4f})")
print("Early Fusion combines heterogeneous features at the input level.")
print("Late Fusion processes each feature type separately before combining representations.")
print("Late Ensemble combines learned representations from both fusion strategies.")
print("Note: This notebook uses synthetic data for demonstration, not a real intrusion dataset.")

# ---------------------------------------------------------
# 8. Graphs
# ---------------------------------------------------------
# Graph 1: Model accuracy comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(results["Model"], results["Accuracy"])
plt.title("Accuracy Comparison of Fusion Models")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, results["Accuracy"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()

# Graph 2: Training and validation accuracy
plt.figure(figsize=(9, 5))
plt.plot(h_early.history["accuracy"], label="Early - Train")
plt.plot(h_early.history["val_accuracy"], label="Early - Validation")
plt.plot(h_late.history["accuracy"], label="Late - Train")
plt.plot(h_late.history["val_accuracy"], label="Late - Validation")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Graph 3: Training and validation loss
plt.figure(figsize=(9, 5))
plt.plot(h_early.history["loss"], label="Early - Train")
plt.plot(h_early.history["val_loss"], label="Early - Validation")
plt.plot(h_late.history["loss"], label="Late - Train")
plt.plot(h_late.history["val_loss"], label="Late - Validation")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
