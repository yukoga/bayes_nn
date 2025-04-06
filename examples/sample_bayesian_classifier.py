# -*- coding: utf-8 -*-
# Copyright (c) 2025 yukoga. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from bayes_nn import BayesianClassifier
except ImportError:
    print("Please install the bayes_nn package first (e.g., 'pip install .')")
    print("Or adjust the Python path to include the project root.")
    sys.exit(1)


def generate_classification_data(
    n_samples: int = 300,
    n_features: int = 2,
    n_classes: int = 2,
    n_clusters_per_class: int = 1,
    class_sep: float = 1.0,
    random_state: int = 42,
):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state,
    )
    print(
        f"Generated data: {n_samples} samples, {n_features} features, "
        f"{n_classes} classes."
    )
    return X, y


def plot_decision_boundary(
    X,
    y_true,
    model,
    title="Bayesian Classification Decision Boundary",
    figsize=(10, 6),
    cmap="viridis",
    step=0.02,
):
    """Plot the decision boundary of a classifier."""
    if X.shape[1] != 2:
        print("Decision boundary plot only available for 2D feature space.")
        return

    plt.figure(figsize=figsize)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_proba = model.predict_proba(grid_points)
    Z_pred = np.argmax(Z_proba, axis=1)
    Z_pred = Z_pred.reshape(xx.shape)

    plt.contourf(xx, yy, Z_pred, cmap=cmap, alpha=0.6)
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y_true, cmap=cmap, edgecolors="k", marker="o"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    if len(np.unique(y_true)) > 1:
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=[f"Class {i}" for i in np.unique(y_true)],
            title="Classes",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 1. Data Generation ---
    N_CLASSES = 2
    X, y = generate_classification_data(
        n_samples=400, n_classes=N_CLASSES, class_sep=1.5, random_state=42
    )
    # Split data for potential evaluation later if needed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 2. Model Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bnn_classifier = BayesianClassifier(
        input_dim=X_train.shape[1],
        num_classes=N_CLASSES,
        hidden_dims=[32, 16],
        activation=torch.nn.Tanh(),
        n_epochs=150,
        batch_size=32,
        lr=0.01,
        kl_weight=0.1,
        n_samples_predict=100,
        optimizer_cls=torch.optim.AdamW,
        validation_split=0.15,
        early_stopping_patience=20,
        device=device,
        random_state=42,
    )

    # --- 2.5 Plot Network Architecture (Optional) ---
    # print("\nPlotting network architecture...")
    # bnn_classifier.plot_network_architecture(
    #     filename="classifier_bnn_architecture", view=False
    # )

    # --- 3. Model Training ---
    print("\nStarting training...")
    bnn_classifier.fit(X_train, y_train, verbose=True)
    print("Training finished.")

    # --- 4. Plotting Training Results ---
    # Plot loss history
    bnn_classifier.plot_loss_history(
        title="Classifier BNN: Training and Validation Loss"
    )

    # Plot decision boundary using training data
    plot_decision_boundary(
        X_train,
        y_train,
        bnn_classifier,
        title="Bayesian Classifier Decision Boundary (Training Data)",
    )

    # --- 5. Evaluation ---
    print("\nEvaluating on Training Data:")
    y_pred_train = bnn_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print("Training Classification Report:")
    print(classification_report(y_train, y_pred_train))

    # Evaluate on Test Data
    print("\nEvaluating on Test Data:")
    y_pred_test = bnn_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred_test))

    # --- 6. Prediction on New Data (Example) ---
    X_new = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0], [-1.0, 1.0]])
    pred_probs = bnn_classifier.predict_proba(X_new)
    pred_labels = bnn_classifier.predict(X_new)

    print("\nPredictions for new data points:")
    for i in range(X_new.shape[0]):
        print(
            f"Input: {X_new[i]}, "
            f"Predicted Probabilities: {pred_probs[i]}, "
            f"Predicted Label: {pred_labels[i]}"
        )
