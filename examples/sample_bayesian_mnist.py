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
import torchvision
import torchvision.transforms as transforms

import sys
from sklearn.metrics import accuracy_score, classification_report

# Import the created bayes_nn module
try:
    # If installed via pip install .
    from bayes_nn.models import BayesianClassifier
except ImportError:
    print("Please install the bayes_nn package first (e.g., 'pip install .')")
    print("Or adjust the Python path to include the project root.")
    sys.exit(1)


def load_mnist_data(batch_size: int = 128, data_root: str = "./data"):
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST data...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    trainset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    X_train = trainset.data.view(len(trainset), -1).float().numpy() / 255.0
    y_train = trainset.targets.numpy()
    X_test = testset.data.view(len(testset), -1).float().numpy() / 255.0
    y_test = testset.targets.numpy()

    print(f"MNIST data loaded: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, y_train, X_test, y_test


def plot_mnist_predictions(
    X_test,
    y_test,
    model,
    n_samples_plot: int = 10,
    n_mc_samples: int = 100,
    figsize=(12, 5),
):
    """Plot sample MNIST images with predictions and probability distributions."""
    indices = np.random.choice(len(X_test), n_samples_plot, replace=False)
    sample_images = X_test[indices]
    sample_labels = y_test[indices]

    pred_probs = model.predict_proba(sample_images, n_samples=n_mc_samples)
    pred_labels = np.argmax(pred_probs, axis=1)

    fig, axes = plt.subplots(
        2, n_samples_plot // 2, figsize=figsize, constrained_layout=True
    )
    axes = axes.ravel()

    for i in range(n_samples_plot):
        ax = axes[i]
        img = sample_images[i].reshape(28, 28)
        true_label = sample_labels[i]
        pred_label = pred_labels[i]

        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Sample MNIST Predictions (n_mc_samples={n_mc_samples})", fontsize=16)
    plt.show()


if __name__ == "__main__":
    # --- 1. Data Loading ---
    # Note: Using numpy arrays directly for simplicity, matching other
    # examples.
    X_train, y_train, X_test, y_test = load_mnist_data(batch_size=256)

    # --- 2. Model Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    INPUT_DIM = X_train.shape[1]  # 784
    NUM_CLASSES = 10

    bnn_classifier = BayesianClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dims=[128, 64],
        activation=torch.nn.ReLU(),
        n_epochs=30,
        batch_size=256,
        lr=0.005,
        kl_weight=1.0 / len(X_train),
        n_samples_predict=100,
        optimizer_cls=torch.optim.AdamW,
        validation_split=0.1,
        early_stopping_patience=5,
        device=device,
        random_state=42,
    )

    # --- 2.5 Plot Network Architecture (Optional) ---
    # print("\nPlotting network architecture...")
    # bnn_classifier.plot_network_architecture(
    #     filename="mnist_bnn_architecture", view=False
    # )

    # --- 3. Model Training ---
    print("\nStarting training...")
    # Pass numpy arrays directly to fit
    bnn_classifier.fit(X_train, y_train, verbose=True)
    print("Training finished.")

    # --- 4. Plotting Training Results ---
    # Plot loss history
    bnn_classifier.plot_loss_history(title="MNIST BNN: Training and Validation Loss")

    # --- 5. Evaluation ---
    print("\nEvaluating on Test Data:")
    y_pred_test = bnn_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Classification Report:")
    # Use zero_division=0 to handle potential cases with no predicted samples
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # --- 6. Visualize Predictions ---
    print("\nPlotting sample predictions from test set...")
    plot_mnist_predictions(X_test, y_test, bnn_classifier, n_samples_plot=10)

    # --- 7. Predict Proba Example ---
    print("\nPredicting probabilities for the first 5 test images:")
    sample_indices = np.arange(5)
    X_sample = X_test[sample_indices]
    pred_probs_sample = bnn_classifier.predict_proba(X_sample)

    for i in range(len(X_sample)):
        print(f"Image {i} (True Label: {y_test[i]}):")
        print(f"  Predicted Probabilities: {pred_probs_sample[i].round(3)}")
        print(f"  Predicted Label: {np.argmax(pred_probs_sample[i])}")
