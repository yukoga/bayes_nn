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

# Import the created bayes_nn module
# (Assuming execution from the project root)
import sys

# Add project root to the path (adjust according to your environment)
# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# )

try:
    # If installed via pip install .
    from bayes_nn.models import BayesianRegressor
    from bayes_nn.utils import plot_observed_vs_predicted
except ImportError:
    print("Please install the bayes_nn package first (e.g., 'pip install .')")
    print("Or adjust the Python path to include the project root.")
    sys.exit(1)


def generate_normal_data(
    n_samples: int = 200,
    true_a: float = 2.0,
    true_b: float = 1.0,
    true_sigma: float = 1.5,
    random_state: int = 42,
):
    """Generate data: y = ax + b + N(0, sigma^2)"""
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 1) * 10 - 5  # Input X in the range [-5, 5]
    noise = np.random.normal(0, true_sigma, size=(n_samples, 1))
    y = true_a * X + true_b + noise
    print(f"Generated data: y = {true_a}*x + {true_b} + N(0, {true_sigma**2:.2f})")
    return X, y.flatten()


def plot_predictions_gaussian(
    X,
    y_true,
    model,
    n_samples_plot=100,
    title="Bayesian Regression (Gaussian)",
    figsize=(10, 6),
):
    """Plot the results of Gaussian regression."""
    plt.figure(figsize=figsize)

    # Plot training data
    plt.scatter(X, y_true, alpha=0.5, label="Training Data")

    # X-axis data points for prediction
    X_pred_points = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    # Obtain predictive distribution by sampling multiple times
    y_samples = model.predict_proba(X_pred_points, n_samples=n_samples_plot)
    # y_samples shape: (n_samples_plot, num_pred_points, 1)
    y_samples = y_samples.squeeze(-1)  # (n_samples_plot, num_pred_points)

    # Calculate mean and standard deviation of predictions
    # y_mean = y_samples.mean(axis=0) # Unused variable removed
    # y_std = y_samples.std( # Unused variable removed
    #     axis=0
    # )  # This is the std dev of E[y] (close to epistemic uncertainty)

    # Mean and std dev calculated by predict method (includes aleatoric)
    y_pred_mean_total, y_pred_std_total = model.predict(X_pred_points, return_std=True)

    # Plot the predictive mean line
    plt.plot(
        X_pred_points.flatten(),
        y_pred_mean_total,
        "r-",
        label="Predictive Mean",
    )

    # Plot the predictive confidence interval (e.g., mean ± 2*std dev)
    # (Epistemic + Aleatoric uncertainty)
    plt.fill_between(
        X_pred_points.flatten(),
        y_pred_mean_total - 2 * y_pred_std_total,
        y_pred_mean_total + 2 * y_pred_std_total,
        color="r",
        alpha=0.2,
        label="Predictive Uncertainty (Mean ± 2*Std)",
    )

    # (Reference) Case with only Epistemic uncertainty (std dev of y_samples)
    # plt.fill_between(X_pred_points.flatten(),
    #                  y_mean - 2 * y_std,
    #                  y_mean + 2 * y_std,
    #                  color='orange', alpha=0.3,
    #                  label='Epistemic Uncertainty (Mean ± 2*Std[E[y]])')

    plt.xlabel("Input Feature (X)")
    plt.ylabel("Target Value (y)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 1. Data Generation ---
    X_train, y_train = generate_normal_data(n_samples=200, random_state=42)

    # --- 2. Model Initialization ---
    # Auto-select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bnn_regressor = BayesianRegressor(
        input_dim=X_train.shape[1],
        # Break output_type comment
        output_type="gaussian",  # Output is Gaussian (mean, variance)
        hidden_dims=[32, 32],  # Hidden layer configuration
        activation=torch.nn.ReLU(),  # Activation function
        n_epochs=200,  # Number of training epochs
        batch_size=32,  # Batch size
        lr=0.005,  # Learning rate
        # Break kl_weight comment
        kl_weight=0.1,  # KL term weight (adjust based on data size)
        n_samples_predict=100,  # Number of samples for prediction
        optimizer_cls=torch.optim.AdamW,  # Optimizer
        validation_split=0.2,  # Proportion of validation data
        early_stopping_patience=15,  # Early Stopping patience
        device=device,  # Device to use
        random_state=42,  # Random seed
    )

    # --- 2.5 Plot Network Architecture ---
    print("\nPlotting network architecture...")
    bnn_regressor.plot_network_architecture(
        filename="normal_bnn_architecture", view=False
    )

    # --- 3. Model Training ---
    print("\nStarting training...")
    bnn_regressor.fit(X_train, y_train, verbose=True)
    print("Training finished.")

    # --- 4. Plotting Training Results ---
    # Plot loss history
    # Break plot_loss_history call
    bnn_regressor.plot_loss_history(title="Gaussian BNN: Training and Validation Loss")

    # Plot prediction results
    # Break plot_predictions_gaussian call
    plot_predictions_gaussian(
        X_train,
        y_train,
        bnn_regressor,
        title="Bayesian Regression (Gaussian Output)",
    )

    # Plot observed vs predicted for training data
    y_pred_mean_train, y_pred_std_train = bnn_regressor.predict(
        X_train, return_std=True
    )
    plot_observed_vs_predicted(
        y_train,
        y_pred_mean_train,
        y_pred_std_train,
        title="Gaussian BNN: Observed vs. Predicted (Training Data)",
    )

    # --- 5. Prediction on New Data (Example) ---
    X_new = np.array([[-4.0], [0.0], [4.0]])
    y_pred_mean, y_pred_std = bnn_regressor.predict(X_new, return_std=True)

    print("\nPredictions for new data points:")
    for i in range(X_new.shape[0]):
        # Break print line
        print(
            f"Input: {X_new[i, 0]:.2f}, "
            f"Predicted Mean: {y_pred_mean[i]:.2f}, "
            f"Predicted Std: {y_pred_std[i]:.2f}"
        )

    # Get samples from the predictive distribution
    y_samples_new = bnn_regressor.predict_proba(X_new, n_samples=5)
    print("\nSamples from predictive distribution for new data:")
    print(y_samples_new)  # shape: (5, 3, 1)
