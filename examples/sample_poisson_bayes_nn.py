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
import sys

# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# )

try:
    from bayes_nn.models import BayesianRegressor
    from bayes_nn.utils import plot_observed_vs_predicted
except ImportError:
    print("Please install the bayes_nn package first (e.g., 'pip install .')")
    print("Or adjust the Python path to include the project root.")
    sys.exit(1)


def generate_poisson_data(
    n_samples: int = 200,
    true_a: float = 1.5,
    true_b: float = 0.5,
    random_state: int = 123,
):
    """Generate data following a Poisson distribution log(lambda) = ax + b"""
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 1) * 4 - 2  # Input X in the range [-2, 2]
    log_lambda = true_a * X + true_b
    lambda_ = np.exp(log_lambda)
    y = np.random.poisson(lambda_, size=(n_samples, 1))
    print(f"Generated data: y ~ Poisson(lambda), log(lambda) = {true_a}*x + {true_b}")
    return X, y.flatten()


def plot_predictions_poisson(
    X,
    y_true,
    model,
    n_samples_plot=100,
    title="Bayesian Regression (Poisson)",
    figsize=(10, 6),
):
    """Plot the results of Poisson regression."""
    plt.figure(figsize=figsize)

    # Plot training data
    plt.scatter(X, y_true, alpha=0.5, label="Training Data (Counts)")

    # X-axis data points for prediction
    X_pred_points = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    # Obtain predictive distribution by sampling multiple times
    y_samples = model.predict_proba(X_pred_points, n_samples=n_samples_plot)
    # y_samples shape: (n_samples_plot, num_pred_points, 1)
    y_samples = y_samples.squeeze(-1)  # (n_samples_plot, num_pred_points)

    # Mean and std dev of predictions (mean/std dev of predicted counts)
    y_mean = y_samples.mean(axis=0)
    # Std represents uncertainty in the expected count
    y_std = y_samples.std(axis=0)

    # (Optional) Mean rate calculated by predict method (stats of lambda)
    # lambda_pred_mean, _ = model.predict(X_pred_points, return_std=True)

    # Plot the predictive mean count E[y]
    plt.plot(
        X_pred_points.flatten(),
        y_mean,
        "r-",
        label="Predictive Mean Count E[y]",
    )

    # Plot confidence interval for predictive count (e.g., mean ± 2*std dev)
    # This interval reflects the uncertainty about the expected count value
    # due to model parameter uncertainty (epistemic).
    plt.fill_between(
        X_pred_points.flatten(),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="r",
        alpha=0.2,
        label="Predictive Count Uncertainty (E[y] ± 2*Std[E[y]])",
    )

    # (Optional Plotting of Lambda for reference)
    # plt.plot(X_pred_points.flatten(), lambda_pred_mean, 'g--',
    #          label='Mean Predicted Rate (λ)')
    # lambda_pred_mean, lambda_pred_std = model.predict(
    #     X_pred_points, return_std=True
    # )
    # plt.fill_between(X_pred_points.flatten(),
    #                  lambda_pred_mean - 2 * lambda_pred_std,
    #                  lambda_pred_mean + 2 * lambda_pred_std,
    #                  color='g', alpha=0.1,
    #                  label='Predictive Rate Uncertainty (λ ± 2*Std[λ])')

    plt.xlabel("Input Feature (X)")
    plt.ylabel("Target Value (y - Counts)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)  # Counts are non-negative
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 1. Data Generation ---
    X_train, y_train = generate_poisson_data(n_samples=200, random_state=123)

    # --- 2. Model Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bnn_regressor = BayesianRegressor(
        input_dim=X_train.shape[1],
        output_type="poisson",  # Output is Poisson distribution (log rate)
        hidden_dims=[32, 32],
        activation=torch.nn.Tanh(),  # Trying Tanh activation
        n_epochs=300,  # Poisson might take a bit longer to train
        batch_size=32,
        lr=0.003,
        kl_weight=0.1,
        n_samples_predict=100,
        optimizer_cls=torch.optim.AdamW,
        validation_split=0.2,
        early_stopping_patience=20,
        device=device,
        random_state=123,
    )

    # --- 3. Model Training ---
    print("\nStarting training...")
    # For Poisson regression, target y might need to be float
    # (should be handled in loss function)
    bnn_regressor.fit(X_train, y_train.astype(np.float32), verbose=True)
    print("Training finished.")

    # --- 4. Plotting Training Results ---
    # Break plot_loss_history call
    bnn_regressor.plot_loss_history(title="Poisson BNN: Training and Validation Loss")
    # Break plot_predictions_poisson call
    plot_predictions_poisson(
        X_train,
        y_train,
        bnn_regressor,
        title="Bayesian Regression (Poisson Output)",
    )

    # Plot observed counts vs predicted counts for training data
    # Get samples of predicted counts for training data
    y_samples_train = bnn_regressor.predict_proba(
        X_train, n_samples=bnn_regressor.n_samples_predict
    )
    # y_samples_train shape: (n_samples, n_train_points)
    y_samples_train = y_samples_train.squeeze(-1)

    # Calculate mean and std dev of predicted counts
    y_pred_mean_train = y_samples_train.mean(axis=0)
    y_pred_std_train = y_samples_train.std(axis=0)

    plot_observed_vs_predicted(
        y_train,  # Observed counts
        y_pred_mean_train,  # Predicted mean count E[y]
        y_pred_std_train,  # Std dev of predicted mean count Std[E[y]]
        title="Poisson BNN: Observed vs. Predicted Counts (Train)",
        xlabel="Observed Counts",
        ylabel="Predicted Mean Count E[y]",
    )

    # --- 5. Prediction on New Data (Example) ---
    X_new = np.array([[-1.5], [0.0], [1.5]])
    # Get predicted rate (lambda) stats for new data
    lambda_pred_mean, lambda_pred_std = bnn_regressor.predict(X_new, return_std=True)

    print("\nPredictions for new data points (Rate λ):")
    for i in range(X_new.shape[0]):
        print(
            f"Input: {X_new[i, 0]:.2f}, "
            f"Predicted Mean Rate (λ): {lambda_pred_mean[i]:.2f}, "
            f"Predicted Rate Std (λ): {lambda_pred_std[i]:.2f}"
        )

    # Get samples from the predictive distribution (counts) for new data
    y_samples_new = bnn_regressor.predict_proba(
        X_new, n_samples=bnn_regressor.n_samples_predict
    )
    print("\nSamples from predictive distribution (counts) for new data:")
    print(y_samples_new)  # shape: (5, 3, 1)
