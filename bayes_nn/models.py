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

import torch
import torch.nn as nn
import torch.optim as optim

# Import Generator from torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torch import Tensor, Generator  # Correct import location
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Type, Union

# import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .base import BaseEstimator
from .layers import BayesianLinear
from .losses import ELBO, GaussianNLLLoss, PoissonNLLLoss
from .utils import plot_loss_history


class BayesianNetwork(nn.Module):
    """
    Basic BNN structure combining multiple Bayesian linear layers and
    activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [64, 32],
        activation: nn.Module = nn.ReLU(),
    ):
        """
        Args:
            input_dim (int): Number of input dimensions.
            output_dim (int): Number of output dimensions.
            hidden_dims (list): List of hidden layer dimensions.
                                Example: [64, 32]
            activation (nn.Module): Activation function used in hidden layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        last_dim = input_dim

        # Build hidden layers
        for h_dim in hidden_dims:
            self.layers.append(BayesianLinear(last_dim, h_dim))
            self.layers.append(activation)
            last_dim = h_dim

        # Build output layer (no activation function)
        self.layers.append(BayesianLinear(last_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for layer in self.layers:
            x = layer(x)
        return x


class BayesianRegressor(BaseEstimator):
    """
    Regression model using a Bayesian Neural Network.
    Provides a scikit-learn-like interface.
    The output represents the distribution of the predicted value
    (e.g., mean and variance for Gaussian, rate for Poisson).
    """

    def __init__(
        self,
        input_dim: int,
        output_type: str = "gaussian",
        hidden_dims: list = [64, 32],
        activation: nn.Module = nn.ReLU(),
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
        n_samples_predict: int = 100,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        validation_split: float = 0.1,
        early_stopping_patience: Optional[int] = 10,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        """
        Args:
            input_dim (int): Dimensionality of input features.
            output_type (str): Specifies the type of output. 'gaussian' or
                               'poisson'.
                               'gaussian': Output is 2D (mean, log variance).
                               'poisson': Output is 1D (log rate).
            hidden_dims (list): List of hidden layer dimensions.
            activation (nn.Module): Activation function for hidden layers.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            lr (float): Learning rate.
            kl_weight (float): Weight of the KL term in the ELBO loss.
            n_samples_predict (int): Number of samples for ensembling during
                                     prediction.
            optimizer_cls (Type[optim.Optimizer]): Optimizer class to use
                                                   (e.g., optim.Adam).
            validation_split (float): Proportion of training data to use for
                                      validation. 0 means no split.
            early_stopping_patience (Optional[int]): Number of epochs to wait
                                     for validation loss improvement before
                                     stopping. None disables early stopping.
            device (str): Device to use ('auto', 'cpu', 'cuda'). 'auto'
                          selects GPU if available.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        super().__init__()

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.input_dim = input_dim
        self.output_type = output_type
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kl_weight = kl_weight
        self.n_samples_predict = n_samples_predict
        self.optimizer_cls = optimizer_cls
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state  # Used for DataLoader shuffle, etc.

        # Device setting
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Determine output dimension and NLL loss function
        self.nll_loss_fn: nn.Module  # Type hint for mypy
        if self.output_type == "gaussian":
            self.output_dim = 2  # Mean and log variance
            self.nll_loss_fn = GaussianNLLLoss()
        elif self.output_type == "poisson":
            self.output_dim = 1  # Log rate
            self.nll_loss_fn = PoissonNLLLoss()
        else:
            raise ValueError("output_type must be 'gaussian' or 'poisson'")

        # Build BNN model
        self.model = BayesianNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
        ).to(self.device)

        # Optimizer initialization (lr is standard for Adam etc.)
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.elbo_loss_fn: Optional[ELBO] = None  # Initialize later

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        verbose: bool = True,
        **kwargs: Any,  # To match BaseEstimator signature
    ) -> "BayesianRegressor":
        """
        Fits the model to the data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data
                                                 (n_samples, n_features).
            y (Union[np.ndarray, pd.Series]): Target data (n_samples,).
            verbose (bool): Whether to display training progress.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        # Convert data to PyTorch Tensors
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Ensure y is 1D or 2D with last dim 1
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if (
            self.output_type == "poisson"
        ):  # Target for Poisson regression is integer, convert for loss calc
            y_tensor = y_tensor.float()

        dataset = TensorDataset(X_tensor, y_tensor)
        dataset_size = len(dataset)

        train_dataset: Dataset[Tuple[Tensor, ...]]
        val_dataset: Dataset[Tuple[Tensor, ...]]
        train_loader: DataLoader[Tuple[Tensor, ...]]
        val_loader: Optional[DataLoader[Tuple[Tensor, ...]]]

        # Split into training and validation data
        if (
            self.validation_split > 0 and dataset_size > 1
        ):  # Need at least 2 samples to split
            val_size = int(dataset_size * self.validation_split)
            if (
                val_size == 0 and dataset_size > 1
            ):  # Ensure val_size is at least 1 if possible
                val_size = 1
            train_size = dataset_size - val_size
            if train_size == 0:  # Ensure train_size is at least 1
                train_size = 1
                val_size = dataset_size - 1

            # reproducible split
            generator: Optional[Generator] = (
                torch.Generator().manual_seed(self.random_state)
                if self.random_state is not None
                else None
            )
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            if verbose:
                print(
                    f"Training on {train_size} samples, validating on "
                    f"{val_size} samples."
                )
        else:
            # Use full dataset for training if split is 0 or dataset too small
            train_dataset = dataset
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.random_state)
                if self.random_state is not None
                else None,
            )
            val_loader = None
            if verbose:
                print(f"Training on {dataset_size} samples (no validation split).")

        # Initialize ELBO loss function (requires dataset_size)
        # Ensure elbo_loss_fn is not None before use
        self.elbo_loss_fn = ELBO(
            self.model, self.nll_loss_fn, dataset_size, self.kl_weight
        ).to(self.device)

        # --- Training loop ---
        progress_bar = tqdm(range(self.n_epochs), desc="Epochs", disable=not verbose)
        for epoch in progress_bar:
            self.model.train()  # Training mode
            train_loss_epoch = 0.0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_count += 1

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(batch_X)

                # Calculate loss (ELBO)
                # Add assertion for mypy
                assert self.elbo_loss_fn is not None
                loss = self.elbo_loss_fn(y_pred, batch_y)

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                train_loss_epoch += loss.item()  # Sum losses

            # Calculate average loss for the epoch, cast divisor to float
            avg_train_loss = (
                train_loss_epoch / float(batch_count) if batch_count > 0 else 0.0
            )
            self.history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            avg_val_loss = float("nan")  # Default if no validation
            if val_loader:
                self.model.eval()  # Evaluation mode
                val_loss_epoch = 0.0
                val_batch_count = 0
                with torch.no_grad():  # Disable gradient calculation
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val = batch_X_val.to(self.device)
                        batch_y_val = batch_y_val.to(self.device)
                        val_batch_count += 1

                        y_pred_val = self.model(batch_X_val)
                        # Add assertion for mypy
                        assert self.elbo_loss_fn is not None
                        val_loss = self.elbo_loss_fn(y_pred_val, batch_y_val)
                        val_loss_epoch += val_loss.item()  # Sum losses

                # Calculate average loss, cast divisor to float
                avg_val_loss = (
                    val_loss_epoch / float(val_batch_count)
                    if val_batch_count > 0
                    else 0.0
                )
                self.history["val_loss"].append(avg_val_loss)

                if verbose:
                    progress_bar.set_postfix(
                        {
                            "Train Loss": f"{avg_train_loss:.4f}",
                            "Val Loss": f"{avg_val_loss:.4f}",
                        }
                    )

                # --- Early Stopping ---
                if self.early_stopping_patience is not None:
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.epochs_no_improve = 0
                        # Save the best model state
                        self.best_model_state = self.model.state_dict()
                    else:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve >= self.early_stopping_patience:
                            if verbose:
                                print(
                                    f"\nEarly stopping triggered after "
                                    f"{epoch + 1} epochs."
                                )
                            # Restore the best model state
                            if self.best_model_state:
                                self.model.load_state_dict(self.best_model_state)
                            break  # Exit training loop
            else:
                # No validation set
                if verbose:
                    progress_bar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}"})
                # Cannot use early stopping, consider last model as best
                self.best_model_state = self.model.state_dict()

        # After training, load the best model state (if early stopping was used)
        if (
            self.early_stopping_patience is not None
            and val_loader
            and self.best_model_state
        ):
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(
                    f"Loaded best model state with validation loss: "
                    f"{self.best_val_loss:.4f}"
                )

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_std: bool = False,
        n_samples: Optional[int] = None,
        **kwargs: Any,  # To match BaseEstimator signature
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Makes predictions for new data points.
        Performs multiple forward passes (sampling) and aggregates the results.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data for prediction.
            return_std (bool): Whether to return the standard deviation of
                               predictions.
            n_samples (Optional[int]): Number of samples to use for prediction.
                                       Uses `self.n_samples_predict` if None.
            **kwargs: Additional arguments (ignored, for compatibility).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                If return_std=False: Mean of predictions (shape: (n_samples,)).
                If return_std=True: Tuple of (mean predictions, standard
                                    deviation of predictions).
                                    For Gaussian regression, mean is E[mu],
                                    std dev considers Var[mu] and E[sigma^2].
                                    For Poisson regression, mean is E[lambda],
                                    std dev is Std[lambda].
        """
        self.model.eval()  # Evaluation mode
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        n_samples_eff = n_samples if n_samples is not None else self.n_samples_predict

        predictions_list: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples_eff):
                # Weights are sampled each time the model is passed through
                output = self.model(X_tensor)
                predictions_list.append(output.cpu().numpy())

        # predictions_list contains arrays of shape (num_data, output_dim)
        predictions_np = np.array(
            predictions_list
        )  # shape: (n_samples_eff, num_data, output_dim)

        mean_pred: np.ndarray
        std_pred: np.ndarray

        if self.output_type == "gaussian":
            # output_dim = 2 (mu, log_var)
            mus = predictions_np[:, :, 0]  # shape: (n_samples_eff, num_data)
            log_vars = predictions_np[:, :, 1]
            sigmas = np.sqrt(np.exp(log_vars))

            # Mean prediction = E[mu]
            mean_pred = mus.mean(axis=0)  # shape: (num_data,)

            if return_std:
                # Standard deviation of prediction: considers uncertainty
                # Var[y] = E[Var[y|w]] + Var[E[y|w]]
                #        = E[sigma^2] + Var[mu]
                epistemic_uncertainty = mus.var(
                    axis=0
                )  # Model (parameter) uncertainty Var[mu]
                aleatoric_uncertainty = (sigmas**2).mean(
                    axis=0
                )  # Data-inherent noise E[sigma^2]
                total_variance = epistemic_uncertainty + aleatoric_uncertainty
                std_pred = np.sqrt(total_variance)  # shape: (num_data,)
                return mean_pred, std_pred
            else:
                return mean_pred

        elif self.output_type == "poisson":
            # output_dim = 1 (log_lambda)
            log_lambdas = predictions_np[:, :, 0]
            lambdas = np.exp(log_lambdas)  # shape: (n_samples_eff, num_data)

            # Mean prediction = E[lambda]
            mean_pred = lambdas.mean(axis=0)  # shape: (num_data,)

            if return_std:
                # Standard deviation of prediction = Std[lambda]
                std_pred = lambdas.std(axis=0)  # shape: (num_data,)
                return mean_pred, std_pred
            else:
                return mean_pred
        else:
            # Should not happen due to __init__ check, but satisfy mypy
            raise RuntimeError("Invalid output_type encountered in predict")

    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame], n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Returns samples from the predictive distribution.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data for prediction.
            n_samples (Optional[int]): Number of samples. Uses
                                       `self.n_samples_predict` if None.

        Returns:
            np.ndarray: Set of samples from the predictive distribution.
                        Shape: (n_samples_eff, num_data, 1)
                        'gaussian': samples of predicted y
                        'poisson': samples of predicted y
        """
        self.model.eval()  # Evaluation mode
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        n_samples_eff = n_samples if n_samples is not None else self.n_samples_predict

        sampled_outputs_list: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples_eff):
                output = (
                    self.model(X_tensor).cpu().numpy()
                )  # shape: (num_data, output_dim)

                y_sample: np.ndarray
                if self.output_type == "gaussian":
                    mus = output[:, 0]
                    log_vars = output[:, 1]
                    sigmas = np.sqrt(np.exp(log_vars))
                    # Sample from N(mu, sigma^2) for each data point
                    y_sample = np.random.normal(mus, sigmas)
                    sampled_outputs_list.append(
                        y_sample[:, np.newaxis]
                    )  # Make it (num_data, 1)

                elif self.output_type == "poisson":
                    log_lambdas = output[:, 0]
                    lambdas = np.exp(log_lambdas)
                    # Sample from Poisson(lambda) for each data point
                    y_sample = np.random.poisson(lambdas)  # shape: (num_data,)
                    sampled_outputs_list.append(
                        y_sample[:, np.newaxis]
                    )  # Make it (num_data, 1)

        # sampled_outputs_list contains arrays of shape (num_data, 1)
        return np.array(sampled_outputs_list)  # shape: (n_samples_eff, num_data, 1)

    def plot_loss_history(self, title: str = "Training and Validation Loss"):
        """Plots the training and validation loss history."""
        plot_loss_history(self.history, title=title)


# TODO: Implement BayesianClassifier class (if needed)
# class BayesianClassifier(BaseEstimator):
#     def __init__(self, ...):
#         ...
#         self.nll_loss_fn = nn.CrossEntropyLoss() # For classification
#         ...
