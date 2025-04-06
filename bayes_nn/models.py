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

from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torch import Tensor, Generator
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Type, Union

from tqdm.auto import tqdm
import torch.nn.functional as F

from .base import BaseEstimator
from .layers import BayesianLinear

from .losses import (
    ELBO,
    GaussianNLLLoss,
    PoissonNLLLoss,
)
from .utils import plot_loss_history

try:
    import graphviz
except ImportError:
    graphviz = None


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
        self.random_state = random_state

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Determine output dimension and NLL loss function
        self.nll_loss_fn: nn.Module
        if self.output_type == "gaussian":
            self.output_dim = 2
            self.nll_loss_fn = GaussianNLLLoss()
        elif self.output_type == "poisson":
            self.output_dim = 1
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

        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.elbo_loss_fn: Optional[ELBO] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        verbose: bool = True,
        **kwargs: Any,
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
            if train_size == 0:
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

        self.elbo_loss_fn = ELBO(
            self.model, self.nll_loss_fn, dataset_size, self.kl_weight
        ).to(self.device)

        # --- Training loop ---
        progress_bar = tqdm(range(self.n_epochs), desc="Epochs", disable=not verbose)
        for epoch in progress_bar:
            self.model.train()
            train_loss_epoch = 0.0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_count += 1
                self.optimizer.zero_grad()
                y_pred = self.model(batch_X)

                # Calculate loss (ELBO)
                assert self.elbo_loss_fn is not None
                loss = self.elbo_loss_fn(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss_epoch += loss.item()

            # Calculate average loss for the epoch, cast divisor to float
            avg_train_loss = (
                train_loss_epoch / float(batch_count) if batch_count > 0 else 0.0
            )
            self.history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            avg_val_loss = float("nan")
            if val_loader:
                self.model.eval()
                val_loss_epoch = 0.0
                val_batch_count = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val = batch_X_val.to(self.device)
                        batch_y_val = batch_y_val.to(self.device)
                        val_batch_count += 1

                        y_pred_val = self.model(batch_X_val)
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
                            break
            else:
                if verbose:
                    progress_bar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}"})
                self.best_model_state = self.model.state_dict()

        # After training, load the best model state
        # (if early stopping was used)
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
        **kwargs: Any,
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
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        n_samples_eff = n_samples if n_samples is not None else self.n_samples_predict
        predictions_list: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples_eff):
                output = self.model(X_tensor)
                predictions_list.append(output.cpu().numpy())
        predictions_np = np.array(predictions_list)

        mean_pred: np.ndarray
        std_pred: np.ndarray

        if self.output_type == "gaussian":
            # output_dim = 2 (mu, log_var)
            mus = predictions_np[:, :, 0]
            log_vars = predictions_np[:, :, 1]
            sigmas = np.sqrt(np.exp(log_vars))
            # Mean prediction = E[mu]
            mean_pred = mus.mean(axis=0)

            if return_std:
                # Standard deviation of prediction: considers uncertainty
                # Var[y] = E[Var[y|w]] + Var[E[y|w]]
                #        = E[sigma^2] + Var[mu]
                epistemic_uncertainty = mus.var(axis=0)
                aleatoric_uncertainty = (sigmas**2).mean(axis=0)
                total_variance = epistemic_uncertainty + aleatoric_uncertainty
                std_pred = np.sqrt(total_variance)
                return mean_pred, std_pred
            else:
                return mean_pred

        elif self.output_type == "poisson":
            # output_dim = 1 (log_lambda)
            log_lambdas = predictions_np[:, :, 0]
            lambdas = np.exp(log_lambdas)

            # Mean prediction = E[lambda]
            mean_pred = lambdas.mean(axis=0)

            if return_std:
                # Standard deviation of prediction = Std[lambda]
                std_pred = lambdas.std(axis=0)
                return mean_pred, std_pred
            else:
                return mean_pred
        else:
            raise RuntimeError("Invalid output_type encountered in predict")

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        n_samples: Optional[int] = None,
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
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        n_samples_eff = n_samples if n_samples is not None else self.n_samples_predict

        sampled_outputs_list: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples_eff):
                output = self.model(X_tensor).cpu().numpy()

                y_sample: np.ndarray
                if self.output_type == "gaussian":
                    mus = output[:, 0]
                    log_vars = output[:, 1]
                    sigmas = np.sqrt(np.exp(log_vars))
                    # Sample from N(mu, sigma^2) for each data point
                    y_sample = np.random.normal(mus, sigmas)
                    sampled_outputs_list.append(y_sample[:, np.newaxis])

                elif self.output_type == "poisson":
                    log_lambdas = output[:, 0]
                    lambdas = np.exp(log_lambdas)
                    # Sample from Poisson(lambda) for each data point
                    y_sample = np.random.poisson(lambdas)
                    sampled_outputs_list.append(y_sample[:, np.newaxis])
        # sampled_outputs_list contains arrays of shape (num_data, 1)
        return np.array(sampled_outputs_list)

    def plot_loss_history(self, title: str = "Training and Validation Loss"):
        """Plots the training and validation loss history."""
        plot_loss_history(self.history, title=title)

    def plot_network_architecture(
        self,
        filename: str = "bnn_architecture",
        format: str = "png",
        view: bool = False,
        engine: str = "dot",
    ):
        """
        Generates a visualization of the Bayesian Neural Network architecture.

        Requires the 'graphviz' library and the Graphviz system package
        to be installed (`pip install graphviz` and system install).

        Args:
            filename (str): The name of the output file (without extension).
            format (str): The output format (e.g., 'png', 'pdf', 'svg').
            view (bool): If True, automatically opens the generated file.
            engine (str): Layout engine to use (e.g., 'dot', 'neato', 'fdp').
        """
        if graphviz is None:
            print(
                "Error: 'graphviz' library not found. "
                "Please install it (`pip install graphviz`) and ensure the "
                "Graphviz system package is installed."
            )
            return

        dot = graphviz.Digraph(
            comment="Bayesian Neural Network Architecture",
            graph_attr={"rankdir": "LR", "splines": "line"},
            node_attr={
                "shape": "record",
                "style": "filled",
                "fillcolor": "lightblue",
            },
            edge_attr={"color": "black"},
        )

        # Input Node
        input_node_name = "input"
        dot.node(
            input_node_name,
            f"Input\n(dim={self.input_dim})",
            shape="ellipse",
            fillcolor="lightgrey",
        )
        last_node_name = input_node_name

        # Iterate through layers
        layer_idx = 0
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layer_{layer_idx}"
            if isinstance(layer, BayesianLinear):
                in_f = layer.in_features
                out_f = layer.out_features
                label = f"BayesianLinear_{layer_idx // 2}\n(in={in_f}, out={out_f})"
                dot.node(layer_name, label, fillcolor="skyblue")
                dot.edge(last_node_name, layer_name)
                last_node_name = layer_name
            elif isinstance(layer, nn.Module):
                act_name = layer.__class__.__name__
                label = f"Activation\n({act_name})"
                # Add activation as a separate node for clarity
                act_node_name = f"act_{layer_idx // 2}"
                dot.node(
                    act_node_name,
                    label,
                    shape="ellipse",
                    fillcolor="lightyellow",
                )
                dot.edge(last_node_name, act_node_name)
                last_node_name = act_node_name
            layer_idx += 1

        # Output Node (representing the distribution parameters)
        output_node_name = "output"
        output_label = f"Output\n({self.output_type}, dim={self.output_dim})"
        dot.node(
            output_node_name,
            output_label,
            shape="ellipse",
            fillcolor="lightgrey",
        )
        dot.edge(last_node_name, output_node_name)

        try:
            dot.render(
                filename,
                format=format,
                view=view,
                engine=engine,
                cleanup=True,
            )
            print(f"Network architecture saved to {filename}.{format}")
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Ensure Graphviz is installed and in your system's PATH.")


# Implementation of BayesianClassifier
class BayesianClassifier(BaseEstimator):
    """
    Classification model using a Bayesian Neural Network.
    Provides a scikit-learn-like interface.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
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
            num_classes (int): Number of output classes.
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
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        self.num_classes = num_classes
        self.output_dim = num_classes  # Output dim is number of classes
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
        self.random_state = random_state

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # NLL loss function for classification
        # CrossEntropyLoss combines LogSoftmax and NLLLoss, expects raw logits
        self.nll_loss_fn: nn.Module = nn.CrossEntropyLoss()

        # Build BNN model
        self.model = BayesianNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
        ).to(self.device)

        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.elbo_loss_fn: Optional[ELBO] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        verbose: bool = True,
        **kwargs: Any,
    ) -> "BayesianClassifier":
        """
        Fits the model to the data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data
                                                 (n_samples, n_features).
            y (Union[np.ndarray, pd.Series]): Target class labels (n_samples,).
                                              Should contain integers 0 to C-1.
            verbose (bool): Whether to display training progress.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Ensure y is 1D and contains long integers for CrossEntropyLoss
        if y.ndim != 1:
            # Try to flatten if it's like (n, 1)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.flatten()
            else:
                raise ValueError(
                    "Target y must be a 1D array of class labels, "
                    f"but got shape {y.shape}"
                )
        # Ensure target values are within [0, num_classes - 1]
        min_y, max_y = y.min(), y.max()
        if not (0 <= min_y and max_y < self.num_classes):
            raise ValueError(
                f"Target labels y must be integers in the range "
                f"[0, {self.num_classes - 1}], but found range "
                f"[{min_y}, {max_y}]"
            )

        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataset_size = len(dataset)

        train_dataset: Dataset[Tuple[Tensor, ...]]
        val_dataset: Optional[Dataset[Tuple[Tensor, ...]]] = None
        train_loader: DataLoader[Tuple[Tensor, ...]]
        val_loader: Optional[DataLoader[Tuple[Tensor, ...]]] = None
        train_size = 0

        # Split into training and validation data
        if self.validation_split > 0 and dataset_size > 1:
            val_size = int(dataset_size * self.validation_split)
            if val_size == 0 and dataset_size > 1:
                val_size = 1
            train_size = dataset_size - val_size
            if train_size == 0 and dataset_size > 1:
                train_size = 1
                val_size = dataset_size - 1

            # Only split if both train and val sizes are > 0
            if train_size > 0 and val_size > 0:
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
                # Only create val_loader if val_dataset is not None
                if val_dataset:
                    val_loader = DataLoader(
                        val_dataset, batch_size=self.batch_size, shuffle=False
                    )
                if verbose:
                    print(
                        f"Training on {train_size} samples, validating on "
                        f"{val_size} samples."
                    )
            else:  # Not enough data to split, use all for training
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
                    print(
                        f"Training on {dataset_size} samples "
                        "(not enough data to split for validation)."
                    )

        else:  # No validation split requested or dataset too small
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

        # Initialize ELBO loss function (needs dataset_size for KL scaling)
        # Use train_size if validation split occurred, otherwise full dataset_size
        effective_train_size: int
        if val_loader is not None and train_size > 0:
            effective_train_size = train_size
        else:
            effective_train_size = dataset_size

        self.elbo_loss_fn = ELBO(
            self.model, self.nll_loss_fn, effective_train_size, self.kl_weight
        ).to(self.device)

        # --- Training loop ---
        progress_bar = tqdm(range(self.n_epochs), desc="Epochs", disable=not verbose)
        self.history = {"train_loss": [], "val_loss": []}  # Reset history
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_model_state = None

        for epoch in progress_bar:
            self.model.train()
            train_loss_epoch = 0.0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_count += 1
                self.optimizer.zero_grad()
                y_pred_logits = self.model(batch_X)

                # Calculate loss (ELBO)
                assert self.elbo_loss_fn is not None
                # ELBO expects model output (logits) and target (class indices)
                loss = self.elbo_loss_fn(y_pred_logits, batch_y)

                # Check for NaN/inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"Warning: NaN or Inf loss detected at epoch "
                        f"{epoch + 1}, batch {batch_count}. Stopping training."
                    )
                    # Optionally load the last known best state if available
                    if self.best_model_state:
                        self.model.load_state_dict(self.best_model_state)
                    return self

                loss.backward()
                self.optimizer.step()
                train_loss_epoch += loss.item()

            avg_train_loss = (
                train_loss_epoch / float(batch_count) if batch_count > 0 else 0.0
            )
            self.history["train_loss"].append(avg_train_loss)

            # --- Validation ---
            avg_val_loss = float("nan")  # Use NaN as default if no validation
            if val_loader:
                self.model.eval()
                val_loss_epoch = 0.0
                val_batch_count = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val = batch_X_val.to(self.device)
                        batch_y_val = batch_y_val.to(self.device)
                        val_batch_count += 1

                        y_pred_val_logits = self.model(batch_X_val)
                        assert self.elbo_loss_fn is not None
                        # Use the same ELBO loss for validation consistency
                        # Note: KL term might behave differently in eval mode if
                        # layers change behavior, but ELBO class handles the
                        # model's KL divergence
                        # calculation internally.
                        val_loss = self.elbo_loss_fn(y_pred_val_logits, batch_y_val)
                        val_loss_epoch += val_loss.item()

                avg_val_loss = (
                    val_loss_epoch / float(val_batch_count)
                    if val_batch_count > 0
                    else float("inf")  # Avoid division by zero if val_loader is empty
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
                    # Check if avg_val_loss is valid before comparison
                    if not np.isnan(avg_val_loss) and not np.isinf(avg_val_loss):
                        if avg_val_loss < self.best_val_loss:
                            self.best_val_loss = avg_val_loss
                            self.epochs_no_improve = 0
                            # Save the best model state (clone to avoid
                            # reference issues)
                            self.best_model_state = {
                                k: v.clone() for k, v in self.model.state_dict().items()
                            }
                        else:
                            self.epochs_no_improve += 1
                            if self.epochs_no_improve >= self.early_stopping_patience:
                                if verbose:
                                    print(
                                        f"\nEarly stopping triggered after "
                                        f"{epoch + 1} epochs. Best validation "
                                        f"loss: {self.best_val_loss:.4f}"
                                    )
                                # Restore the best model state before breaking
                                if self.best_model_state:
                                    self.model.load_state_dict(self.best_model_state)
                                break
                    else:
                        # Handle invalid validation loss (e.g., print warning)
                        if verbose:
                            print(
                                f"\nWarning: Invalid validation loss "
                                f"({avg_val_loss}) "
                                f"at epoch {epoch + 1}. Skipping early stopping "
                                "check for this epoch."
                            )

            else:  # No validation loader
                if verbose:
                    progress_bar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}"})
                # If no validation, consider the last epoch's model as the 'best'
                # for prediction purposes Or, alternatively, don't implement
                # early stopping without validation. Let's save the state of the
                # last epoch if no validation is done.
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

        # After loop: Load the best model state if early stopping was active and
        # a best state was saved
        if (
            self.early_stopping_patience is not None
            and val_loader
            and self.best_model_state is not None
        ):
            # Check if the loop broke early, if so, the best model is already
            # loaded potentially. However, explicitly loading ensures the best
            # state is active.
            self.model.load_state_dict(self.best_model_state)
            if verbose and self.epochs_no_improve < self.early_stopping_patience:
                print(
                    f"\nFinished training. Loaded best model state with "
                    f"validation loss: {self.best_val_loss:.4f}"
                )
        elif self.best_model_state is not None:
            # If no validation/early stopping, ensure the last saved state is
            # loaded
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\nFinished training after {self.n_epochs} epochs.")

        return self

    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame], n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Predicts class probabilities for new data points.
        Performs multiple forward passes (sampling), applies softmax,
        and averages the probabilities.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data for prediction
                (num_data_points, n_features).
            n_samples (Optional[int]): Number of Monte Carlo samples
                (forward passes)
                                       to use for prediction.
                                       Uses `self.n_samples_predict` if None.

        Returns:
            np.ndarray: Average predicted probabilities for each class.
                        Shape: (num_data_points, num_classes).
        """
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        n_samples_eff = n_samples if n_samples is not None else self.n_samples_predict
        if n_samples_eff <= 0:
            raise ValueError("n_samples must be positive.")

        all_probs_list: List[Tensor] = []
        with torch.no_grad():
            for _ in range(n_samples_eff):
                logits_sample = self.model(X_tensor)
                probs_sample = F.softmax(logits_sample, dim=1)
                all_probs_list.append(probs_sample)

        # Stack probabilities from all samples: shape (n_samples_eff, num_data, num_classes)
        all_probs_tensor = torch.stack(all_probs_list, dim=0)

        # Average probabilities across samples
        avg_probs_tensor = torch.mean(all_probs_tensor, dim=0)

        return avg_probs_tensor.cpu().numpy()  # Shape: (num_data, num_classes)

    def predict(  # type: ignore[override]
        self,
        X: Union[np.ndarray, pd.DataFrame],
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predicts class labels for new data points.
        Predicts probabilities using predict_proba and returns the class
        with the highest average probability.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data for prediction
                (num_data_points, n_features).
            n_samples (Optional[int]): Number of Monte Carlo samples to use for
                                       probability estimation. Uses
                                       `self.n_samples_predict` if None.

        Returns:
            np.ndarray: Predicted class labels. Shape: (num_data_points,).
        """
        avg_probs = self.predict_proba(X, n_samples=n_samples)
        # Return the index (class label) with the highest probability
        return np.argmax(avg_probs, axis=1)

    # --- Optional: Plotting Methods ---
    def plot_loss_history(
        self, title: str = "Classifier Training and Validation Loss", **kwargs
    ):
        """Plots the training and validation loss history."""
        # Ensure val_loss exists and has values before plotting
        plot_loss_history(self.history, title=title, **kwargs)

    def plot_network_architecture(
        self,
        filename: str = "bnn_classifier_architecture",
        format: str = "png",
        view: bool = False,
        engine: str = "dot",
    ):
        """
        Generates a visualization of the Bayesian Neural Network architecture.
        Requires the 'graphviz' library and the Graphviz system package.
        """
        if graphviz is None:
            print(
                "Error: 'graphviz' library not found. "
                "Please install it (`pip install graphviz`) and ensure the "
                "Graphviz system package is installed."
            )
            return

        dot = graphviz.Digraph(
            comment="Bayesian Neural Network Classifier Architecture",
            graph_attr={"rankdir": "LR", "splines": "line"},
            node_attr={"shape": "record", "style": "filled", "fillcolor": "lightblue"},
            edge_attr={"color": "black"},
        )

        # Input Node
        input_node_name = "input"
        dot.node(
            input_node_name,
            f"Input\n(dim={self.input_dim})",
            shape="ellipse",
            fillcolor="lightgrey",
        )
        last_node_name = input_node_name
        # last_linear_node_name = ""  # Keep track of the last linear layer node - Unused

        # Iterate through layers
        linear_layer_count = 0
        activation_layer_count = 0
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, BayesianLinear):
                layer_name = f"linear_{linear_layer_count}"
                in_f = layer.in_features
                out_f = layer.out_features
                label = f"BayesianLinear_{linear_layer_count}\n(in={in_f}, out={out_f})"
                dot.node(layer_name, label, fillcolor="skyblue")
                dot.edge(last_node_name, layer_name)
                last_node_name = layer_name
                linear_layer_count += 1
            elif isinstance(layer, nn.Module):
                layer_name = f"act_{activation_layer_count}"
                act_name = layer.__class__.__name__
                label = f"Activation\n({act_name})"
                dot.node(layer_name, label, shape="ellipse", fillcolor="lightyellow")
                dot.edge(last_node_name, layer_name)
                last_node_name = layer_name
                activation_layer_count += 1

        # Output Logits Node (connected from the last actual layer, which should be the final linear layer)
        output_node_name = "output_logits"
        output_label = f"Output Logits\n(num_classes={self.num_classes})"
        dot.node(
            output_node_name,
            output_label,
            shape="ellipse",
            fillcolor="lightcoral",
        )
        # Connect from the last node in the sequence (which is the final
        # linear layer output)
        dot.edge(last_node_name, output_node_name)

        # Optional: Add a Softmax node visually to indicate the final step for
        # probabilities
        softmax_node_name = "softmax"
        dot.node(
            softmax_node_name,
            "Softmax\n(Probabilities)",
            shape="ellipse",
            fillcolor="lightgreen",
        )
        dot.edge(output_node_name, softmax_node_name)

        try:
            # Use render method which handles file creation
            rendered_path = dot.render(
                filename=filename,
                directory=".",
                format=format,
                view=view,
                cleanup=True,
                engine=engine,
            )
            print(f"Network architecture saved to {rendered_path}")
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Ensure Graphviz is installed and in your system's PATH.")
            print(
                "You might need to install graphviz using your system's "
                "package "
                "manager (e.g., `brew install graphviz` on macOS, "
                "`sudo apt-get install graphviz` on Debian/Ubuntu)."
            )
