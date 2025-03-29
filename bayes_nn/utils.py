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

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(
    history: dict,
    title: str = "Training and Validation Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    figsize: tuple = (8, 5),
):
    """
    Function to plot the learning history (training loss and validation loss).

    Args:
        history (dict): Dictionary in the format
                        {'train_loss': [...], 'val_loss': [...]}.
                        'val_loss' may not always exist.
        title (str): Title of the graph.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        figsize (tuple): Size of the graph.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, history["train_loss"], "bo-", label="Training loss")

    if "val_loss" in history and len(history["val_loss"]) > 0:
        val_epochs = range(1, len(history["val_loss"]) + 1)
        plt.plot(
            val_epochs, history["val_loss"], "ro-", label="Validation loss"
        )
        if len(history["val_loss"]) > 0:
            best_val_epoch = np.argmin(history["val_loss"])
            best_val = history["val_loss"][best_val_epoch]
            plt.plot(
                best_val_epoch + 1,
                best_val,
                "r*",
                markersize=10,
                label=f"Best Val Loss: {best_val:.4f}",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_observed_vs_predicted(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    title: str = "Observed vs. Predicted Values",
    xlabel: str = "Observed Values",
    ylabel: str = "Predicted Values",
    figsize: tuple = (7, 7),
):
    """
    Plots observed vs. predicted values with a 95% confidence interval.

    Args:
        y_true (np.ndarray): Array of true observed values.
        y_pred_mean (np.ndarray): Array of predicted mean values.
        y_pred_std (np.ndarray): Array of predicted standard deviations.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)

    # Scatter plot of observed vs. predicted mean
    plt.scatter(y_true, y_pred_mean, alpha=0.6, label="Predicted Mean")

    # Calculate 95% confidence interval bounds (mean +/- 1.96 * std)
    lower_bound = y_pred_mean - 1.96 * y_pred_std
    upper_bound = y_pred_mean + 1.96 * y_pred_std

    # Plot the 95% confidence interval using error bars.
    plt.errorbar(
        y_true,
        y_pred_mean,
        yerr=1.96 * y_pred_std,
        fmt="none",
        alpha=0.4,
        ecolor="skyblue",
        elinewidth=1,
        capsize=0,
        label="95% Confidence Interval",
    )

    # Plot the y=x line (Ideal prediction)
    min_val = min(np.min(y_true), np.min(y_pred_mean - 2 * y_pred_std))
    max_val = max(np.max(y_true), np.max(y_pred_mean + 2 * y_pred_std))
    # Add some padding to the limits
    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Ideal prediction line (y_observed = y_predicted)",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
