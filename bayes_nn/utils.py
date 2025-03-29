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
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    Plots observed vs. predicted values with a 95% confidence interval
    and displays RMSE and MAE metrics.

    Args:
        y_true (np.ndarray): Array of true observed values.
        y_pred_mean (np.ndarray): Array of predicted mean values.
        y_pred_std (np.ndarray): Array of predicted standard deviations.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        figsize (tuple): Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot of observed vs. predicted mean
    ax.scatter(y_true, y_pred_mean, alpha=0.6, label="Predicted Mean")

    # Plot the 95% confidence interval using error bars.
    ax.errorbar(
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

    # Plot the ideal prediction line
    # Determine plot limits based on data range
    min_val_data = min(np.min(y_true), np.min(y_pred_mean - 2 * y_pred_std))
    max_val_data = max(np.max(y_true), np.max(y_pred_mean + 2 * y_pred_std))
    # Add some padding to the limits using min/max based on data
    padding = (max_val_data - min_val_data) * 0.05
    min_val_plot = min_val_data - padding
    max_val_plot = max_val_data + padding

    ax.plot(
        [min_val_plot, max_val_plot],
        [min_val_plot, max_val_plot],
        "r--",
        label="Ideal Prediction",
    )

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    mae = mean_absolute_error(y_true, y_pred_mean)
    metrics_text = f"RMSE: {rmse:.4f}\nMAE:  {mae:.4f}"

    # Add metrics text to the plot (top-left corner)
    ax.text(
        0.05,  # x-coordinate (5% from left)
        0.95,  # y-coordinate (95% from bottom)
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.set_xlim(min_val_plot, max_val_plot)
    ax.set_ylim(min_val_plot, max_val_plot)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
