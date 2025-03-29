# -*- coding: utf-8 -*-
# Copyright (c) 2025 yukoga. All rights reserved.

import pytest
import numpy as np
import matplotlib.pyplot as plt

# Import functions to be tested
from bayes_nn.utils import plot_loss_history, plot_observed_vs_predicted


@pytest.fixture
def mock_matplotlib_show(monkeypatch):
    """Fixture to mock plt.show() to prevent plots from displaying."""
    def mock_show():
        # print("plt.show() called and mocked.")  # Optional: for debugging
        plt.close()  # Close the figure to free memory

    monkeypatch.setattr(plt, "show", mock_show)


# --- Tests for plot_loss_history ---

def test_plot_loss_history_train_only(mock_matplotlib_show):
    """Test plot_loss_history with only training loss."""
    history = {"train_loss": [0.5, 0.4, 0.3]}
    try:
        plot_loss_history(history)
        # If it runs without error, the basic plotting calls worked.
    except Exception as e:
        pytest.fail(
            f"plot_loss_history raised exception with train_loss only: {e}"
        )


def test_plot_loss_history_train_val(mock_matplotlib_show):
    """Test plot_loss_history with training and validation loss."""
    history = {
        "train_loss": [0.5, 0.4, 0.3, 0.25],
        "val_loss": [0.6, 0.45, 0.35, 0.3],
    }
    try:
        plot_loss_history(history)
        # Check if best val loss marker logic runs
    except Exception as e:
        pytest.fail(
            f"plot_loss_history raised exception with train and val loss: {e}"
        )


def test_plot_loss_history_empty_val(mock_matplotlib_show):
    """Test plot_loss_history with an empty val_loss list."""
    history = {"train_loss": [0.5, 0.4, 0.3], "val_loss": []}
    try:
        plot_loss_history(history)
    except Exception as e:
        pytest.fail(
            f"plot_loss_history raised exception with empty val_loss: {e}"
        )


# --- Tests for plot_observed_vs_predicted ---

@pytest.fixture
def sample_prediction_data():
    """Provides sample data for observed vs. predicted tests."""
    np.random.seed(42)
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred_mean = y_true + np.random.normal(0, 0.5, size=y_true.shape)
    y_pred_std = np.abs(np.random.normal(0.5, 0.2, size=y_true.shape))
    return y_true, y_pred_mean, y_pred_std


def test_plot_observed_vs_predicted_runs(
    mock_matplotlib_show, sample_prediction_data
):
    """Test if plot_observed_vs_predicted runs without errors."""
    y_true, y_pred_mean, y_pred_std = sample_prediction_data
    try:
        plot_observed_vs_predicted(y_true, y_pred_mean, y_pred_std)
        # Basic check: Does it run? Does it calculate metrics?
    except Exception as e:
        pytest.fail(f"plot_observed_vs_predicted raised an exception: {e}")
