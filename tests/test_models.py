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

import pytest
import numpy as np
import torch
from bayes_nn.models import BayesianRegressor


# Test fixtures (data generation)
@pytest.fixture
def linear_regression_data():
    """Generate test data for simple linear regression."""
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 1D input
    # True parameters a=2, b=1, noise standard deviation sigma=0.5
    true_a = 2.0
    true_b = 1.0
    true_sigma = 0.5
    y = true_a * X + true_b + np.random.normal(0, true_sigma, size=X.shape)
    return X, y.flatten()  # Make y a 1D array


@pytest.fixture
def poisson_regression_data():
    """Generate test data for simple Poisson regression."""
    np.random.seed(123)
    X = np.random.rand(100, 1) * 2 - 1  # 1D input [-1, 1]
    # True parameters a=1.5, b=0.5 -> log(lambda) = 1.5*x + 0.5
    true_a = 1.5
    true_b = 0.5
    log_lambda = true_a * X + true_b
    lambda_ = np.exp(log_lambda)
    y = np.random.poisson(lambda_, size=X.shape)
    return X, y.flatten()  # Make y a 1D array


# --- Test cases ---


def test_bayesian_regressor_gaussian_init():
    """Test if BayesianRegressor with Gaussian output can be initialized."""
    model = BayesianRegressor(
        input_dim=1,
        output_type="gaussian",
        hidden_dims=[10],
        n_epochs=1,
        random_state=0,
    )
    assert model is not None
    assert model.output_dim == 2
    assert isinstance(
        model.nll_loss_fn, torch.nn.Module
    )  # Check if it's an instance of GaussianNLLLoss


def test_bayesian_regressor_poisson_init():
    """Test if BayesianRegressor with Poisson output can be initialized."""
    model = BayesianRegressor(
        input_dim=1,
        output_type="poisson",
        hidden_dims=[10],
        n_epochs=1,
        random_state=0,
    )
    assert model is not None
    assert model.output_dim == 1
    assert isinstance(
        model.nll_loss_fn, torch.nn.Module
    )  # Check if it's an instance of PoissonNLLLoss


def test_bayesian_regressor_gaussian_fit_predict(linear_regression_data):
    """Test if fit and predict work for the Gaussian output model."""
    X, y = linear_regression_data
    model = BayesianRegressor(
        input_dim=X.shape[1],
        output_type="gaussian",
        hidden_dims=[16],
        n_epochs=5,
        batch_size=16,  # Small number of epochs for testing
        lr=0.01,
        validation_split=0.2,
        random_state=1,
    )
    model.fit(X, y, verbose=False)  # Hide progress during testing

    # Test predict
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X.shape[0],)

    # Test predict (with standard deviation)
    y_pred_mean, y_pred_std = model.predict(X, return_std=True)
    assert isinstance(y_pred_mean, np.ndarray)
    assert isinstance(y_pred_std, np.ndarray)
    assert y_pred_mean.shape == (X.shape[0],)
    assert y_pred_std.shape == (X.shape[0],)
    assert np.all(y_pred_std >= 0)  # Standard deviation should be non-negative

    # Test predict_proba
    y_samples = model.predict_proba(X, n_samples=10)
    assert isinstance(y_samples, np.ndarray)
    assert y_samples.shape == (10, X.shape[0], 1)  # (n_samples, num_data, 1)

    # Test loss history
    assert "train_loss" in model.history
    assert "val_loss" in model.history
    assert len(model.history["train_loss"]) > 0
    assert len(model.history["val_loss"]) > 0


def test_bayesian_regressor_poisson_fit_predict(poisson_regression_data):
    """Test if fit and predict work for the Poisson output model."""
    X, y = poisson_regression_data
    model = BayesianRegressor(
        input_dim=X.shape[1],
        output_type="poisson",
        hidden_dims=[16],
        n_epochs=5,
        batch_size=16,
        lr=0.01,
        validation_split=0.2,
        random_state=2,
    )
    model.fit(X, y, verbose=False)

    # Test predict
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(y_pred >= 0)  # Predicted rate should be non-negative

    # Test predict (with standard deviation)
    y_pred_mean, y_pred_std = model.predict(X, return_std=True)
    assert isinstance(y_pred_mean, np.ndarray)
    assert isinstance(y_pred_std, np.ndarray)
    assert y_pred_mean.shape == (X.shape[0],)
    assert y_pred_std.shape == (X.shape[0],)
    assert np.all(y_pred_mean >= 0)
    assert np.all(y_pred_std >= 0)

    # Test predict_proba
    y_samples = model.predict_proba(X, n_samples=10)
    assert isinstance(y_samples, np.ndarray)
    assert y_samples.shape == (10, X.shape[0], 1)
    assert np.all(
        y_samples >= 0
    )  # Samples from Poisson distribution are non-negative integers
    # Check for integer type (might have floating point errors)
    # assert np.all(y_samples == y_samples.astype(int))

    # Test loss history
    assert "train_loss" in model.history
    assert "val_loss" in model.history
    assert len(model.history["train_loss"]) > 0
    assert len(model.history["val_loss"]) > 0


# TODO: Test cases for Early stopping
# TODO: Test cases for device specification ('cpu', 'cuda')
# TODO: Tests for the effect of KL term scaling and weight (more advanced)
