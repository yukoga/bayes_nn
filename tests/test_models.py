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
import os
from bayes_nn.models import BayesianRegressor


# Simple mock class to simulate graphviz.Digraph
class MockDigraph:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.nodes = []
        self.edges = []
        # Mock the render method directly on the instance
        self.render = MockRenderMethod()

    def node(self, name, label, **attrs):
        self.nodes.append({"name": name, "label": label, "attrs": attrs})

    def edge(self, start, end, **attrs):
        self.edges.append({"start": start, "end": end, "attrs": attrs})


# Simple mock class for the render method call check
class MockRenderMethod:
    def __init__(self):
        self.called = False
        self.call_args_list = []

    def __call__(self, *args, **kwargs):
        self.called = True
        self.call_args_list.append({"args": args, "kwargs": kwargs})


# Fixture to mock the graphviz module and its Digraph
@pytest.fixture
def mock_graphviz(monkeypatch):
    class MockGraphvizModule:
        Digraph = MockDigraph  # Assign the class itself

    # Replace the graphviz import in the models module
    monkeypatch.setattr("bayes_nn.models.graphviz", MockGraphvizModule)
    MockGraphvizModule.Digraph.render_mock = MockRenderMethod()
    original_digraph_init = MockGraphvizModule.Digraph.__init__

    def patched_init(self, *args, **kwargs):
        original_digraph_init(self, *args, **kwargs)
        self.render = MockGraphvizModule.Digraph.render_mock

    monkeypatch.setattr(MockGraphvizModule.Digraph, "__init__", patched_init)

    yield MockGraphvizModule.Digraph.render_mock  # Yield the mock method

    # TODO: Teardown (monkeypatch handles revert automatically)


# Fixture to simulate graphviz not being installed
@pytest.fixture
def mock_graphviz_unavailable(monkeypatch):
    monkeypatch.setattr("bayes_nn.models.graphviz", None)


# --- Model Instance Fixture ---
@pytest.fixture
def bayesian_regressor_instance():
    """Provides a simple BayesianRegressor instance for testing."""
    return BayesianRegressor(
        input_dim=2,
        output_type="gaussian",
        hidden_dims=[10, 5],
        n_epochs=1,  # Minimal epochs for init
        random_state=42,
    )


# --- Data Fixtures ---
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
    assert isinstance(model.nll_loss_fn, torch.nn.Module)


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
    assert isinstance(model.nll_loss_fn, torch.nn.Module)


def test_bayesian_regressor_gaussian_fit_predict(linear_regression_data):
    """Test if fit and predict work for the Gaussian output model."""
    X, y = linear_regression_data
    model = BayesianRegressor(
        input_dim=X.shape[1],
        output_type="gaussian",
        hidden_dims=[16],
        n_epochs=5,
        batch_size=16,
        lr=0.01,
        validation_split=0.2,
        random_state=1,
    )
    model.fit(X, y, verbose=False)

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
    assert np.all(y_pred_std >= 0)

    # Test predict_proba
    y_samples = model.predict_proba(X, n_samples=10)
    assert isinstance(y_samples, np.ndarray)
    assert y_samples.shape == (10, X.shape[0], 1)

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
    assert np.all(y_pred >= 0)

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
    assert np.all(y_samples >= 0)

    # Test loss history
    assert "train_loss" in model.history
    assert "val_loss" in model.history
    assert len(model.history["train_loss"]) > 0
    assert len(model.history["val_loss"]) > 0


# TODO: Test cases for Early stopping
# TODO: Test cases for device specification ('cpu', 'cuda')
# TODO: Tests for the effect of KL term scaling and weight (more advanced)


# --- Tests for plot_network_architecture ---


def test_plot_network_architecture_calls_render(
    bayesian_regressor_instance, mock_graphviz
):
    """
    Test if plot_network_architecture calls graphviz.Digraph and render
    when graphviz is available.
    """
    model = bayesian_regressor_instance
    mock_render_method = mock_graphviz

    # Define filename to avoid default file creation/cleanup issues
    test_filename = "test_bnn_arch_render"
    output_file = f"{test_filename}.png"
    if os.path.exists(output_file):
        os.remove(output_file)

    model.plot_network_architecture(filename=test_filename, view=False)

    # Check if render was called
    assert mock_render_method.called, "render() was not called"

    # Check the arguments passed to render
    assert len(mock_render_method.call_args_list) == 1
    call_kwargs = mock_render_method.call_args_list[0]["kwargs"]
    call_args = mock_render_method.call_args_list[0]["args"]

    assert call_args[0] == test_filename  # Check filename positional arg
    assert call_kwargs.get("format") == "png"
    assert call_kwargs.get("view") is False
    assert call_kwargs.get("engine") == "dot"
    assert call_kwargs.get("cleanup") is True

    # Clean up the dummy file potentially created by the mock if needed
    if os.path.exists(output_file):
        os.remove(output_file)


def test_plot_network_architecture_handles_no_graphviz(
    bayesian_regressor_instance, mock_graphviz_unavailable, capsys
):
    """
    Test if plot_network_architecture handles the case where graphviz
    is not installed (mocked as None).
    """
    model = bayesian_regressor_instance
    model.plot_network_architecture()

    # Capture printed output
    captured = capsys.readouterr()
    assert "Error: 'graphviz' library not found" in captured.out
    assert "Please install it" in captured.out
