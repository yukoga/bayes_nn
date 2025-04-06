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
from bayes_nn.models import (
    BayesianRegressor,
    BayesianClassifier,
)
from bayes_nn.losses import (
    GaussianNLLLoss,
    PoissonNLLLoss,
)
from torch.nn import CrossEntropyLoss


# Simple mock class to simulate graphviz.Digraph
class MockDigraph:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.nodes = []
        self.edges = []
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
        # Return None to mimic graphviz.render's behavior on success
        # when view=False
        return None


# Fixture to mock the graphviz module and its Digraph
@pytest.fixture
def mock_graphviz(monkeypatch):
    class MockGraphvizModule:
        Digraph = MockDigraph

    # Replace the graphviz import in the models module
    monkeypatch.setattr("bayes_nn.models.graphviz", MockGraphvizModule)
    MockGraphvizModule.Digraph.render_mock = MockRenderMethod()
    original_digraph_init = MockGraphvizModule.Digraph.__init__

    def patched_init(self, *args, **kwargs):
        original_digraph_init(self, *args, **kwargs)
        self.render = MockGraphvizModule.Digraph.render_mock

    monkeypatch.setattr(MockGraphvizModule.Digraph, "__init__", patched_init)

    yield MockGraphvizModule.Digraph.render_mock


# Fixture to simulate graphviz not being installed
@pytest.fixture
def mock_graphviz_unavailable(monkeypatch):
    monkeypatch.setattr("bayes_nn.models.graphviz", None)


# Fixture to mock the plot_loss_history utility function
@pytest.fixture
def mock_plot_loss_history(monkeypatch):
    """Mocks the plot_loss_history function to prevent actual plotting."""

    def mock_plot(history, title, **kwargs):
        pass

    monkeypatch.setattr("bayes_nn.models.plot_loss_history", mock_plot)
    return mock_plot


# --- Model Instance Fixtures ---
@pytest.fixture
def bayesian_regressor_instance():
    """Provides a simple BayesianRegressor instance for testing."""
    return BayesianRegressor(
        input_dim=2,
        output_type="gaussian",
        hidden_dims=[10, 5],
        n_epochs=1,
        random_state=42,
    )


@pytest.fixture
def bayesian_classifier_instance():
    """Provides a simple BayesianClassifier instance for testing."""
    return BayesianClassifier(
        input_dim=2,
        num_classes=3,
        hidden_dims=[10, 5],
        n_epochs=1,
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
    return X, y.flatten()


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
    return X, y.flatten()


@pytest.fixture
def classification_data():
    """Generate simple 2-class classification data."""
    np.random.seed(50)
    torch.manual_seed(50)
    # Class 0: centered around (-1, -1)
    X0 = np.random.randn(50, 2) - 1
    y0 = np.zeros(50, dtype=int)
    # Class 1: centered around (1, 1)
    X1 = np.random.randn(50, 2) + 1
    y1 = np.ones(50, dtype=int)
    # Combine
    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))
    # Shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y


# --- Bayesian Regressor Test cases ---


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
    assert isinstance(model.nll_loss_fn, GaussianNLLLoss)


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
    assert isinstance(model.nll_loss_fn, PoissonNLLLoss)


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


# --- Tests for plot_network_architecture (Regressor) ---


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
    # Clean up potential dot file as well
    dot_file = test_filename
    if os.path.exists(dot_file):
        os.remove(dot_file)

    model.plot_network_architecture(filename=test_filename, view=False)

    # Check if render was called
    assert mock_render_method.called, "render() was not called"

    # Check the arguments passed to render
    assert len(mock_render_method.call_args_list) == 1
    call_kwargs = mock_render_method.call_args_list[0]["kwargs"]
    call_args = mock_render_method.call_args_list[0]["args"]

    assert call_args[0] == test_filename
    assert call_kwargs.get("format") == "png"
    assert call_kwargs.get("view") is False
    assert call_kwargs.get("engine") == "dot"
    assert call_kwargs.get("cleanup") is True

    # Clean up the dummy file potentially created by the mock if needed
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(dot_file):
        os.remove(dot_file)


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


# --- Bayesian Classifier Test cases ---


def test_bayesian_classifier_init():
    """Test if BayesianClassifier can be initialized correctly."""
    input_dim = 4
    num_classes = 3
    model = BayesianClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[10],
        n_epochs=1,
        random_state=0,
    )
    assert model is not None
    assert model.input_dim == input_dim
    assert model.num_classes == num_classes
    assert model.output_dim == num_classes
    assert isinstance(model.nll_loss_fn, CrossEntropyLoss)
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model.device == expected_device


def test_bayesian_classifier_init_invalid_classes():
    """Test initialization fails with non-positive num_classes."""
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        # This check happens inside BayesianNetwork called by
        # BayesianClassifier
        # If num_classes=0, output_dim=0, BayesianNetwork likely fails.
        BayesianClassifier(input_dim=2, num_classes=0)
    with pytest.raises(ValueError):  # Negative should also fail
        BayesianClassifier(input_dim=2, num_classes=-1)


def test_bayesian_classifier_fit_invalid_target_shape(classification_data):
    """Test fit raises error if target y has wrong dimensions."""
    X, y = classification_data
    model = BayesianClassifier(input_dim=X.shape[1], num_classes=2, n_epochs=1)
    # Make y 2D
    # y_2d = y.reshape(-1, 1) # Unused variable
    y_bad_shape = np.stack([y, y], axis=1)
    with pytest.raises(ValueError, match="Target y must be a 1D array"):
        model.fit(X, y_bad_shape, verbose=False)


def test_bayesian_classifier_fit_invalid_target_range(classification_data):
    """Test fit raises error if target y has labels outside [0, C-1]."""
    X, y = classification_data  # y has labels 0, 1
    num_classes = 2
    model = BayesianClassifier(
        input_dim=X.shape[1], num_classes=num_classes, n_epochs=1
    )
    # Modify y to have an invalid label
    y_bad = y.copy()
    y_bad[0] = num_classes  # Should be less than num_classes
    with pytest.raises(
        ValueError,
        match=(
            # Use regex escape for brackets and .* for the variable part
            rf"Target labels y must be integers in the range "
            rf"\[0, {num_classes - 1}\], but found range .*"
        ),
    ):
        model.fit(X, y_bad, verbose=False)

    y_negative = y.copy()
    y_negative[0] = -1  # Negative label
    with pytest.raises(
        ValueError,
        match=(
            # Use regex escape for brackets and .* for the variable part
            rf"Target labels y must be integers in the range "
            rf"\[0, {num_classes - 1}\], but found range .*"
        ),
    ):
        model.fit(X, y_negative, verbose=False)


def test_bayesian_classifier_fit_predict_proba_predict(classification_data):
    """Test fit, predict_proba, and predict work for the classifier."""
    X, y = classification_data
    num_classes = 2
    model = BayesianClassifier(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dims=[8],  # Smaller network for faster test
        n_epochs=3,  # Few epochs just to run
        batch_size=16,
        lr=0.01,
        validation_split=0.2,
        random_state=3,
        n_samples_predict=10,  # Fewer samples for faster prediction
    )
    model.fit(X, y, verbose=False)

    # Test predict_proba
    y_proba = model.predict_proba(X)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (X.shape[0], num_classes)
    assert np.all(y_proba >= 0)
    assert np.all(y_proba <= 1)
    # Check probabilities sum to 1 (approximately)
    assert np.allclose(np.sum(y_proba, axis=1), 1.0, atol=1e-6)

    # Test predict
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(y_pred >= 0)
    assert np.all(y_pred < num_classes)
    # Check consistency between predict and predict_proba
    assert np.all(y_pred == np.argmax(y_proba, axis=1))

    # Test loss history
    assert "train_loss" in model.history
    assert "val_loss" in model.history
    assert len(model.history["train_loss"]) > 0
    assert len(model.history["val_loss"]) > 0


def test_bayesian_classifier_fit_no_validation(classification_data):
    """Test fitting without a validation split."""
    X, y = classification_data
    num_classes = 2
    model = BayesianClassifier(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dims=[8],
        n_epochs=2,
        validation_split=0.0,  # No validation
        random_state=4,
    )
    model.fit(X, y, verbose=False)
    assert "train_loss" in model.history
    assert "val_loss" in model.history
    assert len(model.history["train_loss"]) == 2
    # val_loss should be empty or contain NaNs if validation_split is 0
    assert len(model.history["val_loss"]) == 0 or all(
        np.isnan(vl) for vl in model.history["val_loss"]
    )


def test_bayesian_classifier_plot_loss_history(
    bayesian_classifier_instance, mock_plot_loss_history
):
    """Test calling plot_loss_history."""
    model = bayesian_classifier_instance
    # Add some dummy history
    model.history = {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.55]}
    model.plot_loss_history()
    # Test is primarily that it runs without error due to the mock


def test_bayesian_classifier_plot_network_architecture_calls_render(
    bayesian_classifier_instance, mock_graphviz
):
    """Test plot_network_architecture calls render for classifier."""
    model = bayesian_classifier_instance
    mock_render_method = mock_graphviz

    test_filename = "test_bnn_classifier_arch_render"
    output_file = f"{test_filename}.png"
    dot_file = test_filename
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(dot_file):
        os.remove(dot_file)

    model.plot_network_architecture(filename=test_filename, view=False)

    assert mock_render_method.called, "render() was not called"
    assert len(mock_render_method.call_args_list) == 1
    # call_args = mock_render_method.call_args_list[0]["args"] # Unused
    call_kwargs = mock_render_method.call_args_list[0]["kwargs"]

    # Check filename was passed as a keyword argument
    assert call_kwargs.get("filename") == test_filename
    assert call_kwargs.get("format") == "png"
    assert call_kwargs.get("view") is False
    assert call_kwargs.get("engine") == "dot"
    assert call_kwargs.get("cleanup") is True
    assert call_kwargs.get("directory") == "."

    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(dot_file):
        os.remove(dot_file)


def test_bayesian_classifier_plot_network_architecture_handles_no_graphviz(
    bayesian_classifier_instance, mock_graphviz_unavailable, capsys
):
    """Test plot_network_architecture handles no graphviz for classifier."""
    model = bayesian_classifier_instance
    model.plot_network_architecture()
    captured = capsys.readouterr()
    assert "Error: 'graphviz' library not found" in captured.out
