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
import torch
import torch.nn as nn
import math
import builtins  # Import the builtins module

# Import the classes to be tested
from bayes_nn.losses import GaussianNLLLoss, PoissonNLLLoss, ELBO
# Import BayesianLinear only for the isinstance check target in ELBO
from bayes_nn.layers import BayesianLinear


# --- Tests for GaussianNLLLoss ---

def test_gaussian_nll_loss_init():
    """Test initialization of GaussianNLLLoss."""
    loss_fn = GaussianNLLLoss()
    assert isinstance(loss_fn, nn.Module)


def test_gaussian_nll_loss_forward_basic():
    """Test forward pass with simple values."""
    loss_fn = GaussianNLLLoss()
    # mu=0, log_var=0 (sigma=1)
    # NLL = 0.5 * (log(2*pi*sigma^2) + ((y - mu)/sigma)^2)
    # NLL = 0.5 * (log(2*pi*1) + ((1 - 0)/1)^2) = 0.5 * (log(2*pi) + 1)
    y_pred = torch.tensor([[0.0, 0.0]])  # mu=0, log_var=0
    y_true = torch.tensor([[1.0]])       # y=1
    expected_nll = 0.5 * (math.log(2 * math.pi * 1.0) + ((1.0 - 0.0) / 1.0)**2)
    loss = loss_fn(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(expected_nll))


def test_gaussian_nll_loss_forward_batch():
    """Test forward pass with a batch of values."""
    loss_fn = GaussianNLLLoss()
    # Batch:
    # 1: mu=0, log_var=0 (var=1), y=1
    # 2: mu=1, log_var=ln(4) (var=4), y=3
    y_pred = torch.tensor([[0.0, 0.0], [1.0, math.log(4.0)]])
    y_true = torch.tensor([[1.0], [3.0]])

    # Sample 1: NLL1 = 0.5 * (log(2*pi*1) + ((1-0)/1)^2)
    #                = 0.5 * (log(2*pi) + 1)
    nll1 = 0.5 * (math.log(2 * math.pi * 1.0) + 1.0)
    # Sample 2: NLL2 = 0.5 * (log(2*pi*4) + ((3-1)/2)^2)
    #                = 0.5 * (log(8*pi) + (2/2)^2) = 0.5 * (log(8*pi) + 1)
    nll2 = 0.5 * (math.log(8 * math.pi) + 1.0)

    expected_mean_nll = (nll1 + nll2) / 2.0
    loss = loss_fn(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(expected_mean_nll))


def test_gaussian_nll_loss_forward_y_true_1d():
    """Test forward pass when y_true is 1D."""
    loss_fn = GaussianNLLLoss()
    y_pred = torch.tensor([[0.0, 0.0]])
    y_true = torch.tensor([1.0])  # 1D tensor
    # Expected NLL is same as the single basic case
    expected_nll = 0.5 * (math.log(2 * math.pi * 1.0) + 1.0)
    loss = loss_fn(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(expected_nll))


def test_gaussian_nll_loss_wrong_y_pred_shape():
    """Test error handling for incorrect y_pred shape."""
    loss_fn = GaussianNLLLoss()
    y_pred = torch.tensor([[0.0]])  # Only one column
    y_true = torch.tensor([[1.0]])
    with pytest.raises(ValueError, match="expects y_pred to have 2 columns"):
        loss_fn(y_pred, y_true)


# --- Tests for PoissonNLLLoss ---

def test_poisson_nll_loss_init():
    """Test initialization of PoissonNLLLoss."""
    loss_fn = PoissonNLLLoss()
    assert isinstance(loss_fn, nn.Module)


def test_poisson_nll_loss_forward_basic():
    """Test forward pass with simple values."""
    loss_fn = PoissonNLLLoss()
    # log_lambda = log(5) -> lambda = 5
    y_pred_log_lambda = torch.tensor([[math.log(5.0)]])
    # y=3 (should be integer, but test float for calculation)
    y_true = torch.tensor([[3.0]])
    # NLL = lambda - y*log(lambda) + log(y!)
    # NLL = lambda - y*log(lambda) + lgamma(y+1)
    # NLL = 5 - 3*log(5) + lgamma(3+1) = 5 - 3*log(5) + lgamma(4)
    lgamma_val = torch.lgamma(torch.tensor(3.0 + 1.0)).item()
    expected_nll = 5.0 - 3.0 * math.log(5.0) + lgamma_val
    loss = loss_fn(y_pred_log_lambda, y_true)
    assert torch.isclose(loss, torch.tensor(expected_nll))


def test_poisson_nll_loss_forward_batch():
    """Test forward pass with a batch of values."""
    loss_fn = PoissonNLLLoss()
    # Batch:
    # 1: log_lambda=log(5) (lambda=5), y=3
    # 2: log_lambda=log(1) (lambda=1), y=0
    y_pred_log_lambda = torch.tensor([[math.log(5.0)], [math.log(1.0)]])
    y_true = torch.tensor([[3.0], [0.0]])

    # Sample 1: NLL1 = 5 - 3*log(5) + lgamma(4)
    lgamma1 = torch.lgamma(torch.tensor(3.0 + 1.0)).item()
    nll1 = 5.0 - 3.0 * math.log(5.0) + lgamma1
    # Sample 2: NLL2 = 1 - 0*log(1) + lgamma(0+1) = 1 - 0 + lgamma(1)
    #                = 1.0 (since lgamma(1)=0)
    lgamma2 = torch.lgamma(torch.tensor(0.0 + 1.0)).item()
    nll2 = 1.0 - 0.0 * math.log(1.0) + lgamma2

    expected_mean_nll = (nll1 + nll2) / 2.0
    loss = loss_fn(y_pred_log_lambda, y_true)
    assert torch.isclose(loss, torch.tensor(expected_mean_nll))


def test_poisson_nll_loss_forward_y_true_1d():
    """Test forward pass when y_true is 1D."""
    loss_fn = PoissonNLLLoss()
    y_pred_log_lambda = torch.tensor([[math.log(5.0)]])
    y_true = torch.tensor([3.0])  # 1D tensor
    # Expected NLL is same as the single basic case
    lgamma_val = torch.lgamma(torch.tensor(3.0 + 1.0)).item()
    expected_nll = 5.0 - 3.0 * math.log(5.0) + lgamma_val
    loss = loss_fn(y_pred_log_lambda, y_true)
    assert torch.isclose(loss, torch.tensor(expected_nll))


# --- Fixtures for ELBO Test ---

class MockBayesianLayer(nn.Module):
    """A mock layer inheriting nn.Module with a kl_divergence method."""
    def __init__(self, kl_value: float):
        super().__init__()
        # Store KL value as a tensor for potential device consistency
        self.register_buffer('_kl_value', torch.tensor(kl_value))

    def kl_divergence(self) -> torch.Tensor:
        # Ensure the returned tensor is on the same device as the buffer
        return self._kl_value

    def forward(self, x):
        # Simple pass-through, doesn't need to do anything complex
        return x


@pytest.fixture
def mock_bnn_model_for_elbo(monkeypatch):
    """
    Provides a mock BNN model containing MockBayesianLayer instances
    and patches isinstance check within bayes_nn.losses.
    """
    model = nn.Sequential(
        MockBayesianLayer(kl_value=10.0),
        nn.ReLU(),  # A non-Bayesian layer
        MockBayesianLayer(kl_value=5.0)
    )

    # Patch isinstance used within the ELBO class's forward method
    # Get the original isinstance from the builtins module
    original_isinstance = builtins.isinstance

    # Define the patched function with correct spacing
    def patched_isinstance(obj, classinfo):
        # If checking for BayesianLinear, check for our mock type instead
        if classinfo == BayesianLinear:
            return isinstance(obj, MockBayesianLayer)
        # Otherwise, use the original isinstance
        return original_isinstance(obj, classinfo)

    # Apply the patch to the builtins module, as isinstance is a built-in
    monkeypatch.setattr(builtins, 'isinstance', patched_isinstance)

    return model


@pytest.fixture
def mock_nll_loss_fn_for_elbo():
    """Provides a simple mock NLL loss function returning a fixed value."""
    def _mock_loss(y_pred, y_true):
        # Return a fixed tensor value. Ensure it's on the same device
        # as inputs if needed. For simplicity, assume CPU testing.
        # Return on same device as input
        return torch.tensor(2.5, device=y_pred.device)
    return _mock_loss


# --- Tests for ELBO ---

def test_elbo_init():
    """Test initialization of ELBO."""
    mock_model = nn.Module()  # Simple placeholder
    mock_nll = nn.Module()    # Simple placeholder
    dataset_size = 100
    elbo = ELBO(mock_model, mock_nll, dataset_size)
    assert elbo.model is mock_model
    assert elbo.nll_loss_fn is mock_nll
    assert elbo.dataset_size == dataset_size
    assert elbo.kl_weight == 1.0  # Default value


def test_elbo_init_with_kl_weight():
    """Test initialization of ELBO with a specific kl_weight."""
    mock_model = nn.Module()
    mock_nll = nn.Module()
    dataset_size = 100
    kl_weight = 0.5
    elbo = ELBO(mock_model, mock_nll, dataset_size, kl_weight=kl_weight)
    assert elbo.kl_weight == kl_weight


def test_elbo_forward_calculation(
    mock_bnn_model_for_elbo, mock_nll_loss_fn_for_elbo
):
    """Test the forward pass calculation of ELBO using mocks."""
    dataset_size = 100
    kl_weight = 0.1
    # Get the model with patched isinstance
    model = mock_bnn_model_for_elbo
    nll_fn = mock_nll_loss_fn_for_elbo

    elbo_loss = ELBO(
        model=model,
        nll_loss_fn=nll_fn,
        dataset_size=dataset_size,
        kl_weight=kl_weight
    )

    # Dummy input tensors (shapes don't matter much for these mocks)
    batch_size = 8
    y_pred = torch.randn(batch_size, 1)
    y_true = torch.randn(batch_size, 1)

    # Expected KL divergence calculation
    # Sum KL from MockBayesianLayer instances in mock_bnn_model_for_elbo
    expected_kl_sum = 10.0 + 5.0
    expected_scaled_kl = kl_weight * expected_kl_sum / dataset_size

    # Expected NLL from mock_nll_loss_fn_for_elbo
    expected_nll = 2.5

    # Expected total loss (-ELBO) = NLL + scaled_KL
    expected_loss_value = expected_nll + expected_scaled_kl

    # Calculate actual loss
    actual_loss = elbo_loss(y_pred, y_true)

    # Assertions
    assert isinstance(actual_loss, torch.Tensor)
    assert actual_loss.ndim == 0  # Should be a scalar tensor
    # Use torch.isclose for comparing floating point tensors
    assert torch.isclose(actual_loss, torch.tensor(expected_loss_value))


def test_elbo_forward_device_consistency(
    mock_bnn_model_for_elbo, mock_nll_loss_fn_for_elbo
):
    """Test ELBO handles tensors on different devices (GPU if available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device consistency test")

    device = torch.device("cuda:0")
    dataset_size = 100
    kl_weight = 0.1

    # Move model (which contains mock layers with buffers) to CUDA
    model = mock_bnn_model_for_elbo.to(device)
    nll_fn = mock_nll_loss_fn_for_elbo  # Mock fn already handles device

    elbo_loss = ELBO(
        model=model,
        nll_loss_fn=nll_fn,
        dataset_size=dataset_size,
        kl_weight=kl_weight
    )

    # Input tensors on CUDA
    batch_size = 8
    y_pred = torch.randn(batch_size, 1, device=device)
    y_true = torch.randn(batch_size, 1, device=device)

    # Expected calculation (should be same value, but result on CUDA)
    expected_kl_sum = 10.0 + 5.0
    expected_scaled_kl = kl_weight * expected_kl_sum / dataset_size
    expected_nll = 2.5
    expected_loss_value = expected_nll + expected_scaled_kl

    # Calculate actual loss
    actual_loss = elbo_loss(y_pred, y_true)

    # Assertions
    assert actual_loss.device == device
    expected_tensor = torch.tensor(expected_loss_value, device=device)
    assert torch.isclose(actual_loss, expected_tensor)
