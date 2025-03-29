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
# import torch.nn as nn # Unused import removed
import math

from bayes_nn.layers import BayesianLinear


@pytest.fixture
def layer_params():
    """Provides standard parameters for creating a BayesianLinear layer."""
    return {
        "in_features": 10,
        "out_features": 5,
        "prior_sigma_1": 1.0,
        "prior_sigma_2": 0.1,
        "prior_pi": 0.5,
    }


@pytest.fixture
def bayesian_linear_layer(layer_params):
    """Creates a BayesianLinear layer instance using standard parameters."""
    return BayesianLinear(**layer_params)


class TestBayesianLinear:
    def test_init(self, bayesian_linear_layer, layer_params):
        """Tests the initialization of the BayesianLinear layer."""
        layer = bayesian_linear_layer
        params = layer_params

        assert layer.in_features == params["in_features"]
        assert layer.out_features == params["out_features"]

        # Check parameter shapes
        expected_weight_shape = (params["out_features"], params["in_features"])
        assert layer.weight_mu.shape == expected_weight_shape
        assert layer.weight_rho.shape == expected_weight_shape
        assert layer.bias_mu.shape == (params["out_features"],)
        assert layer.bias_rho.shape == (params["out_features"],)

        # Check if parameters are learnable (requires_grad=True)
        assert layer.weight_mu.requires_grad
        assert layer.weight_rho.requires_grad
        assert layer.bias_mu.requires_grad
        assert layer.bias_rho.requires_grad

        # Check prior parameters storage
        assert layer.prior_sigma_1 == params["prior_sigma_1"]
        assert layer.prior_sigma_2 == params["prior_sigma_2"]
        assert layer.prior_pi == params["prior_pi"]

        # Check registered buffers for prior log sigmas
        assert hasattr(layer, "prior_log_sigma_1")
        assert hasattr(layer, "prior_log_sigma_2")
        assert torch.isclose(
            layer.prior_log_sigma_1,
            torch.tensor(math.log(params["prior_sigma_1"])),
        )
        assert torch.isclose(
            layer.prior_log_sigma_2,
            torch.tensor(math.log(params["prior_sigma_2"])),
        )
        # Buffers should not require gradients
        assert not layer.prior_log_sigma_1.requires_grad
        assert not layer.prior_log_sigma_2.requires_grad

    def test_reset_parameters(self, layer_params):
        """Tests the parameter reset logic."""
        # Create layer; reset_parameters is called in __init__
        # We check the initial state here.
        layer = BayesianLinear(**layer_params)

        # Check if rho values are initialized correctly
        # Note: Initial value changed to -3.0 in layers.py
        assert torch.all(layer.weight_rho == -3.0)
        assert torch.all(layer.bias_rho == -3.0)

        # Check if mu values are initialized and not NaN or Inf
        assert not torch.isnan(layer.weight_mu).any()
        assert not torch.isinf(layer.weight_mu).any()
        assert not torch.isnan(layer.bias_mu).any()
        assert not torch.isinf(layer.bias_mu).any()

        # Explicitly call reset_parameters again (optional sanity check)
        layer.reset_parameters()
        assert torch.all(layer.weight_rho == -3.0)
        assert torch.all(layer.bias_rho == -3.0)
        assert not torch.isnan(layer.weight_mu).any()
        assert not torch.isinf(layer.weight_mu).any()
        assert not torch.isnan(layer.bias_mu).any()
        assert not torch.isinf(layer.bias_mu).any()

    def test_forward_shape_and_type(self, bayesian_linear_layer, layer_params):
        """Tests the forward pass output shape and data type."""
        layer = bayesian_linear_layer
        batch_size = 4
        input_tensor = torch.randn(batch_size, layer_params["in_features"])

        output = layer(input_tensor)

        assert output.shape == (batch_size, layer_params["out_features"])
        assert output.dtype == torch.float32  # Assuming default dtype

    def test_forward_stochasticity(self, bayesian_linear_layer, layer_params):
        """Tests that the forward pass is stochastic due to sampling."""
        layer = bayesian_linear_layer
        batch_size = 4
        input_tensor = torch.randn(batch_size, layer_params["in_features"])

        # Ensure the layer is in training mode for sampling
        layer.train()

        # Get two outputs with the same input
        output1 = layer(input_tensor)
        output2 = layer(input_tensor)

        # The outputs should be different due to sampling
        assert not torch.equal(output1, output2)

        # Check if setting seed makes it deterministic (optional sanity check)
        torch.manual_seed(42)
        output_seed1 = layer(input_tensor)
        torch.manual_seed(42)
        output_seed2 = layer(input_tensor)
        assert torch.equal(output_seed1, output_seed2)

    def test_kl_divergence_shape_and_value(self, bayesian_linear_layer):
        """Tests the KL divergence calculation."""
        layer = bayesian_linear_layer
        kl_div = layer.kl_divergence()

        # KL divergence should be a scalar tensor
        assert isinstance(kl_div, torch.Tensor)
        assert kl_div.ndim == 0
        assert kl_div.dtype == torch.float32

        # KL divergence should be non-negative
        assert kl_div.item() >= 0.0

        # Check that KL divergence requires grad (depends on learnable params)
        assert kl_div.requires_grad

    def test_log_prior_prob_shape(self, bayesian_linear_layer, layer_params):
        """Tests the shape of the log prior probability output."""
        layer = bayesian_linear_layer
        # Create a sample weight tensor with the correct shape
        sample_weights = torch.randn(
            layer_params["out_features"], layer_params["in_features"]
        )

        log_prior = layer.log_prior_prob(sample_weights)

        assert isinstance(log_prior, torch.Tensor)
        assert log_prior.shape == sample_weights.shape
        assert log_prior.dtype == torch.float32

        # Note: log probability *density* can be positive, so the check
        # `assert torch.all(log_prior <= 0.0)` was removed as it's incorrect
        # for density functions, especially with small prior sigmas.

        # Test with bias shape as well (though the method is generic)
        sample_bias = torch.randn(layer_params["out_features"])
        log_prior_bias = layer.log_prior_prob(sample_bias)
        assert log_prior_bias.shape == sample_bias.shape
