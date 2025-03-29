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
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer (Variational Inference).
    Parameterizes weights and biases with mean and log standard deviation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.001,  # Adjusted default
        prior_pi: float = 0.5,
    ):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            prior_sigma_1 (float): Std dev of the larger component in the
                                   scale mixture prior.
            prior_sigma_2 (float): Std dev of the smaller component in the
                                   scale mixture prior. Should be significantly
                                   smaller than prior_sigma_1.
            prior_pi (float): Mixture proportion for the scale mixture prior
                              (probability of using sigma1).
            (Note: While the Flipout paper uses more efficient perturbation
             calculations, this implements a basic Bayesian linear layer
             using the Reparameterization Trick.)
        """
        super().__init__()
        if not (0 < prior_pi < 1):
            raise ValueError("prior_pi must be between 0 and 1")
        if not (prior_sigma_1 > 0 and prior_sigma_2 > 0):
            raise ValueError("Prior sigmas must be positive")
        if prior_sigma_1 <= prior_sigma_2:
            print(
                "Warning: prior_sigma_1 should ideally be larger than "
                "prior_sigma_2 for the scale mixture prior to be effective."
            )

        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters (learnable)
        # Weight mean (weight_mu) and log standard deviation (weight_rho)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias mean (bias_mu) and log standard deviation (bias_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters (inspired by He initialization, etc.)
        self.reset_parameters()

        # Prior distribution parameters (fixed)
        # Using a scale mixture normal distribution as the prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        # Store log sigmas and log mixture weights as buffers for efficiency
        # and numerical stability
        prior_log_sigma_1 = torch.tensor(math.log(prior_sigma_1))
        prior_log_sigma_2 = torch.tensor(math.log(prior_sigma_2))
        log_prior_pi = torch.tensor(math.log(prior_pi))
        log_prior_one_minus_pi = torch.tensor(math.log(1.0 - prior_pi))

        self.register_buffer("prior_log_sigma_1", prior_log_sigma_1)
        self.register_buffer("prior_log_sigma_2", prior_log_sigma_2)
        self.register_buffer("log_prior_pi", log_prior_pi)
        self.register_buffer("log_prior_one_minus_pi", log_prior_one_minus_pi)

    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize weights (inspired by Kaiming He initialization variance)
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialize log standard deviation (rho) with small negative values
        # for small initial std dev (e.g., -3 corresponds to sigma ~ 0.05)
        nn.init.constant_(self.weight_rho, -3.0)  # Changed from -5.0

        # Initialize bias
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -3.0)  # Changed from -5.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Samples weights using the Reparameterization Trick.
        Only samples during training mode. Uses mean during evaluation.
        """
        if self.training:
            # 1. Sample weights and biases from the variational posterior
            # Standard deviation sigma = log(1 + exp(rho)) (softplus)
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)

            # Generate standard normal random numbers (epsilon)
            # .to(self.weight_mu.device) ensures device consistency
            epsilon_weight = torch.randn_like(weight_sigma)
            epsilon_weight = epsilon_weight.to(self.weight_mu.device)
            epsilon_bias = torch.randn_like(bias_sigma)
            epsilon_bias = epsilon_bias.to(self.bias_mu.device)

            # Reparameterization Trick: w = mu + sigma * epsilon
            weight = self.weight_mu + weight_sigma * epsilon_weight
            bias = self.bias_mu + bias_sigma * epsilon_bias
        else:
            # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        # 2. Perform standard linear transformation
        output = F.linear(x, weight, bias)

        return output

    def kl_divergence(self) -> torch.Tensor:
        """
        Calculates the KL divergence between the variational posterior
        q(w|theta) and the prior p(w) for this layer. KL[q(w|theta) || p(w)]
        Uses a scale mixture normal distribution as the prior.

        Note: The exact KL divergence between a Gaussian posterior and a
        Gaussian mixture prior does not have a simple closed-form solution.
        This implementation calculates the KL divergence between the posterior
        and *each component* of the mixture prior, then combines them.
        This is an approximation often used in practice, but might not be
        the theoretically exact KL divergence for the mixture.
        A common simplification is KL[q || N(0, sigma1^2)], ignoring mixture.
        Here we implement the approximation based on log probabilities.
        """
        # Variational posterior q(w|theta) is N(mu, sigma^2)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        kl_weight_comp1 = (
            self.prior_log_sigma_1
            - torch.log(weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_sigma_1**2)
            - 0.5
        ).sum()
        kl_bias_comp1 = (
            self.prior_log_sigma_1
            - torch.log(bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_sigma_1**2)
            - 0.5
        ).sum()

        return kl_weight_comp1 + kl_bias_comp1

    def log_prior_prob(self, w: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of the scale mixture prior p(w)
        for the given weights w using the logsumexp trick for numerical
        stability.
        p(w) = pi * N(w|0, sigma1^2) + (1-pi) * N(w|0, sigma2^2)
        log p(w) = logsumexp( [log(pi) + log N(w|0, sigma1^2),
                               log(1-pi) + log N(w|0, sigma2^2)] )
        """
        # Ensure w is on the same device as the buffers
        w = w.to(self.prior_log_sigma_1.device)

        # Calculate log probability under each Gaussian component
        # log N(w|0, sigma^2) = -0.5 * log(2*pi*sigma^2) - w^2 / (2*sigma^2)
        #                     = -0.5*log(2*pi) - log(sigma) - w^2 / (2*sigma^2)
        const = -0.5 * math.log(2 * math.pi)
        log_prob1 = (
            const - self.prior_log_sigma_1 - (w**2) / (2 * self.prior_sigma_1**2)
        )
        log_prob2 = (
            const - self.prior_log_sigma_2 - (w**2) / (2 * self.prior_sigma_2**2)
        )

        # Combine with log mixture weights
        term1 = self.log_prior_pi + log_prob1
        term2 = self.log_prior_one_minus_pi + log_prob2

        # Stack for logsumexp
        stacked_terms = torch.stack([term1, term2], dim=0)

        # Compute logsumexp over the mixture dimension (dim=0)
        log_mix_prob = torch.logsumexp(stacked_terms, dim=0)

        return log_mix_prob
