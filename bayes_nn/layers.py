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
        prior_sigma_2: float = 0.001,
        prior_pi: float = 0.5,
    ):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            prior_sigma_1 (float): Std dev of the larger component in the
                                   scale mixture prior.
            prior_sigma_2 (float): Std dev of the smaller component in the
                                   scale mixture prior.
            prior_pi (float): Mixture proportion for the scale mixture prior
                              (probability of using sigma1).
            (Note: While the Flipout paper uses more efficient perturbation
             calculations, this implements a basic Bayesian linear layer
             using the Reparameterization Trick.)
        """
        super().__init__()
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
        prior_log_sigma_1 = torch.tensor(math.log(prior_sigma_1))
        prior_log_sigma_2 = torch.tensor(math.log(prior_sigma_2))
        self.register_buffer("prior_log_sigma_1", prior_log_sigma_1)
        self.register_buffer("prior_log_sigma_2", prior_log_sigma_2)

    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize weights (inspired by Kaiming He initialization variance)
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialize log standard deviation (rho) with small negative values
        # for small initial std dev
        nn.init.constant_(self.weight_rho, -5.0)

        # Initialize bias
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Samples weights using the Reparameterization Trick.
        """
        # 1. Sample weights and biases from the variational posterior
        # Standard deviation sigma = log(1 + exp(rho)) (softplus)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # Generate standard normal random numbers (epsilon)
        # .to(self.weight_mu.device) ensures device consistency (GPU/CPU)
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_weight = epsilon_weight.to(self.weight_mu.device)
        epsilon_bias = torch.randn_like(bias_sigma).to(self.bias_mu.device)

        # Reparameterization Trick: w = mu + sigma * epsilon
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # 2. Perform standard linear transformation
        output = F.linear(x, weight, bias)

        return output

    def kl_divergence(self) -> torch.Tensor:
        """
        Calculates the KL divergence between the variational posterior
        q(w|theta) and the prior p(w) for this layer. KL[q(w|theta) || p(w)]
        Uses a scale mixture normal distribution as the prior.
        """
        # Variational posterior q(w|theta) is a Normal distribution
        # N(mu, sigma^2) where sigma = softplus(rho)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # Variational posterior (Normal distribution)
        # - Variables removed as unused
        # q_weight = torch.distributions.Normal(self.weight_mu, weight_sigma)
        # q_bias = torch.distributions.Normal(self.bias_mu, bias_sigma)

        # Prior distribution (Scale mixture normal)
        # p(w) = pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2)
        # Calculating KL divergence analytically is difficult.
        # Monte Carlo approximation or, for simplicity, approximate using
        # a simple Normal prior N(0, prior_sigma_1^2).
        # (More accurately, closed-form approximations proposed in papers
        # should be used)
        # Here, for simplicity, use N(0, prior_sigma_1^2) as the prior.
        prior_std_w = self.prior_sigma_1
        prior_std_b = self.prior_sigma_1

        # KL[N(mu, sigma^2) || N(0, prior_sigma^2)]
        # = log(prior_sigma / sigma)
        #   + (sigma^2 + mu^2) / (2 * prior_sigma^2) - 0.5
        kl_weight = (
            torch.log(prior_std_w / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * prior_std_w**2)
            - 0.5
        ).sum()
        kl_bias = (
            torch.log(prior_std_b / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * prior_std_b**2)
            - 0.5
        ).sum()

        # Simple implementation: KL divergence assuming prior is N(0, 1)
        # kl_weight = -0.5 * torch.sum(
        #     1 + 2 * torch.log(weight_sigma)
        #     - self.weight_mu.pow(2) - weight_sigma.pow(2)
        # )
        # kl_bias = -0.5 * torch.sum(
        #     1 + 2 * torch.log(bias_sigma)
        #     - self.bias_mu.pow(2) - bias_sigma.pow(2)
        # )

        # More accurate KL divergence for scale mixture prior
        # (often uses approximations)
        # log_prior = self.log_prior_prob(weight) + self.log_prior_prob(bias)
        # log_q = q_weight.log_prob(weight).sum() + q_bias.log_prob(bias).sum()
        # kl_div = (log_q - log_prior).mean() # Monte Carlo approximation

        # Using the N(0, prior_sigma_1^2) approximation here
        return kl_weight + kl_bias

    def log_prior_prob(self, w):
        """Log probability of the scale mixture prior."""
        dist1 = torch.distributions.Normal(0, self.prior_sigma_1)
        log_prob1 = dist1.log_prob(w)
        dist2 = torch.distributions.Normal(0, self.prior_sigma_2)
        log_prob2 = dist2.log_prob(w)

        prob1 = log_prob1.exp()
        prob2 = log_prob2.exp()

        return torch.log(self.prior_pi * prob1 + (1 - self.prior_pi) * prob2)
