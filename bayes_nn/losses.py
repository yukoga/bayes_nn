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
from .layers import BayesianLinear  # For KL divergence calculation


class ELBO(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss function. Used in Variational Inference.
    ELBO = E[log P(D|w)] - KL[q(w|theta) || p(w)]
    Instead of maximizing this, we minimize -ELBO.
    Loss = -E[log P(D|w)] + KL[q(w|theta) || p(w)]
         = NLL_Loss + KL_Loss
    """

    def __init__(
        self,
        model: nn.Module,
        nll_loss_fn: nn.Module,
        dataset_size: int,
        kl_weight: float = 1.0,
    ):
        """
        Args:
            model (nn.Module): The BNN model for which to calculate KL
                               divergence.
            nll_loss_fn (nn.Module): Loss function to calculate Negative Log
                                     Likelihood (NLL). e.g., GaussianNLLLoss,
                                     PoissonNLLLoss, nn.CrossEntropyLoss
            dataset_size (int): Total number of samples in the training
                                dataset. Used for scaling the KL term.
            kl_weight (float): Weight for the KL divergence term.
        """
        super().__init__()
        self.model = model
        self.nll_loss_fn = nll_loss_fn
        self.dataset_size = dataset_size
        self.kl_weight = kl_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the ELBO loss (-ELBO, the quantity to be minimized).

        Args:
            y_pred (torch.Tensor): Model output. May contain multiple values
                                   depending on the loss function (e.g., mean
                                   and variance for Gaussian).
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Calculated ELBO loss (scalar value).
        """
        # Negative Log Likelihood term
        nll_loss = self.nll_loss_fn(y_pred, y_true)

        # KL Divergence term
        # Sum KL divergences from all BayesianLinear layers in the model
        kl_div = torch.tensor(
            0.0, device=y_pred.device
        )  # Ensure device consistency (GPU/CPU)
        for module in self.model.modules():
            if isinstance(module, BayesianLinear):
                kl_div += module.kl_divergence()

        # Scale KL divergence by dataset size
        # beta * KL / N (beta = kl_weight)
        scaled_kl_div = self.kl_weight * kl_div / self.dataset_size
        # scaled_kl_div = self.kl_weight * kl_div # If not scaling

        # ELBO loss (-ELBO) = NLL + scaled_KL
        loss = nll_loss + scaled_kl_div

        return loss


# --- Loss functions for likelihood calculation ---


class GaussianNLLLoss(nn.Module):
    """
    Negative Log Likelihood loss for Gaussian distribution.
    Assumes the model outputs mean (mu) and log variance (log_var).
    NLL = 0.5 * (log(2*pi*sigma^2) + ((y - mu)/sigma)^2)
        = 0.5 * (log(2*pi) + log_var + (y - mu)^2 / exp(log_var))
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): shape (batch_size, 2). [:, 0] is mean mu,
                                   [:, 1] is log variance log_var.
            y_true (torch.Tensor): shape (batch_size, 1) or (batch_size, ).
        Returns:
            torch.Tensor: Mean NLL loss over the batch (scalar value).
        """
        if y_pred.shape[1] != 2:
            error_msg = (
                "GaussianNLLLoss expects y_pred to have 2 columns (mu, log_var)."
            )
            raise ValueError(error_msg)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)

        mu = y_pred[:, 0].unsqueeze(1)  # (batch_size, 1)
        log_var = y_pred[:, 1].unsqueeze(1)  # (batch_size, 1)
        var = torch.exp(log_var)
        sigma = torch.sqrt(var)

        term1 = torch.log(2 * torch.pi * var)
        term2 = ((y_true - mu) / sigma) ** 2
        nll = 0.5 * (term1 + term2)

        return nll.mean()  # Take the batch mean


class PoissonNLLLoss(nn.Module):
    """
    Negative Log Likelihood loss for Poisson distribution.
    Assumes the model outputs the log of the rate parameter (log_lambda).
    lambda = exp(log_lambda)
    log P(y | lambda) = y * log(lambda) - lambda - log(y!)
    NLL = lambda - y * log(lambda) + log(y!)
    (log(y!) is constant and doesn't affect optimization,
     but include for loss value)
    log(y!) can be calculated with PyTorch's `torch.lgamma(y + 1)`.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred_log_lambda: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y_pred_log_lambda (torch.Tensor): shape (batch_size, 1). Log of
                                              the rate parameter log(lambda).
            y_true (torch.Tensor): shape (batch_size, 1) or (batch_size, ).
                                   Non-negative integer values.
        Returns:
            torch.Tensor: Mean NLL loss over the batch (scalar value).
        """
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)

        log_lambda = y_pred_log_lambda
        lambda_ = torch.exp(log_lambda)

        # Poisson NLL loss
        # log(y!) is calculated using torch.lgamma(y + 1)
        nll = lambda_ - y_true * log_lambda + torch.lgamma(y_true + 1)

        return nll.mean()  # Take the batch mean
