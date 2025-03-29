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

"""
Bayes NN: A simple Bayesian Neural Network library using PyTorch.
"""

__version__ = "0.1.0"

from .layers import BayesianLinear
from .losses import ELBO, GaussianNLLLoss, PoissonNLLLoss
from .models import (
    BayesianRegressor,
    # BayesianClassifier,
)
from .utils import plot_loss_history

__all__ = [
    "BayesianLinear",
    "ELBO",
    "GaussianNLLLoss",
    "PoissonNLLLoss",
    "BayesianRegressor",
    # "BayesianClassifier",
    "plot_loss_history",
]
