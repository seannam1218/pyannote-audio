#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Juan Manuel Coria
# Hervé Bredin - http://herve.niderb.fr

from typing_extensions import Literal

import torch
import torch.nn as nn


class Pooling(nn.Module):
    """Pooling over the time dimension

    Parameters
    ----------
    method : {"last", "max", "average", "stats"}, optional
        Use "max" for max pooling, "average" for average pooling.
        Use "average" for average pooling.
        Use "last" for returning the last element of the sequence.
        Use "stats" for statistics pooling (à la x-vector).
    bidirectional : bool, optional
        When using "last" pooling, indicate whether the input sequence should
        be considered as the output of a bidirectional recurrent layer, in which
        case the last element in both directions are concatenated.
    eps : float, optional
        When using "stats" pooling, add normally-distributed noise with mean 0
        and variance eps, during training. Defaults to 1e-5.
    """

    def __init__(
        self,
        n_features,
        method: Literal["last", "max", "average", "stats"] = None,
        bidirectional: bool = None,
        eps: float = 1e-5,
    ):
        super().__init__()

        if method == "last" and bidirectional is None:
            msg = "'last' pooling expects an additional 'bidirectional' parameter."
            raise ValueError(msg)

        self.n_features = n_features
        self.method = method
        self.bidirectional = bidirectional
        self.eps = eps

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Temporal pooling

        Parameters
        ----------
        sequences : torch.Tensor
            Input sequences with shape (batch_size, n_frames, n_features)

        Returns
        -------
        pooled : torch.Tensor
            Pooled sequences with shape (batch_size, n_features)
        """

        if self.method is None:
            return sequences

        if self.method == "last":
            if self.bidirectional:
                batch_size, n_frames, _ = sequences.shape
                reshaped = sequences.view(batch_size, n_frames, 2, -1)
                return torch.cat([reshaped[:, -1, 0], reshaped[:, 0, 1]], dim=1)
            else:
                return sequences[:, -1]

        if self.method == "max":
            return torch.max(sequences, dim=1, keepdim=False, out=None)[0]

        if self.method == "average":
            return torch.mean(sequences, dim=1, keepdim=False, out=None)

        if self.method == "stats":
            mu = torch.mean(sequences, dim=1)
            if self.training:
                noise = torch.randn_like(sequences) * self.eps
                sigma = torch.std(sequences + noise, dim=1)
            else:
                sigma = torch.std(sequences, dim=1)

            return torch.cat((mu, sigma), dim=1)

    @property
    def dimension(self):
        "Dimension of output features"
        if self.method == "stats":
            return self.n_features * 2
        else:
            return self.n_features
