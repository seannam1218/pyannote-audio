#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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
# Hervé BREDIN

from pyannote.audio.train.model import Model
from pyannote.audio.train.task import Problem
from pyannote.audio.features import RawAudio

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tdnn import TDNN
from .pooling import Pooling


class XVector(Model):
    def init(self):

        if self.task.problem != Problem.REPRESENTATION:
            msg = "XVector architecture can only be used for representation learning."
            raise ValueError(msg)

        if isinstance(self.task.feature_extraction, RawAudio):
            msg = "XVector architecture does not work from the waveform directly."
            raise ValueError(msg)

        n_features = self.task.feature_extraction.dimension

        self.normalize_ = nn.InstanceNorm1d(n_features, affine=False)

        self.frame1_ = TDNN(
            context=[-2, 2],
            input_channels=n_features,
            output_channels=512,
            full_context=True,
        )
        self.batchnorm1_ = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.frame2_ = TDNN(
            context=[-2, 0, 2],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.batchnorm2_ = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.frame3_ = TDNN(
            context=[-3, 0, 3],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.batchnorm3_ = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.frame4_ = TDNN(
            context=[0], input_channels=512, output_channels=512, full_context=True
        )
        self.batchnorm4_ = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.frame5_ = TDNN(
            context=[0], input_channels=512, output_channels=1500, full_context=True
        )
        self.batchnorm5_ = nn.BatchNorm1d(1500, momentum=0.1, affine=False)

        self.stats_pooling_ = Pooling(1500, method="stats")
        self.segment6_ = nn.Linear(3000, 512)

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        chunks : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of chunks features (batch_size, n_samples, n_features).

        Returns
        -------
        xvector : torch.Tensor
            (batch_size, 512) x-vectors.
        """
        output = self.normalize_(chunks.transpose(1, 2)).transpose(1, 2)
        output = self.batchnorm1_(
            F.relu(self.frame1_(output)).transpose(1, 2)
        ).transpose(1, 2)
        output = self.batchnorm2_(
            F.relu(self.frame2_(output)).transpose(1, 2)
        ).transpose(1, 2)
        output = self.batchnorm3_(
            F.relu(self.frame3_(output)).transpose(1, 2)
        ).transpose(1, 2)
        output = self.batchnorm4_(
            F.relu(self.frame4_(output)).transpose(1, 2)
        ).transpose(1, 2)
        output = self.batchnorm5_(
            F.relu(self.frame5_(output)).transpose(1, 2)
        ).transpose(1, 2)
        output = self.stats_pooling_(output)
        output = self.segment6_(output)
        return output

    def get_dimension(self) -> int:
        return 512
