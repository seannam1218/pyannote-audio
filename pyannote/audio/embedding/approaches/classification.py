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
# Hervé BREDIN - http://herve.niderb.fr

from typing import List, Union, Dict
from argparse import Namespace

from pyannote.database import Protocol
from pyannote.database import ProtocolFile
from pyannote.database import Subset

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSpeakerEmbedding


class SpeakerEmbeddingCrossEntropyLoss(BaseSpeakerEmbedding):
    def __init__(
        self, hparams: Union[Namespace, Dict], **kwargs,
    ):

        super().__init__(hparams, **kwargs)
        self.classifier = nn.Linear(self.model.dimension, len(self.classes), bias=False)

    def get_loss(self):
        def loss(fX: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.nll_loss(F.log_softmax(self.classifier(fX), dim=-1), y)

        return loss
