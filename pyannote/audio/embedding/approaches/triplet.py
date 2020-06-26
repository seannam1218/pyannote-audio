#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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


from typing import List, Tuple, Union, Dict
from argparse import Namespace
import numpy as np
import torch
import torch.nn.functional as F
from .base import BaseSpeakerEmbedding
from pyannote.core.utils.distance import to_condensed
from scipy.spatial.distance import squareform


class SpeakerEmbeddingTripletLoss(BaseSpeakerEmbedding):
    def __init__(
        self, hparams: Union[Namespace, Dict], **kwargs,
    ):

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        if "per_label" not in hparams:
            hparams.per_label = 3

        if "sampling" not in hparams:
            hparams.sampling = "all"

        if "clamp" not in hparams:
            hparams.clamp = "sigmoid"

        if "margin" not in hparams:
            hparams.margin = 0.0

        if "scale" not in hparams:
            hparams.scale = 10.0

        super().__init__(hparams, **kwargs)

    def batch_easy(
        self, y: torch.Tensor, distances: np.ndarray
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build easy triplets"""

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                d = distances[anchor, positive]

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    if d > distances[anchor, negative]:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives

    def batch_hard(
        self, y: torch.Tensor, distances: np.ndarray
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build triplet with both hardest positive and hardest negative"""

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):

            d = distances[anchor]

            # hardest positive
            pos = np.where(y == y_anchor)[0]
            pos = [p for p in pos if p != anchor]
            positive = int(pos[np.argmax(d[pos])])

            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives

    def batch_negative(
        self, y: torch.Tensor, distances: np.ndarray
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build triplet with hardest negative"""

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):

            # hardest negative
            d = distances[anchor]
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            for positive in np.where(y == y_anchor)[0]:
                if positive == anchor:
                    continue

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        return anchors, positives, negatives

    def batch_all(
        self, y: torch.Tensor, distances: np.ndarray
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build all possible triplet"""

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives

    def get_loss(self):

        get_triplets = getattr(self, "batch_{0}".format(self.hparams.sampling))

        def loss(fX: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

            # pairwise cosine distance
            distances = 0.5 * torch.pow(F.pdist(F.normalize(fX)), 2)

            anchors, positives, negatives = get_triplets(
                y, squareform(distances.detach().numpy())
            )

            # convert indices from squared to condensed matrix base
            pos = to_condensed(self.hparams.batch_size, anchors, positives)
            neg = to_condensed(self.hparams.batch_size, anchors, negatives)

            # compute raw triplet loss (no margin, no clamping, the lower, the better)
            delta = distances[pos] - distances[neg]

            if self.hparams.clamp == "positive":
                clamped = torch.clamp(delta + self.hparams.margin, min=0)

            elif self.hparams.clamp == "softmargin":
                clamped = torch.log1p(torch.exp(delta))

            elif self.hparams.clamp == "sigmoid":
                clamped = torch.sigmoid(
                    self.hparams.scale * (delta + self.hparams.margin)
                )

            return torch.mean(clamped)

        return loss
