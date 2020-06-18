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

from typing import Dict, Text, List
import random

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution

from argparse import Namespace

from pyannote.core import Segment

from pyannote.database import Protocol
from pyannote.database import ProtocolFile
from pyannote.database import Subset
from torch.utils.data import IterableDataset

import torch

# import torch.nn.functional as F

from collections import Counter


class BaseSpeakerEmbedding(BaseTask):

    problem = Problem.REPRESENTATION
    resolution_input = Resolution.FRAME
    resolution_output = Resolution.CHUNK

    def __init__(
        self,
        hparams: Namespace,
        protocol: Protocol = None,
        subset: Subset = "train",
        files: List[ProtocolFile] = None,
    ):

        if "duration" not in hparams:
            hparams.duration = 2.0

        if "min_duration" not in hparams:
            hparams.min_duration = 2.0

        if "per_fold" not in hparams:
            hparams.per_fold = 32

        if "per_label" not in hparams:
            hparams.per_label = 1

        if "per_turn" not in hparams:
            hparams.per_turn = 1

        if "label_min_duration" not in hparams:
            hparams.label_min_duration = 0.0

        hparams.batch_size = hparams.per_fold * hparams.per_label * hparams.per_turn

        if protocol is not None:
            protocol.preprocessors["metadata"] = self.get_metadata

        super().__init__(hparams, protocol=protocol, subset=subset, files=files)

    def get_metadata(self, file: ProtocolFile) -> Dict[Text, Dict]:

        metadata = dict()

        for label in file["annotation"].labels():
            timeline = file["annotation"].label_timeline(label)
            segments = [s for s in timeline if s.duration > self.hparams.duration]
            if not segments:
                continue
            duration = sum(s.duration for s in segments)
            metadata[label] = {"file": file, "segments": segments, "duration": duration}

        return metadata

    def get_classes(self):

        total_duration = Counter()

        for file in self.files:
            duration = {
                label: metadata["duration"]
                for label, metadata in file["metadata"].items()
            }
            total_duration.update(duration)

        return sorted(
            label
            for label, duration in total_duration.items()
            if duration > self.hparams.label_min_duration
        )

    def prepare_data(self):

        self._dataloader_metadata: Dict[Text, List[Dict]] = dict()

        for file in self.files:
            for label, metadatum in file["metadata"].items():
                if not label in self.classes:
                    continue
                self._dataloader_metadata.setdefault(label, []).append(metadatum)

    def train_dataset(self) -> IterableDataset:
        class Dataset(IterableDataset):
            def __iter__(dataset):

                labels = list(self.classes)

                # batch_counter counts samples in current batch.
                # as soon as it reaches batch_size, a new random duration is selected
                # so that the next batch will use a different chunk duration
                batch_counter = 0
                batch_duration = self.hparams.min_duration + random.random() * (
                    self.hparams.duration - self.hparams.min_duration
                )

                while True:

                    # shuffle labels
                    random.shuffle(labels)

                    for label in labels:

                        # choose "per_label" files in which "label" occurs
                        # NOTE: probability of choosing a file is proportional
                        # to the duration of "label" in the file
                        metadata = self._dataloader_metadata[label]
                        metadata = random.choices(
                            metadata,
                            weights=[metadatum["duration"] for metadatum in metadata],
                            k=self.hparams.per_label,
                        )

                        for metadatum in metadata:
                            file = metadatum["file"]

                            # choose one "label" segment
                            # NOTE: probability of choosing a segment is
                            # proportional to the duration of the segment
                            segment, *_ = random.choices(
                                metadatum["segments"],
                                weights=[s.duration for s in metadatum["segments"]],
                                k=1,
                            )

                            # choose "per_turn" chunks
                            for _ in range(self.hparams.per_turn):
                                start_time = random.uniform(
                                    segment.start, segment.end - batch_duration
                                )
                                chunk = Segment(start_time, start_time + batch_duration)

                                # extract features
                                X = self.feature_extraction.crop(
                                    file, chunk, mode="center", fixed=batch_duration,
                                )

                                # extract target
                                y = self.classes.index(label)

                                yield {"X": X, "y": y}

                                # increment number of samples in current batch
                                batch_counter += 1

                                # as soon as the batch is complete, a new random
                                # duration is selected so that the next batch will use
                                # a different chunk duration
                                if batch_counter == self.hparams.batch_size:
                                    batch_counter = 0
                                    batch_duration = self.hparams.min_duration + random.random() * (
                                        self.hparams.duration
                                        - self.hparams.min_duration
                                    )

            def __len__(dataset):
                return 100

        return Dataset()

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        return self.model(chunks)

    def training_step(self, batch, batch_idx):
        X = batch["X"]
        y = batch["y"]

        fX = self(X)
        if self.hparams.per_turn > 1:

            # TODO. add support for other aggregation functions, e.g. replacing
            # mean by product may encourage sparse representation
            fX = fX.view(
                self.hparams.per_fold * self.hparams.per_label,
                self.hparams.per_turn,
                -1,
            ).mean(axis=1)
            y = y[:: self.hparams.per_turn]

        loss = self.loss(fX, y)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    # def pdist(self, fX):
    #     """Compute pdist à-la scipy.spatial.distance.pdist

    #     Parameters
    #     ----------
    #     fX : (n, d) torch.Tensor
    #         Embeddings.

    #     Returns
    #     -------
    #     distances : (n * (n-1) / 2,) torch.Tensor
    #         Condensed pairwise distance matrix
    #     """

    #     if self.hparams.metric == "euclidean":
    #         return F.pdist(fX)

    #     elif self.hparams.metric in ("cosine", "angular"):

    #         distance = 0.5 * torch.pow(F.pdist(F.normalize(fX)), 2)
    #         if self.hparams.metric == "cosine":
    #             return distance

    #         return torch.acos(torch.clamp(1.0 - distance, -1 + 1e-12, 1 - 1e-12))
