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

from typing import Dict, Text, List, Union
import random
import numpy as np
import math

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution

from argparse import Namespace

from pyannote.core import Segment

from pyannote.database import Protocol
from pyannote.database import ProtocolFile
from pyannote.database import Subset
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerVerificationProtocol

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from pyannote.core.utils.hierarchy import linkage
import scipy.optimize
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure
from pyannote.metrics.binary_classification import det_curve
from pyannote.core import Annotation

from torch.utils.data import IterableDataset


import torch

# import torch.nn.functional as F

from collections import Counter


class Dataset(IterableDataset):
    def __init__(self, task: "BaseSpeakerEmbedding"):
        super().__init__()
        self.task = task

    def __iter__(self):

        labels = list(self.task.classes)

        # batch_counter counts samples in current batch.
        # as soon as it reaches batch_size, a new random duration is selected
        # so that the next batch will use a different chunk duration
        batch_counter = 0
        batch_duration = self.task.hparams.min_duration + random.random() * (
            self.task.hparams.duration - self.task.hparams.min_duration
        )

        while True:

            # shuffle labels
            random.shuffle(labels)

            for label in labels:

                # choose "per_label" files in which "label" occurs
                # NOTE: probability of choosing a file is proportional
                # to the duration of "label" in the file
                metadata = self.task._dataloader_metadata[label]
                metadata = random.choices(
                    metadata,
                    weights=[metadatum["duration"] for metadatum in metadata],
                    k=self.task.hparams.per_label,
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
                    for _ in range(self.task.hparams.per_turn):
                        start_time = random.uniform(
                            segment.start, segment.end - batch_duration
                        )
                        chunk = Segment(start_time, start_time + batch_duration)

                        # extract features
                        X = self.task.feature_extraction.crop(
                            file, chunk, mode="center", fixed=batch_duration,
                        )

                        # extract target
                        y = self.task.classes.index(label)

                        yield {"X": X, "y": y}

                        # increment number of samples in current batch
                        batch_counter += 1

                        # as soon as the batch is complete, a new random
                        # duration is selected so that the next batch will use
                        # a different chunk duration
                        if batch_counter == self.task.hparams.batch_size:
                            batch_counter = 0
                            batch_duration = self.task.hparams.min_duration + random.random() * (
                                self.task.hparams.duration
                                - self.task.hparams.min_duration
                            )

    def __len__(self):

        average_chunk_duration = 0.5 * (
            self.task.hparams.min_duration + self.task.hparams.duration
        )

        median_speaker_duration = np.median(
            [
                sum(metadatum["duration"] for metadatum in metadata)
                for label, metadata in self.task._dataloader_metadata.items()
            ]
        )

        num_speakers = len(self.task._dataloader_metadata)
        num_samples = math.ceil(
            (num_speakers * median_speaker_duration) / average_chunk_duration
        )

        # TODO: remove when https://github.com/pytorch/pytorch/pull/38925 is released
        num_samples = max(1, num_samples // self.task.hparams.batch_size)
        return num_samples


class BaseSpeakerEmbedding(BaseTask):

    problem = Problem.REPRESENTATION
    resolution_input = Resolution.FRAME
    resolution_output = Resolution.CHUNK

    def __init__(
        self, hparams: Union[Namespace, Dict], protocol: Protocol = None, **kwargs,
    ):

        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

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

        super().__init__(hparams, protocol=protocol, **kwargs)

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

    def prepare_metadata(self):

        self._dataloader_metadata: Dict[Text, List[Dict]] = dict()

        for file in self.files:
            for label, metadatum in file["metadata"].items():
                if label not in self.classes:
                    continue
                self._dataloader_metadata.setdefault(label, []).append(metadatum)

    def train_dataset(self) -> IterableDataset:
        return Dataset(self)

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

    @staticmethod
    def validation_criterion(protocol: Protocol):

        if isinstance(protocol, SpeakerVerificationProtocol):
            return "equal_error_rate"
        elif isinstance(protocol, SpeakerDiarizationProtocol):
            return "diarization_fscore"

        msg = (
            "Only SpeakerDiarization and SpeakerVerification protocols are "
            "supported."
        )
        raise ValueError(msg)

    def validation(
        self,
        files: List[ProtocolFile],
        protocol: Protocol = None,
        subset: Subset = "development",
        warm_start: Dict = None,
        epoch: int = None,
    ):

        if isinstance(protocol, SpeakerVerificationProtocol):
            return self.validation_verification(
                files,
                protocol=protocol,
                subset=subset,
                warm_start=warm_start,
                epoch=epoch,
            )

        elif isinstance(protocol, SpeakerDiarizationProtocol):
            return self.validation_diarization(
                files,
                protocol=protocol,
                subset=subset,
                warm_start=warm_start,
                epoch=epoch,
            )

        msg = (
            "Only SpeakerDiarization and SpeakerVerification protocols are "
            "supported."
        )
        raise ValueError(msg)

    def validation_diarization(
        self,
        files: List[ProtocolFile],
        protocol: SpeakerDiarizationProtocol = None,
        subset: Subset = "development",
        warm_start: Dict = None,
        epoch: int = None,
    ):

        # compute clustering dendrogram for all files
        for file in files:
            X = []
            for segment, _ in file["annotation"].itertracks():
                for mode in ["center", "loose"]:
                    x = file["scores"].crop(segment, mode=mode)
                    if len(x) > 0:
                        break
                X.append(np.mean(x, axis=0))
            X = np.array(X)
            file["dendrogram"] = linkage(X, method="pool", metric="cosine")

        def objective(threshold):
            metric = DiarizationPurityCoverageFMeasure(weighted=False)
            for file in files:
                clusters = fcluster(file["dendrogram"], threshold, criterion="distance")
                diarization = Annotation()
                for (segment, track), cluster in zip(
                    file["annotation"].itertracks(), clusters
                ):
                    diarization[segment, track] = cluster
                metric(file["annotation"], diarization, uem=file["annotated"])
            return 1.0 - abs(metric)

        res = scipy.optimize.minimize_scalar(
            objective, bounds=(0.0, 2.0), method="bounded", options={"maxiter": 10}
        )

        threshold = res.x.item()

        return {
            "metric": "diarization_fscore",
            "minimize": False,
            "value": float(1.0 - res.fun),
            "params": {"threshold": threshold},
        }

    def validation_verification(
        self,
        files: List[ProtocolFile],
        protocol: SpeakerVerificationProtocol = None,
        subset: Subset = "development",
        warm_start: Dict = None,
        epoch: int = None,
    ):

        files = {file["uri"]: file for file in files}

        def get_hash(file: ProtocolFile):
            return hash(tuple((file["uri"], tuple(file["try_with"]))))

        def get_embedding(file: ProtocolFile):
            return np.mean(
                files[file["uri"]]["scores"].crop(file["try_with"], mode="center"),
                axis=0,
                keepdims=True,
            )

        y_true, y_pred, cache = [], [], dict()

        for trial in getattr(protocol, f"{subset}_trial")():

            # compute average embedding for file1
            file1 = trial["file1"]
            hash1 = get_hash(file1)
            if hash1 in cache:
                emb1 = cache[hash1]
            else:
                emb1 = get_embedding(file1)
                cache[hash1] = emb1

            # compute average embedding for file2
            file2 = trial["file2"]
            hash2 = get_hash(file2)
            if hash2 in cache:
                emb2 = cache[hash2]
            else:
                emb2 = get_embedding(file2)
                cache[hash2] = emb2

            # compare average embeddings
            y_pred.append(cdist(emb1, emb2, metric="cosine")[0, 0])
            y_true.append(trial["reference"])

        # compute EER
        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred), distances=True)

        return {
            "metric": "equal_error_rate",
            "minimize": True,
            "value": float(eer),
        }
