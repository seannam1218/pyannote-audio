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

"""Domain classification"""

import random
import math
import numpy as np

from torch.utils.data import IterableDataset
from typing import List, Dict

from pyannote.database import ProtocolFile
from pyannote.database import Protocol
from pyannote.database import Subset

from pyannote.core import Segment

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution

from tqdm import tqdm
from collections import Counter


class Dataset(IterableDataset):
    def __init__(self, task: "DomainClassification"):
        super().__init__()
        self.task = task

    def __iter__(self):
        random.seed()

        files = self.task.train_metadata["files"]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = random.choices(
                files, weights=[f["__duration"] for f in files], k=1
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = random.choices(
                file["annotated"], weights=[s.duration for s in file["annotated"]], k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = random.uniform(
                segment.start, segment.end - self.task.hparams.duration
            )
            chunk = Segment(start_time, start_time + self.task.hparams.duration)

            # extract features
            X = self.task.feature_extraction.crop(
                file, chunk, mode="center", fixed=self.task.hparams.duration
            )

            # extract target
            y = file["__domain"]

            # yield batch
            yield {"X": X, "y": y}

    def __len__(self):

        epoch_duration = self.task.train_metadata["epoch_duration"]

        num_samples = math.ceil(epoch_duration / self.task.hparams.duration)

        # TODO: remove when https://github.com/pytorch/pytorch/pull/38925 is released
        num_samples = max(1, num_samples // self.task.hparams.batch_size)

        return num_samples


class DomainClassification(BaseTask):

    problem = Problem.MULTI_CLASS_CLASSIFICATION
    resolution_input = Resolution.FRAME
    resolution_output = Resolution.CHUNK

    def prepare_metadata(self, files: List[ProtocolFile]) -> Dict:

        # gather list of domains
        domains = sorted(set(f[self.hparams.domain] for f in files))

        # add __duration and __domain keys to files
        # * __duration is the annotated duration minus too short segments
        # * __domain is the domain index
        for f in tqdm(iterable=files, desc="Loading training metadata", unit="file"):
            f["__duration"] = sum(
                s.duration for s in f["annotated"] if s.duration > self.hparams.duration
            )
            f["__domain"] = domains.index(f[self.hparams.domain])

            del f["annotation"]

        return {
            "classes": domains,
            "epoch_duration": sum(f["__duration"] for f in files),
            "files": [dict(f) for f in files],
        }

    def train_dataset(self) -> IterableDataset:
        return Dataset(self)

    @staticmethod
    def validation_criterion(protocol: Protocol):
        return "accuracy"

    def validation(
        self,
        files: List[ProtocolFile],
        protocol: Protocol = None,
        subset: Subset = "development",
        warm_start: Dict = None,
        epoch: int = None,
    ):
        """Validation

        Validation consists in computing file-wise domain classification accuracy.
        """

        criterion = self.validation_criterion(protocol)
        domains = self.hparams.classes

        y_true_file, y_pred_file = [], []

        for file in files:
            y_pred = np.argmax(file["scores"], axis=1)
            y_pred_file.append(Counter(y_pred).most_common(1)[0][0])

            y_true = domains.index(file[self.hparams.domain])
            y_true_file.append(y_true)

        accuracy = np.mean(np.array(y_true_file) == np.array(y_pred_file))

        return {
            "metric": criterion,
            "minimize": False,
            "value": float(accuracy),
        }
