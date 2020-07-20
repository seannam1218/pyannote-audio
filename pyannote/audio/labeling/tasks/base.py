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


import random
import math
from torch.utils.data import IterableDataset
from pyannote.core import Segment
from pyannote.audio.train.task import BaseTask


class LabelingDataset(IterableDataset):
    def __init__(self, task: BaseTask):
        super().__init__()
        self.task = task

    def __iter__(self):
        random.seed()

        files = self.task.train_metadata["files"]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = random.choices(
                files, weights=[f["__duration"] for f in files], k=1,
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
            y = file["__target"].crop(
                chunk, mode=self.task.model.alignment, fixed=self.task.hparams.duration
            )

            # yield sample
            yield {"X": X, "y": y}

    def __len__(self):

        epoch_duration = self.task.train_metadata["epoch_duration"]

        num_samples = math.ceil(epoch_duration / self.task.hparams.duration)

        # TODO: remove when https://github.com/pytorch/pytorch/pull/38925 is released
        num_samples = max(1, num_samples // self.task.hparams.batch_size)

        return num_samples
