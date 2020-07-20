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

"""Speaker change detection"""

import numpy as np
from typing import List, Dict, Union
from tqdm import trange, tqdm
import scipy.signal

from torch.utils.data import IterableDataset

from pyannote.core import Segment, Timeline, SlidingWindowFeature
from pyannote.core.utils.numpy import one_hot_encoding

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution

from argparse import Namespace

from pyannote.audio.pipeline.speaker_change_detection import (
    SpeakerChangeDetection as SpeakerChangeDetectionPipeline,
)
from pyannote.pipeline import Optimizer
from pyannote.database import ProtocolFile
from pyannote.database import Protocol
from pyannote.database import Subset

from .base import LabelingDataset


class SpeakerChangeDetection(BaseTask):
    """Speaker change detection"""

    problem = Problem.MULTI_CLASS_CLASSIFICATION
    resolution_input = Resolution.FRAME
    resolution_output = Resolution.FRAME

    def __init__(
        self, hparams: Union[Namespace, Dict], **kwargs,
    ):

        super().__init__(hparams, **kwargs)
        if "collar" not in self.hparams:
            self.hparams.collar = 0.1

    def prepare_metadata(self, files: List[ProtocolFile]) -> Dict:

        # number of samples in collar
        resolution = self.model.resolution
        collar_samples = resolution.duration_to_samples(self.hparams.collar)

        # window
        window = scipy.signal.triang(collar_samples)[:, np.newaxis]

        for f in tqdm(iterable=files, desc="Loading training metadata", unit="file"):

            f["__duration"] = sum(
                s.duration for s in f["annotated"] if s.duration > self.hparams.duration
            )

            orig_y = one_hot_encoding(
                f["annotation"],
                Timeline(segments=[Segment(0, f["duration"])]),
                resolution,
                mode="center",
            )

            # replace NaNs by 0s
            orig_y = np.nan_to_num(orig_y)
            n_samples, n_speakers = orig_y.shape

            # True = change. False = no change
            y = np.sum(np.abs(np.diff(orig_y, axis=0)), axis=1, keepdims=True)
            y = np.vstack(([[0]], y > 0))

            # mark change points neighborhood as positive
            y = np.minimum(1, scipy.signal.convolve(y, window, mode="same"))

            # HACK for some reason, y rarely equals zero
            y = 1 * (y > 1e-10)

            # at this point, all segment boundaries are marked as change
            # (including non-speech/speaker changes.
            # let's remove non-speech/speaker change

            # append (half collar) empty samples at the beginning/end
            expanded_Y = np.vstack(
                [
                    np.zeros(
                        ((collar_samples + 1) // 2, n_speakers), dtype=orig_y.dtype
                    ),
                    orig_y,
                    np.zeros(
                        ((collar_samples + 1) // 2, n_speakers), dtype=orig_y.dtype
                    ),
                ]
            )

            # stride trick. data[i] is now a sliding window of collar length
            # centered at time step i.
            data = np.lib.stride_tricks.as_strided(
                expanded_Y,
                shape=(n_samples, n_speakers, collar_samples),
                strides=(orig_y.strides[0], orig_y.strides[1], orig_y.strides[0]),
            )

            # y[i] = 1 if more than one speaker are speaking in the
            # corresponding window. 0 otherwise
            x_speakers = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
            x_speakers = x_speakers.reshape(-1, 1)

            y *= x_speakers

            f["__target"] = SlidingWindowFeature(y, resolution, labels=["change"])
            del f["annotation"]

        return {
            "classes": ["non_change", "change"],
            "epoch_duration": sum(f["__duration"] for f in files),
            "files": [dict(f) for f in files],
        }

    def train_dataset(self) -> IterableDataset:
        return LabelingDataset(self)

    @staticmethod
    def validation_criterion(protocol: Protocol):
        return "segmentation_fscore"

    def validation_pipeline(self):
        pipeline = SpeakerChangeDetectionPipeline(
            scores="@scores", fscore=True, diarization=False
        )
        pipeline.freeze({"min_duration": 0.1})
        return pipeline

    def validation(
        self,
        files: List[ProtocolFile],
        protocol: Protocol = None,
        subset: Subset = "development",
        warm_start: Dict = None,
        epoch: int = None,
    ):
        """Validation

        Validation consists in looking for the value of the peak threshold
        that maximizes the f-score of segmentation purity and coverage.
        """

        criterion = self.validation_criterion(protocol)
        pipeline = self.validation_pipeline()

        show_progress = {"unit": "file", "leave": False, "position": 2}

        optimizer = Optimizer(pipeline)
        iterations = optimizer.tune_iter(
            files, warm_start=warm_start, show_progress=show_progress
        )

        for i in trange(
            10,
            unit="iteration",
            position=1,
            leave=False,
            desc=f"epoch #{epoch} | optimizing threshold...",
        ):
            result = next(iterations)

        return {
            "metric": criterion,
            "minimize": False,
            "value": result["loss"],
            "params": result["params"],
        }
