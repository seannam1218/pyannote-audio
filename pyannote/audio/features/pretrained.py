# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr

import warnings
from typing import Optional
from typing import Union
from typing import Text
from pathlib import Path

import torch
import yaml
import numpy as np

from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature

from pyannote.audio.train.task import Resolution

from pyannote.audio.augmentation import Augmentation
from pyannote.audio.features.base import FeatureExtraction

from pyannote.audio.train.task import BaseTask


class Pretrained(FeatureExtraction):
    """Pretrained model as feature extractor

    Parameters
    ----------
    validate_dir : Path
        Path to a validation directory.
    epoch : int, optional
        If provided, force loading this epoch.
        Defaults to reading epoch in validate_dir/params.yml.
    augmentation : Augmentation, optional
    duration : float, optional
        Use audio chunks with that duration. Defaults to the duration used for
        training, when available.
    step : float, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.25.
    batch_size : int, optional
        Batch size. Defaults to the batch size used for training when available
        or to 32 otherwise.
    device : optional
        Defaults to "cuda" when GPU is available, "cpu" otherwise.
    """

    # TODO: add progress bar (at least for demo purposes)

    def __init__(
        self,
        validate_dir: Path = None,
        epoch: int = None,
        augmentation: Optional[Augmentation] = None,
        duration: float = None,
        step: float = None,
        batch_size: int = None,
        device: Optional[Union[Text, torch.device]] = None,
        progress_hook=None,
    ):

        try:
            validate_dir = Path(validate_dir)
        except TypeError as e:
            msg = (
                f'"validate_dir" must be str, bytes or os.PathLike object, '
                f"not {type(validate_dir).__name__}."
            )
            raise TypeError(msg)

        validate_dir = validate_dir.expanduser().resolve(strict=epoch is None)

        train_dir = validate_dir.parents[1]
        hparams_yml = train_dir / "hparams.yaml"

        root_dir = train_dir.parents[1]
        config_yml = root_dir / "config.yml"

        task_class, hparams, preprocessors = BaseTask.load_config(
            config_yml, hparams_yml=hparams_yml
        )

        if epoch is None:
            validate_params_yml = validate_dir / "params.yml"
            with open(validate_params_yml, "r") as fp:
                validate_params = yaml.load(fp, Loader=yaml.SafeLoader)
            epoch = validate_params["epoch"]
            self.epoch_ = epoch
            self.pipeline_params_ = validate_params.get("params", dict())

        checkpoint_path = train_dir / "weights" / f"epoch={epoch:04d}.ckpt"
        task = task_class.load_from_checkpoint(
            str(checkpoint_path), map_location=lambda storage, loc: storage
        )

        super().__init__(
            augmentation=augmentation, sample_rate=task.feature_extraction.sample_rate,
        )
        task.feature_extraction.augmentation = augmentation

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.task_ = task.eval().to(device)

        # initialize chunks duration and batch size with that used during training
        self.duration = getattr(self.task_.hparams, "duration", None)
        self.min_duration = getattr(self.task_.hparams, "min_duration", self.duration)
        self.batch_size = getattr(self.task_.hparams, "batch_size", 32)

        # override chunks duration by user-provided value
        if duration is not None:
            # warn that this might be sub-optimal
            if self.duration is not None and not (
                self.min_duration <= duration <= self.duration
            ):
                if self.min_duration != self.duration:
                    msg = (
                        f"Model was trained with {self.min_duration:g}s to "
                        f"{self.duration:g}s chunks and is applied on "
                        f"{duration:g} chunks. This might lead to sub-optimal "
                        f"results."
                    )
                else:
                    msg = (
                        f"Model was trained with {self.duration:g}s chunks and "
                        f"is applied on {duration:g}s chunks. This might lead "
                        f"to sub-optimal results."
                    )
                warnings.warn(msg)
            # do it anyway
            self.duration = duration

        if step is None:
            step = 0.25
        self.step = step

        # override batch size by user-provided value
        if batch_size is not None:
            self.batch_size = batch_size

        self.progress_hook = progress_hook

    @property
    def duration(self):
        return self.duration_

    @duration.setter
    def duration(self, duration: float):
        self.duration_ = duration
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

    @property
    def step(self):
        return getattr(self, "step_", 0.25)

    @step.setter
    def step(self, step: float):
        self.step_ = step
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

    @property
    def classes(self):
        return self.task_.classes

    def get_dimension(self) -> int:
        try:
            dimension = self.task_.model.dimension
        except AttributeError:
            dimension = len(self.classes)
        return dimension

    def get_resolution(self) -> SlidingWindow:

        resolution = self.task_.model.resolution
        if resolution == Resolution.CHUNK:
            resolution = self.chunks_

        return resolution

    def get_features(self, y, sample_rate) -> np.ndarray:

        features = SlidingWindowFeature(
            self.task_.feature_extraction.get_features(y, sample_rate),
            self.task_.feature_extraction.sliding_window,
        )

        return self.task_.model.slide(
            features,
            self.chunks_,
            batch_size=self.batch_size,
            device=self.device,
            progress_hook=self.progress_hook,
        ).data

    def get_context_duration(self) -> float:
        # FIXME: add half window duration to context?
        return self.task_.feature_extraction_.get_context_duration()
