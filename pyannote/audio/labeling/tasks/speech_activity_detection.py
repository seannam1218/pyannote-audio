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

"""Speech activity detection"""

import numpy as np
from typing import List, Dict
from tqdm import trange
from tqdm import tqdm

from torch.utils.data import IterableDataset

from pyannote.core import Segment, Timeline
from pyannote.core.utils.numpy import one_hot_encoding

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution


from pyannote.audio.pipeline import (
    SpeechActivityDetection as SpeechActivityDetectionPipeline,
)
from pyannote.pipeline import Optimizer
from pyannote.database import ProtocolFile
from pyannote.database import Protocol
from pyannote.database import Subset


from .base import LabelingDataset


class SpeechActivityDetection(BaseTask):
    """Speech activity detection"""

    problem = Problem.MULTI_CLASS_CLASSIFICATION
    resolution_input = Resolution.FRAME
    resolution_output = Resolution.FRAME

    def prepare_metadata(self, files: List[ProtocolFile]) -> Dict:

        output_resolution = self.model.get_resolution()

        for f in tqdm(iterable=files, desc="Loading training metadata", unit="file"):

            f["__duration"] = sum(
                s.duration for s in f["annotated"] if s.duration > self.hparams.duration
            )

            y = one_hot_encoding(
                f["annotation"],
                Timeline(segments=[Segment(0, f["duration"])]),
                output_resolution,
                mode="center",
            )

            y.data = np.int64(np.sum(y.data, axis=1, keepdims=True) > 0)
            y.labels = [
                "speech",
            ]
            f["__target"] = y
            del f["annotation"]

        return {
            "classes": ["non_speech", "speech"],
            "epoch_duration": sum(f["__duration"] for f in files),
            "files": [dict(f) for f in files],
        }

    def train_dataset(self) -> IterableDataset:
        return LabelingDataset(self)

    @staticmethod
    def validation_criterion(protocol: Protocol):
        return "detection_fscore"

    def validation_pipeline(self):
        pipeline = SpeechActivityDetectionPipeline(
            scores="@scores", fscore=True, hysteresis=False
        )
        pipeline.freeze(
            {
                "min_duration_on": 0.1,
                "min_duration_off": 0.1,
                "pad_onset": 0.0,
                "pad_offset": 0.0,
            }
        )
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

        Validation consists in looking for the value of the detection threshold
        that maximizes the f-score of recall and precision.
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


# class DomainAwareSpeechActivityDetection(SpeechActivityDetection):
#     """Domain-aware speech activity detection

#     Trains speech activity detection and domain classification jointly.

#     Parameters
#     ----------
#     domain : `str`, optional
#         Batch key to use as domain. Defaults to 'domain'.
#         Could be 'database' or 'uri' for instance.
#     attachment : `int`, optional
#         Intermediate level where to attach the domain classifier.
#         Defaults to -1. Passed to `return_intermediate` in models supporting it.
#     rnn: `dict`, optional
#         Parameters of the RNN used in the domain classifier.
#         See `pyannote.audio.models.models.RNN` for details.
#     domain_loss : `str`, optional
#         Loss function to use. Defaults to 'NLLLoss'.
#     """

#     DOMAIN_PT = "{train_dir}/weights/{epoch:04d}.domain.pt"

#     def __init__(
#         self, domain="domain", attachment=-1, rnn=None, domain_loss="NLLLoss", **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.domain = domain
#         self.attachment = attachment

#         if rnn is None:
#             rnn = dict()
#         self.rnn = rnn

#         self.domain_loss = domain_loss
#         if self.domain_loss == "NLLLoss":
#             # Default value
#             self.domain_loss_ = nn.NLLLoss()
#             self.activation_ = nn.LogSoftmax(dim=1)

#         elif self.domain_loss == "MSELoss":
#             self.domain_loss_ = nn.MSELoss()
#             self.activation_ = nn.Sigmoid()

#         else:
#             msg = f"{domain_loss} has not been implemented yet."
#             raise NotImplementedError(msg)

#     def more_parameters(self):
#         """Initialize trainable trainer parameters

#         Yields
#         ------
#         parameter : nn.Parameter
#             Trainable trainer parameters
#         """

#         domain_classifier_rnn = RNN(
#             n_features=self.model.intermediate_dimension(self.attachment), **self.rnn
#         )

#         n_classes = len(self.specifications[self.domain]["classes"])
#         domain_classifier_linear = nn.Linear(
#             domain_classifier_rnn.dimension, n_classes, bias=True
#         ).to(self.device)

#         self.domain_classifier_ = nn.Sequential(
#             domain_classifier_rnn, domain_classifier_linear
#         ).to(self.device)

#         # TODO: check if we really need to do this .to(self.device) twice

#         return self.domain_classifier_.parameters()

#     def load_more(self, model_pt=None) -> bool:
#         """Load classifier from disk"""

#         if model_pt is None:
#             domain_pt = self.DOMAIN_PT.format(
#                 train_dir=self.train_dir_, epoch=self.epoch_
#             )
#         else:
#             domain_pt = model_pt.with_suffix(".domain.pt")

#         domain_classifier_state = torch.load(
#             domain_pt, map_location=lambda storage, loc: storage
#         )
#         self.domain_classifier_.load_state_dict(domain_classifier_state)

#         # FIXME add support for different domains
#         return True

#     def save_more(self):
#         """Save domain classifier to disk"""

#         domain_pt = self.DOMAIN_PT.format(train_dir=self.train_dir_, epoch=self.epoch_)
#         torch.save(self.domain_classifier_.state_dict(), domain_pt)

#     def batch_loss(self, batch):
#         """Compute loss for current `batch`

#         Parameters
#         ----------
#         batch : `dict`
#             ['X'] (`numpy.ndarray`)
#             ['y'] (`numpy.ndarray`)

#         Returns
#         -------
#         batch_loss : `dict`
#             ['loss'] (`torch.Tensor`) : Loss
#         """

#         # forward pass
#         X = torch.tensor(batch["X"], dtype=torch.float32, device=self.device_)
#         fX, intermediate = self.model_(X, return_intermediate=self.attachment)

#         # speech activity detection
#         fX = fX.view((-1, self.n_classes_))
#         target = (
#             torch.tensor(batch["y"], dtype=torch.int64, device=self.device_)
#             .contiguous()
#             .view((-1,))
#         )

#         weight = self.weight
#         if weight is not None:
#             weight = weight.to(device=self.device_)
#         loss = self.loss_func_(fX, target, weight=weight)

#         # domain classification
#         domain_target = torch.tensor(
#             batch[self.domain], dtype=torch.int64, device=self.device_
#         )

#         domain_scores = self.activation_(self.domain_classifier_(intermediate))

#         domain_loss = self.domain_loss_(domain_scores, domain_target)

#         return {
#             "loss": loss + domain_loss,
#             "loss_domain": domain_loss,
#             "loss_task": loss,
#         }


# class DomainAdversarialSpeechActivityDetection(DomainAwareSpeechActivityDetection):
#     """Domain Adversarial speech activity detection

#     Parameters
#     ----------
#     domain : `str`, optional
#         Batch key to use as domain. Defaults to 'domain'.
#         Could be 'database' or 'uri' for instance.
#     attachment : `int`, optional
#         Intermediate level where to attach the domain classifier.
#         Defaults to -1. Passed to `return_intermediate` in models supporting it.
#     alpha : `float`, optional
#         Coefficient multiplied with the domain loss
#     """

#     def __init__(self, domain="domain", attachment=-1, alpha=1.0, **kwargs):
#         super().__init__(domain=domain, attachment=attachment, **kwargs)
#         self.alpha = alpha
#         self.gradient_reversal_ = GradientReversal()

#     def batch_loss(self, batch):
#         """Compute loss for current `batch`

#         Parameters
#         ----------
#         batch : `dict`
#             ['X'] (`numpy.ndarray`)
#             ['y'] (`numpy.ndarray`)

#         Returns
#         -------
#         batch_loss : `dict`
#             ['loss'] (`torch.Tensor`) : Loss
#         """
#         # forward pass
#         X = torch.tensor(batch["X"], dtype=torch.float32, device=self.device_)

#         fX, intermediate = self.model_(X, return_intermediate=self.attachment)

#         # speech activity detection
#         fX = fX.view((-1, self.n_classes_))

#         target = (
#             torch.tensor(batch["y"], dtype=torch.int64, device=self.device_)
#             .contiguous()
#             .view((-1,))
#         )

#         weight = self.weight
#         if weight is not None:
#             weight = weight.to(device=self.device_)

#         loss = self.loss_func_(fX, target, weight=weight)

#         # domain classification
#         domain_target = torch.tensor(
#             batch[self.domain], dtype=torch.int64, device=self.device_
#         )

#         domain_scores = self.activation_(
#             self.domain_classifier_(self.gradient_reversal_(intermediate))
#         )

#         if self.domain_loss == "MSELoss":
#             # One hot encode domain_target for Mean Squared Error Loss
#             nb_domains = domain_scores.shape[1]
#             identity_mat = torch.sparse.torch.eye(nb_domains, device=self.device_)
#             domain_target = identity_mat.index_select(dim=0, index=domain_target)

#         domain_loss = self.domain_loss_(domain_scores, domain_target)

#         return {
#             "loss": loss + self.alpha * domain_loss,
#             "loss_domain": domain_loss,
#             "loss_task": loss,
#         }
