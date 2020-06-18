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


"""
Neural building blocks for speaker diarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
  pyannote-audio train    [--cpu | --gpu] [options] <root>     <protocol>
  pyannote-audio validate [--cpu | --gpu] [options] <train>    <protocol>
  pyannote-audio apply    [--cpu | --gpu] [options] <validate> <protocol>
  pyannote-audio -h | --help
  pyannote-audio --version

This command line tool can be used to train, validate, and apply neural networks
for most module of a speaker diarization pipeline

Running a complete speech activity detection experiment on the provided
"debug" dataset would go like this:

    * Run experiment on this pyannote.database protocol
      $ export DATABASE=Debug.SpeakerDiarization.Debug

    * This directory will contain experiments artifacts:
      $ mkdir my_experiment && cd my_experiment

    * A unique configuration file describes the experiment hyper-parameters
      (see "Configuration file" below for details):
      $ edit config.yml

    * This will train the model on the training set:
      $ pyannote-audio train ${PWD} ${DATABASE}

    * Training artifacts (including model weights) are stored in a sub-directory
      whose name makes it clear which dataset and subset (train, by default)
      were used for training the model.
      $ cd train/${DATABASE}.train

    * This will validate the model on the development set:
      $ pyannote-audio validate ${PWD} ${DATABASE}

    * Validation artifacts (including the selection of the best epoch) are
      stored in a sub-directory named after the dataset and subset (development,
      by default) used for validating the model.
      $ cd validate/${DATABASE}.development

    * This will apply the best model (according to the validation step) to the
      test set:
      $ pyannote-audio apply ${PWD} ${DATABASE}

    * Inference artifacts are stored in a sub-directory whose name makes it
      clear which epoch has been used (e.g. apply/0125). Artifacts include:
        * raw output of the best model (one numpy array per file  than can be
          loaded with pyannote.audio.features.Precomputed API and handled with
          pyannote.core.SlidingWindowFeature API)
        * (depending on the task) a file "${DATABASE}.test.rttm" containing the
          post-processing of raw output.
        * (depending on the task) a file "${DATABASE}.test.eval" containing the
          evaluation result computed with pyannote.metrics.

pyannote.database support
~~~~~~~~~~~~~~~~~~~~~~~~~

PYANNOTE_DATABASE_CONFIG=

Configuration file <root>/config.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Reproducible research is facilitated by the systematic use of configuration
    files stored in <root>/config.yml in YAML format.

    .......................... <root>/config.yml ..........................
    task:
        name:
        params:

    feature_extraction:
        name:
        params:

    augmentation:
        name:
        params:

    architecture:
        name:
        params:

    scheduler:
        name:
        params:

    preprocessors:

    callbacks:
    ...................................................................

    File <root>/config.yml is mandatory, unless option --pretrained is used.

    When fine-tuning a model with option --pretrained=<model>, one can omit it
    and the original <model> configuration file is used instead. If (a possibly
    partial) <root>/config.yml file is provided anyway, it is used to override
    <model> configuration file.

Tensorboard support
~~~~~~~~~~~~~~~~~~~

    A bunch of metrics are logged during training and validation (e.g. loss,
    learning rate, computation time, validation metric). They can be visualized
    using tensorboard:

        $ tensorboard --logdir=<root>

Common options
~~~~~~~~~~~~~~

  <root>                  Experiment root directory. Should contain config.yml
                          configuration file, unless --pretrained option is
                          used (for which config.yml is optional).

  <protocol>              Name of protocol to use for training, validation, or
                          inference. Have a look at pyannote.database
                          documentation for instructions on how to define a
                          protocol with your own dataset:
                          https://github.com/pyannote/pyannote-database

  <train>                 Path to <root> sub-directory containing training
                          artifacts (e.g. <root>/train/<protocol>.train)

  <validate>              Path to <train> sub-directory containing validation
                          artifacts (e.g. <train>/validate/<protocol>.development)
                          In case option --pretrained=<model> is used, the
                          output of the pretrained model is dumped into the
                          <validate> directory.

  --subset=<subset>       Subset to use for training (resp. validation,
                          inference). Defaults to "train" (resp. "development",
                          "test") for strict enforcement of machine learning
                          good practices.

  --gpu                   Run on all available GPUs. Use CUDA_VISIBLE_DEVICES 
                          environment variable to force using specific ones.

  --cpu                   Run on CPU. Defaults to using GPUs when available.

  --debug                 Run using PyTorch's anomaly detection. This will throw
                          an error if a NaN value is produced, and the stacktrace
                          will point to the origin of it. This option can
                          considerably slow execution.

  --from=<epoch>          Start training (resp. validating) at epoch <epoch>.
                          Use --from=last to start from last available epoch at
                          launch time. Not used for inference. [default: 0].

  --to=<epoch>            End training (resp. validating) at epoch <epoch>.
                          Use --end=last to validate until last available epoch
                          at launch time. Not used for inference [default: 100].

  --batch=<size>          Set batch size used for validation and inference.
                          Has no effect when training as this parameter should
                          be defined in the configuration file [default: 32].

  --step=<ratio>          Ratio of audio chunk duration used as step between
                          two consecutive audio chunks [default: 0.25]

  --parallel=<n_jobs>     Use at most that many threads for generating training
                          samples or validating files. Defaults to using all
                          CPUs but one.

Speaker embedding
~~~~~~~~~~~~~~~~~

  --duration=<duration>   Use audio chunks with that duration. Defaults to the
                          fixed duration used during training, when available.

  --metric=<metric>       Use this metric (e.g. "cosine" or "euclidean") to
                          compare embeddings. Defaults to the metric defined in
                          <root>/config.yml configuration file.

Pretrained model options
~~~~~~~~~~~~~~~~~~~~~~~~

  --pretrained=<model>    Warm start training with pre-trained model. Can be
                          either a path to an existing checkpoint (e.g.
                          <train>/weights/0050.pt) or the name of a model
                          available in torch.hub.list('pyannote/pyannote.audio')
                          This option can also be used to apply a pretrained
                          model. See description of <validate> for more details.

Validation options
~~~~~~~~~~~~~~~~~~

  --every=<epoch>         Validate model every <epoch> epochs [default: 1].

  --evergreen             Prioritize validation of most recent epoch.

  For speech activity and overlapped speech detection, validation consists in
  looking for the value of the detection threshold that maximizes the f-score
  of recall and precision.

  For speaker change detection, validation consists in looking for the value of
  the peak detection threshold that maximizes the f-score of purity and
  coverage:

  --diarization           Use diarization purity and coverage instead of
                          (default) segmentation purity and coverage.

  For speaker embedding and verification protocols, validation runs the actual
  speaker verification experiment (representing each recording by its average
  embedding) and reports equal error rate.

  For speaker embedding and diarization protocols, validation runs a speaker
  diarization pipeline based on oracle segmentation and "pool-linkage"
  agglomerative clustering of speech turns (represented by their average
  embedding), and looks for the threshold that maximizes the f-score of purity
  and coverage.

"""

import yaml
from docopt import docopt
from pathlib import Path
from typing import Tuple, Type
from argparse import Namespace

from pyannote.audio.train.task import BaseTask
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from pyannote.database import Preprocessors
from pyannote.database.custom import gather_loaders
from pyannote.audio.features.utils import get_audio_duration

import pytorch_lightning as pl


def load_hparams(hparams_yml: Path) -> Tuple[Type[BaseTask], Namespace, Preprocessors]:

    with open(hparams_yml, "r") as fp:
        configuration = yaml.load(fp, Loader=yaml.SafeLoader)

    task_section = configuration.pop("task")
    task_class = get_class_by_name(task_section["name"])
    for key, value in task_section["params"].items():
        configuration[key] = value

    # preprocessors
    preprocessors_section = configuration.pop("preprocessors", dict())
    preprocessors, loaders = dict(), dict()
    for key, preprocessor in preprocessors_section.items():
        #    key:
        #       name: package.module.ClassName
        #       params:
        #          param1: value1
        #          param2: value2
        if isinstance(preprocessor, dict):
            preprocessor_class = get_class_by_name(preprocessor["name"])
            preprocessors[key] = preprocessor_class(
                **preprocessor.get("params", dict())
            )
        #    key: /path/to/file.suffix
        else:
            loaders[key] = preprocessor
    preprocessors.update(gather_loaders(loaders))

    hparams = Namespace(**configuration)
    return task_class, hparams, preprocessors


def main():

    # TODO: update version automatically
    arg = docopt(__doc__, version="pyannote-audio 2.0")

    if arg["train"]:

        root_dir = Path(arg["<root>"]).expanduser().resolve(strict=True)
        config_yml = root_dir / "config.yml"
        task_class, hparams, preprocessors = load_hparams(config_yml)

        # initialize training set
        protocol_name = arg["<protocol>"]
        protocol = get_protocol(protocol_name, preprocessors=preprocessors)
        protocol.name = protocol_name
        subset = "train" if arg["--subset"] is None else arg["--subset"]

        # add "audio" and/or "duration" preprocessors when missing
        file = next(getattr(protocol, subset)())
        missing_preprocessors = dict()
        if "audio" not in file:
            missing_preprocessors["audio"] = FileFinder()
        if "duration" not in file:
            missing_preprocessors["duration"] = get_audio_duration
        if missing_preprocessors:
            preprocessors.update(missing_preprocessors)
            protocol = get_protocol(protocol_name, preprocessors=preprocessors)
            protocol.name = protocol_name

        # initialize trainer
        trainer_params = dict()
        trainer_params["checkpoint_callback"] = pl.callbacks.ModelCheckpoint(
            filepath=f"{root_dir}/train/{protocol_name}.{subset}/weights/{{epoch:04d}}",
            save_top_k=-1,
            save_last=True,
            period=1,
        )
        trainer_params["logger"] = pl.loggers.TensorBoardLogger(
            f"{root_dir}/train/{protocol_name}.{subset}", name="", version=""
        )
        if getattr(hparams, "batch_size", None) == "auto":
            trainer_params["auto_scale_batch_size"] = "power"

        # TODO: set "benchmark" to True if and only if all training chunks
        # have the same size
        trainer_params["benchmark"] = False

        # TODO: try this distributed thing...
        # trainer_params["distributed_backend"] = ...
        # trainer_params["num_nodes"] = ...
        # trainer_params["num_processes"] = ...

        # TODO: add a debug option that activates this parameter
        # trainer_params["fast_dev_run"] = ...
        # trainer_params["overfit_batches"] = ...
        # trainer_params["profiler"] = ...
        # trainer_params["limit_train_batches"] = ...

        # TODO: try this "precision" thing
        # trainer_params["precision"] = ...

        trainer_params["gpus"] = None if arg["--cpu"] else -1
        trainer_params["max_epochs"] = int(arg["--to"])

        resume = arg["--from"]
        if resume == "last":
            checkpoint = f"{root_dir}/train/{protocol_name}.{subset}/weights/last.ckpt"
        else:
            epoch = int(resume)
            if epoch == 0:
                checkpoint = None
            else:
                checkpoint = f"{root_dir}/train/{protocol_name}.{subset}/weights/epoch={epoch:04d}.ckpt"
        trainer_params["resume_from_checkpoint"] = checkpoint

        trainer = pl.Trainer(**trainer_params)

        if getattr(hparams, "learning_rate", "auto") == "auto":

            # initialize model with an arbitray learning rate (or lr_find will complain)
            hparams.learning_rate = 1e-3
            task = task_class(hparams, protocol=protocol, subset=subset)

            # suggest good learning rate
            lr_finder = trainer.lr_find(task, min_lr=1e-7, max_lr=10, num_training=50)
            suggested_lr = lr_finder.suggestion()

            # initialize model with suggested learning rate
            hparams.learning_rate = suggested_lr
            task = task_class(hparams, files=list(task.files))

        else:
            task = task_class(hparams, protocol=protocol, subset=subset)

        trainer.fit(task)

    # if arg["validate"]:

    #     train_dir = Path(arg["<train>"]).expanduser().resolve(strict=True)
    #     app = Application.from_train_dir(train_dir, training=False)

    #     params["subset"] = "development" if subset is None else subset

    #     start = arg["--from"]
    #     if start != "last":
    #         start = int(start)
    #     params["start"] = start

    #     end = arg["--to"]
    #     if end != "last":
    #         end = int(end)
    #     params["end"] = end

    #     params["every"] = int(arg["--every"])
    #     params["chronological"] = not arg["--evergreen"]
    #     params["batch_size"] = int(arg["--batch"])

    #     params["diarization"] = arg["--diarization"]

    #     duration = arg["--duration"]
    #     if duration is None:
    #         duration = getattr(app.task_, "duration", None)
    #         if duration is None:
    #             msg = (
    #                 "Task has no 'duration' defined. "
    #                 "Use '--duration' option to provide one."
    #             )
    #             raise ValueError(msg)
    #     else:
    #         duration = float(duration)
    #     params["duration"] = duration

    #     params["step"] = float(arg["--step"])

    #     if arg["emb"]:

    #         metric = arg["--metric"]
    #         if metric is None:
    #             metric = getattr(app.task_, "metric", None)
    #             if metric is None:
    #                 msg = (
    #                     "Approach has no 'metric' defined. "
    #                     "Use '--metric' option to provide one."
    #                 )
    #                 raise ValueError(msg)
    #         params["metric"] = metric

    #     # FIXME: parallel is broken in pyannote.metrics
    #     params["n_jobs"] = 1

    #     app.validate(protocol, **params)

    # if arg["apply"]:

    #     validate_dir = Path(arg["<validate>"]).expanduser().resolve(strict=True)

    #     params["subset"] = "test" if subset is None else subset
    #     params["batch_size"] = int(arg["--batch"])

    #     duration = arg["--duration"]
    #     if duration is not None:
    #         duration = float(duration)
    #     params["duration"] = duration

    #     params["step"] = float(arg["--step"])
    #     params["Pipeline"] = getattr(Application, "Pipeline", None)

    #     params["pretrained"] = arg["--pretrained"]

    #     apply_pretrained(validate_dir, protocol, **params)
