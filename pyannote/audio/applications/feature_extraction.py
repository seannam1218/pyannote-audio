#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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
Feature extraction

Usage:
  pyannote-speech-feature [--robust --database=<db.yml>] <experiment_dir> <database.task.protocol>
  pyannote-speech-feature check [--database=<db.yml>] <experiment_dir> <database.task.protocol>
  pyannote-speech-feature -h | --help
  pyannote-speech-feature --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --robust                   When provided, skip files for which feature extraction fails.
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process
    (e.g. MFCCs).

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives
    ...................................................................

"""

import yaml
import h5py
import os.path
import numpy as np
from docopt import docopt

import pyannote.core
import pyannote.database
from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier
from pyannote.database import get_protocol

from pyannote.audio.util import mkdir_p
from pyannote.audio.features.utils import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.utils import PyannoteFeatureExtractionError


def extract(protocol_name, file_finder, experiment_dir, robust=False):

    protocol = get_protocol(protocol_name, progress=False)

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    sliding_window = feature_extraction.sliding_window()
    dimension = feature_extraction.dimension()

    # create metadata file at root that contains
    # sliding window and dimension information
    path = Precomputed.get_config_path(experiment_dir)
    f = h5py.File(path)
    f.attrs['start'] = sliding_window.start
    f.attrs['duration'] = sliding_window.duration
    f.attrs['step'] = sliding_window.step
    f.attrs['dimension'] = dimension
    f.close()

    for subset in ['development', 'test', 'train']:

        try:
            protocol.progress = False
            file_generator = getattr(protocol, subset)()
            first_item = next(file_generator)
        except NotImplementedError as e:
            continue

        protocol.progress = True
        file_generator = getattr(protocol, subset)()

        for current_file in file_generator:

            try:
                current_file['audio'] = file_finder(current_file)
            except ValueError as e:
                if not robust:
                    raise e
                print(e)
                continue

            uri = get_unique_identifier(current_file)
            path = Precomputed.get_path(experiment_dir, current_file)

            if os.path.exists(path):
                continue

            try:
                features = feature_extraction(current_file)
            except PyannoteFeatureExtractionError as e:
                msg = 'Feature extraction failed for file "{uri}".'
                msg = msg.format(uri=uri)
                if not robust:
                    raise PyannoteFeatureExtractionError(msg)
                print(msg)
                continue

            if features is None:
                msg = 'Feature extraction returned None for file "{uri}".'
                msg = msg.format(uri=uri)
                if not robust:
                    raise PyannoteFeatureExtractionError(msg)
                print(msg)
                continue

            data = features.data

            if np.any(np.isnan(data)):
                msg = 'Feature extraction returned NaNs for file "{uri}".'
                msg = msg.format(uri=uri)
                if not robust:
                    raise PyannoteFeatureExtractionError(msg)
                print(msg)
                continue

            # create parent directory
            mkdir_p(os.path.dirname(path))

            f = h5py.File(path)
            f.attrs['start'] = sliding_window.start
            f.attrs['duration'] = sliding_window.duration
            f.attrs['step'] = sliding_window.step
            f.attrs['dimension'] = dimension
            f.create_dataset('features', data=data)
            f.close()

def check(protocol_name, file_finder, experiment_dir):

    protocol = get_protocol(protocol_name)
    precomputed = Precomputed(experiment_dir)

    for subset in ['development', 'test', 'train']:

        try:
            file_generator = getattr(protocol, subset)()
            first_item = next(file_generator)
        except NotImplementedError as e:
            continue

        for current_file in getattr(protocol, subset)():

            try:
                audio = file_finder(current_file)
                current_file['audio'] = audio
            except ValueError as e:
                print(e)
                continue

            duration = get_audio_duration(current_file)

            try:
                features = precomputed(current_file)
            except PyannoteFeatureExtractionError as e:
                print(e)
                continue

            if not np.isclose(duration,
                              features.getExtent().duration,
                              atol=1.):
                uri = get_unique_identifier(current_file)
                print('Duration mismatch for "{uri}"'.format(uri=uri))


def main():

    arguments = docopt(__doc__, version='Feature extraction')

    db_yml = os.path.expanduser(arguments['--database'])
    file_finder = FileFinder(db_yml)

    protocol_name = arguments['<database.task.protocol>']
    experiment_dir = arguments['<experiment_dir>']

    if arguments['check']:
        check(protocol_name, file_finder, experiment_dir)
    else:
        robust = arguments['--robust']
        extract(protocol_name, file_finder, experiment_dir, robust=robust)
