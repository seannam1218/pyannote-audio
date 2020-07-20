#!/usr/bin/env bash

export PYANNOTE_DATABASE_CONFIG=$GITHUB_WORKSPACE/tests/data/database.yml
export DEBUG=Debug.SpeakerDiarization.Debug
pyannote-audio train --to=2 $GITHUB_WORKSPACE/tests/cli/scd $DEBUG
pyannote-audio validate --to=last $GITHUB_WORKSPACE/tests/cli/scd/train/$DEBUG.train $DEBUG
pyannote-audio apply $GITHUB_WORKSPACE/tests/cli/scd/train/$DEBUG.train/validate_segmentation_fscore/$DEBUG.development $DEBUG
