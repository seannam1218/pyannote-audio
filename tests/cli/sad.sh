#!/usr/bin/env bash

export PYANNOTE_DATABASE_CONFIG=$GITHUB_WORKSPACE/tests/data/database.yml
export DEBUG=Debug.SpeakerDiarization.Debug
pyannote-audio train --parallel=0 --to=4 $GITHUB_WORKSPACE/tests/cli/sad $DEBUG
pyannote-audio validate --from=2 --to=4 --every=2 $GITHUB_WORKSPACE/tests/cli/sad/train/$DEBUG.train $DEBUG
pyannote-audio apply $GITHUB_WORKSPACE/tests/cli/sad/train/$DEBUG.train/validate_detection_fscore/$DEBUG.development $DEBUG
