#!/bin/bash

set -eo pipefail

# Versions of the datasets that CI downloads, in a form usable in cache keys:
# - TESTING_VERSION: the MNE testing dataset release of the *installed* MNE (which can
#   differ between the stable and dev jobs)
# - OTHER_DATA_VERSION: a hash of the definitions (URLs and hashes) of every other
#   dataset, i.e., the MNE ones we use and all of the mne-nirs ones
TESTING_VERSION=$(python -c "from mne.datasets.config import RELEASES; print(RELEASES['testing'].replace('.', '-'))")
OTHER_DATA_VERSION=$({
    python -c "from mne.datasets.config import MNE_DATASETS as d; print(d['fnirs_motor'], d['sample'])"
    git ls-files "mne_nirs/datasets/*.py" | grep -v /tests/ | sort | xargs cat
} | sha256sum | cut -c 1-16)
if [ -n "$GITHUB_ENV" ]; then
    echo "TESTING_VERSION=$TESTING_VERSION" | tee -a $GITHUB_ENV
    echo "OTHER_DATA_VERSION=$OTHER_DATA_VERSION" | tee -a $GITHUB_ENV
elif [ -n "$CIRCLECI" ]; then
    echo "$TESTING_VERSION" | tee testing_version.txt
    echo "$OTHER_DATA_VERSION" | tee other_data_version.txt
else
    echo "TESTING_VERSION=$TESTING_VERSION"
    echo "OTHER_DATA_VERSION=$OTHER_DATA_VERSION"
fi
