#!/bin/bash

set -eo pipefail

# Write data_key.txt, a hash of everything that determines what
# tools/github_actions_download.sh puts in ~/mne_data, for use in CI cache keys:
# the MNE datasets as defined by the *installed* MNE (which can differ between the
# stable and dev jobs), and the mne-nirs dataset definitions (URLs and hashes)
{
    python -c "from mne.datasets.config import MNE_DATASETS as d; print(d['testing'], d['fnirs_motor'])"
    git ls-files "mne_nirs/datasets/*.py" | grep -v /tests/ | sort | xargs cat
} | sha256sum | cut -d " " -f 1 | tee data_key.txt
