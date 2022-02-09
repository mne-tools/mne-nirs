#!/bin/bash -ef

mkdir -p ~/mne_data/fOLD
chdir ~/mne_data/fOLD
wget -nc -O - https://github.com/nirx/fOLD-public/archive/master.zip | gunzip -
if [ "${CIRCLECI}" == "true" ]; then
    echo "export MNE_FNIRS_FOLD_PATH=~/mne_data/fOLD/fOLD-public-master/Supplementary";
elif [ "${GITHUB_ACTIONS}" == "true" ]; then
    echo "MNE_NIRS_FOLD_PATH=~/mne_data/fOLD/fOLD-public-master/Supplementary" >> ${GITHUB_ENV};
fi
