#!/bin/bash -ef

mkdir -p ~/mne_data/fOLD
cd ~/mne_data/fOLD
curl -L https://github.com/nirx/fOLD-public/archive/master.zip --output data.zip
unzip data.zip
rm data.zip
FOLD_VAR="MNE_NIRS_FOLD_PATH"
FOLD_PATH="~/mne_data/fOLD/fOLD-public-master/Supplementary"
if [ "${CIRCLECI}" == "true" ]; then
    echo "export ${FOLD_VAR}=${FOLD_PATH}" >> ${BASH_ENV};
elif [ "${GITHUB_ACTIONS}" == "true" ]; then
    echo "${FOLD_VAR}=${FOLD_PATH}" >> ${GITHUB_ENV};
elif [ "${AZURE_CI}" == "true" ]; then
    echo "##vso[task.setvariable variable=${FOLD_VAR}]${FOLD_PATH}"
fi
