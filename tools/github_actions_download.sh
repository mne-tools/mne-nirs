#!/bin/bash -ef

if [ "${DEPS}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
fi

python -c 'from mne_nirs.datasets import fnirs_motor_group; fnirs_motor_group.data_path(verbose=True)'
python -c 'from mne_nirs.datasets import block_speech_noise; block_speech_noise.data_path(verbose=True)'
python -c 'from mne_nirs.datasets import audio_or_visual_speech; audio_or_visual_speech.data_path(verbose=True)'