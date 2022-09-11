#!/bin/bash -ef

if [ "${DEPS}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
fi

python -c 'from mne_nirs.datasets import fnirs_motor_group; fnirs_motor_group.data_path(force_update=True)'
python -c 'from mne_nirs.datasets import block_speech_noise; block_speech_noise.data_path(force_update=True)'
python -c 'from mne_nirs.datasets import audio_or_visual_speech; audio_or_visual_speech.data_path(force_update=True)'

python -c 'from mne.datasets import testing; print(testing.data_path(download=False))'
python -c 'from mne.datasets import testing; testing_path = testing.data_path(download=False); print(testing_path / "/NIRx/nirscout/nirx_15_2_recording_w_short")'

ls /home/runner/mne_data
