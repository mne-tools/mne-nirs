#!/bin/bash -ef

if [ "${DEPS}" != "minimal" ]; then
	python -c 'import mne; mne.datasets.testing.data_path(verbose=True)';
fi

python -c 'from mne_nirs.datasets import fnirs_motor_group; fnirs_motor_group.data_path()'
python -c 'from mne_nirs.datasets import block_speech_noise; block_speech_noise.data_path()'
python -c 'from mne_nirs.datasets import audio_or_visual_speech; audio_or_visual_speech.data_path()'
python -c "from mne_nirs.datasets import camh_kf_fnirs_fingertapping; camh_kf_fnirs_fingertapping.data_path()"
python -c "from mne_nirs.datasets import snirf_with_aux; snirf_with_aux.data_path()"
python -c "from mne.datasets import fnirs_motor; fnirs_motor.data_path()"
