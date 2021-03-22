#!/bin/bash -ef

pytest --cov=mne_nirs --cov-report xml -vv -m "examples" "mne_nirs/"
