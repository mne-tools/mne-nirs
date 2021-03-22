#!/bin/bash -ef

pytest --cov=mne_nirs --cov-report xml -vv -m "not examples" "mne_nirs/"
