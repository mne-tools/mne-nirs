#!/bin/bash -ef
ls
pytest --cov=mne_nirs --cov-report xml -vv "mne_nirs/"
ls
ls mne_nirs