#!/bin/bash -ef

pytest --cov=mne_nirs --cov-report xml -vv "mne_nirs/"
