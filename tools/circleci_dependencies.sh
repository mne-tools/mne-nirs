#!/bin/bash -ef


python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --progress-bar off -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt git+https://github.com/mne-tools/mne-python.git@main "mne-qt-browser[opengl] @ git+https://github.com/mne-tools/mne-qt-browser.git@main"
python -m pip install -e .
