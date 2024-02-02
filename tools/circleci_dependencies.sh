#!/bin/bash

set -eo pipefail
python -m pip install --upgrade pip setuptools wheel
echo "Cloning and installing mne-python"
git clone https://github.com/mne-tools/mne-python.git
cd mne-python
pip install -e .
cd ..
echo "Installing dependencies"
python -m pip install --upgrade --progress-bar off -r requirements.txt -r requirements_testing.txt -r requirements_doc.txt "mne-qt-browser[opengl] @ git+https://github.com/mne-tools/mne-qt-browser.git@main"
python -m pip install -e .
