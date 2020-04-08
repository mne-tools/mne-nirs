PKG_NAME=mne-nirs
PYTHON_NAME=mne_nirs
GH_NAME=rob-luke

unameOut="$(uname -s)"

if [[ "${unameOut}" == "Linux" ]]; then
        git grep -l 'mne_nirs' | xargs sed -i 's/mne_nirs/'"${PYTHON_NAME}"'/g'
        git grep -l 'mne-nirs' | xargs sed -i 's/mne-nirs/'"${PKG_NAME}"'/g'
        sed -i 's/mne-tools/'"${GH_NAME}"'/g' README.rst
        sed -i 's/mne-project-template/'"${PKG_NAME}"'/g' README.rst
        sed -i 's/mne-tools/'"${GH_NAME}"'/g' setup.py
        sed -i 's/mne-project-template/'"${PKG_NAME}"'/g' setup.py
else
    git grep -l 'mne_nirs' | xargs sed -i ' ' -e 's/mne_nirs/'"${PYTHON_NAME}"'/g'
    git grep -l 'mne-nirs' | xargs sed -i ' ' -e 's/mne-nirs/'"${PKG_NAME}"'/g'
    sed -i ' ' -e 's/mne-tools/'"${GH_NAME}"'/g' README.rst
    sed -i ' ' -e 's/mne-project-template/'"${PKG_NAME}"'/g' README.rst
    sed -i ' ' -e 's/mne-tools/'"${GH_NAME}"'/g' setup.py
    sed -i ' ' -e 's/mne-project-template/'"${PKG_NAME}"'/g' setup.py
fi
mv mne_nirs ${PYTHON_NAME}
