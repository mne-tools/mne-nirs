#!/bin/bash -ef

STD_ARGS="--progress-bar off --upgrade"
EXTRA_ARGS=""
if [ ! -z "$CONDA_ENV" ]; then
	pip uninstall -yq mne-nirs
else
	# Changes here should also go in the interactive_test CircleCI job
	python -m pip install $STD_ARGS pip setuptools wheel
	echo "NumPy/SciPy/pandas etc."
	pip install $STD_ARGS --pre --only-binary ":all:" --default-timeout=60 --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" numpy scipy scikit-learn pandas matplotlib pillow statsmodels
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps  --default-timeout=60 -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" dipy
	echo "H5py"
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py
	echo "nilearn"
	pip install $STD_ARGS --pre --only-binary ":all:" nilearn
	echo "VTK"
	pip install $STD_ARGS --pre --only-binary ":all:" vtk
	python -c "import vtk"
	echo "PyVista"
	pip install --progress-bar off git+https://github.com/pyvista/pyvista
	echo "pyvistaqt"
	pip install --progress-bar off git+https://github.com/pyvista/pyvistaqt
	echo "imageio-ffmpeg, xlrd, mffpy"
	pip install --progress-bar off --pre imageio-ffmpeg xlrd mffpy
	if [ "$OSTYPE" == "darwin"* ]; then
	  echo "pyobjc-framework-Cocoa"
	  pip install --progress-bar off pyobjc-framework-Cocoa>=5.2.0
	fi
	EXTRA_ARGS="--pre"
fi

if [ "${MNEPYTHON}" == "dev" ]; then
	MNE_BRANCH="main"
else
	MNE_BRANCH="maint/1.2"
fi
echo "MNE"
pip install $STD_ARGS $EXTRA_ARGS git+https://github.com/mne-tools/mne-python.git@${MNE_BRANCH}

if [ -z "$CONDA_ENV" ]; then
	echo "requirements.txt"
	pip install $STD_ARGS $EXTRA_ARGS --progress-bar off -r requirements.txt
fi

echo "requirements_testing.txt"
pip install --progress-bar off -r requirements_testing.txt
