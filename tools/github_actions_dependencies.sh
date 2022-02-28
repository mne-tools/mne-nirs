#!/bin/bash -ef

STD_ARGS="--progress-bar off --upgrade"
EXTRA_ARGS=""
if [ ! -z "$CONDA_DEPENDENCIES" ]; then
	conda install -y $CONDA_DEPENDENCIES
else
	# Changes here should also go in the interactive_test CircleCI job
	python -m pip install $STD_ARGS pip setuptools wheel
	echo "Numpy"
	pip uninstall -yq numpy
	echo "Date utils"
	# https://pip.pypa.io/en/latest/user_guide/#possible-ways-to-reduce-backtracking-occurring
	pip install $STD_ARGS --pre --only-binary ":all:" python-dateutil pytz joblib threadpoolctl six
	echo "PyQt5"
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps --extra-index-url https://www.riverbankcomputing.com/pypi/simple PyQt5 PyQt5-sip PyQt5-Qt5
	echo "NumPy/SciPy/pandas etc."
	# TODO: Currently missing dipy for 3.10 https://github.com/dipy/dipy/issues/2489
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -i "https://pypi.anaconda.org/scipy-wheels-nightly/simple" numpy scipy pandas "scikit-learn>=0.24.2" statsmodels
	echo "H5py, pillow, matplotlib"
	pip install $STD_ARGS --pre --only-binary ":all:" --no-deps -f "https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com" h5py pillow matplotlib
	echo "Numba, nilearn"
	pip install $STD_ARGS --pre --only-binary ":all:" numba llvmlite https://github.com/nilearn/nilearn/zipball/main
	echo "VTK"
	# Have to use our own version until VTK releases a 3.10 build
	if `python -c "import sys; assert sys.version_info >= (3, 10), sys.version_info"`; then
		wget -q https://osf.io/hjyvx/download -O vtk-9.1.20211213.dev0-cp310-cp310-linux_x86_64.whl
		pip install $STD_ARGS --pre --only-binary ":all:" vtk-9.1.20211213.dev0-cp310-cp310-linux_x86_64.whl
	else
		pip install $STD_ARGS --pre --only-binary ":all:" vtk
	fi;
	echo "PyVista"
	pip install --progress-bar off https://github.com/pyvista/pyvista/zipball/main
	echo "pyvistaqt"
	pip install --progress-bar off https://github.com/pyvista/pyvistaqt/zipball/main
	echo "imageio-ffmpeg, xlrd, mffpy"
	pip install --progress-bar off --pre imageio-ffmpeg xlrd mffpy
	if [ "$OSTYPE" == "darwin"* ]; then
	  echo "pyobjc-framework-Cocoa"
	  pip install --progress-bar off pyobjc-framework-Cocoa>=5.2.0
	fi
	EXTRA_ARGS="--pre"
fi

pip install $STD_ARGS $EXTRA_ARGS -r requirements.txt
pip install $STD_ARGS $EXTRA_ARGS -r requirements_testing.txt
if [ "${MNEPYTHON}" == "dev" ]; then
	pip install $STD_ARGS $EXTRA_ARGS https://github.com/mne-tools/mne-python/zipball/main
fi
