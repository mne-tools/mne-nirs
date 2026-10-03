# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from ._peak_power import peak_power
from ._scalp_coupling_segmented import scalp_coupling_index_windowed
from ._mayer import quantify_mayer_fooof
from .motion_removal import (
    auto_wavelet,
    detect_motion_artifacts,
    detect_motion_artifacts_by_channel,
    motion_artifact,
    motion_artifact_by_channel,
    motion_correct_spline,
    motion_correct_wavelet,
    motion_detect_and_correct_wavelet,
    spline,
    wavelet,
)
