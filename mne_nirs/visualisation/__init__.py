"""NIRS specific plotting functions."""

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from ._plot_nirs_source_detector import plot_nirs_source_detector
from._plot_GLM_topo import (plot_glm_topo, plot_glm_contrast_topo,
                            plot_glm_group_topo)
from ._plot_quality_metrics import plot_timechannel_quality_metric