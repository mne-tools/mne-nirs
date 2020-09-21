"""Statistical analysis."""

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from ._run_GLM import run_GLM, compute_contrast
from ._roi import glm_region_of_interest
from ._statsmodels import statsmodels_to_results
