# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose

import pytest
import pandas as pd
import statsmodels.formula.api as smf

from ...simulation import simulate_nirs_raw
from ...experimental_design import make_first_level_design_matrix
from ...statistics import run_GLM, statsmodels_to_results
from ...utils._io import glm_to_tidy


@pytest.mark.parametrize('func', ('mixedlm', 'ols', 'rlm'))
@pytest.mark.filterwarnings('ignore:.*optimization.*:')
@pytest.mark.filterwarnings('ignore:.*on the boundary.*:')
@pytest.mark.filterwarnings('ignore:.*The Hessian matrix at the estimated.*:')
def test_statsmodel_to_df(func):
    func = getattr(smf, func)
    np.random.seed(0)

    amplitude = 1.432

    df_cha = pd.DataFrame()
    for n in range(5):

        raw = simulate_nirs_raw(sfreq=3., amplitude=amplitude,
                                sig_dur=300., stim_dur=5.,
                                isi_min=15., isi_max=45.)
        raw._data += np.random.normal(0, np.sqrt(1e-12), raw._data.shape)
        design_matrix = make_first_level_design_matrix(raw, stim_dur=5.0)
        glm_est = run_GLM(raw, design_matrix)
        with pytest.warns(RuntimeWarning, match='Non standard source detect'):
            cha = glm_to_tidy(raw, glm_est, design_matrix)
        cha["ID"] = '%02d' % n
        df_cha = df_cha.append(cha)
    df_cha["theta"] = df_cha["theta"] * 1.0e6
    roi_model = func("theta ~ -1 + Condition", df_cha,
                     groups=df_cha["ID"]).fit()
    df = statsmodels_to_results(roi_model)
    assert type(df) == pd.DataFrame
    assert_allclose(df["Coef."]["Condition[A]"], amplitude, rtol=0.1)
    assert df["Significant"]["Condition[A]"]
    assert df.shape == (8, 8)

    roi_model = smf.rlm("theta ~ -1 + Condition", df_cha,
                        groups=df_cha["ID"]).fit()
    df = statsmodels_to_results(roi_model)
    assert type(df) == pd.DataFrame
    assert_allclose(df["Coef."]["Condition[A]"], amplitude, rtol=0.1)
    assert df["Significant"]["Condition[A]"]
    assert df.shape == (8, 8)
