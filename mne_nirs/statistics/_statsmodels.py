# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import pandas as pd
import numpy as np


def summary_to_dataframe(summary):
    '''Convert statsmodels summary to pandas dataframe.

    .. warning:: The summary has precision issues, use the numerical values
                 from it with caution.
    '''
    results = summary.tables[1]
    if type(results) is not pd.core.frame.DataFrame:
        results = summary.tables[1].as_html()
        results = pd.read_html(results, header=0, index_col=0)[0]
    return results


def expand_summary_dataframe(summary):
    '''Expand dataframe index column in to individual columns'''

    # Determine new columns
    new_cols = summary.index[0].split(':')
    col_names = []
    for col in new_cols:
        col_name = col.split('[')[0]
        summary[col_name] = 'NaN'
        col_names.append(col_name)

    # Fill in values
    indices = summary.copy(deep=True).index
    if 'Group Var' in summary.index:
        summary = summary[:-1]
        indices = summary.index
    for row_idx, row in enumerate(indices):
        col_vals = row.split(':')
        for col_idx, col in enumerate(col_names):
            if "]" in col_vals[col_idx]:
                val = col_vals[col_idx].split('[')[1].split(']')[0]
            else:
                val = col
            summary.at[row, col] = val

    summary = summary.copy()  # Copies required to suppress .loc warnings
    sum_copy = summary.copy(deep=True)
    key = 'P>|t|' if 'P>|t|' in summary.columns else 'P>|z|'
    float_p = [float(p) for p in sum_copy[key]]
    summary.loc[:, key] = float_p
    summary.loc[:, "Significant"] = False
    summary.loc[summary[key] < 0.05, 'Significant'] = True

    # Standardise returned column name, it seems to vary per test
    if 'Coef.' in summary.columns:
        summary.loc[:, "Coef."] = [float(c) for c in summary["Coef."]]
    elif 'coef' in summary.columns:
        summary = summary.rename(columns={"coef": "Coef."})

    return summary


_REPLACEMENTS = (
    ('P>|z|', 'pvalues'),
    ('Coef.', 'fe_params'),
    ('z', 'tvalues'),
    ('P>|t|', 'pvalues'),
)


def statsmodels_to_results(model, order=None):
    """
    Convert statsmodels summary to a dataframe.

    Parameters
    ----------
    model : statsmodels model output
        The output of a statsmodels analysis. For example rlm or mixedlm.
    order : array of str
        Requested order of the channels.

    Returns
    -------
    df : Pandas dataframe.
        Data frame with the results from the stats model.
    """
    from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
    from scipy.stats.distributions import norm
    df = summary_to_dataframe(model.summary())
    # deal with numerical precision loss in at least some of the values
    for col, attr in _REPLACEMENTS:
        if col in df.columns:
            df[col] = getattr(model, attr, df[col])

    # This one messes up the standard error and quartiles, too
    if isinstance(model, MixedLMResultsWrapper):
        sl = slice(model.k_fe)
        mu = np.asarray(df.iloc[sl, df.columns == 'Coef.'])[:, 0]
        # Adapted from statsmodels, see
        # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/regression/mixed_linear_model.py#L2710-L2736  # noqa: E501
        stderr = np.sqrt(np.diag(model.cov_params()[sl]))
        df.iloc[sl, df.columns == 'Std.Err.'] = stderr
        # Confidence intervals
        qm = -norm.ppf(0.05 / 2)
        df.iloc[sl, df.columns == '[0.025'] = mu - qm * stderr
        df.iloc[sl, df.columns == '0.975]'] = mu + qm * stderr
        # All random effects variances and covariances
        sdf = np.zeros((model.k_re2 + model.k_vc, 2))
        jj = 0
        for i in range(model.k_re):
            for j in range(i + 1):
                sdf[jj, 0] = np.asarray(model.cov_re)[i, j]
                sdf[jj, 1] = np.sqrt(model.scale) * model.bse[model.k_fe + jj]
                jj += 1

        # Variance components
        for i in range(model.k_vc):
            sdf[jj, 0] = model.vcomp[i]
            sdf[jj, 1] = np.sqrt(model.scale) * model.bse[model.k_fe + jj]
            jj += 1

        df.iloc[model.k_fe:, df.columns == 'Coef.'] = sdf[:, 0]
        df.iloc[model.k_fe:, df.columns == 'Std.Err.'] = sdf[:, 1]

    df = expand_summary_dataframe(df)

    if order is not None:
        df['old_index'] = df.index
        df = df.set_index('ch_name')
        df = df.loc[order, :]
        df['ch_name'] = df.index
        df.index = df['old_index']
        df.drop(columns='old_index', inplace=True)
        df.rename_axis(None, inplace=True)

    return df
