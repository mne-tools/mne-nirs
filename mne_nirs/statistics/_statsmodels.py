# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import pandas as pd


def summary_to_dataframe(summary):
    '''Convert statsmodels summary to pandas dataframe'''
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
    float_p = [float(p) for p in sum_copy["P>|z|"]]
    summary.loc[:, "P>|z|"] = float_p
    summary.loc[:, "Significant"] = False
    summary.loc[summary["P>|z|"] < 0.05, 'Significant'] = True

    # Standardise returned column name, it seems to vary per test
    if 'Coef.' in summary.columns:
        summary.loc[:, "Coef."] = [float(c) for c in summary["Coef."]]
    elif 'coef' in summary.columns:
        summary = summary.rename(columns={"coef": "Coef."})

    return summary


def statsmodels_to_results(model, order=None):
    """
    Convert statsmodels summary to a dataframe.

    Parameters
    ----------
    model : statsmodels model output
        The output of a statsmodels analysis. For example rlm or mixedlm.
    order : array of strings
        Requested order of the channels.

    Returns
    -------
    df : Pandas dataframe.
    """
    df = summary_to_dataframe(model.summary())
    df = expand_summary_dataframe(df)

    if order is not None:
        df['old_index'] = df.index
        df = df.set_index('ch_name')
        df = df.loc[order, :]
        df['ch_name'] = df.index
        df.index = df['old_index']
        df.drop(columns='old_index', inplace=True)

    return df
