# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import pandas as pd


def summary_to_dataframe(summary):
    '''Convert statsmodels summary to pandas dataframe'''
    results_as_html = summary.tables[1].as_html()
    return pd.read_html(results_as_html, header=0, index_col=0)[0]


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
    for row_idx, row in enumerate(summary.index):
        col_vals = row.split(':')
        for col_idx, col in enumerate(col_names):
            val = col_vals[col_idx].split('[')[1].split(']')[0]
            summary[col][row_idx] = val

    summary["sig"] = [("NS", "S")[p < 0.05] for p in summary["P>|z|"]]

    return summary


def statsmodels_to_results(rlm_model):
    as_df = summary_to_dataframe(rlm_model.summary())
    as_df = expand_summary_dataframe(as_df)
    return as_df
