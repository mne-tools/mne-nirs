# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne.preprocessing.nirs import _validate_nirs_info


def plot_timechannel_quality_metric(raw, scores, times, threshold=0.1, title=None):
    """
    Plot time x channel based quality metrics.

    The left figure shows the raw score per channel and time.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    scores : array
        The quality metric scores.
    times : list of pairs
        Start and end time for each quality metric.
    threshold : float
        Value below which a segment will be marked as bad.
    title : str
        Title of plot. If not specified a default title will be used.

    Returns
    -------
    fig : figure
        Matplotlib figure displaying raw scores and thresholded scores.
    """
    # `scores` holds one row per fNIRS channel, in `raw` order.
    picks = np.sort(_validate_nirs_info(raw.info))
    ch_names = [raw.ch_names[pick] for pick in picks]
    cols = [np.round(t[0]) for t in times]

    if title is None:
        title = "Automated noisy channel detection: fNIRS"

    data_to_plot = pd.DataFrame(
        data=scores,
        columns=pd.Index(cols, name="Time (s)"),
        index=pd.Index(ch_names, name="Channel"),
    )

    n_chans = len(ch_names)
    vsize = 0.2 * n_chans

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(20, vsize), layout="constrained")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    sns.heatmap(
        data=data_to_plot,
        cmap="Reds_r",
        vmin=0,
        vmax=1,
        cbar_kws=dict(label="Score"),
        ax=ax[0],
    )
    [
        ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(times))
    ]
    ax[0].set_title("All Scores", fontweight="bold")
    markbad(ax[0], ch_names, raw.info["bads"])

    # Now, adjust the color range to highlight segments that exceeded the
    # limit.

    data_to_plot = pd.DataFrame(
        data=scores > threshold,
        columns=pd.Index(cols, name="Time (s)"),
        index=pd.Index(ch_names, name="Channel"),
    )
    sns.heatmap(
        data=data_to_plot,
        vmin=0,
        vmax=1,
        cmap="Reds_r",
        cbar_kws=dict(label="Score"),
        ax=ax[1],
    )
    [
        ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(times))
    ]
    ax[1].set_title("Scores < Limit", fontweight="bold")
    markbad(ax[1], ch_names, raw.info["bads"])

    return fig


def markbad(ax, ch_names, bads):
    [
        ax.axhline(y + 0.5, ls="solid", lw=2, color="black")
        for y in np.where([ch in bads for ch in ch_names])[0]
    ]

    return ax
