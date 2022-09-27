# Author: Robert Luke <code@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
import logging
import h5py
from scipy import interpolate
from pandas import DataFrame
from mne.io import Raw


def read_snirf_aux_data(fname: str, raw: Raw):
    """Read auxiliary data from SNIRF file.

    Reads the auxiliary channel data (e.g. heart rate data,
    accelerometer, etc). The auxiliary data will be resampled
    to match the raw data.

    Parameters
    ----------
    fname : str
        Path to the SNIRF data file.
    raw : Raw
        Instance of raw snirf data as created by read_raw_snirf.

    Returns
    -------
    fname : str
        Path to the SNIRF data file.
    """

    with h5py.File(fname, 'r') as dat:
        if 'nirs' in dat:
            basename = "nirs"
        elif 'nirs1' in dat:
            basename = "nirs1"
        else:
            raise RuntimeError("Data does not contain nirs field")

        all_keys = list(dat.get(basename).keys())
        aux_keys = [i for i in all_keys if i.startswith('aux')]
        aux_names = [_decode_name(dat.get(f'{basename}/{k}/name'))
                     for k in aux_keys]
        logging.debug(f"Found auxiliary channels {aux_names}")

        d = {'times': raw.times}
        for idx, aux in enumerate(aux_keys):
            aux_data = np.array(dat.get(f'{basename}/{aux}/dataTimeSeries'))
            aux_time = np.array(dat.get(f'{basename}/{aux}/time'))
            aux_data_interp = interpolate.interp1d(aux_time, aux_data,
                                                   axis=0, bounds_error=False,
                                                   fill_value='extrapolate')
            aux_data_matched_to_raw = aux_data_interp(raw.times)
            d[aux_names[idx]] = aux_data_matched_to_raw

        df = DataFrame(data=d)
        df = df.set_index('times')

    return df


def _decode_name(key):
    return np.array(key)[0].decode()
