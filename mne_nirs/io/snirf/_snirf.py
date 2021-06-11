# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import h5py as h5py
import re
import numpy as np
from mne.io.pick import _picks_to_idx


def write_raw_snirf(raw, fname):
    """Writer for continuous wave SNIRF data.

    Parameters
    ----------
    raw : instance of Raw
        Data to write to file. Must contain only `fnirs_cw_amplitude` type.
    fname : str
        Path to the SNIRF data file.
    """

    picks = _picks_to_idx(raw.info, 'fnirs_cw_amplitude', exclude=[])
    assert len(picks) == len(raw.ch_names), "Data must be fnirs_cw_amplitude"

    # Reordering channels
    num_chans = len(raw.ch_names)
    raw = raw.copy()
    raw.pick(picks=list(range(num_chans)[0::2]) + list(range(num_chans)[1::2]))

    with h5py.File(fname, "w") as f:
        f.create_dataset("nirs/data1/measurementList1/dataType", data=1)
        f.create_dataset("/nirs/data1/dataTimeSeries", data=raw.get_data().T)
        f.create_dataset("/nirs/data1/time", data=raw.times)

        # Store measurement and birth date
        datestr = raw.info["meas_date"].strftime("%Y-%m-%d")
        timestr = raw.info["meas_date"].strftime("%H:%M:%SZ")
        birthstr = '{0:02d}-{1:02d}-{2:02d}'.format(
            raw.info["subject_info"]['birthday'][0],
            raw.info["subject_info"]['birthday'][1],
            raw.info["subject_info"]['birthday'][2])
        f.create_dataset("nirs/metaDataTags/"
                         "MeasurementDate", data=[datestr.encode('UTF-8')])
        f.create_dataset("nirs/metaDataTags/"
                         "MeasurementTime", data=[timestr.encode('UTF-8')])
        f.create_dataset("nirs/metaDataTags/"
                         "DateOfBirth", data=[birthstr.encode('UTF-8')])

        # Extract info from file names
        rgx = r'S(\d+)_D(\d+) (\d+)'
        chs = raw.info['chs']
        sources = [float(re.match(rgx, r['ch_name']).groups()[0])
                   for r in chs]
        detectors = [float(re.match(rgx, r['ch_name']).groups()[1])
                     for r in chs]
        wavelengths = [float(re.match(rgx, r['ch_name']).groups()[2])
                       for r in chs]

        # Create info summary and recode
        sources_sorted = np.sort(np.unique(sources))
        detectors_sorted = np.sort(np.unique(detectors))
        wavelengths_sorted = np.sort(np.unique(wavelengths))
        sources_sorted = [str(int(src)).encode('UTF-8')
                          for src in sources_sorted]
        detectors_sorted = [str(int(det)).encode('UTF-8')
                            for det in detectors_sorted]
        wavelengths_sorted = [str(wve).encode('UTF-8')
                              for wve in wavelengths_sorted]

        # Store source/detector/wavelength info
        f.create_dataset("nirs/probe/sourceLabels",
                         data=[('S'.encode('UTF-8') + src)
                               for src in sources_sorted])
        f.create_dataset("nirs/probe/detectorLabels",
                         data=[('D'.encode('UTF-8') + det)
                               for det in detectors_sorted])
        f.create_dataset("nirs/probe/wavelengths",
                         data=[float(wve)
                               for wve in wavelengths_sorted])

        # Create 3d locs and store
        srclocs = np.empty((len(np.unique(sources_sorted)), 3))
        detlocs = np.empty((len(np.unique(detectors_sorted)), 3))
        for i, src in enumerate(sources_sorted):
            idx = sources.index(float(src))
            srclocs[i, :] = raw.info['chs'][idx]['loc'][3:6]
        for i, det in enumerate(detectors_sorted):
            idx = detectors.index(float(det))
            detlocs[i, :] = raw.info['chs'][idx]['loc'][6:9]
        f.create_dataset("nirs/probe/sourcePos3D", data=srclocs)
        f.create_dataset("nirs/probe/detectorPos3D", data=detlocs)
        f.create_dataset("nirs/metaDataTags/LengthUnit",
                         data=['m'.encode('UTF-8')])

        # Prep data for storing each MNE channel as SNIRF measurementList
        channels = ["measurementList" + str(idx + 1)
                    for idx in range(len(raw.ch_names))]
        sources = np.array([float(src) for src in sources])
        detectors = np.array([float(det) for det in detectors])
        sources_sorted = [float(src) for src in sources_sorted]
        detectors_sorted = [float(det) for det in detectors_sorted]
        wavelengths_sorted = [float(wve) for wve in wavelengths_sorted]
        w = [float(wve) for wve in wavelengths]
        wavelengths_index = [wavelengths_sorted.index(wve) + 1 for wve in w]

        for idx, ch in enumerate(channels):
            f.create_dataset('nirs/data1/' + ch + '/sourceIndex',
                             data=[sources_sorted.index(sources[idx]) + 1])
            f.create_dataset('nirs/data1/' + ch + '/detectorIndex',
                             data=[detectors_sorted.index(detectors[idx]) + 1])
            f.create_dataset('nirs/data1/' + ch + '/wavelengthIndex',
                             data=[wavelengths_index[idx]])

        # Store demographic info
        subject_id = raw.info["subject_info"]['first_name']
        f.create_dataset("nirs/metaDataTags/SubjectID",
                         data=[subject_id.encode('UTF-8')])

        # Convert MNE annotations to SNIRF stims
        for desc in np.unique(raw.annotations.description):
            key = "stim" + desc
            trgs = np.where(raw.annotations.description == desc)[0]
            stims = np.zeros((len(trgs), 3))
            for idx, trg in enumerate(trgs):
                stims[idx, :] = [raw.annotations.onset[trg], 5.0,
                                 raw.annotations.duration[trg]]
            f.create_dataset('/nirs/' + key + '/data', data=stims)

        # Store probe landmarks
        if raw.info['dig'] is not None:
            diglocs = np.empty((len(raw.info['dig']), 3))
            digname = list()
            for idx, dig in enumerate(raw.info['dig']):
                ident = re.match(r"\d+ \(FIFFV_POINT_(\w+)\)",
                                 str(dig.get("ident")))
                if ident is not None:
                    digname.append(ident[1])
                else:
                    digname.append(str(dig.get("ident")))
                diglocs[idx, :] = dig.get("r")
            digname = [d.encode('UTF-8') for d in digname]
            f.create_dataset("nirs/probe/landmarkPos3D", data=diglocs)
            f.create_dataset("nirs/probe/landmarkLabels", data=digname)
            # Add non standard (but allowed) custom metadata tags
            f.create_dataset("nirs/metaDataTags/MNE_coordFrame",
                             data=[int(raw.info['dig'][0].get("coord_frame"))])

        # Add non standard (but allowed) custom metadata tags
        if 'middle_name' in raw.info["subject_info"]:
            mname = [raw.info["subject_info"]['middle_name'].encode('UTF-8')]
            f.create_dataset("nirs/metaDataTags/middleName", data=mname)
        if 'last_name' in raw.info["subject_info"]:
            lname = [raw.info["subject_info"]['last_name'].encode('UTF-8')]
            f.create_dataset("nirs/metaDataTags/lastName", data=lname)
        if 'sex' in raw.info["subject_info"]:
            sex = str(int(raw.info["subject_info"]['sex'])).encode('UTF-8')
            f.create_dataset("nirs/metaDataTags/sex", data=[sex])
