# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import datetime

import h5py as h5py
import re
import numpy as np
from mne.io.pick import _picks_to_idx
from mne.transforms import apply_trans, _get_trans
from mne.channels import make_standard_montage


# The currently-implemented spec can be found here:
# https://raw.githubusercontent.com/fNIRS/snirf/v1.1/snirf_specification.md
SPEC_FORMAT_VERSION = '1.1'


def write_raw_snirf(raw, fname, add_montage=False):
    f"""Write continuous wave data to disk in SNIRF format.

    Writes the data from a raw object to a SNIRF file using
    SNIRF specification version {SPEC_FORMAT_VERSION}.
    The data can be either raw continuous wave amplitude,
    optical density or haemoglobin data.
    The data is written to the SNIRF file as a single data block
    and should pass the validation provided by the official
    snirf validator.
    
    Notes
    -----
    The MNE SNIRF reader requires that the data file contains either
    two wavelengths or both hbo and hbr data. This function can export
    data with just one wavelength or just hbo or hbr data.
    However, until modifications are made to the SNIRF reader, the
    exported data will not be readable by the MNE SNIRF reader.

    Parameters
    ----------
    raw : instance of Raw
        Data to write to file. Can be either continuous wave
        (`fnirs_cw_amplitude`) or processed (`fnirs_od`, 'hbo', 'hbr').
    fname : str
        Path to the SNIRF data file.
    add_montage : bool
        Should the montage be added as landmarks to
        facilitate compatibility with AtlasViewer.
    """

    supported_types = ['fnirs_cw_amplitude', 'fnirs_od', 'hbo', 'hbr']
    picks = _picks_to_idx(raw.info, supported_types, exclude=[])
    assert len(picks) == len(raw.ch_names),\
        'Data must only be of type fnirs_cw_amplitude,  fnirs_od, hbo or hbr'
    if ('fnirs_cw_amplitude' in raw) or ('fnirs_od' in raw):
        assert len(np.unique(raw.info.get_channel_types())) == 1,\
            'All channels must be of the same type'
    elif ('hbo' in raw) or ('hbr' in raw):
        assert len(np.unique(raw.info.get_channel_types())) <= 2,\
            'Channels must be of type hbo and hbr'

    with h5py.File(fname, 'w') as f:
        nirs = f.create_group('/nirs')
        f.create_dataset('formatVersion',
                         data=_str_encode(SPEC_FORMAT_VERSION))

        _add_metadata_tags(raw, nirs)
        _add_single_data_block(raw, nirs)
        _add_probe_info(raw, nirs, add_montage=add_montage)
        _add_stim_info(raw, nirs)


def _str_encode(str_val):
    """Encode a string for use in an h5py Dataset.

    Parameters
    ----------
    str_val : str
        The string to encode.
    """
    return str_val.encode('UTF-8')


def _add_metadata_tags(raw, nirs):
    """Creates and adds elements to the nirs metaDataTags group.

    Parameters
    ----------
    raw : instance of Raw
        Data to add to the snirf file.
    nirs : hpy5.Group
        The root hdf5 nirs group to which the metadata should beadded.
    """
    metadata_tags = nirs.create_group('metaDataTags')

    # Store measurement
    datestr = raw.info['meas_date'].strftime('%Y-%m-%d')
    timestr = raw.info['meas_date'].strftime('%H:%M:%SZ')
    metadata_tags.create_dataset('MeasurementDate',
                                 data=_str_encode(datestr))
    metadata_tags.create_dataset('MeasurementTime',
                                 data=_str_encode(timestr))

    # Store demographic info
    subject_id = raw.info['subject_info']['first_name']
    metadata_tags.create_dataset('SubjectID', data=_str_encode(subject_id))

    # Store the units of measurement
    metadata_tags.create_dataset('LengthUnit', data=_str_encode('m'))
    metadata_tags.create_dataset('TimeUnit', data=_str_encode('s'))
    metadata_tags.create_dataset('FrequencyUnit', data=_str_encode('Hz'))

    # Add non standard (but allowed) custom metadata tags
    if 'birthday' in raw.info['subject_info']:
        birthday = datetime.date(*raw.info['subject_info']['birthday'])
        birthstr = birthday.strftime('%Y-%m-%d')
        metadata_tags.create_dataset('DateOfBirth',
                                     data=[_str_encode(birthstr)])
    if 'middle_name' in raw.info['subject_info']:
        middle_name = raw.info['subject_info']['middle_name']
        metadata_tags.create_dataset('middleName',
                                     data=[_str_encode(middle_name)])
    if 'last_name' in raw.info['subject_info']:
        last_name = raw.info['subject_info']['last_name']
        metadata_tags.create_dataset('lastName', data=[_str_encode(last_name)])
    if 'sex' in raw.info['subject_info']:
        sex = str(int(raw.info['subject_info']['sex']))
        metadata_tags.create_dataset('sex', data=[_str_encode(sex)])
    if raw.info['dig'] is not None:
        coord_frame_id = int(raw.info['dig'][0].get('coord_frame'))
        metadata_tags.create_dataset('MNE_coordFrame', data=[coord_frame_id])


def _add_single_data_block(raw, nirs):
    """Adds the data from raw to the nirs data1 group.

    While SNIRF supports multiple datablocks, this writer only supports
    a single data block named data1.

    Parameters
    ----------
    raw : instance of Raw
        Data to add to the snirf file.
    nirs : hpy5.Group
        The root hdf5 nirs group to which the data should be added.
    """
    data_block = nirs.create_group('data1')
    data_block.create_dataset('dataTimeSeries', data=raw.get_data().T)
    data_block.create_dataset('time', data=raw.times)

    _add_measurement_lists(raw, data_block)


def _add_measurement_lists(raw, data_block):
    """Adds the measurement list groups to the nirs data1 group.

    Parameters
    ----------
    raw : instance of Raw
        Data to add to the snirf file.
    data_block : hpy5.Group
        The hdf5 data1 group to which the measurement lists should be added.
    """
    sources = _get_unique_source_list(raw)
    detectors = _get_unique_detector_list(raw)
    wavelengths = _get_unique_wavelength_list(raw)

    raw_types = raw.info.get_channel_types()
    raw_wavelengths = [c["loc"][9] for c in raw.info["chs"]]

    for idx, ch_name in enumerate(raw.ch_names, start=1):
        ml_id = f'measurementList{idx}'
        ch_group = data_block.require_group(ml_id)

        source_idx = sources.index(_extract_source(ch_name)) + 1
        detector_idx = detectors.index(_extract_detector(ch_name)) + 1
        wavelength_idx = wavelengths.index(raw_wavelengths[idx - 1]) + 1
        ch_group.create_dataset('sourceIndex', data=source_idx, dtype='int32')
        ch_group.create_dataset('detectorIndex', data=detector_idx,
                                dtype='int32')
        ch_group.create_dataset('wavelengthIndex', data=wavelength_idx,
                                dtype='int32')

        # The data type coding is described at
        # https://github.com/fNIRS/snirf/blob/master/snirf_specification.md#appendix
        # The currently implemented data types are:
        # 1 = Continuous Wave
        # 99999 = Processed
        data_type = 1 if raw_types[idx - 1] == 'fnirs_cw_amplitude' else 99999

        ch_group.create_dataset('dataType', data=data_type, dtype='int32')
        ch_group.create_dataset('dataTypeIndex', data=1, dtype='int32')
        if raw_types[idx - 1] == 'fnirs_od':
            ch_group.create_dataset('dataTypeLabel', data="dOD")
        elif raw_types[idx - 1] == 'fnirs_cw_amplitude':
            # The SNIRF specification does not specify a label for this type
            continue
        elif raw_types[idx - 1] == 'hbo':
            ch_group.create_dataset('dataTypeLabel', data="HbO")
        elif raw_types[idx - 1] == 'hbr':
            ch_group.create_dataset('dataTypeLabel', data="HbR")


def _add_probe_info(raw, nirs, add_montage):
    """Adds details of the probe to the nirs group.

    Parameters
    ----------
    raw : instance of Raw
        Data to add to the snirf file.
    nirs : hpy5.Group
        The root hdf5 nirs group to which the probe info should be added.
    add_montage : bool
        Should the montage be added as landmarks to
        facilitate compatibility with AtlasViewer.
    """
    sources = _get_unique_source_list(raw)
    detectors = _get_unique_detector_list(raw)
    wavelengths = _get_unique_wavelength_list(raw)

    probe = nirs.create_group('probe')

    # Store source/detector/wavelength info
    encoded_source_labels = [_str_encode(f'S{src}') for src in sources]
    encoded_detector_labels = [_str_encode(f'D{det}') for det in detectors]
    probe.create_dataset('sourceLabels', data=encoded_source_labels)
    probe.create_dataset('detectorLabels', data=encoded_detector_labels)
    probe.create_dataset('wavelengths', data=wavelengths)

    # Create 3d locs and store
    ch_sources = [_extract_source(ch) for ch in raw.ch_names]
    ch_detectors = [_extract_detector(ch) for ch in raw.ch_names]
    srclocs = np.empty((len(sources), 3))
    detlocs = np.empty((len(detectors), 3))
    for i, src in enumerate(sources):
        idx = ch_sources.index(src)
        srclocs[i, :] = raw.info['chs'][idx]['loc'][3:6]
    for i, det in enumerate(detectors):
        idx = ch_detectors.index(det)
        detlocs[i, :] = raw.info['chs'][idx]['loc'][6:9]
    probe.create_dataset('sourcePos3D', data=srclocs)
    probe.create_dataset('detectorPos3D', data=detlocs)

    # Store probe landmarks
    if raw.info['dig'] is not None:
        _store_probe_landmarks(raw, probe, add_montage)


def _store_probe_landmarks(raw, probe, add_montage):
    """Adds the probe landmarks to the probe group.

    The SNIRF specification provides flexibility around
    what is stored in this field. Some software expect specific
    landmarks which are not part of the SNIRF specification.
    We endeavour to provide as much information as possible,
    without causing conflicts with the specification,
    in order to achieve compliance with other software packages.

    Atlas Viewer expects a list of 10-20 locations to be provided
    in the landmark fields.

    We also store the values in `raw.info['dig']` as these
    may represent positions that have been digitised by the
    researcher and should be retained. These are prepended
    by the string "HP" to represent a head point.

    Parameters
    ----------
    raw : instance of Raw
        Data to add to the snirf file.
    probe : hpy5.Group
        The hdf5 probe group to which the landmark info should be added.
    add_montage : bool
        Should the montage be added as landmarks to
        facilitate compatibility with AtlasViewer.
    """
    diglocs = np.empty((len(raw.info['dig']), 3))
    digname = list()
    for idx, dig in enumerate(raw.info['dig']):
        ident = re.match(r'\d+ \(FIFFV_POINT_(\w+)\)',
                         str(dig.get('ident')))
        if ident is not None:
            digname.append(ident[1])
        else:
            digname.append(f"HP_{str(dig.get('ident'))}")
        diglocs[idx, :] = dig.get('r')

    if add_montage:
        # First, get the template positions in mni fsaverage space
        montage = make_standard_montage(
            'standard_1020', head_size=0.09700884729534559)
        ch_names = montage.ch_names
        montage_locs = np.array(list(montage._get_ch_pos().values()))

        # montage_locs = np.array([d['r'] for d in montage.dig])
        head_mri_t, _ = _get_trans('fsaverage', 'mri', 'head')
        locations_head = apply_trans(head_mri_t, montage_locs)

        digname.extend(ch_names)
        diglocs = np.concatenate((diglocs, locations_head))

    digname = [_str_encode(d) for d in digname]
    probe.create_dataset('landmarkPos3D', data=diglocs)
    probe.create_dataset('landmarkLabels', data=digname)


def _add_stim_info(raw, nirs):
    """Adds details of the stimuli to the nirs group.

    Parameters
    ----------
    raw : instance of Raw
        Data to add to the snirf file.
    nirs : hpy5.Group
        The root hdf5 nirs group to which the stimuli info should be added.
    """
    # Convert MNE annotations to SNIRF stims
    descriptions = np.unique(raw.annotations.description)
    for idx, desc in enumerate(descriptions, start=1):
        stim_group = nirs.create_group(f'stim{idx}')
        trgs = np.where(raw.annotations.description == desc)[0]
        stims = np.zeros((len(trgs), 3))
        for idx_t, trg in enumerate(trgs):
            stims[idx_t, :] = [raw.annotations.onset[trg],
                               raw.annotations.duration[trg],
                               idx]
        stim_group.create_dataset('data', data=stims)
        stim_group.create_dataset('name', data=_str_encode(desc))


def _get_unique_source_list(raw):
    """Returns the sorted list of distinct source ids.

    Parameters
    ----------
    raw : instance of Raw
        Data object containing the list of channels.
    """
    ch_sources = [_extract_source(ch_name) for ch_name in raw.ch_names]
    return list(sorted(set(ch_sources)))


def _get_unique_detector_list(raw):
    """Returns the sorted list of distinct detector ids.

    Parameters
    ----------
    raw : instance of Raw
        Data object containing the list of channels.
    """
    ch_detectors = [_extract_detector(ch_name) for ch_name in raw.ch_names]
    return list(sorted(set(ch_detectors)))


def _get_unique_wavelength_list(raw):
    """Returns the sorted list of distinct wavelengths.

    Parameters
    ----------
    raw : instance of Raw
        Data object containing the list of channels.
    """
    ch_wavelengths = [c["loc"][9] for c in raw.info["chs"]]
    return list(sorted(set(ch_wavelengths)))


def _match_channel_pattern(channel_name):
    """Returns a regex match against the expected channel name format.

    The returned match object contains three named groups: source, detector,
    and wavelength/type. If no match is found, a ValueError is raised.

    For example, if the name is "S1_D2 hbo" the source is 1,
    the detector is 2, and the type is hbo.
    If the name is "S1_D2 850" the source is 1,
    the detector is 2, and the wavelength is 850.

    Parameters
    ----------
    channel_name : str
        The name of the channel.
    """
    rgx = r'^S(?P<source>\d+)_D(?P<detector>\d+) (?P<wavelength_type>[\w]+)$'
    match = re.fullmatch(rgx, channel_name)
    if match is None:
        msg = f'channel name does not match expected pattern: {channel_name}'
        raise ValueError(msg)
    return match


def _extract_source(channel_name):
    """Extracts and returns the source id from the channel name.

    The id is returned as an integer value.

    Parameters
    ----------
    channel_name : str
        The name of the channel.
    """
    return int(_match_channel_pattern(channel_name).group('source'))


def _extract_detector(channel_name):
    """Extracts and returns the detector id from the channel name.

    The id is returned as an integer value.

    Parameters
    ----------
    channel_name : str
        The name of the channel.
    """
    return int(_match_channel_pattern(channel_name).group('detector'))
