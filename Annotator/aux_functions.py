import json

import numpy as np
from biosppy import tools as sti

def vm_extractor(signal: np.ndarray):
    """Extracts the vector magnitude and signal magnitude features from an input ACC signal, given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.

    Returns
    -------
    vm_features : array
        Extracted Vector Magnitude (VM) feature.
    """

    # get acceleration features
    vm_features = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
        vm_features[i] = np.linalg.norm(
            np.array([signal[i][0], signal[i][1], signal[i][2]])
        )

    return vm_features

def acc_filtering(signal, window_sz_ms=500, fs=1000.):
    """Filters one signals acceleration signal using the standard moving average filter.
    """""
    # smooth
    sm_size = int(fs * (window_sz_ms / 1000))
    filtered, _ = sti.smoother(signal=signal, kernel="boxzen", size=sm_size, mirror=True)

    return filtered

def acc_multi_filtering(signals, window_sz_ms=500, fs=1000.):
    """Filters the 3 signals of 3-axial acceleration signals using the standard moving average filter.
    """""
    signals_filtered = []
    for s in signals:
        signals_filtered.append(acc_filtering(s, window_sz_ms=window_sz_ms, fs=fs))

    return signals_filtered

def filter_eda(signal, sampling_rate=1000.0, EDR=False):
    # filter signal
    aux, _, _ = sti.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5, #5,
        sampling_rate=sampling_rate,
    )

    if EDR:
        # filter signal
        aux, _, _ = sti.filter_signal(
            signal=aux,
            ftype="butter",
            band="highpass",
            order=4,
            frequency=0.05,
            sampling_rate=sampling_rate,
        )

    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = sti.smoother(signal=aux, kernel="boxzen", size=sm_size, mirror=True)

    return filtered

def normalize(input_signal: np.ndarray, new_min, new_max):
    """Normalizes the signal to new minimum and maximum values.
    """

    # first normalize from 0 to Condition - 1
    input_signal = (input_signal - np.min(input_signal)) / (np.max(input_signal) - np.min(input_signal))

    # then normalize to the [new_min, new_max] range
    output_signal = input_signal * (new_max - new_min) + new_min

    return output_signal


def milliseconds_to_samples(time_milliseconds: int, sampling_rate: float):
    """Converts a duration value in milliseconds to samples."""
    return int(time_milliseconds * (int(sampling_rate) / 1000))

def get_signals_as_dict_v4(signal_path, signal_type='bitalino', bit_signal_map=None):
    """
        This is an improved implementation of the functions to read the EDA signals collected from a BITalino or a Sympathia.
        It uses the file path directly instead of the enclosing folder path.
    """""

    signals = {'sympathia': {}, 'bitalino': {}}

    if signal_type == 'bitalino':
        bitalino_data = np.loadtxt(signal_path, skiprows=3)
        bitalino_mdata = bitalino_file_header_as_dict(signal_path)

        if bit_signal_map is None:
            bit_signal_map = {
                "A1": "EDA",
                "A2": "LUX"
            }

        for channel, modal in bit_signal_map.items():
            bitalino_columns = bitalino_mdata['column']
            for i in range(len(bitalino_columns)):
                if channel == bitalino_columns[i]:
                    signals['bitalino'][modal] = bitalino_data[:, i]

    elif signal_type == 'sympathia':

        scientisst_data = np.loadtxt(signal_path, skiprows=2)
        scientisst_columns = get_scientissti_header_from_secondL(signal_path)

        for i in range(len(scientisst_columns)):
            signals['sympathia'][scientisst_columns[i]] = scientisst_data[:, i]

    else:
        print("Sympathia signals path not provided.")

    return signals

def get_scientissti_header_from_secondL(scientisst_data_path):
    # read first three lines. The second is the mdata.
    with open(scientisst_data_path, "r") as file:
        lines = [next(file).strip() for i in range(2)][1]

    # remove initial char, and convert single quote to double
    lines = lines.split('#')[1].split('\t')

    return lines

def bitalino_file_header_as_dict(bitalino_data_path: str):
    # read first three lines. The second is the mdata.
    with open(bitalino_data_path, "r") as file:
        lines = [next(file).strip() for i in range(3)][1]

    # remove initial char
    lines = lines.split('# ')[1]

    # convert to dictionary
    mdata_dict = json.loads(lines)

    mac_address = list(mdata_dict.keys())[0]

    return mdata_dict[mac_address]

