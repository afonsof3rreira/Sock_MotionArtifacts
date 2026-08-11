import csv
import json
import numpy as np
import pandas as pd

def activate_pd_full_print():
    """
    Activates pandas full printing options.
    """""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)  # prevents line wrapping
    pd.set_option("display.max_colwidth", None)  # shows full cell content


import numpy as np
import pandas as pd


def _read_table(path, skiprows):
    """Fast whitespace-delimited numeric read."""
    return pd.read_csv(path, sep=r'\s+', skiprows=skiprows,
                       header=None, dtype=np.float64,
                       na_filter=False).to_numpy()


def get_signals_as_dict(signal_path: str, device_name='bitalino', bit_signal_map=None, raw_full=False):
    """(docstring unchanged)"""

    signals = {'sympathia': {}, 'bitalino': {}}

    if device_name == 'bitalino':
        bitalino_data = _read_table(signal_path, skiprows=3)
        bitalino_columns = bitalino_file_header_as_dict(signal_path)['column']

        if raw_full:
            for i, col in enumerate(bitalino_columns):
                signals['bitalino'][col] = bitalino_data[:, i]
        else:
            if bit_signal_map is None:
                bit_signal_map = {"A1": "EDA", "A2": "LUX"}
            index = {col: i for i, col in enumerate(bitalino_columns)}
            for channel, modal in bit_signal_map.items():
                if channel in index:
                    signals['bitalino'][modal] = bitalino_data[:, index[channel]]

    elif device_name == 'sympathia':
        scientisst_data = _read_table(signal_path, skiprows=2)
        scientisst_columns = get_sympathia_header(signal_path)

        for i, col in enumerate(scientisst_columns):
            name = col if raw_full else ('LED' if col == 'O2' else col)
            signals['sympathia'][name] = scientisst_data[:, i]

    else:
        print("Sympathia signals path not provided.")

    return signals

def get_header_raw(data_path: str, device: str):

    if device == 'sympathia':
        with open(data_path, 'r', newline='') as f:
            rows = [next(f).strip() for _ in range(2)]

    elif device == 'bitalino':
        with open(data_path, 'r') as f:
            rows = [next(f).strip() for _ in range(3)]

    return rows

def get_sympathia_header(data_path: str):
    """Obtains the header of the Sympathia data file, which is in the second line, and returns it as a list of column names.
    
    Parameters
    ----------
    data_path : str
        Path to signals file.
        
    Returns
    -------
    lines : dict
        Resulting dictionary containing the signals.
    """""
    # read first three lines. The second is the mdata.
    with open(data_path, "r") as file:
        lines = [next(file).strip() for i in range(2)][1]

    # remove initial char, and convert single quote to double
    lines = lines.split('#')[1].split('\t')

    return lines


def bitalino_file_header_as_dict(bitalino_data_path: str):
    """Computes patient classification performance in a comprehensive manner (AUC, Accuracy, F1-score).

    Parameters
    ----------
    data_path : str
        Path to data.
    win_len : int
        Window length (in seconds, used in feature extraction).
    win_overlap : float
        Window overlap (decimal, used in feature extraction).
    feature_sels : list
        List containing the feature selection ('eda', 'edr', or 'all' for both).
    norm_method : str
        Normalization method used in feature extraction.

    Returns
    -------
    df_results : pd.DataFrame
        Resulting dataframe.
    """""
    # read first three lines. The second is the mdata.
    with open(bitalino_data_path, "r") as file:
        lines = [next(file).strip() for i in range(3)][1]

    # remove initial char
    lines = lines.split('# ')[1]

    # convert to dictionary
    mdata_dict = json.loads(lines)

    mac_address = list(mdata_dict.keys())[0]

    return mdata_dict[mac_address]

def write_signals_from_dict(output_path: str, signals: dict, header: list):
    """
    Writes processed signals to a file, preserving the original header and
    writing signal data as tab-separated values in dictionary insertion order.

    Parameters
    ----------
    output_path : str
        Full path to the output file.
    signals : dict
        Dictionary of modality_name -> np.ndarray (1D signal arrays).
    header : list
        Header rows as returned by get_header_raw().
        Sympathia: list of lists. BITalino: list of strings.
    device : str
        Device name: 'sympathia' or 'bitalino'.
    """

    with open(output_path, 'w', newline='') as f:

        for row in header:
            f.write(row + '\n')

        modalities = list(signals.keys())
        data = np.column_stack([signals[m] for m in modalities])
        for row in data:
            f.write('\t'.join(str(v) for v in row) + '\n')