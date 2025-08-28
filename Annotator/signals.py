from biosppy import tools as st
from biosppy import utils
import numpy as np

def preprocess_pcg(signal=None, sampling_rate=1000.):
    """Pre-processes a raw PCG signal.

    Parameters
    ----------
    signal : array
        Raw PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    filtered : array
        Filtered PCG signal.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # Filter Design
    order = 2
    passBand = np.array([25, 400])

    # Band-Pass filtering of the PCG:
    filtered, fs, params = st.filter_signal(signal, 'butter', 'bandpass', order, passBand, sampling_rate)

    return utils.ReturnTuple((filtered,), ("filtered",))

def preprocess_ppg(signal=None, sampling_rate=1000.):
    """Pre-processes a raw PPG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    filtered : array
        Filtered PPG signal.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=4,
                                      frequency=[1, 8],
                                      sampling_rate=sampling_rate)

    return utils.ReturnTuple((filtered,), ("filtered",))

def preprocess_eda(signal=None, sampling_rate=1000.0):
    """Pre-processes a raw EDA signal (the stage before feature extraction).

    Parameters
    ----------
    signal : array
        Raw EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered EDA signal.
    onsets : array
        Indices of SCR pulse onsets.
    peaks : array
        Indices of the SCR peaks.
    amplitudes : array
        SCR pulse amplitudes.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    aux, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5,
        sampling_rate=sampling_rate,
    )

    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = st.smoother(signal=aux, kernel="boxzen", size=sm_size, mirror=True)

    # output
    return utils.ReturnTuple((filtered,), ("filtered",))

def preprocess_ecg(signal=None, sampling_rate=1000.0):
    """Pre-processes a raw ECG signal.

    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    filtered : array
        Filtered ECG signal.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    order = int(1.5 * sampling_rate)
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=[0.67, 45],
        sampling_rate=sampling_rate,
    )

    filtered = filtered - np.mean(filtered)  # remove DC offset

    # output
    return utils.ReturnTuple((filtered,), ("filtered",))

def preprocess_emg(signal=None, sampling_rate=1000.):
    """Pre-processes a raw EMG signal.

    Parameters
    ----------
    signal : array
        Raw EMG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    filtered : array
        Filtered EMG signal.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='highpass',
                                      order=4,
                                      frequency=100,
                                      sampling_rate=sampling_rate)

    # output
    return utils.ReturnTuple((filtered,), ("filtered",))

def preprocess_abp(signal=None, sampling_rate=1000.0):
    """Pre-processes a raw ABP signal.

    Parameters
    ----------
    signal : array
        Raw ABP signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    filtered : array
        Filtered ABP signal.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="bandpass",
        order=4,
        frequency=[1, 8],
        sampling_rate=sampling_rate,
    )

    # output
    return utils.ReturnTuple((filtered,), ("filtered",))