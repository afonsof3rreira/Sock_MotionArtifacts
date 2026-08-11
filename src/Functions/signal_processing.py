import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from biosppy.signals.acc import time_domain_feature_extractor
from biosppy.signals.eda import eda
from biosppy import tools as sti

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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
    """""

    # get acceleration features
    vm_features = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
        vm_features[i] = np.linalg.norm(
            np.array([signal[i][0], signal[i][1], signal[i][2]])
        )

    return vm_features


def sm_extractor(signal: np.ndarray):
    """Extracts the vector magnitude and signal magnitude features from an input ACC signal, given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.

    Returns
    -------
    sm_features : array
        Extracted Signal Magnitude (SM) feature.

    """""
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # get acceleration features
    sm_features = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
        sm_features[i] = (abs(signal[i][0]) + abs(signal[i][1]) + abs(signal[i][2])) / 3

    return sm_features

def smooth_signal_section(signal: np.ndarray, start_idx: int, end_idx: int):
    """
    Smooths a section of the signal section.
    
    Parameters
    ----------
    signal : pd.DataFrame
        Signal to be smoothed.
    start_idx : int
        Starting index of the section to be smoothed.
    end_idx : int
        Ending index of the section to be smoothed.
        
    Returns
    -------
    smoothed_signal : np.ndarray
        The smoothed signal.

    """""
    # Validate indices
    if start_idx < 0 or end_idx > len(signal) or start_idx >= end_idx:
        raise ValueError("Invalid start or end index")

    # Extract the sub-signal
    sub_signal = signal[start_idx:end_idx]

    # Apply convolution using the custom kernel
    smoothed_sub_signal, _ = sti.smoother(sub_signal, kernel='parzen', size=1000)

    # Replace the section in the original signal
    smoothed_signal = signal.copy()
    smoothed_signal[start_idx:end_idx] = smoothed_sub_signal

    return smoothed_signal

def signal_corrector(eda_signal: np.ndarray, dac, show=False, debug=False):
    """EDA Signal corrector. This function splits the EDA signal in patches (with a constant DAC value) and glues
    them together to form a continuous signal.
    
    Parameters
    ----------
    eda_signal : np.ndarray
        Un-corrected sympathia EDA signal.
    dac : np.ndarray
        The DAC signal.
    show : bool
        Whether to show the correction process for each transition.
    debug : bool
        Whether to print debug messages during the correction process.

    Returns
    -------
    eda_signal_output : np.ndarray
        Corrected EDA signal.
    """""

    eda_signal_output = eda_signal.copy()

    dac_changes = np.where(np.diff(dac) != 0)[0] + 1
    signal_length = eda_signal.shape[0]
    amp_offset = 0

    # STEP 1 - identifying the mid-point of transitions in the EDA signal, using derivative and DAC
    transition_mid_points = []
    transition_ranges = []
    # signal_blocks = []

    for i in range(dac_changes.shape[0]):
        try:
            # STEP 1 - locate the transition precisely (it's not exactly the timestamp of the DAC)
            # Why don't we just use the value of the DAC? Because there is a delay between the DAC and the effect on the EDA signal;
            # the transient produced on the EDA signal is also not 100% replicable (varies)

            idx_i = dac_changes[i] - 2000

            if idx_i < 0:
                idx_i = 0

            idx_f = dac_changes[i] + 2000

            if idx_f > signal_length:
                idx_f = signal_length

            # Segment the signal around the dac change (this is the window we will work with)
            signal_block = eda_signal[idx_i:idx_f]

            # Compute the first derivative
            derivative = np.diff(signal_block)

            # Find the index of the maximum absolute derivative (sharpest transition)
            max_transition_index = np.argmax(np.abs(derivative))

            # STEP 2 - for each midpoint, find transition range (from the bottom of the slope to the top) finding the
            # start and end of the transition equals to finding when the derivative slows down. To do this,
            # we will walk back from the mid-point until a moderate slope is achieved (start), and then we do the
            # same but forward (end).
            mid_point_idx = max_transition_index

            # Define a minimum derivative threshold (e.g., 0.0001% of the max derivative)
            th_val = 0.1  # 0.000001
            min_derivative_threshold = th_val * np.abs(derivative[mid_point_idx])

            # Walk backward to find the start of the transition
            start_index = mid_point_idx - 20
            while start_index > 0 and np.abs(derivative[start_index]) > min_derivative_threshold:
                start_index -= 1

            # Walk forward to find the end of the transition
            end_index = mid_point_idx + 20
            while end_index < len(derivative) - 1 and np.abs(derivative[end_index]) > min_derivative_threshold:
                end_index += 1

            # add additional 1 second to cover the transition bump forward
            end_index += 200  # 1000

            # STEP 3: mirror the signal into the identified transition range.
            # To do this, we shift back in time the time range of the transition. We then obtain a segment that
            # precedes the transition, and the same length as the transition range.
            # Then, we copy and paste the signal patch (from the preceding segment) into the transition range.
            # To do this we need to offset the amplitude so that the newly replaced transition is continuous
            # with the preceding segment.

            range_of_transition = end_index - start_index
            idx_i_patch = start_index - range_of_transition
            idx_f_patch = start_index

            eda_signal_before_snapshot = eda_signal_output.copy()

            first_data_point = eda_signal_output[idx_i + idx_i_patch]
            last_data_point = eda_signal_output[idx_i + idx_f_patch]

            # signal_patch = signal_block[idx_i_patch:idx_f_patch]
            signal_patch = eda_signal_output[idx_i + idx_i_patch: idx_i + idx_f_patch]

            amp_offset = last_data_point - first_data_point

            # signal_block[start_index:end_index] = signal_patch + amp_offset

            # assign transition range
            eda_signal_output[idx_i + start_index:idx_i + end_index] = signal_patch + amp_offset

            eda_signal_patched_snapshot = eda_signal_output.copy()

            # compute difference between amps
            first_r_sample = eda_signal_output[idx_i + end_index - 1]
            # signal_block)[start_index]
            last_r_sample = eda_signal_output[idx_i + end_index + 1]

            # STEP 4 - then also offset the rest of the signal
            amp_offset_range = last_r_sample - first_r_sample

            # assign jump
            eda_signal_output[idx_i + end_index:] = eda_signal_output[idx_i + end_index:] - amp_offset_range

            # smooth-out any remaining distortions
            eda_signal_output_smoothed = smooth_signal_section(eda_signal_output, idx_i + start_index, idx_i + end_index)
            eda_signal_output_smoothed = smooth_signal_section(eda_signal_output_smoothed, idx_i + start_index - 2000, idx_i + end_index + 2000)

            if show:
                if i > -1:  # > 12

                    avr_forward = 8000
                    avr_backward = 10000

                    start_index += avr_backward
                    end_index += avr_backward
                    idx_i_patch += avr_backward
                    idx_f_patch += avr_backward
                    mid_point_idx += avr_backward

                    offset_signal = np.abs(np.min(eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]))

                    eda_signal_output += offset_signal
                    eda_signal_before_snapshot += offset_signal
                    eda_signal_patched_snapshot += offset_signal
                    eda_signal_output_smoothed += offset_signal

                    min_1 = np.min(eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]) - 0.1 * (
                                np.max(eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]) - np.min(
                            eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]))
                    min_2 = np.min(eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward]) - 0.1 * (
                                np.max(eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward]) - np.min(
                            eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward]))
                    min_3 = np.min(eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward]) - 0.1 * (
                                np.max(eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward]) - np.min(
                            eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward]))

                    max_1 = np.max(eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]) + 0.1 * (
                                np.max(eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]) - np.min(
                            eda_signal_output[idx_i - avr_backward:idx_f + avr_forward]))
                    max_2 = np.max(eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward]) + 0.1 * (
                                np.max(eda_signal_before_snapshot[idx_i - avr_backward:idx_f + 2000]) - np.min(
                            eda_signal_before_snapshot[idx_i - avr_backward:idx_f + 2000]))
                    max_3 = np.max(eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward]) + 0.1 * (
                                np.max(eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward]) - np.min(
                            eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward]))

                    ymin = min(min_1, min_2, min_3)
                    ymax = max(max_1, max_2, max_3)

                    fig3, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), height_ratios=[2, 1, 2, 2])

                    ax1.axvspan(start_index, end_index + 1, color='salmon', alpha=0.3,
                                label='Transition Region')

                    ax1.axvspan(idx_i_patch, idx_f_patch, color='lightskyblue', alpha=0.3,
                                label='Transition Region')

                    ax1.vlines(start_index, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5, label='1')
                    # ax1.vlines(end_index - 500, ymin, ymax, colors='red', linestyles='dashed', alpha=0.5, label='1')
                    ax1.vlines(end_index, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5, label='1')

                    ax1.vlines(idx_i_patch, ymin, ymax, colors='k', linestyles='solid', alpha=0.3, label='1')

                    original_eda, = ax1.plot(eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward])
                    sc1 = ax1.scatter(mid_point_idx,
                                      eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward][
                                          mid_point_idx], c='darkred', zorder=3, alpha=0.5)

                    sc2 = ax1.scatter(idx_i_patch,
                                      eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward][idx_i_patch],
                                      s=30, c='k', marker="^", zorder=3, alpha=0.5)
                    sc3 = ax1.scatter(idx_f_patch,
                                      eda_signal_before_snapshot[idx_i - avr_backward:idx_f + avr_forward][idx_f_patch],
                                      s=30, c='k', marker="v", zorder=3, alpha=0.5)

                    ax1.set_ylim(ymin, ymax)
                    ax1.set_title("(1) Detection of the Transition Range")
                    # ax1.legend([original_eda], ["EDA Signal"])
                    ax1.legend([original_eda], ["EDA Signal"])

                    ax1.set_ylabel("a.u. [-]")

                    dac_changes_local = np.diff(dac[idx_i - avr_backward:idx_f + avr_forward])
                    local_dac_ch_idx = np.where(dac_changes_local != 0)[0] + 1

                    # ax2.vlines(local_dac_ch_idx, 150.5, 152.5, colors='k', linestyles='solid', alpha=0.3, label='1')
                    # ax2.vlines(start_index, 150.5, 152.5, colors='k', linestyles='dashed', alpha=0.5, label='1')

                    # ax2.vlines(idx_i, 150.5, 152.5, colors='red', linestyles='dashed', alpha=0.5, label='1')
                    # ax2.vlines(idx_f, 150.5, 152.5, colors='red', linestyles='dashed', alpha=0.5, label='1')

                    dac_line, = ax2.plot(dac[idx_i - avr_backward:idx_f + avr_forward], color=colors[2])

                    ax2.set_ylim(np.min(dac[idx_i - avr_backward:idx_f + avr_forward]) - 0.5,
                                 np.max(dac[idx_i - avr_backward:idx_f + avr_forward]) + 0.5)
                    ax2.legend([dac_line], ["DAC Signal"])
                    ax2.set_ylabel("Integer [0-255]")

                    ax3.axvspan(idx_i_patch, idx_f_patch + 1, color='lightskyblue', alpha=0.3,
                                label='Transition Region')

                    ax3.axvspan(start_index, end_index + 1, color='palegreen', alpha=0.3,
                                label='Transition Region')

                    ax3.vlines(start_index, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5, label='1')
                    ax3.vlines(end_index, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5, label='1')
                    ax3.vlines(idx_i_patch, ymin, ymax, colors='k', linestyles='solid', alpha=0.3, label='1')

                    eda_line2, = ax3.plot(eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward])

                    ax3.scatter(start_index,
                                eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward][start_index],
                                s=30, c='k', marker="^", zorder=3, alpha=0.5)
                    ax3.scatter(end_index,
                                eda_signal_patched_snapshot[idx_i - avr_backward:idx_f + avr_forward][end_index], s=30,
                                c='k', marker="v", zorder=3, alpha=0.5)

                    ax3.set_ylim(ymin, ymax)
                    ax3.set_title("(2) Mirroring Stage")
                    ax3.legend([eda_line2], ["EDA Signal"])
                    ax3.set_ylabel("a.u. [-]")

                    ax4.vlines(start_index, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5, label='1')
                    ax4.vlines(end_index, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5, label='1')

                    eda_line3, = ax4.plot(eda_signal_output[idx_i - avr_backward:idx_f + avr_forward])

                    ax4.axvspan(end_index + 1, idx_f + avr_forward, color='palegreen', alpha=0.3,
                                label='Transition Region')

                    ax4.set_ylim(ymin, ymax)
                    ax4.set_title("(3) Final Amplitude Offset Adjustment")
                    ax4.legend([eda_line3], ["EDA Signal"])
                    ax4.set_ylabel("a.u. [-]")
                    ax4.set_xlabel("Time [ms]")

                    # eda_line4, = ax5.plot(eda_signal_output_smoothed[idx_i - avr_backward:idx_f+avr_forward])
                    # ax4.set_ylim(ymin, ymax)
                    # ax4.set_title("(4) Final Amplitude Offset Adjustment")
                    # ax4.legend([eda_line4], ["EDA Signal"])
                    # ax5.set_ylim(ymin, ymax)

                    # ax5.set_xlabel("Time [ms]")

                    ax1.set_xlim(0, eda_signal_output[idx_i - avr_backward:idx_f + avr_forward].shape[0])
                    ax2.set_xlim(0, eda_signal_output[idx_i - avr_backward:idx_f + avr_forward].shape[0])
                    ax3.set_xlim(0, eda_signal_output[idx_i - avr_backward:idx_f + avr_forward].shape[0])
                    ax4.set_xlim(0, eda_signal_output[idx_i - avr_backward:idx_f + avr_forward].shape[0])
                    # ax5.set_xlim(0, eda_signal_output[idx_i - avr_backward:idx_f+avr_forward].shape[0])

                    ax1.set_xticklabels([])
                    ax1.set_xticks([])

                    ax2.set_xticklabels([])
                    ax2.set_xticks([])

                    ax3.set_xticklabels([])
                    ax3.set_xticks([])

                    # print(figures_path)
                    # fig3.savefig(os.path.join(figures_path, 'EDA_DAC_adjustment_{}.png'.format(i)), dpi=1200)

                # plt.show()

            transition_mid_points.append(idx_i + mid_point_idx + 1)
            transition_ranges.append([idx_i + start_index + 1, idx_i + end_index + 1])
            # signal_blocks.append(signal_block)

            if debug:
                print("Transition was processed {} successfully".format(i))

        except:
            if debug:
                print("Could not process transition {}".format(i))

    return eda_signal_output


def normalize(signal_arr, min=None, max=None):
    """Normalizes a signal between using min-max normalization (0-1).

    Parameters
    ----------
    signal_arr : np.ndarray
        The original signal.
    min : int
        The minimum value (default is 0).
    max : int
        The maximum value (default is 1).

    Returns
    -------
    signal_arr : np.ndarray
        Normalized signal.
    """""
    if np.min(signal_arr) == np.max(signal_arr):
        signal_arr = np.ones_like(signal_arr)
    else:
        signal_arr = (signal_arr - np.min(signal_arr)) / (np.max(signal_arr) - np.min(signal_arr))

    if min is not None and max is not None:
        signal_arr = (signal_arr * (max - min)) + min

    return signal_arr


def filter_eda(signal: np.ndarray, fs=1000., EDR=False):
    """Filters the EDA signal into a filtered EDA signal or into EDR signal.

    Parameters
    ----------
    signal : np.ndarray
        The original signal.
    fs : int
        The sampling rate of the signal.
    EDR : bool
        Whether to output the signal as EDR (True) or EDA (False).

    Returns
    -------
    filt_arr : np.ndarray
        Filtered signal.
    """""
    # filter signal
    aux, _, _ = sti.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5,  # 5,
        sampling_rate=fs,
    )

    if EDR:
        # filter signal
        aux, _, _ = sti.filter_signal(
            signal=aux,
            ftype="butter",
            band="highpass",
            order=4,
            frequency=0.05,
            sampling_rate=fs,
        )

    # smooth
    sm_size = int(0.75 * fs)
    filt_signal, _ = sti.smoother(signal=aux, kernel="boxzen", size=sm_size, mirror=True)

    return filt_signal


def acc_multi_filtering(signals: np.ndarray, fs=1000.):
    """Filters the 3 signals of 3-axial acceleration signals using the standard moving average filter.
    
    Parameters
    ----------
    signals : np.ndarray
        The 3-axial acceleration signals.
    fs : int
        The sampling rate of the signal.

    Returns
    -------
    filt_signals : np.ndarray
        Filtered accelerometer signals.
    """""
    filt_signals = []
    for s in signals:
        filt_signals.append(acc_filtering(s, fs=fs))

    return filt_signals


def acc_filtering(signal: np.ndarray, fs=1000.):
    """Filters one Acceleration (ACC) signal using the standard Butterworth band-pass filter (cut-off band [0.3 - 20] Hz), 
    preserving most human motion.
    
    Parameters
    ----------
    signal : np.ndarray
        The ACC signal.
    fs : float
        The sampling rate of the signal.

    Returns
    -------
    acc_filtered : np.ndarray
        The filtered ACC signal.
    """""
    # Band-pass limits
    lowcut = 0.3  # Hz
    highcut = 20.0  # Hz
    order = 4

    # Design SOS Butterworth band-pass filter
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

    # Apply zero-phase filtering
    acc_filtered = sosfiltfilt(sos, signal)

    return acc_filtered