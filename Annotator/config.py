# -*- coding: utf-8 -*-
"""
biosppy.inter_plotting.ecg
-------------------

This module provides configuration functions and variables for the Biosignal Annotator.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""
from matplotlib.backend_bases import MouseButton
import matplotlib.colors as mcolors
from biosppy.signals import eda
from biosppy.signals import ecg
from biosppy.signals import abp
from biosppy.signals import emg
from biosppy.signals import pcg
from biosppy.signals import ppg

import numpy as np

from Annotator import signals


def UI_intention(event, var_edit_plots=None, var_toggle_Ctrl=None, var_zoomed_in=None, window_in_border=None,
                 closest_event=False):
    """Returns the user intention and action given an event.

    Parameters
    ----------
    var_edit_plots : None
        Enables (if set to 1) or disables (if set to 0) annotation editing.
    var_toggle_Ctrl : None
        Stores whether currently zoomed-in (if set to 1) or zoomed-out (if set to 0) in amplitude.
    var_zoomed_in : None
        Stores whether currently zoomed-in (if true) or zoomed-out (if false) in time.
    window_in_border: None
        Stores whether the current zoomed-in window is in the borders of the time range (x-scale).
    closest_event : None
        Stores whether closest event is within defined range (if true) or not (false).

    Returns
    -------
    return_dict : dict
        The returned dictionary with keys 'triggered_intention' and 'triggered_action'.

    """

    triggered_intention = None
    triggered_action = None

    if hasattr(event, 'inaxes'):

        if event.inaxes is not None and not event.dblclick:  # and var_edit_plots.get() == 1

            if event.button == MouseButton.RIGHT:
                triggered_intention = "rmv_annotation"

                if var_edit_plots is not None:

                    if var_edit_plots.get() == 1 and closest_event:
                        triggered_action = "rmv_annotation"

                    elif var_edit_plots.get() == 1 and not closest_event:
                        triggered_action = "rmv_annotation_not_close"

                # else the action remain None

            elif event.button == MouseButton.LEFT:
                triggered_intention = "add_annotation"

                if var_edit_plots is not None:

                    if var_edit_plots.get() == 1:
                        triggered_action = "add_annotation"

    elif hasattr(event, 'keysym'):

        # Moving to the Right (right arrow)
        if event.keysym == 'Right':
            triggered_intention = "move_right"

            if window_in_border is not None:
                if not window_in_border:
                    triggered_action = "move_right"

                # else, window doesn't move to the right (stays the same)

        # Moving to the left (left arrow)
        elif event.keysym == 'Left':
            triggered_intention = "move_left"

            if window_in_border is not None:
                if not window_in_border:
                    triggered_action = "move_left"
                # else, window doesn't move to the left (stays the same)

        # Zooming out to original xlims or to previous zoomed in lims
        elif event.keysym == 'Shift_L':
            triggered_intention = "adjust_in_time"

            if var_zoomed_in is not None:
                # if the window was not zoomed in, it should zoom in
                if not var_zoomed_in:
                    triggered_action = "zooming_in"

                # else, it should zoom out
                else:
                    triggered_action = "zooming_out"

        # if Left Control is pressed
        elif event.keysym == 'Control_L':

            triggered_intention = "adjust_in_amplitude"

            if var_toggle_Ctrl is not None:

                # if the window was not zoomed in, it should zoom in
                if var_toggle_Ctrl.get() == 0:
                    triggered_action = "zooming_in"

                # else, it should zoom out
                else:
                    triggered_action = "zooming_out"

        elif event.keysym == 'Escape':
            triggered_intention = "quit_ui"

    return_dict = {'triggered_intention': triggered_intention, 'triggered_action': triggered_action}
    return return_dict

# possible intentions and actions
UI_intentions_actions = {

    "adjust_in_amplitude": {None: "Error", "zooming_in": "Amplitude Zoom-in", "zooming_out": "Amplitude Zoom-out"},

    "adjust_in_time": {None: "Error", "zooming_in": "Time zoom-in", "zooming_out": "Time zoom-out"},

    "move_right": {None: "Cannot move further to the right (signal ends there).", "move_right": "Moving right"},
    "move_left": {None: "Cannot move further to the left (signal ends there).", "move_left": "Moving left"},

    "add_annotation": {None: "Click on \'Edit Annotations\' checkbox", "add_annotation": "Adding Annotation"},
    "rmv_annotation": {None: "Click on \'Edit Annotations\' checkbox", "rmv_annotation": "Removing Annotation",
                       "rmv_annotation_not_close": "Click closer to the annotation."},

}

# listed BioSPPy default filtering methods and main feature extraction methods.
list_functions = {'EDA': {
    'Basic SCR Extractor - Onsets': {'preprocess': signals.preprocess_eda, 'function': eda.basic_scr,
                                     'template_key': 'onsets'},
    'Basic SCR Extractor - Peaks': {'preprocess': signals.preprocess_eda, 'function': eda.basic_scr,
                                    'template_key': 'onsets'},
    'KBK SCR Extractor - Onsets': {'preprocess': signals.preprocess_eda, 'function': eda.kbk_scr,
                                   'template_key': 'peaks'},
    'KBK SCR Extractor - Peaks': {'preprocess': signals.preprocess_eda, 'function': eda.kbk_scr,
                                  'template_key': 'peaks'}},
    'ECG': {'R-peak Hamilton Segmenter': {'preprocess': signals.preprocess_ecg,
                                          'function': ecg.hamilton_segmenter, 'template_key': 'rpeaks'},
            'R-peak SSF Segmenter': {'preprocess': signals.preprocess_ecg, 'function': ecg.ssf_segmenter,
                                     'template_key': 'rpeaks'},
            'R-peak Christov Segmenter': {'preprocess': signals.preprocess_ecg,
                                          'function': ecg.christov_segmenter, 'template_key': 'rpeaks'},
            'R-peak Engzee Segmenter': {'preprocess': signals.preprocess_ecg,
                                        'function': ecg.engzee_segmenter, 'template_key': 'rpeaks'},
            'R-peak Gamboa Segmenter': {'preprocess': signals.preprocess_ecg,
                                        'function': ecg.gamboa_segmenter, 'template_key': 'rpeaks'},
            'R-peak ASI Segmenter': {'preprocess': signals.preprocess_ecg, 'function': ecg.ASI_segmenter,
                                     'template_key': 'rpeaks'}},
    'ABP': {'Onset Extractor': {'preprocess': signals.preprocess_abp, 'function': abp.find_onsets_zong2003,
                                'template_key': 'onsets'}},
    'EMG': {'Basic Onset Finder': {'preprocess': signals.preprocess_emg, 'function': emg.find_onsets,
                                   'template_key': 'onsets'}},
    'PCG': {
        'Basic Peak Finger': {'preprocess': None, 'function': pcg.find_peaks, 'template_key': 'peaks'}},
    'PPG': {'Elgendi Onset Finder': {'preprocess': signals.preprocess_ppg,
                                     'function': ppg.find_onsets_elgendi2013, 'template_key': 'onsets'},
            'Kavsaoglu Onset Finder': {'preprocess': signals.preprocess_ppg,
                                       'function': ppg.find_onsets_kavsaoglu2016,
                                       'template_key': 'onsets'}}
}

plot_colors = list(mcolors.TABLEAU_COLORS.values())

def rescale_signals(input_signal: np.ndarray, new_min, new_max):
    """Normalizes the signal to new minimum and maximum values.

    Parameters
    ----------
    input_signal : np.ndarray
        Input signal.
    new_min : float
        New minimum of the output signal.
    new_max : float
        New minimum of the output signal.

    Returns
    -------
    output_signal : np.ndarray
        The output signal.

    """

    # first normalize from 0 to 1
    input_signal = (input_signal - np.min(input_signal)) / (np.max(input_signal) - np.min(input_signal))

    # then normalize to the [new_min, new_max] range
    output_signal = input_signal * (new_max - new_min) + new_min

    return output_signal


def milliseconds_to_samples(time_milliseconds: int, sampling_rate: float):
    """Converts a duration value in milliseconds to samples.

    Parameters
    ----------
    time_milliseconds : int
        Time in milliseconds to be converted.
    sampling_rate : float
        Sampling rate in Hertz.

    Returns
    -------
    samples : int
        The converted number of samples.

    """
    return int(time_milliseconds * (int(sampling_rate) / 1000))