import base64
import json
import math
import os
import platform

import matplotlib
from matplotlib.figure import Figure
import time
import numpy as np
from tkinter import *
import pandas as pd
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from src.Annotator.config import *
from src.Functions.bit_eda_sensor import bitalino_eda_to_us
from src.Functions.read_write import get_signals_as_dict
from src.Functions.signal_processing import *
from src.Functions.sym_eda_sensor import raw_to_conductance, dac_settling_time
import random

class event_annotator_ranges:
    """Opens an editor of event annotations of the input biosignal modality."""

    def __init__(self, sampling_rate, os_name, signals_dir, saving_dir, device_name, tf_weights, window_size=6.0, window_stride=1.5,
                 annotations_dir=None, annotator_id=None):
        """Initializes the event annotator.

        Parameters
        ----------
        signals : np.ndarray
            Input raw signal.
        modalities : list
            Metadata is a list of modalities ['EDA', 'ECG', 'TMP']
        window_size : float, optional
            Size of the moving window used for navigation in seconds.
        window_stride : float, optional
            Sride of the moving window used for navigation in seconds.
        moving_avg_wind_sz : int, optional
            Window size of the moving average filter used to obtain the filtered signals in milliseconds.
        """""

        self.os_name = os_name
        self.annotations_dir = annotations_dir
        self.segments_fpath = os.path.join(saving_dir, f'{device_name}_checker.txt')
        self.max_view_nr = 0
        self.range_index = None
        self.device_name = device_name
        self.saving_dir = saving_dir
        self.signals_dir = signals_dir
        self.current_segment_nr = 0
        self.current_subj_id = 0
        self.tf_weights = tf_weights

        self.annotator_id = annotator_id

        # Extracting metadata
        self.sampling_rate = sampling_rate

        self.load_signals()
        self._init_annotation_files()

        matplotlib.rcParams['path.simplify'] = True
        matplotlib.rcParams['path.simplify_threshold'] = 1.0
        matplotlib.rcParams['agg.path.chunksize'] = 10000

        # restore last subject before get_signal
        _tmp = _read_checker(self.segments_fpath)
        if _tmp.get("last_subj") and _tmp["last_subj"] in self.csv_fnames:
            self.current_subj_id = self.csv_fnames.index(_tmp["last_subj"])

        self.get_signal()
        self.init_segments()

        self.checker["order"] = self.csv_fnames
        self.checker["annotator"] = str(self.annotator_id)

        self.nr_acc_signals = sum("ACC" in item for item in self.modalities)

        self.raw_signals = self.input_signals

        # single-channel signals arrive 1-D; expand so shape[1] is always valid
        if self.raw_signals.ndim == 1:
            self.raw_signals = np.expand_dims(self.raw_signals, axis=-1)

        # Sub-signals or channels of the same signal modality
        self.nr_signals = self.raw_signals.shape[1]

        # when plotting, we always reduce the 1-3 ACC signal(s) to just one, to simplify
        self.nr_plots = self.nr_signals - self.nr_acc_signals + 1 if self.nr_acc_signals >= 1 else self.nr_signals

        # then we also need to reduce the modalities
        acc_indexes = [i for i, item in enumerate(self.modalities) if "ACC" in item]

        if self.nr_acc_signals > 1:
            first_index = acc_indexes[0]
            # Remove all ACC-containing elements
            self.modalities = [item for item in self.modalities if "ACC" not in item]
            # Insert "ACC" at the original index of the first
            self.modalities.insert(first_index, "ACC")

        self.moving_window_size = window_size
        self.window_shift = window_stride

        self.nr_samples = self.raw_signals.shape[0]

        # --- window setup
        self.root = Tk()
        _who = f" — {self.annotator_id}" if self.annotator_id else ""
        self.root.title(f"{self.device_name.capitalize()} EDA Annotator{_who}")

        _tkver = self.root.tk.call('info', 'patchlevel')
        try:
            _parts = tuple(int(x) for x in _tkver.split('.')[:3])
        except ValueError:
            _parts = (0, 0, 0)

        if _parts < (8, 6, 13):
            messagebox.showwarning(
                "Outdated Tk",
                f"Tk {_tkver} detected — mouse clicks may be silently dropped on macOS.\n"
                "Please use a Python 3.12+ build from python.org.")

        if self.os_name == 'Windows':
            self.root.tk.call("tk", "scaling", 3)

        # self.root.resizable(False, False)
        self.root.state("zoomed")

        # Controllable On / Off variables (needed before the first draw)
        self.var_edit_plots = IntVar()
        self.var_edit_plots.set(1)
        self.var_autoscale = IntVar()
        self.var_autoscale.set(1)
        self.var_toggle_Ctrl = IntVar()
        self.var_view_raw_signal = IntVar()
        self.var_view_raw_signal.set(1)

        self.previous_rmv_annotation_intention = False

        for _k in [k for k in plt.rcParams if k.startswith('keymap.')]:
            plt.rcParams[_k] = []

        # rows
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)  # header
        self.root.grid_rowconfigure(1, weight=0)  # progress bars
        self.root.grid_rowconfigure(2, weight=0)  # nav buttons
        self.root.grid_rowconfigure(3, weight=0)  # legend
        self.root.grid_rowconfigure(4, weight=2)  # plot
        self.root.grid_rowconfigure(5, weight=0)  # toolbar
        self.root.grid_rowconfigure(6, weight=0)  # real time log

        # --- header (row 0)
        self.f1 = Frame(self.root)
        self.f1.grid(row=0, column=0, sticky=W)

        # --- navigation buttons (row 1)
        self._build_progress_bar()
        self.update_percent(0)

        self._build_nav_frame()
        self._build_legend()

        self.var_zoomed_in = False  # Zoom-in / Zoom-out
        self._switching = False
        self._window_overlay = []

        self.draw_plots(row_plot=4, row_toolbar=5)
        self.load_annotations()

        self.zoomed_in_lims = [0.0, float(self.moving_window_size)]
        self._load_subject_progress()  # ← add here (progress bar loads on open)

        # Check Button to allow editing annotations (left here as an option)
        self.annotation_checkbox = Checkbutton(self.f1, text='Edit Annotations', variable=self.var_edit_plots,
                                               onvalue=1,
                                               offvalue=0)

        self.autoscale_checkbox = Checkbutton(self.f1, text='Auto-scale', variable=self.var_autoscale,
                                               onvalue=1,
                                               offvalue=0)

        # Drop menu for "file" section
        self.file_menu = Menubutton(self.f1, text="File", relief='raised')

        self.file_menu.menu = Menu(self.file_menu, tearoff=0)
        self.file_menu["menu"] = self.file_menu.menu
        self.file_menu.update()

        file_menu_options = ["Open", "Save As..."]

        for menu_option in file_menu_options:
            self.file_menu.menu.add_command(label=menu_option,
                                            command=lambda option=menu_option: self.file_options(option))

        # Drop menu for "file" section
        self.edit_menu = Menubutton(self.f1, text="Edit", relief='raised')

        self.edit_menu.menu = Menu(self.edit_menu, tearoff=0)
        self.edit_menu["menu"] = self.edit_menu.menu
        self.edit_menu.update()

        self.edit_menu_options = {"Reset Annotations": ["Reset Annotations"]}

        for menu_option, menu_values in self.edit_menu_options.items():
            self.edit_menu.menu.add_command(label=menu_values[0],
                                            command=lambda option=menu_option: self.edit_options(option))

        self.view_raw_button = Checkbutton(self.f1, text='View Raw', variable=self.var_view_raw_signal,
                                            onvalue=1,
                                            offvalue=0,
                                            command=self.view_raw_signals)

        # packing
        self.file_menu.pack(side="left")
        self.edit_menu.pack(side="left")
        self.annotation_checkbox.pack(side="left")
        self.autoscale_checkbox.pack(side="left")
        self.view_raw_button.pack(side="left")

        self.pressed_key_display = Label(self.root, text=" ", font=('Helvetica', 14, 'bold'))
        self.pressed_key_display.grid(row=6, column=0)

        self.pressed_key_display.update()

        self.last_save = True
        self._undo_stack = []

        self.range_index = None # None is None (adding the first line), any other value is the index of the axes where the first line is idle.

        for x in self.axs:
            x.set_autoscale_on(False)

        self.root.bind("<Key>", self.on_key_press)

        self.root.bind_all('<Key>', self.annotation_navigator)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Choose the correct modifier based on the OS
        if platform.system() == 'Darwin':
            modifier = 'Command'
        else:
            modifier = 'Control'

        # Bind the shortcut
        self.root.bind(f'<{modifier}-s>', self.save_action)
        self.root.bind(f'<{modifier}-z>', self.undo_last_deletion)

        def _report_exc(exc, val, tb):
            import traceback
            traceback.print_exception(exc, val, tb)
            self._flash(f"Error: {val}")

        self.root.report_callback_exception = _report_exc

        self.root.lift()
        self.root.focus_force()
        self.canvas_plot.get_tk_widget().focus_set()

        self.root.mainloop()

    def load_signals(self):
        file_name_ext = '.txt' if self.device_name == 'bitalino' else '.csv'
        self.csv_fnames = [x for x in sorted(os.listdir(self.signals_dir))
                           if x.endswith(f"{self.device_name}{file_name_ext}")]

        if self.annotator_id:
            random.Random(str(self.annotator_id)).shuffle(self.csv_fnames)

    def get_signal(self):

        if self.current_subj_id is None:
            self.current_subj_id = 0

        signals_fname = self.csv_fnames[self.current_subj_id]

        self.subject_id = str(signals_fname.split("_")[0])
        print(f"-> Sample: {self.subject_id}")

        s_path = os.path.join(self.signals_dir, signals_fname)
        signals = None

        if self.device_name == 'bitalino':
            signals = get_signals_as_dict(s_path, device_name=self.device_name)

        elif self.device_name == 'sympathia':
            signals = get_signals_as_dict(s_path, device_name=self.device_name)

        if self.device_name == 'sympathia':
            signals = signals["sympathia"]

            # EDA
            # eda_signal = eda_signal
            eda_sym = signals['EDA']
            dac_sym = signals['DAC']
            eda_sym, crop_idx = raw_to_conductance(eda_sym, dac_sym, self.tf_weights, crop_leading_nans=True)

            eda_sym = eda_sym[crop_idx:]  # unfiltered ADC, for saturation masks
            dac_sym = dac_sym[crop_idx:]

            # --- second crop: DAC settling + tolerance ---
            settle_idx, transitions, intervals_s = dac_settling_time(dac_sym)
            crop2 = settle_idx + int(10 * self.sampling_rate)

            eda_signal = eda_sym[crop2:]
            eda_signal = filter_eda(eda_signal, self.sampling_rate, EDR=False)

            # ACC
            # raw_acc_x_sym = signals['ACC_X'][total_crop:]
            # raw_acc_y_sym = signals['ACC_Y'][total_crop:]
            # raw_acc_z_sym = signals['ACC_Z'][total_crop:]
            #
            # [acc_x_sym, acc_y_sym, acc_z_sym] = acc_multi_filtering(np.asarray([raw_acc_x_sym, raw_acc_y_sym, raw_acc_z_sym]), fs=self.sampling_rate)
            #
            # acc_combines = np.column_stack((acc_x_sym, acc_y_sym, acc_z_sym))
            # acc_vm = vm_extractor(signal=acc_combines)

            # self.input_signals = np.stack((eda_signal, acc_vm), axis=-1)
            # self.modalities = ['EDA', 'Accelerometer']

            self.input_signals = np.stack((eda_signal), axis=-1)
            self.modalities = ['EDA']

        else:
            signals = signals["bitalino"]
            eda_signal = signals['EDA']

            eda_signal = bitalino_eda_to_us(eda_signal, n_bits=10, vcc=3.3)
            eda_signal = filter_eda(eda_signal, self.sampling_rate, EDR=False)

            self.input_signals = np.stack((eda_signal), axis=-1)
            self.modalities = ['EDA']

        saving_suffix = f'_{self.device_name}_artifacts.csv'
        self.saving_path_fname = os.path.join(signals_fname.split("_")[0] + saving_suffix)

    def draw_plots(self, row_plot, row_toolbar):
        global list_functions

        if hasattr(self, 'canvas_plot'):
            try:
                self.toolbar.destroy()
                self.toolbarFrame.destroy()
                self.canvas_plot.get_tk_widget().destroy()
            except Exception:
                pass

        self.raw_signals = self.input_signals

        # it's easier to expand one dim when single-channel than to check the dimensions for every loop in which
        # signals are plotted.
        if self.raw_signals.ndim == 1:
            self.raw_signals = np.expand_dims(self.raw_signals, axis=-1)

        self.time_arr = np.arange(self.raw_signals.shape[0]) / self.sampling_rate
        self.nr_signals = self.raw_signals.shape[1]

        # --- figure
        self.figure = Figure(figsize=(8, 3), dpi=100)  # no layout='constrained'
        self.axs = self.figure.subplots(self.nr_plots, 1)
        self.figure.subplots_adjust(left=0.055, right=0.99, top=0.96, bottom=0.11)

        if self.nr_plots == 1:
            self.axs = [self.axs]

        for ax in self.axs:
            ax.set_ylabel("Amplitude [$\mu$S]")
            ax.set_xmargin(0)
            ax.set_xlim(float(self.time_arr[0]), float(self.time_arr[-1]))

        # Saving x- and y-lims of original signals

        self.axs[-1].set_xlabel("Time [s]")

        # --- canvas
        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas_plot.get_tk_widget().grid(row=row_plot, column=0,
                                              padx=10, pady=(2, 2), sticky="nsew")
        # callbacks
        self.canvas_plot.callbacks.connect('button_press_event', self.on_click_ax)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_key_press)

        # --- toolbar (centered under canvas)
        self.toolbarFrame = Frame(master=self.root)
        self.toolbarFrame.grid(row=row_toolbar, column=0)  # no sticky
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbarFrame)

        self.main_plots = {}

        for modality_name, i in zip(self.modalities, range(self.nr_signals)):

            tmp_axis, = self.axs[i].plot(self.time_arr, self.raw_signals[:, i], linewidth=UI_settings[self.os_name]['linewidth'], color=plot_colors[i])

            self.main_plots['{}_raw'.format(modality_name)] = [tmp_axis ,self.raw_signals[:, i]]

        # Saving x- and y-lims of original signals
        self.original_xlims = self.axs[0].get_xlim()
        self.original_ylims = []

        for i in range(self.nr_plots):
            self.original_ylims.append(self.axs[i].get_ylim())

        # List of Dictionaries to store annotated events
        self.annotation_pack = []
        for i in range(self.nr_plots):
            self.annotation_pack.append({
                'values': [],
                'axes': []
            })

        self.canvas_plot.draw()

    def hide_text(self):
        self._hide_text_after = None
        try:
            self.pressed_key_display.config(text=" ")
        except Exception:
            pass

    def on_key_press(self, event):
        ks = getattr(event, 'keysym', None)

        if ks in ('Left', 'a', 'A'):
            in_border = self.zoomed_in_lims[0] <= float(self.time_arr[0])
        elif ks in ('Right', 'd', 'D'):
            in_border = self.zoomed_in_lims[1] >= float(self.time_arr[-1])
        else:
            in_border = False

        triggered_result = UI_intention(
            event,
            var_edit_plots=self.var_edit_plots,
            var_toggle_Ctrl=self.var_toggle_Ctrl,
            var_zoomed_in=self.var_zoomed_in,
            window_in_border=in_border,
            closest_event=self.previous_rmv_annotation_intention,
        )
        triggered_intention = triggered_result['triggered_intention']
        triggered_action = triggered_result['triggered_action']

        if triggered_intention is not None:
            display_text = UI_intentions_actions[triggered_intention][triggered_action]
            self._flash(display_text)

    def view_raw_signals(self):
        """Enables or disables the display of the filtered signal(s)."""

        # if user toggled "view filtered signals"
        if self.var_view_raw_signal.get() == 1:

            for modality_name, i in zip(self.modalities, range(self.nr_plots)):

                if self.nr_signals == 1:
                    color_tmp = 'black'
                else:
                    color_tmp = plot_colors[i]

                tmp_ax, = self.axs[i].plot(self.time_arr,
                                           self.main_plots['{}_raw'.format(modality_name)][1],
                                           alpha=1.0,
                                           color=color_tmp)

                self.main_plots['{}_raw'.format(modality_name)][0] = tmp_ax

        else:
            for modality_name, i in zip(self.modalities, range(self.nr_signals)):
                # Note: this .remove() function is from matplotlib!
                self.main_plots['{}_raw'.format(modality_name)][0].remove()
                del self.main_plots['{}_raw'.format(modality_name)][0]
                self.main_plots['{}_raw'.format(modality_name)].insert(0, None)

        self.canvas_plot.draw()

    def reset_plot(self):

        for modality_name, i in zip(self.modalities, range(self.nr_signals)):
            # Note: this .remove() function is from matplotlib!
            self.main_plots['{}_raw'.format(modality_name)][0].remove()
            del self.main_plots['{}_raw'.format(modality_name)][0]
            self.main_plots['{}_raw'.format(modality_name)].insert(0, None)

    def reset_annotations(self, draw_canva=True):
        """Deletes all existing annotations and refreshes the plot."""
        # Remove all existing annotations
        try:

            for i in range(len(self.annotation_pack)):

                annotation_set = self.annotation_pack[i]

                for ax in self.annotation_pack[i]['axes']:
                    for artist in ax:
                        try:
                            artist.remove()
                        except Exception:
                            pass

                # Clear the lists inside the dict
                annotation_set['axes'].clear()
                annotation_set['values'].clear()

            # finally re-draw everything back
            if draw_canva:
                self.canvas_plot.draw()

        except Exception as e:
            print("Failed to reset annotations:", e)

    def edit_options(self, option):
        """Dropdown menu for Edit options. Currently only supporting loading."""

        if option == "Reset Annotations":
            self.reset_annotations()


    def load_annotations(self):

        curr_fname = self.csv_fnames[self.current_subj_id].split('.')[0] + '_artifacts.csv'
        curr_csv_fpath = os.path.join(self.annotations_dir, curr_fname)

        try:
            df = pd.read_csv(curr_csv_fpath, delimiter=",")
            print("Loading annotations...")
            # Clear any existing annotations
            self.annotation_pack = [{'values': [], 'axes': []} for _ in range(self.nr_plots)]

            for _, row in df.iterrows():
                modality = 'EDA'

                r_i = int(row["range_i"])
                r_f = int(row["range_f"])

                if modality in self.modalities:
                    idx = self.modalities.index(modality)

                    cert = str(row["certainty"]) if "certainty" in df.columns else "legacy"
                    self.annotation_pack[idx]["axes"].append(self._draw_range(idx, r_i, r_f, cert))
                    self.annotation_pack[idx]["values"].append([r_i, r_f, cert])

        except Exception as e:
            print("Failed to load annotations:", e)

        self.canvas_plot.draw()
        print("Annotations successfully custom-loaded!")

    def file_options(self, option):
        """Dropdown menu for File options. Currently only supporting loading."""

        files = [('Text Document', '*.txt'),
                 ('Comma-Separated Values file', '*.csv')]

        if option == "Open":
            # If more than one sub-signal, guess the name of the first as default
            # init_filename_guess = self.modalities[0] + "_annotations.txt"

            annotations_path = filedialog.askopenfile(filetypes=files, defaultextension=files,
                                                      title="Loading Annotations file",
                                                      initialdir=self.annotations_dir)
            # initialfile=init_filename_guess)

            if annotations_path is not None:
                try:

                    df = pd.read_csv(annotations_path.name, delimiter=",")

                    print("Loading annotations...")

                    # Clear any existing annotations
                    self.annotation_pack = [{'values': [], 'axes': []} for _ in range(self.nr_plots)]

                    for _, row in df.iterrows():
                        modality = row["modality"]
                        r_i = int(row["range_i"])
                        r_f = int(row["range_f"])

                        if modality in self.modalities:
                            idx = self.modalities.index(modality)

                            cert = str(row["certainty"]) if "certainty" in df.columns else "legacy"
                            self.annotation_pack[idx]["axes"].append(self._draw_range(idx, r_i, r_f, cert))
                            self.annotation_pack[idx]["values"].append([r_i, r_f, cert])

                except Exception as e:
                    print("Failed to load annotations:", e)

            self.canvas_plot.draw()
            print("Annotations successfully loaded!")

        elif option == "Save As...":

            # If more than one sub-signal, guess the name of the first as default
            init_filename_guess = os.path.join(self.saving_dir, self.saving_path_fname)

            saving_path_main = filedialog.asksaveasfile(filetypes=files, defaultextension=files,
                                                        title="Saving Annotations file",
                                                        initialdir=self.annotations_dir,
                                                        initialfile=init_filename_guess)

            try:
                print("Saving annotations...")

                self.save_data()

            except:
                pass


    def save_action(self, event):
        self.save_data()

    def save_data(self):

        saving_path_main = os.path.join(os.path.join(self.saving_dir, self.saving_path_fname))
        annotation_rows = []

        for i in range(self.nr_plots):

            modality = self.modalities[i]
            annotations = self.annotation_pack[i]["values"]

            for sample in annotations:
                if sample[1] is None:
                    continue  # half-open range, not yet committed
                annotation_rows.append({
                    "modality": modality,
                    "range_i": int(round(sample[0])),
                    "range_f": int(round(sample[1])),
                    "certainty": sample[2],
                })

        annotations_df = pd.DataFrame(annotation_rows)

        if annotation_rows:
            annotations_df = annotations_df.sort_values(by=["modality", "range_i"]).reset_index(drop=True)
        else:
            annotations_df = pd.DataFrame(columns=["modality", "range_i", "range_f", "certainty"])

        annotations_df.to_csv(saving_path_main, index=False)

        print("...done")
        self.last_save = True

    def on_closing(self):
        """If there are unsaved annotation changes, prompts the user if he wants to quit the GUI."""
        self._flush_checker()

        if not self.last_save:

            if messagebox.askokcancel("Quit", "Do you want to quit without saving?"):
                self.checker["last_subj"] = self.csv_fnames[self.current_subj_id]
                _write_checker(self.segments_fpath, self.checker)
                self.root.destroy()
            else:
                print("canceled...")

        else:
            self.checker["last_subj"] = self.csv_fnames[self.current_subj_id]
            _write_checker(self.segments_fpath, self.checker)
            self.root.destroy()

    def annotation_navigator(self, event):
        """Navigates the signal based on a moving window and pressing Right and Left arrow keys."""
        if getattr(self, '_switching', False):
            return

        intent = UI_intention(event)['triggered_intention']

        if intent == "toggle_certainty":
            self.toggle_certainty()

        elif intent == "move_right":
            was_zoomed_out = not self.var_zoomed_in
            self.var_zoomed_in = True

            if was_zoomed_out:
                pass
            elif self.zoomed_in_lims[1] + self.window_shift <= self.time_arr[-1]:
                self.zoomed_in_lims[0] += self.window_shift
                self.zoomed_in_lims[1] += self.window_shift
            else:
                self.zoomed_in_lims[1] = self.time_arr[-1]
                self.zoomed_in_lims[0] = self.zoomed_in_lims[1] - self.moving_window_size

            self.current_segment_nr = math.ceil(self.zoomed_in_lims[1] / self.window_shift)
            self.update_df()
            self._schedule_render()

        # Moving to the left (left arrow)
        elif intent == "move_left":

            was_zoomed_out = not self.var_zoomed_in
            self.var_zoomed_in = True

            if was_zoomed_out:
                pass  # entering the window IS the move — don't shift
            elif self.zoomed_in_lims[0] - self.window_shift >= 0:
                self.zoomed_in_lims[0] -= self.window_shift
                self.zoomed_in_lims[1] -= self.window_shift
            else:
                self.zoomed_in_lims[0] = 0
                self.zoomed_in_lims[1] = self.moving_window_size

            self.current_segment_nr = math.ceil(self.zoomed_in_lims[1] / self.window_shift)
            self.update_df()
            self._schedule_render()

        # Zooming out to original xlims or to previous zoomed in lims
        elif intent == "adjust_in_time":
            if self.var_zoomed_in:
                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.original_xlims[0], self.original_xlims[1])
                self.var_zoomed_in = False
            else:
                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                self.var_zoomed_in = True

            self.update_df()
            if self.var_autoscale.get() == 1:
                self._autoscale_y()
            self._draw_window_overlay()
            self.canvas_plot.draw_idle()

        # if Left Control is pressed, the signals should be normalized in amplitude, within the displayed window
        elif intent == "adjust_in_amplitude":

            # if currently zoomed-out, then zoom-in
            if self.var_toggle_Ctrl.get() == 0:

                # Setting baseline min/max values is needed because if there are no filtered signals, these values
                # won't interfer with the min/max computation on the raw signals
                # values for min set at unreasonably high number (1000)
                min_max_within_window = 10e6 * np.ones((2 * self.nr_signals, 2))

                # values for max set at unreasonably high number (-1000)
                min_max_within_window[:, 1] = -1 * min_max_within_window[:, 1]

                for label, i in zip(self.modalities, range(self.nr_signals)):
                    # let's find the maximum and minimum values within the current window

                    # if the "view filtered signal" is toggled, means that filtered signals are currently displayed,
                    # therefore the axes exist

                    # min and max indices of x-axis within current window to get min and max in amplitude
                    idx_i = max(int(self.sampling_rate * self.axs[i].get_xlim()[0]), 0)
                    idx_f = min(int(self.sampling_rate * self.axs[i].get_xlim()[1]), self.nr_samples)

                    # anyways, we always compute for raw signals
                    min_max_within_window[self.nr_signals + i, 0] = np.min(
                        self.main_plots['{}_raw'.format(label)][1][idx_i:idx_f])

                    min_max_within_window[self.nr_signals + i, 1] = np.max(
                        self.main_plots['{}_raw'.format(label)][1][idx_i:idx_f])

                    min_in_window = min(min_max_within_window[self.nr_signals + i, 0], min_max_within_window[i, 0])
                    max_in_window = max(min_max_within_window[self.nr_signals + i, 1], min_max_within_window[i, 1])

                    self.axs[i].set_ylim(min_in_window, max_in_window)

                self.var_toggle_Ctrl.set(1)
                self.canvas_plot.draw()

            else:
                for i in range(self.nr_plots):
                    self.axs[i].set_ylim(self.original_ylims[i][0], self.original_ylims[i][1])

                self.var_toggle_Ctrl.set(0)
                self.canvas_plot.draw()

        elif event.keysym == 'Escape':
            self.on_closing()

    def on_click_ax(self, event):
        """Left click: opens a range on the first click, closes it on the second.
        Right click: deletes the range under the cursor."""

        if getattr(self, '_switching', False):
            return

        clicked_ax = event.inaxes
        if clicked_ax is None:
            return  # click was outside the plotting area

        # which subplot was clicked
        for i, ax in enumerate(self.axs):
            if ax == clicked_ax:
                clicked_index = i
                break
        else:
            return  # no matching axis

        if self.var_edit_plots.get() != 1:
            return

        intent = UI_intention(event)['triggered_intention']
        pending = self._pending_range(clicked_index)  # source of truth
        sub_pack = self.annotation_pack[clicked_index]
        click_val = int(event.xdata * self.sampling_rate)

        # ---------------------------------------------------------- delete
        if intent == "rmv_annotation" and pending is None:
            closest = self.closest_nearby_event(event, clicked_index)
            if closest is not None:
                self._undo_stack.append({
                    "plot_index": clicked_index,
                    "values": list(sub_pack["values"][closest]),
                })
                for artist in sub_pack["axes"][closest]:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                del sub_pack["values"][closest]
                del sub_pack["axes"][closest]
                self.save_data()

        # ------------------------------------------------ open a new range
        elif intent == "add_annotation" and pending is None:
            inside = next((v for v in sub_pack['values']
                           if v[1] is not None and v[0] <= click_val <= v[1]), None)
            if inside is not None:
                self._flash("Can't start a range inside an existing one")
            else:
                line = self.axs[clicked_index].axvline(
                    event.xdata, color='#FF6B66',
                    linewidth=UI_settings[self.os_name]['linewidth'])
                sub_pack['values'].append([click_val, None, 'certain'])
                sub_pack['axes'].append([line, None, None])
                self.range_index = clicked_index

        # --------------------------------------------- close the open range
        elif intent == "add_annotation":
            r_i, r_f = sorted([sub_pack['values'][pending][0], click_val])

            if r_f <= r_i:
                reason = "Zero-width range — discarded"
            elif any(r_i < v[1] and v[0] < r_f
                     for j, v in enumerate(sub_pack['values'])
                     if j != pending and v[1] is not None):
                reason = "Range overlaps an existing one — discarded"
            else:
                reason = None

            # the pending first-click line goes either way
            try:
                sub_pack['axes'][pending][0].remove()
            except Exception:
                pass

            if reason is None:
                sub_pack['values'][pending] = [r_i, r_f, 'certain']
                sub_pack['axes'][pending] = self._draw_range(
                    clicked_index, r_i, r_f, 'certain')
                self.save_data()
            else:
                del sub_pack['values'][pending]
                del sub_pack['axes'][pending]
                self._flash(reason)

            self.range_index = None

        self._redraw_progress_bar()
        self.canvas_plot.draw_idle()

    def closest_nearby_event(self, event, clicked_index):
        """Checks whether a clicked location is close enough to a nearby annotation."""

        nr_ranges = len(self.annotation_pack[clicked_index]['values'])
        ranges_for_this_idx = self.annotation_pack[clicked_index]['values']

        self.previous_rmv_annotation_intention = False
        closest_annotation_idx = None

        for r, j in zip(ranges_for_this_idx, range(nr_ranges)):

            if r[1] is None: continue

            idx_i, idx_f = r[0] / self.sampling_rate, r[1] / self.sampling_rate

            # test if event.xdata is closeby the idx_i or idx_f or if it's within the range
            cond_1 = idx_i <= event.xdata <= idx_f
            cond_2 = (idx_i - 0.2 < event.xdata < idx_i + 0.2)
            cond_3 = (idx_f - 0.2 < event.xdata < idx_f + 0.2)

            # 1) if xdata is within range
            if cond_1 or cond_2 or cond_3:
                self.previous_rmv_annotation_intention = True
                closest_annotation_idx = j

        return closest_annotation_idx

    def undo_last_deletion(self, event=None):
        if getattr(self, '_switching', False):
            return
        if not self._undo_stack:
            return
        item = self._undo_stack.pop()
        idx = item["plot_index"]

        r_i, r_f, cert = item["values"]
        self.annotation_pack[idx]["axes"].append(self._draw_range(idx, r_i, r_f, cert))
        self.annotation_pack[idx]["values"].append([r_i, r_f, cert])

        self.save_data()
        self._redraw_progress_bar()
        self.canvas_plot.draw()

    def _build_progress_bar(self):
        self._bar_pct_win_start = 0.0
        self._bar_pct_win_end = 0.0
        self._bar_pct_viewed = 0.0

        frame = Frame(self.root)
        frame.grid(row=1, column=0, pady=(10, 6), sticky="ew", padx=40)
        frame.columnconfigure(0, weight=1)

        # ── overview bar (row 0) ──────────────────────────────────────────
        self.overview_canvas = Canvas(frame, height=16, highlightthickness=1,
                                      highlightbackground="#cccccc", bg="#e0e0e0")
        self.overview_canvas.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.overview_canvas.bind("<Configure>", lambda e: self._redraw_overview_bar())
        self.overview_canvas.bind("<Motion>", self._show_overview_tooltip)
        self.overview_canvas.bind("<Leave>", self._hide_tooltip)
        self.overview_canvas.bind("<Button-1>", self._on_overview_click)
        self.overview_canvas.config(cursor="hand2")

        self._tooltip = None

        # ── subject progress bar (row 1) ─────────────────────────────────
        self.progress_canvas = Canvas(frame, height=20, highlightthickness=1,
                                      highlightbackground="#cccccc", bg="#e0e0e0")
        self.progress_canvas.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        self.progress_canvas.bind("<Configure>", lambda e: self._redraw_progress_bar())

        self.progress_label = Label(frame, text="0% viewed",
                                    font=UI_settings[self.os_name]["viewed_font"])
        self.progress_label.grid(row=2, column=0, pady=(2, 0))

        self.subject_label = Label(frame, text="", font=UI_settings[self.os_name]["viewed_font"])
        self.subject_label.grid(row=3, column=0, pady=(0, 2))

    def _update_subject_label(self):
        n = len(self.csv_fnames)

        self.subject_label.config(text=f"[{self.device_name.capitalize()} EDA Signal] Subject: {self.subject_id}  ({self.current_subj_id + 1} / {n})")

    def _redraw_overview_bar(self):
        c = self.overview_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1 or not hasattr(self, 'csv_fnames') or not hasattr(self, 'checker'):
            return

        n = len(self.csv_fnames)
        if n == 0:
            return

        gap = 2
        rect_w = max(1, (w - gap * (n - 1)) / n)

        for i, fname in enumerate(self.csv_fnames):
            subj = str(fname.split("_")[0])
            entry = self.checker["subjects"].get(subj, {})
            pct = float(entry.get("max_view_nr", 0))

            if pct >= 100:
                color = "#a8f0b8"
            elif pct > 0:
                color = "#f0a8a8"
            else:
                color = "#cccccc"

            x0 = i * (rect_w + gap)
            x1 = x0 + rect_w
            c.create_rectangle(x0, 1, x1, h - 1, fill=color, outline="#999999", width=1)

            # dark blue dot on current subject
            if subj == self.subject_id:
                cx = (x0 + x1) / 2
                cy = h / 2
                r = min(4, (h - 4) / 2, rect_w / 2 - 1)
                c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#0c447c", outline="")

    def _redraw_progress_bar(self):
        c = self.progress_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1:
            return

        # trough
        c.create_rectangle(0, 0, w, h, fill="#e0e0e0", outline="")

        # viewed fill — always soft red, green only at 100%
        pv = self._bar_pct_viewed
        color = "#a8f0b8" if pv >= 100 else "#f0a8a8"
        fill_x = w if pv >= 100 else int(w * pv / 100)
        if fill_x > 0:
            c.create_rectangle(0, 0, fill_x, h, fill=color, outline="")

        # annotation ranges — dark bands proportional to signal length
        if hasattr(self, 'annotation_pack') and hasattr(self, 'time_arr'):
            signal_duration = float(self.time_arr[-1])
            for pack in self.annotation_pack:
                for r_i, r_f, *rest in pack['values']:
                    if r_f is None:
                        continue
                    cert = rest[0] if rest else 'certain'
                    st = CERTAINTY_STYLES.get(cert, CERTAINTY_STYLES['certain'])
                    ax0 = int(w * (r_i / self.sampling_rate) / signal_duration * 100 / 100)
                    ax1 = int(w * (r_f / self.sampling_rate) / signal_duration * 100 / 100)
                    ax1 = max(ax1, ax0 + 2)  # minimum 2px so it's always visible
                    c.create_rectangle(ax0, 2, ax1, h - 2, fill=st['bar'],
                                       outline="", stipple=st['stipple'])

        # current viewing window — blue rectangle, minimum 6px wide so it's always visible
        x0 = int(w * self._bar_pct_win_start / 100)
        x1 = int(w * self._bar_pct_win_end / 100)
        x1 = max(x1, x0 + 6)
        # c.create_rectangle(x0, 1, x1, h - 1, fill="#378add", outline="#0c447c", width=2)
        # current viewing window — outline only, no fill
        c.create_rectangle(x0, 1, x1, h - 1, fill="", outline="#0c447c", width=2)

        # black tick centered in the window
        x_mid = (x0 + x1) // 2
        c.create_line(x_mid, 0, x_mid, h, fill="#0c447c", width=2)

    def update_percent(self, percent):
        self._bar_pct_viewed = float(percent)
        return percent

    def update_position(self, pct_start, pct_end):
        self._bar_pct_win_start = float(pct_start)
        self._bar_pct_win_end = float(pct_end)

    def _update_progress_label(self):
        pv = self._bar_pct_viewed
        if pv >= 100:
            viewed_msg = f"{pv:.0f}% — done! ⟶"
        elif pv >= 70:
            viewed_msg = f"{pv:.0f}% viewed — almost there"
        elif pv >= 40:
            viewed_msg = f"{pv:.0f}% viewed — keep going"
        else:
            viewed_msg = f"{pv:.0f}% viewed"
        self.progress_label.config(text=viewed_msg)

    def update_df(self):
        self.max_view_nr = max(self.current_segment_nr, self.max_view_nr)
        total_windows = self.time_arr.shape[0] / (self.sampling_rate * self.window_shift)
        percent_curr = min(round(100 * self.max_view_nr / total_windows, 2), 100)
        last_pos = round(float(self.zoomed_in_lims[0]), 3)

        stored = self.checker["subjects"].get(self.subject_id, {})
        percent_curr = max(percent_curr, float(stored.get("max_view_nr", 0)))
        self.max_view_nr = max(self.max_view_nr, int(percent_curr / 100 * total_windows))

        # accumulate time silently
        elapsed = round(time.time() - self._session_start, 1)
        self._session_start = time.time()
        total_time = float(stored.get("time_spent", 0)) + elapsed

        self.checker["subjects"][self.subject_id] = {
            "max_view_nr": percent_curr,
            "last_pos": last_pos,
            "time_spent": total_time,
            "zoomed_in": bool(self.var_zoomed_in),
        }

        self._queue_checker_write()

        pct_start = min(round(100 * float(self.zoomed_in_lims[0]) / float(self.time_arr[-1]), 2), 100)
        pct_end = min(round(100 * float(self.zoomed_in_lims[1]) / float(self.time_arr[-1]), 2), 100)

        self.update_percent(percent_curr)
        self.update_position(pct_start, pct_end)
        self._schedule_ui_refresh()

    def init_segments(self):
        self.checker = _read_checker(self.segments_fpath)
        if self.subject_id not in self.checker["subjects"]:
            self.checker["subjects"][self.subject_id] = {"max_view_nr": 0.0, "last_pos": 0.0}
            _write_checker(self.segments_fpath, self.checker)

    def _build_legend(self):
        frame = Frame(self.root)
        frame.grid(row=3, column=0, pady=(0, 4))

        for name in ('certain', 'uncertain'):
            st = CERTAINTY_STYLES[name]
            Label(frame, text="      ", bg=st['swatch'],
                  relief='solid', borderwidth=1).pack(side="left", padx=(12, 4))
            Label(frame, text=st['label']).pack(side="left")

    def _build_nav_frame(self):
        nav_frame = Frame(self.root)
        nav_frame.grid(row=2, column=0, pady=(10, 5))

        # ── signal navigation ──────────────────────────────────────────
        Button(nav_frame, text="◀ Prev signal", command=self.on_back,
               font=UI_settings[self.os_name]["prev_next_font"],
               bg="#b0c4de", relief="raised").pack(side="left", padx=(5, 15))

        # ── window navigation ──────────────────────────────────────────
        Button(nav_frame, text="⟵ Back", command=lambda: self.jump_windows(-1),
               font=UI_settings[self.os_name]["prev_next_font"]).pack(side="left", padx=3)

        Button(nav_frame, text="⟵⟵ -10", command=lambda: self.jump_windows(-10),
               font=UI_settings[self.os_name]["prev_next_font"]).pack(side="left", padx=3)

        Button(nav_frame, text="⏮ Start", command=self.jump_to_start,
               font=UI_settings[self.os_name]["prev_next_font"]).pack(side="left", padx=3)

        Button(nav_frame, text="+10 ⟶⟶", command=lambda: self.jump_windows(10),
               font=UI_settings[self.os_name]["prev_next_font"]).pack(side="left", padx=3)

        Button(nav_frame, text="Forward ⟶", command=lambda: self.jump_windows(1),
               font=UI_settings[self.os_name]["prev_next_font"]).pack(side="left", padx=3)

        # ── signal navigation ──────────────────────────────────────────
        Button(nav_frame, text="Next signal ▶", command=self.on_forward,
               font=UI_settings[self.os_name]["prev_next_font"],
               bg="#b0c4de", relief="raised").pack(side="left", padx=(15, 5))

    def jump_to_start(self):
        """Snaps the view back to the very beginning of the signal."""
        self.zoomed_in_lims = [0.0, float(self.moving_window_size)]
        self.var_zoomed_in = True
        self.current_segment_nr = math.ceil(self.zoomed_in_lims[1] / self.window_shift)
        self.update_df()
        self._schedule_render()

    def jump_windows(self, n):
        new_start = self.zoomed_in_lims[0] + n * self.window_shift
        new_end = new_start + self.moving_window_size

        if new_start < 0:
            new_start = 0.0
            new_end = float(self.moving_window_size)
        if new_end > float(self.time_arr[-1]):
            new_end = float(self.time_arr[-1])
            new_start = new_end - float(self.moving_window_size)

        self.zoomed_in_lims = [new_start, new_end]
        self.var_zoomed_in = True
        self.current_segment_nr = math.ceil(self.zoomed_in_lims[1] / self.window_shift)
        self.update_df()
        self._schedule_render()

    def _load_subject_progress(self):
        self.max_view_nr = 0
        self.zoomed_in_lims = [0.0, float(self.moving_window_size)]

        entry = self.checker["subjects"].get(self.subject_id)
        if entry:
            stored_percent = float(entry["max_view_nr"])
            stored_pos = float(entry["last_pos"])
            total_windows = self.time_arr.shape[0] / (self.sampling_rate * self.window_shift)
            self.max_view_nr = int(stored_percent / 100 * total_windows)
            self.update_percent(stored_percent)
            pct_start = min(round(100 * stored_pos / float(self.time_arr[-1]), 2), 100)
            pct_end = min(round(100 * (stored_pos + self.moving_window_size) / float(self.time_arr[-1]), 2), 100)
            self.update_position(pct_start, pct_end)

            stored_zoom = bool(entry.get("zoomed_in", True))

            if stored_percent > 0:
                # restore the window position either way
                if stored_pos > 0:
                    self.zoomed_in_lims = [stored_pos,
                                           min(stored_pos + self.moving_window_size, float(self.time_arr[-1]))]

                if stored_zoom:
                    # was zoomed IN — restore the window view and autoscale
                    for ax in self.axs:
                        ax.set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                    if self.var_autoscale.get() == 1:
                        self._autoscale_y()
                    self.var_zoomed_in = True

                else:
                    # was zoomed OUT — full signal, overlay marks the locked window
                    for ax in self.axs:
                        ax.set_xlim(self.original_xlims[0], self.original_xlims[1])
                    self.var_zoomed_in = False
            else:
                for ax in self.axs:
                    ax.set_xlim(self.original_xlims[0], self.original_xlims[1])
                self.var_zoomed_in = False
        else:
            self.checker["subjects"][self.subject_id] = {"max_view_nr": 0.0, "last_pos": 0.0}
            _write_checker(self.segments_fpath, self.checker)
            self.update_percent(0)
            self.update_position(0, round(100 * self.moving_window_size / float(self.time_arr[-1]), 2))
            # never seen — show full signal overview
            for ax in self.axs:
                ax.set_xlim(self.original_xlims[0], self.original_xlims[1])
            self.var_zoomed_in = False

        self._update_subject_label()
        self._draw_window_overlay()
        self._redraw_progress_bar()
        self._update_progress_label()
        self.canvas_plot.draw()
        self._redraw_overview_bar()
        self._session_start = time.time()

    def _show_overview_tooltip(self, event):
        i = self._subject_index_at_x(event.x)
        if i is None:
            self._hide_tooltip()
            return
        if i == getattr(self, '_tooltip_idx', None):
            return  # same square — nothing changed
        self._tooltip_idx = i

        subj = str(self.csv_fnames[i].split("_")[0])
        entry = self.checker["subjects"].get(subj, {})
        pct = float(entry.get("max_view_nr", 0))
        self._show_tooltip(event, f"Subject: {subj}\n{pct:.0f}% viewed")

    def _show_tooltip(self, event, text):
        x = self.overview_canvas.winfo_rootx() + event.x + 12
        y = self.overview_canvas.winfo_rooty() + event.y + 12

        if self._tooltip is None:
            self._tooltip = Toplevel(self.root)
            self._tooltip.wm_overrideredirect(True)
            if self.os_name == 'Darwin':
                # non-activating help window — stops the activation churn
                self._tooltip.tk.call(
                    "::tk::unsupported::MacWindowStyle", "style",
                    self._tooltip._w, "help", "noActivates")
            self._tooltip_label = Label(
                self._tooltip, text=text, background="#ffffe0",
                relief="solid", borderwidth=1, font=('Helvetica', 11))
            self._tooltip_label.pack()
        else:
            self._tooltip_label.config(text=text)

        self._tooltip.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self, event=None):
        if self._tooltip:
            self._tooltip.destroy()
            self._tooltip = None

    def _check_all_complete(self):
        if not hasattr(self, 'csv_fnames') or not hasattr(self, 'checker'):
            return
        all_done = all(
            float(self.checker["subjects"].get(str(f.split("_")[0]), {}).get("max_view_nr", 0)) >= 100
            for f in self.csv_fnames
        )
        if all_done:
            self.progress_label.config(
                text="🎉 All signals complete! Great work!",
                fg="#2e7d32"
            )
    def _init_annotation_files(self):
        """Create empty annotation files for any subjects that don't have one yet."""
        for fname in self.csv_fnames:
            subject_id = fname.split("_")[0]
            ann_fname = f"{subject_id}_{self.device_name}_artifacts.csv"
            ann_path = os.path.join(self.annotations_dir, ann_fname)
            if not os.path.exists(ann_path):
                with open(ann_path, "w") as f:
                    f.write("modality,range_i,range_f,certainty\n")
                print(f"Created empty annotation file: {ann_fname}")

    def _draw_window_overlay(self):
        """Draw a rectangle on the axes showing current window position when zoomed out."""
        # remove previous overlay if exists
        if hasattr(self, '_window_overlay') and self._window_overlay:
            for span in self._window_overlay:
                try:
                    span.remove()
                except Exception:
                    pass
        self._window_overlay = []

        if self.var_zoomed_in:
            return  # only show overlay when zoomed out

        for ax in self.axs:
            span = ax.axvspan(
                self.zoomed_in_lims[0],
                self.zoomed_in_lims[1],
                alpha=0.15,
                facecolor="#0c447c",
                edgecolor="#0c447c",
                linewidth=1.5,
                zorder=5
            )
            self._window_overlay.append(span)

    def _subject_index_at_x(self, x):
        w = self.overview_canvas.winfo_width()
        n = len(self.csv_fnames)
        if n == 0 or w <= 1:
            return None
        gap = 2
        rect_w = max(1, (w - gap * (n - 1)) / n)
        i = int(x / (rect_w + gap))
        return i if 0 <= i < n else None

    def _switch_subject(self, new_index):

        if getattr(self, '_switching', False):
            return  # a switch is already running; drop queued clicks

        if new_index == self.current_subj_id or not (0 <= new_index < len(self.csv_fnames)):
            return

        self._flush_checker()

        self._switching = True
        self._render_pending = False
        self._ui_pending = False
        previous = self.current_subj_id
        try:
            self.save_data()
            self.update_df()  # must run BEFORE the id changes — it writes self.subject_id's entry
            self.current_subj_id = new_index
            self.range_index = None
            try:
                self.get_signal()
            except Exception as e:
                print(f"Failed to load subject index {new_index}: {e}")
                self.current_subj_id = previous
                self._flash("Could not load that signal")
                return

            self.draw_plots(4, 5)
            self.load_annotations()
            self.current_segment_nr = 0
            self._load_subject_progress()
            self.checker["last_subj"] = self.csv_fnames[self.current_subj_id]
            _write_checker(self.segments_fpath, self.checker)
            self._hide_tooltip()
        finally:
            self._switching = False

    def _on_overview_click(self, event):
        i = self._subject_index_at_x(event.x)
        if i is not None:
            self._switch_subject(i)

    def on_back(self):
        self._switch_subject(self.current_subj_id - 1)

    def on_forward(self):
        self._switch_subject(self.current_subj_id + 1)

    def _draw_range(self, idx, r_i, r_f, certainty='certain'):
        """Draws the two boundary lines and the shaded span. Returns the 3 artists."""
        st = CERTAINTY_STYLES.get(certainty, CERTAINTY_STYLES['certain'])
        lw = UI_settings[self.os_name]['linewidth']
        sr = self.sampling_rate

        ax_i = self.axs[idx].axvline(r_i / sr, color=st['color'], linewidth=lw)
        ax_f = self.axs[idx].axvline(r_f / sr, color=st['color'], linewidth=lw)

        range_ax = self.axs[idx].axvspan(
            r_i / sr, r_f / sr,
            alpha=st['alpha'], facecolor=st['color'],
            hatch=st['hatch'],
            edgecolor=st['color'] if st['hatch'] else 'none',
            linewidth=0,
        )
        return [ax_i, ax_f, range_ax]

    def toggle_certainty(self, event=None):

        if self._pending_range(0) is not None:
            return

        idx = 0
        vals = self.annotation_pack[idx]['values']
        if not vals:
            return

        j = len(vals) - 1  # most recently created range
        r_i, r_f, cert = vals[j]
        if r_f is None:
            return

        # only allow toggling a range that's currently on screen
        x_lo, x_hi = self.axs[idx].get_xlim()
        t_i, t_f = r_i / self.sampling_rate, r_f / self.sampling_rate
        if t_f < x_lo or t_i > x_hi:
            self._flash("Last range is off-screen — no change")
            return

        new_cert = 'uncertain' if cert != 'uncertain' else 'certain'

        for artist in self.annotation_pack[idx]['axes'][j]:
            try:
                artist.remove()
            except Exception:
                pass

        self.annotation_pack[idx]['axes'][j] = self._draw_range(idx, r_i, r_f, new_cert)
        vals[j] = [r_i, r_f, new_cert]

        self._flash(f"Last range → {new_cert.upper()}")
        self.save_data()
        self._redraw_progress_bar()
        self.canvas_plot.draw()

    def _flash(self, text):
        if not hasattr(self, 'pressed_key_display'):
            return

        self.pressed_key_display.config(text=text, font=UI_settings[self.os_name]['intention'])
        if getattr(self, '_hide_text_after', None):
            self.root.after_cancel(self._hide_text_after)
        self._hide_text_after = self.root.after(2000, self.hide_text)

    def _flush_checker(self):
        if getattr(self, '_checker_pending', None):
            self.root.after_cancel(self._checker_pending)
        self._checker_pending = None
        _write_checker(self.segments_fpath, self.checker)

    def _queue_checker_write(self):
        if getattr(self, '_checker_pending', None):
            self.root.after_cancel(self._checker_pending)
        self._checker_pending = self.root.after(800, self._flush_checker)

    def _schedule_render(self):
        if getattr(self, '_render_pending', False):
            return                      # one already queued — let it fire
        self._render_pending = True
        self.root.after(33, self._render_now)      # ~30 fps

    def _render_now(self):
        self._render_pending = False
        if getattr(self, '_switching', False):
            return
        for i in range(self.nr_plots):
            self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])


        if self.var_autoscale.get() == 1:
            self._autoscale_y()

        self._draw_window_overlay()
        self.canvas_plot.draw_idle()

    def _autoscale_y(self):
        for label, i in zip(self.modalities, range(self.nr_signals)):
            x_lo, x_hi = self.axs[i].get_xlim()
            idx_i = max(int(self.sampling_rate * x_lo), 0)
            idx_f = min(int(self.sampling_rate * x_hi), self.nr_samples)
            if idx_f <= idx_i:
                continue
            seg = self.main_plots[f'{label}_raw'][1][idx_i:idx_f]
            self.axs[i].set_ylim(np.min(seg), np.max(seg))

    def _schedule_ui_refresh(self):
        if getattr(self, '_ui_pending', False):
            return
        self._ui_pending = True
        self.root.after(100, self._ui_refresh_now)  # ~10 fps

    def _ui_refresh_now(self):
        self._ui_pending = False
        self._redraw_progress_bar()
        self._update_progress_label()
        self._redraw_overview_bar()
        self._check_all_complete()

    def _pending_range(self, idx):
        """Index of the half-open range in this plot, or None."""
        for j, v in enumerate(self.annotation_pack[idx]['values']):
            if v[1] is None:
                return j
        return None

    def _install_click_probes(self):
        w = self.canvas_plot.get_tk_widget()

        def _tk_probe(e):
            print(f"[tk ] x={e.x} y={e.y} num={getattr(e, 'num', '?')} "
                  f"grab={self.root.grab_current()} "
                  f"switching={getattr(self, '_switching', False)} "
                  f"mode={getattr(self.toolbar, 'mode', None)!r}", flush=True)

        w.bind("<Button-1>", _tk_probe, add="+")
        w.bind("<Button-2>", _tk_probe, add="+")
        w.bind("<Button-3>", _tk_probe, add="+")

        def _mpl_probe(e):
            print(f"[mpl] btn={e.button} inaxes={e.inaxes is not None} "
                  f"xdata={e.xdata}", flush=True)

        self.canvas_plot.callbacks.connect('button_press_event', _mpl_probe)

def _read_checker(fpath: str) -> dict:
    if not os.path.exists(fpath):
        return {"last_subj": None, "subjects": {}}
    try:
        with open(fpath, 'rb') as f:
            return json.loads(base64.b64decode(f.read()).decode())
    except Exception:
        return {"last_subj": None, "subjects": {}}

def _write_checker(fpath: str, data: dict) -> None:
    with open(fpath, 'wb') as f:
        f.write(base64.b64encode(json.dumps(data).encode()))
