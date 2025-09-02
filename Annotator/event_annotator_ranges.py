# Imports
import math
import os
import platform
import sys
import threading
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from biosppy import tools as st

from .aux import get_signals_as_dict_v4, filter_eda, acc_multi_filtering, vm_extractor
from .config import UI_intention, UI_intentions_actions, plot_colors, milliseconds_to_samples, \
    rescale_signals
import pandas as pd

class event_annotator_ranges:
    """Opens an editor of event annotations of the input biosignal modality."""
    global list_functions

    def __init__(self, sampling_rate, signals_dir, saving_dir, device_name, window_size=6.0, window_stride=1.5,
                 annotations_dir=None):
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
        """

        # If no directory is provided, it remains None -> when saving / loading annotations, default dir will open
        self.annotations_dir = annotations_dir
        self.segments_fpath = os.path.join(saving_dir, 'checker.txt')
        self.max_view_nr = 0
        self.range_index = None
        self.device_name = device_name
        self.saving_dir = saving_dir
        self.signals_dir = signals_dir
        self.current_segment_nr = 0
        self.current_subj_id = 0

        self.load_signals()
        self.get_signal()

        self.init_segments()


        # Extracting metadata
        self.sampling_rate = sampling_rate

        self.nr_acc_signals = sum("ACC" in item for item in self.modalities)

        self.raw_signals = self.input_signals

        # it's easier to expand one dim when single-channel than to check the dimensions for every loop in which signals are plotted.
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

        self.time_arr = np.arange(0, self.raw_signals.shape[0] / self.sampling_rate, 1 / self.sampling_rate)

        self.nr_samples = self.raw_signals.shape[0]

        # --- window setup
        self.root = Tk()
        self.root.resizable(True, True)
        self.root.state("zoomed")

        # rows
        self.root.grid_rowconfigure(0, weight=0)  # headerx
        self.root.grid_rowconfigure(1, weight=0)  # message
        self.root.grid_rowconfigure(2, weight=0)  # arrows
        self.root.grid_rowconfigure(3, weight=1)  # plot
        self.root.grid_rowconfigure(4, weight=0)  # toolbar
        self.root.grid_rowconfigure(5, weight=0)  # real time log
        self.root.grid_rowconfigure(6, weight=1)  # image instructions


        # --- header (row 0)
        self.f1 = Frame(self.root)
        self.f1.grid(row=0, column=0, sticky=W)

        # --- navigation buttons (row 1)
        self.header = Label(self.root, text="0%", font=("Arial", 25))
        self.header.grid(row=1, column=0, pady=(10, 10))
        self.update_percent(0)

        nav_frame = Frame(self.root)
        nav_frame.grid(row=2, column=0, pady=(10, 5))

        self.back_btn = Button(nav_frame, text="⟵ Back", command=self.on_back)
        self.back_btn.pack(side="left", padx=5)

        self.forward_btn = Button(nav_frame, text="Forward ⟶", command=self.on_forward)
        self.forward_btn.pack(side="left", padx=5)

        self.draw_plots(row_plot=3, row_toolbar=4)

        # Controllable On / Off variables
        self.var_edit_plots = IntVar()  # enable / disable annotation editing
        self.var_edit_plots.set(1)  # edit is disabled at start so that the user won't add

        self.var_autoscale = IntVar()  # enable / disable annotation editing
        self.var_autoscale.set(1)  # edit is disabled at start so that the user won't add

        self.var_toggle_Ctrl = IntVar()  # Fit ylims to amplitude within displayed window / show original ylims
        self.var_view_raw_signal = IntVar()  # Enable / disable view overlapped raw signal
        self.var_view_raw_signal.set(1)

        self.var_zoomed_in = False  # Zoom-in / Zoom-out
        self.previous_rmv_annotation_intention = False  # False = Not within closest range, True = within range

        # Variable to store last Zoom-in configuration, so that when zooming-in again it returns back to previous
        # zoomed-in view
        self.zoomed_in_lims = [0, self.moving_window_size]

        # Add keyword image
        # Create an object of tkinter ImageTk
        root_path = os.path.dirname(sys.argv[0])
        img = ImageTk.PhotoImage(Image.open(os.path.join(root_path, "Annotator", "rsc", "biosppy_annotator_instructions.png")))

        # Create a Label Widget to display the text or Image
        self.image_label = Label(self.root, image=img)
        self.image_label.grid(row=6, column=0)


        # Check Button to allow editing annotations (left here as an option)
        self.annotation_checkbox = Checkbutton(self.f1, text='Edit Annotations', variable=self.var_edit_plots,
                                               onvalue=1,
                                               offvalue=0)

        self.autoscale_checkbox = Checkbutton(self.f1, text='Auto-scale', variable=self.var_autoscale,
                                               onvalue=1,
                                               offvalue=0)

        self.annotation_ind = 0
        self.moving_window_ind = 0

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
        self.pressed_key_display.grid(row=5, column=0)

        self.pressed_key_display.update()

        self.last_save = True
        self.moving_onset = None
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

        self.root.mainloop()

    def load_signals(self):
        file_name_ext = '.txt' if self.device_name == 'bitalino' else '.csv'
        self.csv_fnames = [x for x in os.listdir(self.signals_dir) if x.endswith(f"{self.device_name}{file_name_ext}")]

    def get_signal(self):

        if self.current_subj_id is None:
            self.current_subj_id = 0

        csv_fname = self.csv_fnames[self.current_subj_id]

        self.subject_id = str(csv_fname.split("_")[0])
        print(f"-> Sample: {self.subject_id}")

        tmp_path = os.path.join(self.signals_dir, csv_fname)

        # skip comment lines starting with '#', keep header row
        df = pd.read_csv(tmp_path, comment="#")

        signals = {}
        print(df)
        signals['EDA'] = df["EDA"].to_numpy()
        signals["ACC_X"] = df["ACC_X"].to_numpy()
        signals["ACC_Y"] = df["ACC_Y"].to_numpy()
        signals["ACC_Z"] = df["ACC_Z"].to_numpy()

        # signals = get_signals_as_dict_v4(tmp_path, signal_type=self.device_name)

        device_signals = signals #signals[self.device_name]
        eda_signal = device_signals['EDA']
        eda_filt = filter_eda(eda_signal, sampling_rate=100., EDR=False)

        if self.device_name == 'sympathia':

            raw_acc_x_sym = device_signals['ACC_X']
            raw_acc_y_sym = device_signals['ACC_Y']
            raw_acc_z_sym = device_signals['ACC_Z']

            [acc_x_sym, acc_y_sym, acc_z_sym] = acc_multi_filtering([raw_acc_x_sym, raw_acc_y_sym, raw_acc_z_sym],
                                                                    window_sz_ms=500,
                                                                    fs=100.)

            acc_combines = np.column_stack((acc_x_sym, acc_y_sym, acc_z_sym))
            acc_vm = vm_extractor(signal=acc_combines)

            self.input_signals = np.stack((eda_filt, acc_vm), axis=-1)
            self.modalities = ['EDA', 'Accelerometer']

        else:
            eda_signal = signals['bitalino']['EDA']
            eda_filt = filter_eda(eda_signal, sampling_rate=1000., EDR=False)
            edr_filt = filter_eda(eda_signal, sampling_rate=1000., EDR=True)

            self.input_signals = np.stack((eda_filt, edr_filt), axis=-1)
            self.modalities = ['EDA', 'EDR']

        saving_suffix = f'_{self.device_name}_martfs.csv'
        self.saving_path_fname = os.path.join(csv_fname.split("_")[0] + saving_suffix)


    def load_checker(self):
        self.checker_fpath = os.path.join(self.saving_dir, 'checker.txt')
        if not os.path.exists(self.checker_fpath):
            with open(self.checker_fpath, "w") as f:
                f.write("# Checker file\n")
                f.write("subject_id, max_view_nr\n")

    def draw_plots(self, row_plot, row_toolbar):
        global list_functions

        self.raw_signals = self.input_signals

        # it's easier to expand one dim when single-channel than to check the dimensions for every loop in which signals are plotted.
        if self.raw_signals.ndim == 1:
            self.raw_signals = np.expand_dims(self.raw_signals, axis=-1)

        self.time_arr = np.arange(0, self.raw_signals.shape[0] / self.sampling_rate, 1 / self.sampling_rate)

        # Sub-signals or channels of the same signal modality
        self.nr_signals = self.raw_signals.shape[1]

        # --- figure
        self.figure, self.axs = plt.subplots(self.nr_plots, 1, figsize=(8, 3), dpi=100, constrained_layout=True)

        if self.nr_plots == 1:
            self.axs = [self.axs]
        for ax in self.axs:
            ax.set_ylabel("Amplitude [-]")
        self.axs[-1].set_xlabel("Time [s]")

        # --- canvas (CENTERED)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.root)
        # no sticky -> stays centered within the expanding cell
        self.canvas_plot.get_tk_widget().grid(row=row_plot, column=0, padx=10, pady=10)

        # callbacks
        self.canvas_plot.callbacks.connect('button_press_event', self.on_click_ax)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_key_press)

        # --- toolbar (centered under canvas)
        self.toolbarFrame = Frame(master=self.root)
        self.toolbarFrame.grid(row=row_toolbar, column=0)  # no sticky
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbarFrame)

        self.main_plots = {}

        for modality_name, i in zip(self.modalities, range(self.nr_signals)):

            tmp_axis, = self.axs[i].plot(self.time_arr, self.raw_signals[:, i], linewidth=0.8, color=plot_colors[i])
            self.main_plots['{}_raw'.format(modality_name)] = [tmp_axis ,self.raw_signals[:, i]]

        self.axs[0].axhline(y=7e6, linestyle='--', label='8M', color='orange', alpha=0.7, linewidth=0.8,)
        self.axs[0].axhline(y=300e3, linestyle='--', label='8M', color='orange', alpha=0.7, linewidth=0.8,)

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

        # Setting plot title for one or multiple signal modalities
        self.axs[0].set_title('Main Signal: ' + self.modalities[0],
                              fontsize=8)

        self.axs[1].set_title('Aux Signal: ' + self.modalities[1],
                              fontsize=8)

        self.canvas_plot.draw()

    def hide_text(self):
        """Hides existing displayed log message."""
        self.pressed_key_display.config(text=" ")
        self.pressed_key_display.update()

    def on_key_press(self, event):
        """Displays a log message to help the user."""
        triggered_result = UI_intention(event, var_edit_plots=self.var_edit_plots, var_toggle_Ctrl=self.var_toggle_Ctrl,
                                        var_zoomed_in=self.var_zoomed_in,
                                        window_in_border=self.moving_window_is_in_border(),
                                        closest_event=self.previous_rmv_annotation_intention)

        triggered_intention = triggered_result['triggered_intention']
        triggered_action = triggered_result['triggered_action']

        if triggered_intention is not None:
            display_text = UI_intentions_actions[triggered_intention][triggered_action]

            self.pressed_key_display.config(text=display_text)

            t = threading.Timer(2, self.hide_text)
            t.start()

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
                                           self.main_plots['{}_raw'.format(modality_name)][1], linewidth=0.8,
                                           alpha=1.0,
                                           color=color_tmp)

                self.main_plots['{}_raw'.format(modality_name)][0] = tmp_ax

        else:
            for modality_name, i in zip(self.modalities, range(self.nr_signals)):
                # Note: this .remove() function is from matplotlib!
                print(self.main_plots['{}_raw'.format(modality_name)])
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
                    ax[0].remove()
                    ax[1].remove()
                    ax[2].remove()

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


    def load_annotations(self, annotations_path):

        try:
            df = pd.read_csv(annotations_path, delimiter=",")

            print("Loading annotations...")
            # Clear any existing annotations
            self.annotation_pack = [{'values': [], 'axes': []} for _ in range(self.nr_plots)]

            for _, row in df.iterrows():
                # modality = row["modality"]
                if 'sympathia' in annotations_path:
                    modality = 'EDA_sympathia'
                else:
                    modality = 'EDA_bitalino'

                r_i = int(row["range_i"])
                r_f = int(row["range_f"])

                if modality in self.modalities:
                    idx = self.modalities.index(modality)

                    ax_i = self.axs[idx].vlines(
                        x=r_i / self.sampling_rate,
                        ymin=-1e6,
                        ymax=10e6,
                        colors='#FF6B66'
                    )

                    ax_f = self.axs[idx].vlines(
                        x=r_f / self.sampling_rate,
                        ymin=-1e6,
                        ymax=10e6,
                        colors='#FF6B66'
                    )

                    range_ax = self.axs[idx].axvspan(r_i / self.sampling_rate, r_f / self.sampling_rate, ymin=-1e6, ymax=10e6,
                                                               alpha=0.4, edgecolor='none',
                                                               facecolor='#FF6B66')

                    self.annotation_pack[idx]["axes"].append([ax_i, ax_f, range_ax])
                    self.annotation_pack[idx]["values"].append([r_i, r_f])

            print(self.annotation_pack)
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

                            ax_i = self.axs[idx].vlines(
                                x=r_i / self.sampling_rate,
                                ymin=-1e6,
                                ymax=10e6,
                                colors='#FF6B66'
                            )

                            ax_f = self.axs[idx].vlines(
                                x=r_f / self.sampling_rate,
                                ymin=-1e6,
                                ymax=10e6,
                                colors='#FF6B66'
                            )

                            range_ax = self.axs[idx].axvspan(r_i / self.sampling_rate, r_f / self.sampling_rate, ymin=-1e6, ymax=10e6,
                                                                       alpha=0.4, edgecolor='none',
                                                                       facecolor='#FF6B66')

                            self.annotation_pack[idx]["axes"].append([ax_i, ax_f, range_ax])
                            self.annotation_pack[idx]["values"].append([r_i, r_f])

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
                annotation_rows.append(
                    {"modality": modality, "range_i": int(round(sample[0])), "range_f": int(round(sample[1]))})

        annotations_df = pd.DataFrame(annotation_rows)

        if self.annotation_pack[0]['values']:
            # Optional: sort by modality/sample
            annotations_df = annotations_df.sort_values(by=["modality", "range_i"]).reset_index(drop=True)
            annotations_df.to_csv(saving_path_main, index=False)

            print("...done")
            self.last_save = True

    def on_closing(self):
        """If there are unsaved annotation changes, prompts the user if he wants to quit the GUI."""

        if not self.last_save:

            if messagebox.askokcancel("Quit", "Do you want to quit without saving?"):
                self.root.destroy()
            else:
                print("canceled...")

        else:
            self.root.destroy()

    def moving_window_is_in_border(self):
        """Return whether the current moving window is already placed at the border of the signal."""

        if self.zoomed_in_lims[1] == self.time_arr[-1] or self.zoomed_in_lims[0] == self.time_arr[0]:
            return True
        else:
            return False

    def annotation_navigator(self, event):
        """Navigates the signal based on a moving window and pressing Right and Left arrow keys."""

        # Moving to the right (right arrow) unless the left most limit + shift surpasses the length of signal
        if UI_intention(event)['triggered_intention'] == "move_right":

            if self.zoomed_in_lims[1] + self.window_shift <= self.time_arr[-1]:

                # d * self.moving_window_size) < len(self.acc_signal_filt):
                self.zoomed_in_lims[0] += self.window_shift
                self.zoomed_in_lims[1] += self.window_shift

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

            else:
                self.zoomed_in_lims[1] = self.time_arr[-1]
                self.zoomed_in_lims[0] = self.zoomed_in_lims[1] - self.moving_window_size

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

            self.current_segment_nr = math.ceil(self.zoomed_in_lims[1] / self.window_shift)
            self.update_df()

            if self.var_autoscale.get() == 1:

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

            self.canvas_plot.draw()


        # Moving to the left (left arrow)
        elif UI_intention(event)['triggered_intention'] == "move_left":

            if self.zoomed_in_lims[0] - self.window_shift >= 0:

                self.zoomed_in_lims[0] -= self.window_shift
                self.zoomed_in_lims[1] -= self.window_shift

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

            else:

                self.zoomed_in_lims[0] = 0
                self.zoomed_in_lims[1] = self.zoomed_in_lims[0] + self.moving_window_size

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

            self.current_segment_nr = math.ceil(self.zoomed_in_lims[1] / self.window_shift)
            self.update_df()

            if self.var_autoscale.get() == 1:

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

            self.canvas_plot.draw()

        # Zooming out to original xlims or to previous zoomed in lims
        elif UI_intention(event)['triggered_intention'] == "adjust_in_time":

            # if the window was already zoomed in, it should zoom out to original xlims
            if self.var_zoomed_in:

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.original_xlims[0], self.original_xlims[1])

                self.canvas_plot.draw()

                self.var_zoomed_in = False

            elif not self.var_zoomed_in:

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])
                self.canvas_plot.draw()
                self.var_zoomed_in = True

            if self.var_autoscale.get() == 1:

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

            self.canvas_plot.draw()

        # if Left Control is pressed, the signals should be normalized in amplitude, within the displayed window
        elif UI_intention(event)['triggered_intention'] == "adjust_in_amplitude":

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
        """Adds a annotation at the clicked location within the plotting area (Left mouse click) or deletes annotation that are close to
        the clicked coordinates (Right mouse click). """

        clicked_ax = event.inaxes

        if clicked_ax is None:
            return  # Click was outside the plotting area

        # Find which axes was clicked
        for i, ax in enumerate(self.axs):  # assuming self.axs is a list of your subplot axes
            if ax == clicked_ax:
                clicked_index = i
                break
        else:
            return  # No matching axis found

        if UI_intention(event)['triggered_intention'] == "rmv_annotation" and self.var_edit_plots.get() == 1 and self.range_index is None:

            closest_annotation = self.closest_nearby_event(event, clicked_index)

            # using a "detection" window of 0.2s (=200 ms)
            if closest_annotation is not None:
                del self.annotation_pack[clicked_index]["values"][closest_annotation]

                # del self.annotation_pack[clicked_index]["values"][closest_annotation]
                self.annotation_pack[clicked_index]["axes"][closest_annotation][0].remove()
                self.annotation_pack[clicked_index]["axes"][closest_annotation][1].remove()
                self.annotation_pack[clicked_index]["axes"][closest_annotation][2].remove()

                del self.annotation_pack[clicked_index]["axes"][closest_annotation]

                self.last_save = False

        elif UI_intention(event)['triggered_intention'] == "add_annotation" and self.var_edit_plots.get() == 1:
            print("add annotation")
            first_line_val = int(event.xdata * self.sampling_rate)

            # if no lines, add first line
            if self.range_index is None:

                range_ok = True

                for range_set in self.annotation_pack[clicked_index]['values']:
                    # if any of the range_set values is empty, then
                    if range_set[0] <= first_line_val <= range_set[1]:
                        range_set = False
                        break

                if range_ok:

                    self.annotation_pack[clicked_index]['values'].append([first_line_val, None])

                    first_line_ax = self.axs[clicked_index].vlines(
                        event.xdata,
                        ymin=-1e6,
                        ymax=10e6,
                        colors='#FF6B66')

                    self.annotation_pack[clicked_index]['axes'].append([first_line_ax, None, None])

                    self.range_index = clicked_index

            # if 1 line exists, add second line
            elif clicked_index == self.range_index:

                second_line_val = int(event.xdata * self.sampling_rate)
                sub_pack = self.annotation_pack[self.range_index]

                range_ok = True

                # there must be one set with a None value, otherwise, the range cannot be established for any axis
                # then this one must be the candidate since there are not any other half-empty ranges. Save index
                # lista = [0,1,2,3]
                # xml = [example for example in lista if example > 0]

                current_line_idx = [j for j in range(len(sub_pack['values'])) if not all(sub_pack['values'][j])][0]
                first_line_val = sub_pack['values'][current_line_idx][0]

                for i in range(len(sub_pack['values'])):

                    if i != current_line_idx:
                        if (sub_pack['values'][i][0] <= second_line_val <= sub_pack['values'][i][1]) or (second_line_val == sub_pack['values'][i][0] or second_line_val == sub_pack['values'][i][1]) or second_line_val == first_line_val:
                            range_ok = False

                # range ok, let's plot 1) the line and 2) the range
                if range_ok:

                    second_line_ax = self.axs[clicked_index].vlines(
                        event.xdata,
                        ymin=-1e6,
                        ymax=10e6,
                        colors='#FF6B66')

                    # paint the second line first
                    if second_line_val > first_line_val:
                        self.annotation_pack[self.range_index]['values'][current_line_idx][1] = second_line_val
                        self.annotation_pack[self.range_index]['axes'][current_line_idx][1] = second_line_ax
                        range_i = first_line_val / self.sampling_rate
                        range_f = second_line_val / self.sampling_rate

                    else:
                        self.annotation_pack[self.range_index]['values'][current_line_idx] = [second_line_val, first_line_val]
                        self.annotation_pack[self.range_index]['axes'][current_line_idx][1] = self.annotation_pack[self.range_index]['axes'][current_line_idx][0]
                        self.annotation_pack[self.range_index]['axes'][current_line_idx][0] = second_line_ax
                        range_i = second_line_val / self.sampling_rate
                        range_f = first_line_val / self.sampling_rate


                    self.annotation_pack[self.range_index]['axes'][current_line_idx][2] = second_line_ax

                    range_ax = self.axs[clicked_index].axvspan(range_i, range_f, ymin=-1e6, ymax=10e6, alpha=0.4, edgecolor='none',
                                 facecolor='#FF6B66')

                    self.annotation_pack[self.range_index]['axes'][current_line_idx][2] = range_ax

                # reset this var
                self.range_index = None
                self.last_save = False

                print(self.annotation_pack)

        self.canvas_plot.draw()

    def closest_nearby_event(self, event, clicked_index):
        """Checks whether a clicked location is close enough to a nearby annotation."""
        closest_annotation = None

        nr_ranges = len(self.annotation_pack[clicked_index]['values'])
        ranges_for_this_idx = self.annotation_pack[clicked_index]['values']

        self.previous_rmv_annotation_intention = False
        closest_annotation_idx = None

        try:
            for r, j in zip(ranges_for_this_idx, range(nr_ranges)):
                idx_i, idx_f = r[0] / self.sampling_rate, r[1] / self.sampling_rate

                # test if event.xdata is closeby the idx_i or idx_f or if it's within the range
                cond_1 = idx_i <= event.xdata <= idx_f
                cond_2 = (idx_i - 0.2 < event.xdata < idx_i + 0.2)
                cond_3 = (idx_f - 0.2 < event.xdata < idx_f + 0.2)

                # 1) if xdata is within range
                if cond_1 or cond_2 or cond_3:
                    self.previous_rmv_annotation_intention = True
                    closest_annotation_idx = j

        # if it doesn't work, it's because tmeplate_pakc is empty
        except:
            pass

        return closest_annotation_idx

    def update_percent(self, percent):
        if percent >= 99:
            color = "#80c0ff"  # light blue
        elif percent >= 70:
            color = "#80ff80"  # light green
        elif percent >= 40:
            color = "#ffd280"  # light orange
        else:
            color = "#ff8080"  # light red

        self.header.config(
            text=f"Viewed {percent}%" if color != "#80c0ff" else f"Viewed {percent}% - Done! (move onto next one ⟶)",
            bg=color
        )
        return percent

    def update_df(self):
        self.max_view_nr = max(self.current_segment_nr, self.max_view_nr)
        percent_curr = np.round(100 * (self.max_view_nr / ((self.time_arr.shape[0]) / (self.sampling_rate * self.window_shift))), 2)

        if percent_curr > 100:
            percent_curr = 100

        self.segments_df.loc[
            self.segments_df["subject_id"] == str(self.subject_id),
            "max_view_nr"
        ] = percent_curr

        self.segments_df.to_csv(
            self.segments_fpath,
            index=False  # avoids writing the DataFrame index
        )
        self.update_percent(percent_curr)

    def add_df_entry(self):
        percent_curr = np.round(100 * (self.max_view_nr / ((self.time_arr.shape[0]) / (self.sampling_rate * self.window_shift))), 2)

        mask = self.segments_df["subject_id"] == str(self.subject_id)

        if mask.any():
            # Take the max between existing value(s) and new one
            self.segments_df.loc[mask, "max_view_nr"] = np.maximum(
                self.segments_df.loc[mask, "max_view_nr"].astype(float),
                percent_curr
            )
        else:
            # Add new row
            new_row = {
                "subject_id": str(self.subject_id),
                "max_view_nr": percent_curr
            }
            self.segments_df = pd.concat(
                [self.segments_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

        self.update_df()

    def on_back(self):
        self.save_data()

        self.update_df()
        self.get_previous_fname()
        self.reset_plot()
        self.get_signal()
        self.draw_plots(3, 4)
        self.zoomed_in_lims = [0, self.moving_window_size]
        self.current_segment_nr = 0
        self.max_view_nr = 0
        self.add_df_entry()

    def on_forward(self):
        self.save_data()

        self.update_df()
        self.get_next_fname()
        self.reset_plot()
        self.get_signal()
        self.draw_plots(3, 4)
        self.zoomed_in_lims = [0, self.moving_window_size]
        self.current_segment_nr = 0
        self.max_view_nr = 0
        self.add_df_entry()

    def init_segments(self):
        print(self.segments_fpath)

        if not os.path.exists(self.segments_fpath):
            # File does not exist → create new with header and first subject
            with open(self.segments_fpath, "w") as f:
                f.write("subject_id,max_view_nr\n")
                f.write(f"{self.subject_id},0\n")

            self.segments_df = pd.read_csv(
                self.segments_fpath,
                comment="#",
                skipinitialspace=True,
                dtype={"subject_id": str}
            )

        self.segments_df = pd.read_csv(self.segments_fpath, comment="#", skipinitialspace=True, dtype={"subject_id": str})

        # check if any subject_id
        if (self.segments_df["subject_id"] == self.subject_id).any():
            print(f"There are rows with subject_id = {self.subject_id}")

        else:
            with open(self.segments_fpath, "a") as f:
                f.write(f"{self.subject_id}, 0\n")

    def get_previous_fname(self):
        if self.current_subj_id > 0:
            self.current_subj_id -= 1

    def get_next_fname(self):
        if self.current_subj_id < len(self.csv_fnames) - 1:
            self.current_subj_id += 1