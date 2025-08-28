# Imports
import os
import sys
import threading
import time
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, filedialog
from biosppy import tools as st
from .config import UI_intention, UI_intentions_actions, plot_colors, milliseconds_to_samples, rescale_signals, list_functions
import pandas as pd

class event_annotator_timestamps:
    """Opens an editor of event annotations of the input biosignal modality."""
    import pandas as pd
    global list_functions

    def __init__(self, signals, sampling_rate, modalities, subject_id, window_size=6.0, window_stride=1.5, moving_avg_wind_sz=700,
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

        # Extracting metadata
        self.sampling_rate = sampling_rate
        self.modalities = modalities

        self.nr_acc_signals = sum("ACC" in item for item in self.modalities)

        self.raw_signals = signals

        # it's easier to expand one dim when single-channel than to check the dimensions for every loop in which signals are plotted.
        if self.raw_signals.ndim == 1:
            self.raw_signals = np.expand_dims(self.raw_signals, axis=-1)

        # Sub-signals or channels of the same signal modality
        self.nr_signals = self.raw_signals.shape[1]

        # when plotting, we always reduce the 1-3 ACC signal(s) to just one, to simplify
        self.nr_plots = self.nr_signals - self.nr_acc_signals + 1 if self.nr_acc_signals >= 1 else self.nr_signals

        print(self.nr_plots)

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

        print(self.time_arr)

        self.nr_samples = self.raw_signals.shape[0]

        # tkinter figures and plots
        self.root = Tk()
        self.root.resizable(False, False)

        # self.ax = self.figure.add_subplot(111)

        self.figure, self.axs = plt.subplots(self.nr_plots, 1, figsize=(5, 3), dpi=100)

        if self.nr_plots == 1:
            self.axs = [self.axs]

        for i in range(self.nr_plots):
            self.axs[i].set_ylabel("Amplitude [-]")

        self.axs[-1].set_xlabel("Time [s]")

        self.canvas_plot = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas_plot.get_tk_widget().grid(row=1, column=0, columnspan=1, sticky='w', padx=10)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_click_ax)
        self.canvas_plot.callbacks.connect('button_press_event', self.on_key_press)

        self.toolbarFrame = Frame(master=self.root)
        self.toolbarFrame.grid(row=2, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbarFrame)

        self.main_plots = {}

        for modality_name, i in zip(self.modalities, range(self.nr_signals)):

            # in next version we need to have this structure different, for example to have 1 raw and 1 filt per channel
            try:
                preprocessing_function = list_functions[modality_name]["preprocess"]
                tmp_filt_signal = preprocessing_function(self.raw_signals[:, i], sampling_rate=self.sampling_rate)
                tmp_filt_signal = np.squeeze(tmp_filt_signal)

            except:
                # filter signal with a standard moving average with window=1s
                tmp_filt_signal, _ = st.smoother(self.raw_signals[:, i],
                                                 size=milliseconds_to_samples(moving_avg_wind_sz,
                                                                              self.sampling_rate))  # 500 ms

            tmp_filt_signal = rescale_signals(tmp_filt_signal, np.min(self.raw_signals[:, i]),
                                              np.max(self.raw_signals[:, i]))

            self.main_plots['{}_filt'.format(modality_name)] = [None, tmp_filt_signal]

            self.main_plots['{}_raw'.format(modality_name)] = [
                self.axs[i].plot(self.time_arr, self.raw_signals[:, i], linewidth=0.8, color=plot_colors[i]),
                self.raw_signals[:, i]]

        # Saving x- and y-lims of original signals
        self.original_xlims = self.axs[0].get_xlim()
        self.original_ylims = []

        for i in range(self.nr_plots):
            self.original_ylims.append(self.axs[i].get_ylim())

        # List of Dictionaries to store annotated events
        self.annotation_pack = []
        for i in range(self.nr_plots):
            self.annotation_pack.append({})

        # Setting plot title for one or multiple signal modalities
        for i in range(self.nr_plots):
            self.axs[i].set_title('Main Signal: ' + self.modalities[i])

        # Controllable On / Off variables
        self.var_edit_plots = IntVar()  # enable / disable annotation editing
        self.var_edit_plots.set(0)  # edit is disabled at start so that the user won't add

        self.var_toggle_Ctrl = IntVar()  # Fit ylims to amplitude within displayed window / show original ylims
        self.var_view_filtered_signal = IntVar()  # Enable / disable view overlapped filtered signal
        self.var_zoomed_in = False  # Zoom-in / Zoom-out
        self.previous_rmv_annotation_intention = False  # False = Not within closest range, True = within range

        # Variable to store last Zoom-in configuration, so that when zooming-in again it returns back to previous
        # zoomed-in view
        self.zoomed_in_lims = [0, self.moving_window_size]

        # Add keyword image
        # Create an object of tkinter ImageTk
        img = ImageTk.PhotoImage(Image.open(os.path.join(os.path.dirname(sys.argv[0]), "Annotator", "rsc", "biosppy_annotator_instructions.png")))

        # Create a Label Widget to display the text or Image
        self.image_label = Label(self.root, image=img)
        self.image_label.grid(row=4, column=0)

        self.f1 = Frame(self.root)
        self.f1.grid(row=0, column=0, sticky=W)

        # Check Button to allow editing annotations (left here as an option)
        self.annotation_checkbox = Checkbutton(self.f1, text='Edit Annotations', variable=self.var_edit_plots,
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

        # self.edit_menu_options = {"Edit Annotations": ["Lock Annotations", "Unlock Annotations"], "Reset Annotations": ["Reset Annotations"]}
        self.edit_menu_options = {"Reset Annotations": ["Reset Annotations"]}

        for menu_option, menu_values in self.edit_menu_options.items():
            self.edit_menu.menu.add_command(label=menu_values[0],
                                            command=lambda option=menu_option: self.edit_options(option))

        # Button to optionally view filtered signal
        self.overlap_raw = Checkbutton(self.f1, text='View filtered signal', variable=self.var_view_filtered_signal,
                                       onvalue=1,
                                       offvalue=0,
                                       command=self.view_filtered_signals)

        # Drop menu for computing or loading annotations
        self.annotation_opt_menu = Menubutton(self.f1, text="Compute annotations", relief='raised')
        self.annotation_opt_menu.menu = Menu(self.annotation_opt_menu, tearoff=0)
        self.annotation_opt_menu["menu"] = self.annotation_opt_menu.menu
        self.annotation_opt_menu.update()

        # packing
        self.file_menu.pack(side="left")
        self.edit_menu.pack(side="left")
        self.annotation_opt_menu.pack(side="left")
        self.annotation_checkbox.pack(side="left")
        self.overlap_raw.pack(side="left")

        self.pressed_key_display = Label(self.root, text=" ", font=('Helvetica', 14, 'bold'))
        self.pressed_key_display.grid(row=3, column=0)

        self.pressed_key_display.update()

        for k, v in list_functions.items():

            sub_menu = Menu(self.annotation_opt_menu.menu)

            self.annotation_opt_menu.menu.add_cascade(label=k, menu=sub_menu)

            for sub_k, sub_v in v.items():
                sub_menu.add_command(label=sub_k,
                                     command=lambda
                                         option="{},{}".format(k, sub_k): self.update_annotations_options(
                                         option))

        self.last_save = True
        self.moving_onset = None

        self.root.bind("<Key>", self.on_key_press)

        self.root.bind_all('<Key>', self.annotation_navigator)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

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

    def view_filtered_signals(self):
        """Enables or disables the display of the filtered signal(s)."""

        # if user toggled "view filtered signals"
        if self.var_view_filtered_signal.get() == 1:

            for modality_name, i in zip(self.modalities, range(self.nr_plots)):

                if self.nr_signals == 1:
                    color_tmp = 'black'
                else:
                    color_tmp = plot_colors[i]

                tmp_ax, = self.axs[i].plot(self.time_arr,
                                           self.main_plots['{}_filt'.format(modality_name)][1], linewidth=0.8,
                                           alpha=1.0,
                                           color=color_tmp)

                self.main_plots['{}_filt'.format(modality_name)][0] = tmp_ax

        else:
            for modality_name, i in zip(self.modalities, range(self.nr_signals)):
                # Note: this .remove() function is from matplotlib!
                self.main_plots['{}_filt'.format(modality_name)][0].remove()
                del self.main_plots['{}_filt'.format(modality_name)][0]
                self.main_plots['{}_filt'.format(modality_name)].insert(0, None)

        self.canvas_plot.draw()

    def reset_annotations(self, draw_canva=True):
        """Deletes all existing annotations and refreshes the plot."""
        # Remove all existing annotations
        for i in range(len(self.annotation_pack)):
            annotation_set = self.annotation_pack[i]
            for annotation in list(annotation_set.keys()):
                try:
                    self.annotation_pack[i][annotation].remove()
                    del self.annotation_pack[i][annotation]

                except:
                    pass

        # finally re-draw everything back
        if draw_canva:
            self.canvas_plot.draw()

    def edit_options(self, option):
        """Dropdown menu for Edit options. Currently only supporting loading."""

        if option == "Reset Annotations":
            self.reset_annotations()

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
                    import pandas as pd

                    df = pd.read_csv(annotations_path.name)

                    print("Loading annotations...")

                    # Clear any existing annotations
                    self.annotation_pack = [{} for _ in range(self.nr_plots)]

                    for _, row in df.iterrows():
                        modality = row["modality"]
                        sample = int(row["sample"])

                        if modality in self.modalities:
                            idx = self.modalities.index(modality)

                            annotation = self.axs[idx].vlines(
                                x=sample,
                                ymin=self.original_ylims[0],
                                ymax=self.original_ylims[1],
                                colors='#FF6B66'
                            )

                            self.annotation_pack[idx][sample] = annotation

                except Exception as e:
                    print("Failed to load annotations:", e)

        elif option == "Save As...":

            # If more than one sub-signal, guess the name of the first as default
            exp_time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
            init_filename_guess = "annotation" + exp_time_stamp + ".txt"

            saving_path_main = filedialog.asksaveasfile(filetypes=files, defaultextension=files,
                                                        title="Saving Annotations file",
                                                        initialdir=self.annotations_dir,
                                                        initialfile=init_filename_guess)

            if saving_path_main is not None:
                print("Saving annotations...")

                annotation_rows = []

                for i in range(self.nr_plots):
                    modality = self.modalities[i]
                    annotation_set = list(self.annotation_pack[i].keys())
                    annotations = [int(round(x)) for x in annotation_set]

                    for sample in annotations:
                        annotation_rows.append({"modality": modality, "sample": sample})

                # Convert to DataFrame
                import pandas as pd

                annotations_df = pd.DataFrame(annotation_rows)

                # Optional: sort by modality/sample
                annotations_df = annotations_df.sort_values(by=["modality", "sample"]).reset_index(drop=True)
                annotations_df.to_csv(saving_path_main.name, index=False)

                print("...done")
                self.last_save = True

        self.canvas_plot.draw()

    def update_annotations_options(self, option):
        """Dropdown menu for annotation options (load or compute annotations). Currently only supporting loading."""

        # start by removing all existing annotations (without plotting)
        self.reset_annotations(draw_canva=False)

        # then apply signal processing as requested by user
        if option == "load Annotations":
            files = [('Text Document', '*.txt'),
                     ('Comma-Separated Values file', '*.csv')]

            # If more than one sub-signal, guess the name of the first as default
            # init_filename_guess = self.modalities[0] + "_annotations.txt"

            annotations_path = filedialog.askopenfile(filetypes=files, defaultextension=files,
                                                      title="Loading Annotations file",
                                                      initialdir=self.annotations_dir)
            # initialfile=init_filename_guess)

            if annotations_path is not None:
                try:
                    import pandas as pd

                    df = pd.read_csv(annotations_path.name)

                    print("Loading annotations...")

                    # Clear any existing annotations
                    self.annotation_pack = [{} for _ in range(self.nr_plots)]

                    for _, row in df.iterrows():
                        modality = row["modality"]
                        sample = int(row["sample"])

                        if modality in self.modalities:
                            idx = self.modalities.index(modality)

                            annotation = self.axs[idx].vlines(
                                x=sample,
                                ymin=self.original_ylims[0],
                                ymax=self.original_ylims[1],
                                colors='#FF6B66'
                            )

                            self.annotation_pack[idx][sample] = annotation

                except Exception as e:
                    print("Failed to load annotations:", e)

        else:
            chosen_modality = option.split(",")[0]
            chosen_algorithm = option.split(",")[1]

            if chosen_modality in self.modalities:
                idx = self.modalities.index(chosen_modality)

                preprocess = list_functions[chosen_modality][chosen_algorithm]["preprocess"]
                function = list_functions[chosen_modality][chosen_algorithm]["function"]
                annotation_key = list_functions[chosen_modality][chosen_algorithm]["template_key"]

                # only computing for one of the sub-signals (the first one)
                if function is None:
                    input_extraction_signal = self.raw_signals[:, idx]
                else:
                    input_extraction_signal = preprocess(self.raw_signals[:, idx], sampling_rate=self.sampling_rate)
                    input_extraction_signal = input_extraction_signal['filtered']

                annotations = function(input_extraction_signal, sampling_rate=self.sampling_rate)[annotation_key]

                # adding loaded annotations
                for annotation in annotations:
                    annotation_temp = self.axs[idx].vlines(annotation / self.sampling_rate, self.original_ylims[0],
                                                           self.original_ylims[1],
                                                           colors='#FF6B66')

                    self.annotation_pack[idx][annotation] = annotation_temp

        # finally re-draw everything back
        self.canvas_plot.draw()

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

                self.canvas_plot.draw()

            else:
                self.zoomed_in_lims[1] = self.time_arr[-1]
                self.zoomed_in_lims[0] = self.zoomed_in_lims[1] - self.moving_window_size

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

                self.canvas_plot.draw()

        # Moving to the left (left arrow)
        elif UI_intention(event)['triggered_intention'] == "move_left":

            if self.zoomed_in_lims[0] - self.window_shift >= 0:

                self.zoomed_in_lims[0] -= self.window_shift
                self.zoomed_in_lims[1] -= self.window_shift

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

                self.canvas_plot.draw()

            else:

                self.zoomed_in_lims[0] = 0
                self.zoomed_in_lims[1] = self.zoomed_in_lims[0] + self.moving_window_size

                for i in range(self.nr_plots):
                    self.axs[i].set_xlim(self.zoomed_in_lims[0], self.zoomed_in_lims[1])

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

        # if Left Control is pressed, the signals should be normalized in amplitude, within the displayed window
        elif UI_intention(event)['triggered_intention'] == "adjust_in_amplitude":

            # if currently zoomed-out, then zoom-in
            if self.var_toggle_Ctrl.get() == 0:

                # Setting baseline min/max values is needed because if there are no filtered signals, these values
                # won't interfer with the min/max computation on the raw signals
                # values for min set at unreasonably high number (1000)
                min_max_within_window = 1000000 * np.ones((2 * self.nr_signals, 2))

                # values for max set at unreasonably high number (-1000)
                min_max_within_window[:, 1] = -1 * min_max_within_window[:, 1]

                for label, i in zip(self.modalities, range(self.nr_signals)):
                    # let's find the maximum and minimum values within the current window

                    # if the "view filtered signal" is toggled, means that filtered signals are currently displayed,
                    # therefore the axes exist

                    # min and max indices of x-axis within current window to get min and max in amplitude
                    idx_i = max(int(self.sampling_rate * self.axs[i].get_xlim()[0]), 0)
                    idx_f = min(int(self.sampling_rate * self.axs[i].get_xlim()[1]), self.nr_samples)

                    if self.var_view_filtered_signal.get() == 1:
                        # mins are in 2nd index = 0
                        min_max_within_window[i, 0] = np.min(self.main_plots['{}_filt'.format(label)][1][idx_i:idx_f])
                        min_max_within_window[i, 1] = np.max(self.main_plots['{}_filt'.format(label)][1][idx_i:idx_f])

                    # anyways, we always compute for raw signals
                    min_max_within_window[self.nr_signals + i, 0] = np.min(
                        self.main_plots['{}_raw'.format(label)][1][idx_i:idx_f])

                    min_max_within_window[self.nr_signals + i, 1] = np.max(
                        self.main_plots['{}_raw'.format(label)][1][idx_i:idx_f])

                    min_in_window = np.min(min_max_within_window[:, 0])
                    max_in_window = np.max(min_max_within_window[:, 1])

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

        if UI_intention(event)['triggered_intention'] == "rmv_annotation" and self.var_edit_plots.get() == 1:

            closest_annotation = self.closest_nearby_event(event, clicked_index)

            # using a "detection" window of 0.2s (=200 ms)
            if closest_annotation is not None:
                self.annotation_pack[clicked_index][round(closest_annotation * self.sampling_rate, 3)].remove()
                del self.annotation_pack[clicked_index][round(closest_annotation * self.sampling_rate, 3)]

                self.last_save = False

        elif UI_intention(event)['triggered_intention'] == "add_annotation" and self.var_edit_plots.get() == 1:

            self.annotation_pack[clicked_index][int(event.xdata * self.sampling_rate)] = self.axs[clicked_index].vlines(
                event.xdata,
                self.original_ylims[clicked_index][0],
                self.original_ylims[clicked_index][1],
                colors='#FF6B66')

            self.last_save = False

        self.canvas_plot.draw()

    def closest_nearby_event(self, event, clicked_index):
        """Checks whether a clicked location is close enough to a nearby annotation."""
        closest_annotation = None

        try:
            annotations_in_samples_temp = np.asarray(
                list(self.annotation_pack[clicked_index].keys())) / self.sampling_rate
            closest_annotation = annotations_in_samples_temp[
                np.argmin(np.abs(annotations_in_samples_temp - event.xdata))]

            nearby_condition = (closest_annotation - 0.2 < event.xdata < closest_annotation + 0.2)

            if not nearby_condition:
                closest_annotation = None
                self.previous_rmv_annotation_intention = False

            else:
                self.previous_rmv_annotation_intention = True

        # if it doesn't work, it's because tmeplate_pakc is empty
        except:
            pass

        return closest_annotation
