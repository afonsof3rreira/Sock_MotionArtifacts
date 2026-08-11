import os
import platform
import sys
from Annotator.event_annotator_ranges import event_annotator_ranges
import pandas as pd

import matplotlib, tkinter
print(matplotlib.__version__, tkinter.TkVersion, tkinter.TclVersion)

root_path = sys.argv[0]
root_path, _ = os.path.split(root_path)

signals_path = os.path.join(root_path, "Data/Downsampled")
ranges_path = os.path.join(root_path, "Data/Annotations")
tf_weights_path = os.path.join(root_path, "Data/TF_weights/fit_coeffs_4-1A__All_max.csv")

for device in ['sympathia', 'bitalino']:

    # window_size_mins = 3
    window_size_seconds = 100
    window_stride_seconds = int(window_size_seconds * 0.25)

    print(f"Device = {device}")

    os_name = platform.system()

    # load coefficients for conversion
    tf_weights = pd.read_csv(tf_weights_path, comment='#')

    event_annotator_ranges(50., os_name, signals_dir=signals_path, saving_dir=ranges_path,
                           device_name=device, tf_weights=tf_weights, window_size=window_size_seconds, window_stride=window_stride_seconds,
                           annotations_dir=ranges_path, annotator_id="A1")
