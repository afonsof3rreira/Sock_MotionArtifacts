import os
import sys
import numpy as np
from Annotator.event_annotator_ranges import event_annotator_ranges
from Annotator.aux_functions import get_signals_as_dict_v4, filter_eda, acc_multi_filtering, vm_extractor

# pip install numpy, matplotlib, pandas, peakutils, biosppy

root_path = sys.argv[0]
root_path, _ = os.path.split(root_path)

signals_path = os.path.join(root_path, "Data/Raw")
ranges_path = os.path.join(root_path, "Data/Annotations")

device = 'sympathia'
mode = 0
assert mode in [0, 1]
# 0 is annotation 1 is revision

# window_size_mins = 3
window_size_seconds = 100
window_stride_seconds = int(window_size_seconds * 0.25)


mode_str = "annotation" if mode == 0 else "revision"

print(f"Selected mode = {mode} ({mode_str})")
print(f"Device = {device}")

event_annotator_ranges(100., signals_dir=signals_path, saving_dir=ranges_path,
                       device_name='sympathia', window_size=window_size_seconds, window_stride=window_stride_seconds,
                       annotations_dir=ranges_path)
