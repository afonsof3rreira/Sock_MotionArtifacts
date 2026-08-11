import numpy as np

def _downsample_simple(x: np.ndarray, factor: int) -> np.ndarray:
    """Take every Nth sample."""
    return x[::factor]


def _downsample_aa(x, factor):
    from scipy.signal import resample_poly
    return np.asarray(resample_poly(x, up=1, down=factor, padtype='line'))


def downsample_signals(signals: dict, target_hz: int, original_hz: int = 1000, anti_alias: bool = True) -> dict:
    """
    Downsample each 1D numpy array in 'signals' by 'factor'.
    Returns a new dict with the same keys → numpy arrays.
    """
    if original_hz % target_hz != 0:
        # If 1000 isn't divisible by your target (e.g., 30Hz),
        # you'd technically need resampling (interpolation), not just downsampling.
        raise ValueError(f"Target {target_hz}Hz must be a factor of {original_hz}Hz.")

    factor = original_hz // target_hz

    ds_fn = _downsample_aa if anti_alias else _downsample_simple
    out = {}

    DISCRETE = {'DAC'}  # add any other setpoint/state channels

    for k, v in signals.items():
        arr = np.asarray(v).astype(float)
        fn = _downsample_simple if k in DISCRETE else ds_fn
        out[k] = fn(arr, factor)

    # Sanity check for consistent lengths
    lengths = [len(v) for v in out.values()]
    if len(set(lengths)) > 1:
        # Note: sometimes rounding can cause 1-sample differences depending on ds_fn logic
        min_len = min(lengths)
        out = {k: v[:min_len] for k, v in out.items()}

    # Convert to integers after all float operations are done
    out = {k: np.round(v).astype(int) for k, v in out.items()}

    return out


