import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat, savemat
import os

# -------------------------
# Helper utilities
# -------------------------
def unwrap(v):
    """
    Recursively unwrap MATLAB objects imported by scipy.
    Returns plain numpy objects / lists / scalars where possible.
    """
    # None-like checks
    if v is None:
        return None
    # Scalars
    if np.isscalar(v):
        return v
    # object arrays (cell arrays, struct arrays)
    if isinstance(v, np.ndarray) and v.dtype == object:
        # if single element: unwrap it
        if v.size == 1:
            return unwrap(v.reshape(-1)[0])
        # multi-element: return list of unwrapped elements
        return [unwrap(x) for x in v.reshape(-1)]
    # numpy structured array / void (MATLAB structs)
    if isinstance(v, np.void) or (isinstance(v, np.ndarray) and v.dtype.names is not None):
        return v
    # standard numpy array
    return v

def get_field_from_struct(ds, name_variants):
    """
    Attempt to extract a field from a MATLAB struct-like numpy void or dict.
    name_variants: list of candidate names (e.g. ['X','x'])
    Returns None if not found.
    """
    if ds is None:
        return None

    # If ds is a plain dict (loadmat top-level)
    if isinstance(ds, dict):
        for n in name_variants:
            if n in ds:
                return unwrap(ds[n])
            # case-insensitive
            for k in ds.keys():
                if isinstance(k, str) and k.lower() == n.lower():
                    return unwrap(ds[k])
        return None

    # If ds is a numpy void/structured array with field names
    dtype_names = getattr(ds, 'dtype', None)
    if dtype_names is not None and dtype_names.names is not None:
        names = ds.dtype.names
        for n in name_variants:
            # direct
            if n in names:
                return unwrap(ds[n])
            # case-insensitive match
            for fn in names:
                if fn.lower() == n.lower():
                    return unwrap(ds[fn])
    return None

def ensure_2d_time_channels(X):
    """
    Ensures X is (n_samples, n_channels). If 1D, turn into (n_samples,1).
    If shape seems swapped (channels x samples) we transpose.
    """
    X = np.asarray(X)
    if X is None:
        return None
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        n0, n1 = X.shape
        # Heuristic: if first dim >> second and likely time in first -> OK
        # But if first dim < second (fewer samples than channels) assume transpose needed
        if n0 < n1 and n0 < 200:  # small number of rows -> probably channels x samples
            return X.T
        # another heuristic: if n0 is very large (time) leave as is
        return X
    # If >2 dims, try to squeeze
    return np.squeeze(X)

# -------------------------
# Load and extract
# -------------------------
mat = loadmat("SNN_EEG/data/data1.mat", struct_as_record=False, squeeze_me=False)
#mat = loadmat("locked-in/P1_high1.mat", struct_as_record=False, squeeze_me=False)
# Try both: some MAT files put the struct under 'data' others top-level variables.
top_keys = mat.keys()

# find candidate struct (common name is 'data' in your previous script)
if 'data' in mat:
    ds = unwrap(mat['data'])
else:
    # try to find the largest struct-like object
    ds = None
    for k in mat:
        if k.startswith('__'):
            continue
        candidate = mat[k]
        # pick first structured object or large ndarray of object
        if isinstance(candidate, np.ndarray) and candidate.dtype == object:
            ds = unwrap(candidate)
            break
        if isinstance(candidate, np.void):
            ds = candidate
            break
    # fallback: use entire dict
    if ds is None:
        ds = mat

# Now extract fields robustly (try common names)
X = get_field_from_struct(ds, ['X', 'x', 'y', 'signal', 'EEG', 'data'])  # note some files use 'y' for signal
trig = get_field_from_struct(ds, ['trig', 'trigger', 'Triggers'])
y_labels = get_field_from_struct(ds, ['y', 'labels', 'triallabels', 'class'])
channels = get_field_from_struct(ds, ['channels', 'chanlocs', 'labels'])
fs = get_field_from_struct(ds, ['fs', 'Fs', 'sampling_rate', 'srate', 'sr'])

# If X is still None, try top-level 'y' or 'Y' from loaded mat
if X is None:
    for alt in ['y', 'Y', 'data', 'EEG']:
        if alt in mat:
            X = unwrap(mat[alt])
            break

# ensure shapes and types
X = ensure_2d_time_channels(X)
if X is None:
    raise RuntimeError("Could not find 'X' or data matrix in the MAT file. Keys: " + ", ".join([k for k in mat.keys() if not k.startswith('__')]))

# sampling frequency
if fs is None:
    # try to infer from header if exists
    fs_val = 250.0
else:
    try:
        fs_val = float(np.asarray(fs).reshape(-1)[0])
    except Exception:
        try:
            fs_val = float(np.asarray(fs).squeeze())
        except Exception:
            fs_val = 250.0

# triggers and labels normalization
if trig is not None:
    trig = np.asarray(trig).squeeze()
if y_labels is not None:
    y_labels = np.asarray(y_labels).squeeze()

n_samples,n_channels = X.shape[0],X.shape[1]

print("X shape:", X.shape)

