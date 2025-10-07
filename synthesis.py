import sys
sys.path.append('../')
from mi_bci import RNNSpikeSignal
from framework_utils import *
import numpy as np
from scipy.io import loadmat
import pickle
import os

n_epochs = 10001
layer_wise_neurons = [600, 1500, 400]
dt = 1e-3

def unwrap(v):
    """Recursively unwrap MATLAB objects imported by scipy."""
    if v is None:
        return None
    if np.isscalar(v):
        return v
    if isinstance(v, np.ndarray) and v.dtype == object:
        if v.size == 1:
            return unwrap(v.reshape(-1)[0])
        return [unwrap(x) for x in v.reshape(-1)]
    if isinstance(v, np.void) or (isinstance(v, np.ndarray) and v.dtype.names is not None):
        return v
    return v

def get_field_from_struct(ds, name_variants):
    """Attempt to extract a field from a MATLAB struct-like numpy void or dict."""
    if ds is None:
        return None
    if isinstance(ds, dict):
        for n in name_variants:
            if n in ds:
                return unwrap(ds[n])
            for k in ds.keys():
                if isinstance(k, str) and k.lower() == n.lower():
                    return unwrap(ds[k])
        return None
    dtype_names = getattr(ds, 'dtype', None)
    if dtype_names is not None and dtype_names.names is not None:
        names = ds.dtype.names
        for n in name_variants:
            if n in names:
                return unwrap(ds[n])
            for fn in names:
                if fn.lower() == n.lower():
                    return unwrap(ds[fn])
    return None

def convert_p_to_npy(p_file):
    """Convert a .p file to .npy"""
    try:
        with open(p_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle PyTorch tensors if present
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        
        npy_file = p_file.replace('.p', '.npy')
        np.save(npy_file, data)
        print(f"  Converted: {p_file} -> {npy_file}")
        
        # Optionally remove the .p file
        os.remove(p_file)
        
        return npy_file
    except Exception as e:
        print(f"  Failed to convert {p_file}: {e}")
        return None

# Load data
mat = loadmat("clean/P2_high1_clean_epochs.mat", struct_as_record=False, squeeze_me=False)
mat1 = loadmat("clean/P2_high2_clean_epochs.mat", struct_as_record=False, squeeze_me=False)

X = get_field_from_struct(mat, ['X', 'x', 'y', 'signal', 'EEG', 'data','epochs_clean'])
X1 = get_field_from_struct(mat1, ['X', 'x', 'y', 'signal', 'EEG', 'data','epochs_clean'])
fs = 256
print("\nChecking input channels:")
for ch in range(X.shape[1]):
    print(f"X Ch {ch}: min={X[:, ch].min():.4f}, max={X[:, ch].max():.4f}, std={X[:, ch].std():.6f}")
for ch in range(X1.shape[1]):
    print(f"X1 Ch {ch}: min={X1[:, ch].min():.4f}, max={X1[:, ch].max():.4f}, std={X1[:, ch].std():.6f}")
print(f"X shape: {X.shape}")
print(f"X1 shape: {X1.shape}")
print(f"Sampling rate: {fs}")

# FIX: Get correct n_steps and handle different lengths
n_steps_x = X.shape[0]
n_steps_x1 = X1.shape[0]
n_steps = min(n_steps_x, n_steps_x1)  # Use the shorter length

print(f"\nOriginal lengths: X={n_steps_x}, X1={n_steps_x1}")
print(f"Using n_steps={n_steps} (trimmed to shorter)")

# Trim both to same length and transpose so time is last dimension
class1_eeg = X[:n_steps, :].T  # Shape becomes (n_channels, n_steps)
class2_eeg = X1[:n_steps, :].T  # Shape becomes (n_channels, n_steps)

# Set n_steps to match time dimension
n_steps = class1_eeg.shape[1]

print(f"Trimmed shapes: class1={class1_eeg.shape}, class2={class2_eeg.shape}")
print(f"Using n_steps={n_steps} (matches time dimension)")

# Only synthesize class 1 signal (class2_eeg)
desired_signal = [class2_eeg]
T = dt * n_steps

# Assign number of noise neurons and noise factors
noise_neurons = np.array([1000])
noise_factor = np.array([1, 2,3,4,5])
#noise_neurons = np.hstack((np.arange(10, 105, 50), np.arange(100, 260, 50)))
#noise_factor = np.hstack((1, np.arange(5, 45, 20)))

print(f"\nGenerating synthetic data...")
print(f"Noise neurons: {noise_neurons}")
print(f"Noise factors: {noise_factor}")
print(f"Total combinations: {len(noise_neurons) * len(noise_factor)}")

# Generate and save artificial data for each combination
for nn in noise_neurons:
    for nf in noise_factor:
        print(f"\nProcessing: noise_neurons={nn}, noise_factor={nf}")
        
        model = RNNSpikeSignal(
            neurons=layer_wise_neurons, 
            n_steps=n_steps, 
            n_epochs=n_epochs,
            desired_spike_signal=desired_signal, 
            noise_neurons=nn, 
            noise_factor=nf, 
            freq=10,
            lr=5e-4, 
            syn_trials=100, 
            last_epoch=n_epochs - 1, 
            perturb=True
        )
        
        model.create_trials()
        
        # Find and convert the generated .p file to .npy
        p_filename = f'NEWESTmi__noise_{nn}_noise_factor_{nf}_synth_trials.p'
        if os.path.exists(p_filename):
            convert_p_to_npy(p_filename)
        
        print(f'Completed: noise_neuron={nn}, noise_factor={nf}')

print("\n" + "="*60)
print("ALL SYNTHETIC DATA GENERATED AND CONVERTED TO .npy")
print("="*60)