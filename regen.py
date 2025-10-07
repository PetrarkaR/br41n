#!/usr/bin/env python3
"""
regen_synthetic.py

Regenerate synthetic EEG data using saved trained weights (WF and WR) and correct template shapes.

What this script does:
 - backs up old syntheticdata/*.p files into syntheticdata/old_bad/
 - loads template1.npy and template2.npy and enforces channels x time
 - finds the latest trained epoch by inspecting saveddata/*_WR_* files
 - loads WF (list of tensors) and WR (tensor) for that epoch
 - constructs an RNNSpikeSignal instance (imported from mi_bci)
 - injects the loaded WF/WR into the model and regenerates synthetic trials for a set of noise configs
 - converts resulting pickled synthetic files to .npy for easy inspection

Usage:
    python regen_synthetic.py

Make sure you run this from your project root (the same place that has saveddata/ and syntheticdata/ and template1.npy).
"""

import os
import glob
import pickle
import re
import shutil
import numpy as np
import torch

# --- User-tweakable parameters ---
SYNTH_DIR = 'syntheticdata'
SAVED_DIR = 'saveddata'
TEMPLATE1 = 'template1.npy'
TEMPLATE2 = 'template2.npy'
# noise configs to (re)generate - matches previous experiments
NOISE_NEURONS = list(np.hstack((np.arange(10, 105, 50), np.arange(100, 260, 50))).astype(int))
NOISE_FACTORS = list(np.hstack((1, np.arange(5, 45, 20))).astype(int))
SYN_TRIALS = 10  # number of synthetic trials created per call (create_trials uses self.syn_trials)
# if you want to force a particular epoch, set this to an int. Otherwise it's auto-detected.
TARGET_EPOCH = None
# --------------------------------


def ensure_channels_first(a, max_channels=128):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"expected 2D array, got {a.shape}")
    # If rows look like many timesteps and cols like fewer channels -> transpose
    if a.shape[0] > max_channels and a.shape[1] <= max_channels:
        print(f"[ensure] Transposing array shape {a.shape} -> {(a.T.shape)}")
        return a.T
    return a


def backup_old_synth(synth_dir=SYNTH_DIR):
    os.makedirs(synth_dir, exist_ok=True)
    old_dir = os.path.join(synth_dir, 'old_bad')
    os.makedirs(old_dir, exist_ok=True)
    moved = []
    for f in glob.glob(os.path.join(synth_dir, '*_synth_trials.p')):
        dest = os.path.join(old_dir, os.path.basename(f))
        shutil.move(f, dest)
        moved.append(dest)
    print(f"Moved {len(moved)} old synthetic files to {old_dir}")
    return moved


def find_latest_epoch(saved_dir=SAVED_DIR):
    # looks for files containing _epoch_###_WR_
    wr_files = glob.glob(os.path.join(saved_dir, '*_WR_*'))
    epoch_nums = []
    pat = re.compile(r'_epoch_(\d+)_WR_')
    for f in wr_files:
        m = pat.search(f)
        if m:
            epoch_nums.append(int(m.group(1)))
    if not epoch_nums:
        return None
    return max(epoch_nums)


def find_weight_files_for_epoch(epoch, saved_dir=SAVED_DIR):
    # returns tuples (wf_path, wr_path) if found, else (None,None)
    str1_pattern = '*'  # we'll try to find matching files by epoch and *_WF_* / *_WR_* patterns
    wf_candidates = glob.glob(os.path.join(saved_dir, f'*_epoch_{epoch}_WF_*'))
    wr_candidates = glob.glob(os.path.join(saved_dir, f'*_epoch_{epoch}_WR_*'))
    wf = wf_candidates[0] if wf_candidates else None
    wr = wr_candidates[0] if wr_candidates else None
    return wf, wr


def load_pickle_maybe_tensor(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    # convert torch tensors if present
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
        return [o.cpu() for o in obj]
    return obj


def convert_p_to_npy(p_file):
    """Convert a .p pickle (possibly containing torch tensors) into a .npy file safely."""
    try:
        with open(p_file, 'rb') as f:
            data = pickle.load(f)

        import torch

        def to_numpy(x):
            # Recursively convert torch tensors -> detached cpu numpy, lists -> mapped, otherwise -> np.asarray
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            if isinstance(x, (list, tuple)):
                return [to_numpy(y) for y in x]
            return np.asarray(x)

        # data may be list-of-trials, or a numpy array; handle both
        if isinstance(data, (list, tuple)):
            arr = [to_numpy(tr) for tr in data]
            # try to stack into an ndarray if uniform shapes, else keep as object array
            try:
                stacked = np.stack(arr, axis=0)
                np.save(p_file.replace('.p', '.npy'), stacked, allow_pickle=False)
            except Exception:
                np.save(p_file.replace('.p', '.npy'), np.asarray(arr, dtype=object), allow_pickle=True)
        else:
            # single object: convert directly
            np.save(p_file.replace('.p', '.npy'), to_numpy(data), allow_pickle=True)

        print(f"  Converted: {p_file} -> {p_file.replace('.p', '.npy')}")
        # Optionally remove the .p file; keep it for safety
        # os.remove(p_file)
        return p_file.replace('.p', '.npy')

    except Exception as e:
        print(f"  Failed to convert {p_file}: {e}")
        return None


def generate_poisson_input(batch_size, n_steps, n_inputs, freq=20, dt=1e-3, device='cpu', dtype=torch.float):
    prob = freq * dt
    mask = torch.rand((batch_size, n_steps, n_inputs), dtype=dtype, device=device)
    x_data = torch.zeros((batch_size, n_steps, n_inputs), dtype=dtype, device=device)
    x_data[mask < prob] = 1.0
    return x_data


if __name__ == '__main__':
    print("=== Regen synthetic helper ===")

    # 1) backup old synthetic files
    backup_old_synth(SYNTH_DIR)

    # 2) load templates
    if not os.path.exists(TEMPLATE1) or not os.path.exists(TEMPLATE2):
        raise SystemExit(f"Templates {TEMPLATE1} or {TEMPLATE2} not found. Run the template creation step first.")

    t1 = np.load(TEMPLATE1)
    t2 = np.load(TEMPLATE2)
    t1 = ensure_channels_first(t1, max_channels=64)
    t2 = ensure_channels_first(t2, max_channels=64)
    print(f"Loaded templates shapes: {t1.shape}, {t2.shape}")

    desired_signal = [t1, t2]
    n_steps = t1.shape[1]
    n_channels = t1.shape[0]

    # 3) find latest epoch if not provided
    epoch = TARGET_EPOCH if TARGET_EPOCH is not None else find_latest_epoch(SAVED_DIR)
    if epoch is None:
        raise SystemExit("No epoch WR files found in saveddata. Train a model first or set TARGET_EPOCH.")
    print(f"Target epoch for regeneration: {epoch}")

    wf_path, wr_path = find_weight_files_for_epoch(epoch, SAVED_DIR)
    if not wf_path or not wr_path:
        print("Warning: Could not find WF/WR files for epoch. Attempting to locate any WR/WF pair.")
        # try to find any WR & WF and pick the one with nearest epoch
        wf_list = glob.glob(os.path.join(SAVED_DIR, '*_WF_*'))
        wr_list = glob.glob(os.path.join(SAVED_DIR, '*_WR_*'))
        if not wf_list or not wr_list:
            raise SystemExit("No WF/WR files found in saveddata. Cannot regenerate synthetic outputs.")
        wf_path = wf_list[0]
        wr_path = wr_list[0]

    print(f"Using WF: {wf_path}")
    print(f"Using WR: {wr_path}")

    # 4) load weights
    try:
        WF = torch.load(wf_path, map_location='cpu')
        WR = torch.load(wr_path, map_location='cpu')
    except Exception as e:
        raise SystemExit(f"Failed to load WF/WR: {e}")

    # WF expected as list of tensors, WR as tensor
    if not isinstance(WF, list):
        print("WF not a list. Attempting to coerce...")
        if hasattr(WF, 'values'):
            WF = list(WF)
    print(f"WF layers: {len(WF)}, WR shape: {tuple(WR.shape)}")

    # deduce neuron sizes from WF & WR
    neurons = []
    try:
        neurons.append(WF[0].shape[0])
        for w in WF:
            neurons.append(w.shape[1])
    except Exception as e:
        # fallback: use a default
        print("Could not deduce neurons from WF, falling back to default [600,1500,400]")
        neurons = [600, 1500, 400]

    print(f"Deduced neurons: {neurons}")

    # 5) try to import RNNSpikeSignal from mi_bci
    try:
        from mi_bci import RNNSpikeSignal
    except Exception as e:
        raise SystemExit(f"Failed to import RNNSpikeSignal from mi_bci: {e}\nMake sure mi_bci.py is in PYTHONPATH and defines RNNSpikeSignal.")

    # 6) create model instance but prevent cached input collision by overriding input after init
    model = RNNSpikeSignal(neurons=neurons, n_steps=n_steps, n_epochs=1,
                           desired_spike_signal=desired_signal, noise_neurons=NOISE_NEURONS[0],
                           noise_factor=NOISE_FACTORS[0], freq=20, lr=1e-4, last_epoch=epoch, perturb=False,
                           syn_trials=SYN_TRIALS)

    # ensure correct desired signal shape inside model (some legacy versions may not stack properly)
    try:
        # if model expects a list, but we passed numpy arrays, ensure its internal conversion
        model.desired_spike_signal = torch.tensor(np.stack(desired_signal, axis=0), dtype=model.dtype)
    except Exception:
        # not fatal; continue
        pass

    # override random/cached input with freshly generated Poisson input matching the model shape
    batch_size = model.classes
    n_input_neurons = neurons[0]
    device = next(model.WR.parameters()).device if hasattr(model.WR, 'device') and isinstance(model.WR, torch.Tensor) else torch.device('cpu')
    # generate on CPU to avoid GPU surprises
    model.input_current = generate_poisson_input(batch_size, n_steps, n_input_neurons, freq=20, dt=1e-3, device='cpu', dtype=torch.float)
    print(f"Injected fresh Poisson input with shape {model.input_current.shape}")

    # inject loaded weights into the model
    model.WF = WF
    model.WR = WR
    model.last_epoch = epoch

    # set syn_trials in case it differs
    model.syn_trials = SYN_TRIALS

    # 7) Loop combos and regenerate
    print("Starting regeneration loop...")
    for nn in NOISE_NEURONS:
        for nf in NOISE_FACTORS:
            print(f"--> regen noise_neurons={nn}, noise_factor={nf}")
            model.noise_neurons = int(nn)
            model.noise_factor = float(nf)
            model.syn_trials = SYN_TRIALS
            # ensure model.WN has correct shape for noise_neurons; recreate if necessary
            try:
                # WN expected list with one tensor of shape (noise_neurons, neurons[-1])
                if not model.WN or model.WN[0].shape[0] != model.noise_neurons:
                    model.WN = [torch.empty((model.noise_neurons, neurons[-1]), dtype=model.dtype)]
                    torch.nn.init.normal_(model.WN[0], mean=0.0, std=0.01)
            except Exception:
                model.WN = [torch.empty((model.noise_neurons, neurons[-1]), dtype=model.dtype)]
                torch.nn.init.normal_(model.WN[0], mean=0.0, std=0.01)

            # create trials (this saves pickled output in syntheticdata/)
            model.create_trials()

            # find the file(s) that were just created (be lenient about float vs int formatting)
            pattern = os.path.join(SYNTH_DIR, f"{model.file_name}_noise_{model.noise_neurons}_noise_factor_*_synth_trials.p")
            matches = sorted(glob.glob(pattern), key=os.path.getmtime)
            if matches:
                ppath = matches[-1]
                convert_p_to_npy(ppath)
                # quick shape check
                with open(ppath, 'rb') as f:
                    out = pickle.load(f)
                try:
                    print(f"  produced {ppath}, first trial shape: {np.asarray(out[0]).shape}")
                except Exception:
                    print(f"  produced {ppath} (couldn't read trial shape)")
            else:
                # helpful debug output
                print(f"  No synthetic file found for pattern: {pattern}")
                print("  syntheticdata/ dir listing (first 50 entries):")
                for x in sorted(os.listdir(SYNTH_DIR))[:50]:
                    print("   ", x)


    print("\nDone. Regeneration finished. Inspect syntheticdata/ for .npy files and ensure shapes are (n_trials, classes, channels, time)")
