# diagnostics.py
import os, glob, pickle, numpy as np
import torch

def energy_stats(arr):
    # arr shape: (classes, channels, time)
    res = []
    for c in range(arr.shape[0]):
        e = np.sum(np.abs(np.asarray(arr[c])), axis=1)
        res.append({'min': float(e.min()), 'mean': float(e.mean()), 'dead_inds': np.where(e < 1e-6)[0].tolist()})
    return res

def load_pickle_maybe_tensor(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    # if tensor or list of tensors
    try:
        if hasattr(obj, 'detach'):
            return obj.detach().cpu().numpy()
    except:
        pass
    # if list of tensors
    if isinstance(obj, list) and len(obj)>0 and hasattr(obj[0], 'detach'):
        return [o.detach().cpu().numpy() for o in obj]
    return obj

print("=== Template checks ===")
for t in ['template1.npy','template2.npy']:
    if os.path.exists(t):
        a = np.load(t)
        print(t, "shape:", a.shape)
    else:
        print(t, "MISSING")

print("\n=== Saved training outputs ===")
saved_outs = sorted(glob.glob('saveddata/*_outpus_signal_epoch_*.p'))
if not saved_outs:
    print("No saved outputs found in saveddata/")
else:
    for f in saved_outs[-4:]:
        try:
            arr = load_pickle_maybe_tensor(f)
            arr = np.asarray(arr)
            print(f, "->", arr.shape)
            stats = energy_stats(arr)
            for cls_i, s in enumerate(stats):
                print(f"  cls{cls_i}: min {s['min']:.3e}, mean {s['mean']:.3e}, dead {s['dead_inds']}")
        except Exception as e:
            print("  failed to read", f, e)

print("\n=== WR weight files ===")
wr_files = sorted(glob.glob('saveddata/*_WR_*'))
if not wr_files:
    print("No WR files found in saveddata/ (possible typo or never saved).")
else:
    for f in wr_files:
        try:
            w = torch.load(f, map_location='cpu')
            try:
                s = tuple(w.shape)
            except:
                s = str(type(w))
            print(f, "shape/type:", s)
        except Exception as e:
            print("  failed to load", f, e)

print("\n=== syntheticdata checks ===")
synth = sorted(glob.glob('syntheticdata/*_synth_trials.p'))
if not synth:
    print("No syntheticdata files found.")
else:
    for f in synth[-6:]:
        try:
            arr = load_pickle_maybe_tensor(f)
            # arr is usually list of trials: list length syn_trials, each (classes, channels, time)
            if isinstance(arr, list):
                arr0 = np.asarray(arr[0])
                print(f, "saved as list, example trial shape:", arr0.shape)
                stats = energy_stats(arr0)
            else:
                arr = np.asarray(arr)
                print(f, "shape:", arr.shape)
                stats = energy_stats(arr)
            for cls_i, s in enumerate(stats):
                print(f"  cls{cls_i}: min {s['min']:.3e}, mean {s['mean']:.3e}, dead {s['dead_inds']}")
        except Exception as e:
            print("  failed to read", f, e)

print("\n=== Input spike cache ===")
inp = 'mi__input_spike_data'
if os.path.exists(inp):
    try:
        x = load_pickle_maybe_tensor(inp)
        print(inp, "->", np.asarray(x).shape)
    except Exception as e:
        print("  failed to load", e)
else:
    print(inp, "not found")
