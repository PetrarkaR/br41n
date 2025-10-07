import numpy as np
import glob

def learn_channel_mapping(file_list, n_missing_ch=4):
    """Learn linear mapping from working channels -> missing channels."""
    X_list, Y_list = [], []
    for f in file_list:
        data = np.load(f)
        trials, classes, time, ch = data.shape
        for t in range(trials):
            for c in range(classes):
                X = data[t, c, :, n_missing_ch:]  # working channels
                Y = data[t, c, :, :n_missing_ch]  # missing channels
                X_list.append(X)
                Y_list.append(Y)

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    W, _, _, _ = np.linalg.lstsq(X_all, Y_all, rcond=None)
    print(f"Learned mapping matrix: {W.shape}")
    return W


import numpy as np
from scipy.signal import butter, filtfilt

def generate_inflected_channels(data, n_missing_ch=4, segment_len=64, fs=256, seed=42):
    rng = np.random.default_rng(seed)
    recon = np.copy(data)
    
    trials, classes, time, n_total_ch = data.shape
    good_ch = n_total_ch - n_missing_ch
    
    # Small filter for smoothing perturbations
    b, a = butter(2, 0.2)  # lowpass 0.2*Nyquist
    
    for t in range(trials):
        for c in range(classes):
            X_good = recon[t, c, :, good_ch:]  # good channels
            Y_missing = np.zeros((time, n_missing_ch))
            
            for start in range(0, time, segment_len):
                end = min(start + segment_len, time)
                seg_len_actual = end - start
                seg = X_good[start:end, :]
                
                # Rolling weights: slightly change mix per segment
                good_ch = 8 - n_missing_ch

# segment of the good channels
                seg = X_good[start:end, :]  # shape (segment_len, good_ch)

                # random mixing matrix for missing channels
                weights = rng.uniform(-0.5, 0.5, size=(good_ch, n_missing_ch))  # shape (good_ch, n_missing_ch)

                # apply linear mixing
                seg_mixed = seg @ weights    # shape = (segment_len, n_missing_ch)

                
                # Add subtle noise and sine inflections
                for ch in range(n_missing_ch):
                    freq = rng.uniform(1, 12)  # EEG band
                    phase = rng.uniform(0, 2*np.pi)
                    sine = 0.02 * np.sin(2*np.pi*freq*np.arange(seg_len_actual)/fs + phase)
                    
                    pink = rng.standard_normal(seg_len_actual)  # pink noise approx
                    pink = np.cumsum(pink)
                    pink = 0.005 * pink / np.std(pink)
                    
                    seg_mixed[:, ch] = seg_mixed[:, ch] + sine + pink
                
                # Slight segment gain & offset
                gain = 1 + rng.uniform(-0.05, 0.05, n_missing_ch)
                offset = rng.uniform(-0.002, 0.002, n_missing_ch)
                seg_mixed = seg_mixed * gain + offset
                
                # Optional lowpass smoothing
                for ch in range(n_missing_ch):
                    seg_mixed[:, ch] = filtfilt(b, a, seg_mixed[:, ch])
                
                Y_missing[start:end, :] = seg_mixed
            
            recon[t, c, :, :n_missing_ch] = Y_missing
    
    return recon



# --- Example usage ---
train_files = sorted(glob.glob("syntheticdata/mi__noise_1000_noise_factor_*_synth_trials.npy"))
W = learn_channel_mapping(train_files, n_missing_ch=4)

data = np.load("syntheticdata/test.npy", allow_pickle=True)

reconstructed = generate_inflected_channels(
    data,
    n_missing_ch=4
)

out_file = "syntheticdata/test_reconstructed_inflected.npy"
np.save(out_file, reconstructed)
print("Saved:", out_file)
