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
#mat = loadmat("SNN_EEG/data/data1.mat", struct_as_record=False, squeeze_me=False)
mat = loadmat("locked-in/P1_high1.mat", struct_as_record=False, squeeze_me=False)
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

n_samples,n_channels = X.shape

print(f"{'='*70}")
print("DATA LOADED")
print(f"Shape: {n_samples} samples × {n_channels} channels @ {fs_val} Hz")
if trig is not None:
    print(f"Triggers found: shape {trig.shape}")
if y_labels is not None:
    print(f"Labels found: shape {np.shape(y_labels)}")
print(f"{'='*70}\n")

# -------------------------
# Safe plotting parameters
# -------------------------
def clipped_nperseg(n_samples, default=1024):
    if n_samples < default:
        return max(128, 2**(int(np.floor(np.log2(n_samples/8))) if n_samples>=256 else 7))
    return default

# -------------------------
# 1) ALL CHANNELS STACKED VIEW
# -------------------------
duration = 150  # seconds to plot
plot_samples = min(int(duration * fs_val), n_samples)
t = np.arange(plot_samples) / fs_val

fig, ax = plt.subplots(figsize=(18, 14))
offset = 0.0
offsets = []
# color map
colors = plt.cm.viridis(np.linspace(0, 1, n_channels))

for ch in range(n_channels):
    sig = X[:plot_samples, ch].astype(np.float64)
    # safe normalization (avoid div by zero)
    denom = np.std(sig)
    if denom < 1e-12:
        denom = 1.0
    sig_normalized = (sig - np.mean(sig)) / denom
    ax.plot(t, sig_normalized + offset, linewidth=0.6, alpha=0.9, color=colors[ch])
    offsets.append(offset)
    offset += 4.5

ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Channel (stacked)', fontsize=13, fontweight='bold')
ax.set_title(f'All {n_channels} Channels - EEG (first {duration}s)', fontsize=15, fontweight='bold')
# pick ticks every ~max(1, n_channels//15)
tick_step = max(1, n_channels // 1)
yticks = offsets[::tick_step]
yticklabels = [f'Ch {i}' for i in range(0, n_channels, tick_step)]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('als_all_channels_stacked.png', dpi=200, bbox_inches='tight')
plt.show()

# -------------------------
# 2) Motor-specific summary (PSD, band power, spectrogram, correlations)
# -------------------------
# PSD average over channels
nperseg_psd = clipped_nperseg(n_samples, default=2048)
psd_list = []
f_axis = None
for ch in range(n_channels):
    f, psd = signal.welch(X[:, ch], fs=fs_val, nperseg=nperseg_psd)
    psd_list.append(psd)
    f_axis = f
psd_all = np.vstack(psd_list)
psd_mean = np.mean(psd_all, axis=0)

# band power helper
def band_power_matrix(data, fs, band, nperseg=None):
    nperseg = nperseg or clipped_nperseg(data.shape[0], default=1024)
    band_p = []
    for ch in range(data.shape[1]):
        f, psd = signal.welch(data[:, ch], fs=fs, nperseg=nperseg)
        idx = (f >= band[0]) & (f <= band[1])
        if np.any(idx):
            band_p.append(np.trapz(psd[idx], f[idx]))
        else:
            band_p.append(0.0)
    return np.array(band_p), f

mu_band = (8, 13)
beta_band = (13, 30)

mu_power, _ = band_power_matrix(X, fs_val, mu_band)
beta_power, _ = band_power_matrix(X, fs_val, beta_band)

# Prepare figure
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
ax1 = axes[0, 0]
ax1.semilogy(f_axis, psd_mean, linewidth=2, label='Average PSD')
ax1.axvspan(mu_band[0], mu_band[1], alpha=0.2, label='Mu (8-13 Hz)')
ax1.axvspan(beta_band[0], beta_band[1], alpha=0.2, label='Beta (13-30 Hz)')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('PSD')
ax1.set_xlim([0, min(50, fs_val / 2)])
ax1.grid(True)
ax1.legend(fontsize=8)

# Mu vs Beta bar-by-channel
ax2 = axes[0, 1]
x_pos = np.arange(n_channels)
width = 0.35
ax2.bar(x_pos - width/2, mu_power, width, label='Mu', alpha=0.8)
ax2.bar(x_pos + width/2, beta_power, width, label='Beta', alpha=0.8)
ax2.set_xlabel('Channel')
ax2.set_ylabel('Band Power')
ax2.set_title('Mu vs Beta Power by Channel')
ax2.legend()
ax2.grid(True, axis='y')

# Spectrogram for channel 0 (or first available)
ax3 = axes[1, 0]
spec_samples = min(plot_samples * 10, n_samples)
nperseg_spec = 256 if spec_samples >= 256 else 128
f_spec, t_spec, Sxx = signal.spectrogram(X[:spec_samples, 0], fs=fs_val, nperseg=nperseg_spec, noverlap=int(nperseg_spec*0.8))
im = ax3.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-12), shading='gouraud')
ax3.set_ylim([5, min(50, fs_val/2)])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Freq (Hz)')
ax3.set_title('Spectrogram - Ch 0')
plt.colorbar(im, ax=ax3, label='Power (dB)')

# Channel correlation
ax4 = axes[1, 1]
corr_samples = min(20000, n_samples)
corr_matrix = np.corrcoef(X[:corr_samples, :].T)
im2 = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='auto')
ax4.set_title('Channel Correlation')
plt.colorbar(im2, ax=ax4, label='Correlation')

# Signal variance (quality)
ax5 = axes[2, 0]
variances = np.var(X, axis=0)
ax5.bar(range(n_channels), variances, alpha=0.8)
ax5.set_title('Channel Variance (Quality)')
ax5.set_xlabel('Channel')
ax5.set_ylabel('Variance')
ax5.grid(True, axis='y')

# mu/beta ratio
ax6 = axes[2, 1]
mu_beta_ratio = mu_power / (beta_power + 1e-12)
ax6.plot(mu_beta_ratio, marker='o', linestyle='-', linewidth=2)
ax6.set_title('Mu / Beta Power Ratio')
ax6.set_xlabel('Channel')
ax6.set_ylabel('Ratio')
ax6.grid(True)

plt.tight_layout()
plt.savefig('als_motor_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

# -------------------------
# 3) Frequency band summary and text output
# -------------------------
bands = {
    'Delta (0.5-4 Hz)': (0.5, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Mu/Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30),
    'Low Gamma (30-50 Hz)': (30, 50)
}

print("\n" + "="*70)
print("FREQUENCY BAND POWER SUMMARY")
print("="*70)
band_results = {}
for band_name, (low, high) in bands.items():
    band_p, f_used = band_power_matrix(X, fs_val, (low, high))
    band_results[band_name] = band_p
    mean_p = np.mean(band_p)
    std_p = np.std(band_p)
    cv = std_p / (mean_p + 1e-12)
    print(f"{band_name:20s}: Mean={mean_p:.2e}, Std={std_p:.2e}, CV={cv:.3f}")

# -------------------------
# 4) Recommendations + optional save
# -------------------------
# Basic channel quality detection
threshold_high = np.percentile(variances, 95)
threshold_low = np.percentile(variances, 5)
bad_channels = np.where((variances > threshold_high) | (variances < threshold_low))[0]

print("\n" + "="*70)
print("DATA QUALITY SUMMARY")
print("="*70)
print(f"Potential bad channels: {bad_channels.tolist()}")
# mean upper-triangle corr
tri_inds = np.triu_indices_from(corr_matrix, k=1)
mean_corr = np.mean(corr_matrix[tri_inds]) if corr_matrix.size > 1 else 1.0
print(f"Mean channel pairwise correlation (upper triangle): {mean_corr:.3f}")
peak_freq = f_axis[np.argmax(psd_mean)]
print(f"Peak frequency: {peak_freq:.2f} Hz")

# Save cleaned/extracted arrays if desired (MATLAB .mat)
save_dir = "extracted"
os.makedirs(save_dir, exist_ok=True)
out_basename = os.path.join(save_dir, "P1_high1_extracted")
savemat(out_basename + ".mat", {
    'X': X,
    'fs': np.array([[fs_val]]),
    'trig': trig if trig is not None else np.array([]),
    'labels': y_labels if y_labels is not None else np.array([]),
})
print(f"\nSaved extracted data to: {out_basename}.mat")
