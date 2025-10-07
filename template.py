import sys
sys.path.append("../")
import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
# -------------------------
# Parameters
# -------------------------
PRE_S = 0.2  # seconds before trigger
POST_S = 1.0  # seconds after trigger
CHANNELS = [ 0,1,2,3,4,5,6,7]  
N_COMPONENTS = 8  # number of CSP components

# -------------------------
# Trigger-based epoching
# -------------------------
def epoch_by_triggers(y, trig, fs, pre_s=0.5, post_s=3.5):
    pre = int(pre_s * fs)
    post = int(post_s * fs)
    triggers = np.where(trig > 0)[0]  # positive triggers = events (gets 1 and 2, not 0 or -1)
    labels = trig[triggers]

    epochs = []
    kept_labels = []

    for i, t in enumerate(triggers):
        start = t - pre
        end = t + post
        if start >= 0 and end <= len(y):
            epochs.append(y[start:end, :].T[CHANNELS, :])  # select channels
            kept_labels.append(labels[i])
    
    return np.array(epochs), np.array(kept_labels)

# -------------------------
# Pre-processing
# -------------------------
def remove_mean(data):
    """Remove mean from each channel in each trial"""
    x = np.zeros_like(data)
    for tr in range(data.shape[0]):
        for ch in range(data.shape[1]):
            x[tr, ch, :] = data[tr, ch, :] - np.mean(data[tr, ch, :])
    return x

# -------------------------
# Load dataset
# -------------------------
mat1 = sio.loadmat('clean/P2_high1_clean_epochs.mat')
mat2 = sio.loadmat('clean/P2_high2_clean_epochs.mat')

# Extract sampling frequency
fs1 = float(mat1['fs'].squeeze())
print(f"Sampling frequency: {fs1} Hz")

# Data is already epoched: (n_trials, n_samples, n_channels)
# Need to transpose to: (n_trials, n_channels, n_samples)
epochs1 = mat1['epochs_clean'].transpose(0, 2, 1)  # (480, 8, 257)
labels1 = mat1['labels_clean'].squeeze()

# Select only the channels you want
epochs1 = epochs1[:, CHANNELS, :]

print(f"P1 epochs shape: {epochs1.shape}, labels distribution: {np.unique(labels1, return_counts=True)}")

# Remove mean
epochs1 = remove_mean(epochs1)

# Prepare data and labels
data = epochs1.astype(np.float64)
labels = labels1.astype(np.int64)
labels = np.where(labels == 1, 0, 1).astype(np.int64)  # 1 -> 0, 2 -> 1

print(f"\nData shape: {data.shape}, labels shape: {labels.shape}")

# -------------------------
# CSP + LDA
# -------------------------
csp = CSP(n_components=N_COMPONENTS, reg=0.01, log=True, norm_trace=True)
# Remove this line: data = data.reshape(-1,1)
csp.fit(data, labels)  # data is already (n_trials, n_channels, n_samples)
features = csp.transform(data)

clf = LinearDiscriminantAnalysis()
clf.fit(features, labels)
probs = clf.predict_proba(features)
cls0_probs = probs[labels == 0, 0]
cls1_probs = probs[labels == 1, 1]
# -------------------------
# Create templates (class averages)
# -------------------------


best_cls0 = np.mean(data[labels == 0], axis=0)
best_cls1 = np.mean(data[labels == 1], axis=0)


# -------------------------
# Verify
# -------------------------
# Stack and add trial dimension for CSP transform
test_data = np.stack((best_cls0, best_cls1), axis=0)  # Shape: (2, n_channels, n_samples)
test_feat = csp.transform(test_data)
outs = clf.predict_proba(test_feat)

print(f"\nVerification:")
print(f"Class 0 template probability: {outs[0][0]:.4f}")
print(f"Class 1 template probability: {outs[1][1]:.4f}")
# -------------------------
# Verify
# -------------------------

# -------------------------
# Cross-subject test (P1 → P2)
# -------------------------

print("\n=== Cross-Subject Evaluation (P1 → P2) ===")

# Data is already epoched, just load and transpose
epochs2 = mat2['epochs_clean'].transpose(0, 2, 1)  # (n_trials, 8, 257)
labels2 = mat2['labels_clean'].squeeze()

# Select only the channels you want
epochs2 = epochs2[:, CHANNELS, :]

# Remove mean
epochs2 = remove_mean(epochs2)
labels2 = np.where(labels2 == 1, 0, 1).astype(np.int64)

# Transform P2 data using CSP trained on P1
features_test_p2 = csp.transform(epochs2)
y_pred_p2 = clf.predict(features_test_p2)
acc_p2 = np.mean(y_pred_p2 == labels2)

print(f"Cross-subject accuracy (P1 → P2): {acc_p2 * 100:.2f}%")
print(f"Class distribution in P2: {np.unique(labels2, return_counts=True)}")


# -------------------------
# Save templates
# -------------------------
np.save('template1.npy', best_cls0)
np.save('template2.npy', best_cls1)

print("\nTemplates saved successfully!")
print(f"template1.npy shape: {best_cls0.shape}")
print(f"template2.npy shape: {best_cls1.shape}")

# -------------------------
# Plotting
# -------------------------
template1 = np.load('template1.npy')
template2 = np.load('template2.npy')

marker_time = 0.8  # 800 milliseconds after trigger

# For the stacked plots (ax1 and ax2):



n_samples = template1.shape[1]
# Create time axis with trigger at t=0 (shifted by PRE_S)
time_axis = np.arange(n_samples) / fs1 - PRE_S

# Plot Class 0 template
plt.figure(figsize=(12, 6))
for ch in range(template1.shape[0]):
    plt.plot(time_axis, template1[ch], label=f'Channel {CHANNELS[ch]}')
plt.axvline(x=0, color='r', linestyle='--', label='Trigger', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.title('Class 0 Template - Motor Imagery Channels')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Class 1 template
plt.figure(figsize=(12, 6))
for ch in range(template2.shape[0]):
    plt.plot(time_axis, template2[ch], label=f'Channel {CHANNELS[ch]}')
plt.axvline(x=0, color='r', linestyle='--', label='Trigger', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.title('Class 1 Template - Motor Imagery Channels')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

