import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_synthetic_data(file_path):
    """Load synthetic data from .p file and convert to proper shape"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Raw data type: {type(data)}")
    
    # Data is a list of 100 trials, each trial has shape (2_classes, 8_channels, 480_timepoints)
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        print(f"First element shape: {data[0].shape if hasattr(data[0], 'shape') else 'N/A'}")
        
        # Convert each tensor to numpy and stack properly
        trials = []
        for trial in data:
            if isinstance(trial, torch.Tensor):
                trial = trial.detach().cpu().numpy()
            trials.append(trial)
        
        # Stack to get (n_trials, n_classes, n_channels, n_timepoints)
        data = np.stack(trials, axis=0)  # (100, 2, 8, 480)
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    print(f"Final shape: {data.shape} (trials, classes, channels, timepoints)")
    print(f"Expected: 100 trials, 2 classes, 8 channels, 480 timepoints")
    
    return data

def preprocess_data(data, amplification_factor=2.2):
    """Center each channel around 0 and amplify by given factor"""
    # Shape: (trials, classes, channels, timepoints)
    processed_data = np.copy(data)
    
    n_trials, n_classes, n_channels, n_timepoints = data.shape
    
    # Center and amplify each channel for each trial and class
    for trial in range(n_trials):
        for cls in range(n_classes):
            for ch in range(n_channels):
                # Remove DC offset (center around 0)
                channel_data = processed_data[trial, cls, ch, :]
                channel_mean = np.mean(channel_data)
                processed_data[trial, cls, ch, :] = (channel_data - channel_mean) * amplification_factor
    
    return processed_data

def plot_synthetic_trial(data, trial_idx=0, class_idx=1, fs=256):
    """Plot each channel of a single trial in separate subplots with proper time axis"""
    # Shape: (trials, classes, channels, timepoints)
    n_trials, n_classes, n_channels, n_timepoints = data.shape
    
    # Create time vector in seconds
    time = np.arange(n_timepoints) / fs
    
    # Create subplots
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    
    # Handle single channel case
    if n_channels == 1:
        axes = [axes]
    
    for ch in range(n_channels):
        axes[ch].plot(time, data[trial_idx, class_idx, ch, :],
                     color='steelblue', linewidth=1.2)
        axes[ch].set_ylabel(f'Ch {ch}\n(µV)', fontsize=9)
        axes[ch].grid(True, alpha=0.3)
        axes[ch].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[ch].spines['top'].set_visible(False)
        axes[ch].spines['right'].set_visible(False)
    
    # Set xlabel only on bottom plot
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    
    # Add title to top
    fig.suptitle(f'Trial {trial_idx}, Class {class_idx} - All Channels (Centered & Amplified)',
                 fontsize=13, y=0.995)
    
    plt.tight_layout()
    plt.show()

def plot_all_channels_overlay(data, trial_idx=0, class_idx=0, fs=256):
    """Plot all channels overlaid on single plot with proper time axis"""
    n_trials, n_classes, n_channels, n_timepoints = data.shape
    time = np.arange(n_timepoints) / fs
    
    plt.figure(figsize=(14, 6))
    
    for ch in range(n_channels):
        plt.plot(time, data[trial_idx, class_idx, ch, :],
                label=f'Ch {ch}', linewidth=1.2, alpha=0.8)
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude (µV)', fontsize=12)
    plt.title(f'Trial {trial_idx}, Class {class_idx} - All Channels Overlaid (Centered & Amplified)',
              fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    file_path = 'syntheticdata/mi__noise_250_noise_factor_30_synth_trials.p'
    data = load_synthetic_data(file_path)
    
    # Verify the shape is correct
    assert data.shape[2] == 8, f"Expected 8 channels but got {data.shape[2]}"
    
    # Preprocess: center around 0 and amplify
    amplification = 2.2  # Change this value to adjust amplification
    data_processed = preprocess_data(data, amplification_factor=amplification)
    
    print(f"\n=== Data preprocessed: centered and amplified by {amplification}x ===")
    
    # Plot in two different styles
    print("\n=== Plotting separate subplots ===")
    plot_synthetic_trial(data_processed, trial_idx=0, class_idx=1, fs=256)
    
    print("\n=== Plotting overlaid channels ===")
    plot_all_channels_overlay(data_processed, trial_idx=0, class_idx=1, fs=256)