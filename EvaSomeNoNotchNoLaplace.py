# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 08:13:32 2025

@author: QinXinlan
"""

import numpy as np
import scipy.io
import mne
from mne import Epochs, find_events
from mne.io import RawArray
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from scipy.signal import savgol_filter
import time
from datetime import timedelta
import json

def load_vtp_data(mat_file_path):
    """
    Load vibro-tactile P300 data from .mat file and convert to MNE Raw object
    """
    print(f"Loading: {mat_file_path}")
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Extract variables
    eeg_data = mat_data['y']  # EEG data (samples x channels)
    triggers = mat_data['trig'].flatten()  # Trigger channel
    fs = mat_data['fs'].flatten()[0]  # Sampling rate
    
    print(f"  Data shape: {eeg_data.shape}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Trigger values: {np.unique(triggers)}")
    
    # Create channel names based on the paper's electrode positions
    n_channels = eeg_data.shape[1]
    # Paper mentions: Fz, C3, Cz, C4, CP1, CP2, CP2, Pz for P300 paradigms
    standard_channels = ['Fz', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'CPz', 'Pz']
    
    # Use standard channels if available, otherwise create generic names
    if n_channels <= len(standard_channels):
        channel_names = standard_channels[:n_channels]
    else:
        channel_names = [f'EEG{i+1:02d}' for i in range(n_channels)]
    
    # Create MNE info structure
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=fs,
        ch_types='eeg'
    )
    
    # Create Raw object
    raw = RawArray(eeg_data.T, info)  # MNE expects channels x samples
    
    # Add trigger channel as a stimulus channel
    stim_info = mne.create_info(['STI'], fs, ['stim'])
    stim_raw = RawArray(triggers.reshape(1, -1), stim_info)
    raw.add_channels([stim_raw], force_update_info=True)
    
    # Set electrode positions (standard 10-20 system)
    try:
        montage = make_standard_montage('standard_1020')
        # Only set montage for the EEG channels (exclude STI)
        raw.set_montage(montage)
    except Exception as e:
        print(f"  Warning: Could not set montage: {e}")
    
    return raw

def save_data_as_csv(raw, file_path, data_type='raw', csv_dir='csv_files'):
    """
    Save data as CSV files for further analysis
    
    Parameters:
    raw: MNE Raw object
    file_path: Original .mat file path
    data_type: Type of data being saved ('raw', 'dedrifted', 'notched', 'filtered')
    csv_dir: Directory to save CSV files
    """
    os.makedirs(csv_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save data as CSV
    raw_data, times = raw[:, :]
    raw_df = pd.DataFrame(raw_data.T, columns=raw.ch_names)
    raw_df['time'] = times
    raw_df['sample_index'] = np.arange(len(raw_df))
    
    # Reorder columns to have time and sample index first
    cols = ['time', 'sample_index'] + [col for col in raw_df.columns if col not in ['time', 'sample_index']]
    raw_df = raw_df[cols]
    
    # Use data_type in filename
    csv_path = os.path.join(csv_dir, f'{base_name}_{data_type}.csv')
    raw_df.to_csv(csv_path, index=False)
    print(f"  Saved {data_type} data CSV: {csv_path}")
    
    return {
        f'{data_type}_csv': csv_path,
    }

def plot_raw_eeg_check(raw, channel_name='Fz', title="", save_path=None, target_epoch_index=0):
    """
    Plot raw EEG data for a specific channel to check for noise, with detailed view of target epochs
    
    Parameters:
    raw: MNE Raw object
    channel_name: Channel to check
    title: Plot title
    save_path: Where to save the plot
    target_epoch_index: Which target epoch to show in detailed view (0 = first target)
    """
    if channel_name not in raw.ch_names:
        print(f"  Warning: Channel {channel_name} not found. Available channels: {raw.ch_names}")
        channel_name = raw.ch_names[0]
        print(f"  Using {channel_name} instead")
    
    # Create a figure with multiple subplots to assess data quality
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Raw EEG Quality Check - {title} - Channel: {channel_name}', fontsize=16)
    
    # Get the data for the selected channel
    channel_idx = raw.ch_names.index(channel_name)
    data, times = raw[channel_idx, :]
    data = data.flatten()
    
    # 1. Plot raw EEG trace
    axes[0,0].plot(times, data, 'b-', linewidth=0.5)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude (μV)')
    axes[0,0].set_title(f'Raw EEG Trace - {channel_name}')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Plot power spectral density
    picks = [raw.ch_names.index(channel_name)]
    raw_single = raw.copy().pick(picks)
    raw_single.compute_psd().plot(axes=axes[0,1], show=False)
    axes[0,1].set_title(f'Power Spectral Density - {channel_name}')
    
    # 3. Plot histogram of amplitudes
    axes[1,0].hist(data, bins=100, alpha=0.7, color='blue')
    axes[1,0].set_xlabel('Amplitude (μV)')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title(f'Amplitude Distribution - {channel_name}')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. NEW: Plot detailed view of specific target epoch (-1 to +2 seconds around target)
    events = find_events(raw, stim_channel='STI')
    target_events = events[events[:, 2] == 2]  # Filter for target events (code=2)
    
    if len(target_events) > 0:
        if target_epoch_index < len(target_events):
            target_event = target_events[target_epoch_index]
            event_time = target_event[0] / raw.info['sfreq']  # Convert sample to time
            
            # Define epoch window: -1 to +2 seconds around target
            start_time = event_time - 1.0
            end_time = event_time + 2.0
            
            # Convert to samples
            start_sample = max(0, int(start_time * raw.info['sfreq']))
            end_sample = min(len(data), int(end_time * raw.info['sfreq']))
            
            # Extract data for this epoch
            epoch_times = times[start_sample:end_sample] - event_time  # Center at 0
            epoch_data = data[start_sample:end_sample]
            
            axes[1,1].plot(epoch_times, epoch_data, 'b-', linewidth=1)
            axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Target Onset')
            axes[1,1].set_xlabel('Time relative to target (s)')
            axes[1,1].set_ylabel('Amplitude (μV)')
            axes[1,1].set_title(f'Target Epoch {target_epoch_index} - {channel_name}')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
            
            # Calculate statistics for this epoch
            epoch_mean = np.mean(epoch_data)
            epoch_std = np.std(epoch_data)
            epoch_max = np.max(np.abs(epoch_data))
            
            # Add epoch statistics
            epoch_stats = f'Epoch {target_epoch_index} Stats:\n'
            epoch_stats += f'Mean: {epoch_mean:.2f} μV\n'
            epoch_stats += f'Std: {epoch_std:.2f} μV\n'
            epoch_max = np.max(np.abs(epoch_data))
            epoch_stats += f'Max abs: {epoch_max:.2f} μV'
            
            axes[1,1].text(0.02, 0.98, epoch_stats, transform=axes[1,1].transAxes,
                          verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes[1,1].text(0.5, 0.5, f'Target epoch {target_epoch_index} not available\nTotal targets: {len(target_events)}', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title(f'Target Epoch - {channel_name}')
    else:
        axes[1,1].text(0.5, 0.5, 'No target events found', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title(f'Target Epoch - {channel_name}')
    
    # Calculate overall statistics for the text box
    mean_amp = np.mean(data)
    std_amp = np.std(data)
    max_amp = np.max(np.abs(data))
    
    # Add overall statistics to the first subplot
    overall_stats = 'Overall Statistics:\n'
    overall_stats += f'Mean: {mean_amp:.2f} μV\n'
    overall_stats += f'Std: {std_amp:.2f} μV\n'
    overall_stats += f'Max abs: {max_amp:.2f} μV\n'
    overall_stats += f'Duration: {times[-1]:.1f} s\n'
    overall_stats += f'Target events: {len(target_events)}'
    
    axes[0,0].text(0.02, 0.98, overall_stats, transform=axes[0,0].transAxes,
                  verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved raw EEG check: {save_path}")
    
    plt.show()
    
    return {
        'mean_amplitude': mean_amp,
        'std_amplitude': std_amp,
        'max_amplitude': max_amp,
        'channel_used': channel_name,
        'n_target_events': len(target_events),
        'target_epoch_stats': {
            'mean': epoch_mean if 'epoch_mean' in locals() else None,
            'std': epoch_std if 'epoch_std' in locals() else None,
            'max_abs': epoch_max if 'epoch_max' in locals() else None
        }
    }

def plot_trigger_distribution(raw, title, save_path=None):
    """
    Plot the distribution of triggers over time
    """
    events = find_events(raw, stim_channel='STI')
    
    plt.figure(figsize=(12, 4))
    plt.plot(raw.times, raw.get_data(picks=['STI']).flatten(), 'k-', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Trigger Value')
    plt.title(f'Trigger Distribution - {title}')
    plt.grid(True, alpha=0.3)
    
    # Add legend for trigger types
    trigger_types = {-1: 'Distractor', 1: 'Non-target', 2: 'Target'}
    for trigger_val, label in trigger_types.items():
        plt.plot([], [], ' ', label=f'{trigger_val}: {label}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved trigger plot: {save_path}")
    
    plt.show()
    
    return events

def create_epochs(raw, tmin=-0.3, tmax=0.8, baseline=(-0.3, 0)):
    """
    Create epochs from raw data based on triggers
    """
    # Create a copy of the raw data and modify the trigger channel
    raw_modified = raw.copy()
    
    # Get the trigger channel data
    stim_data = raw_modified.get_data(picks=['STI']).flatten()
    
    # Change -1 to 3 for distractors
    stim_data_modified = stim_data.copy()
    stim_data_modified[stim_data_modified == -1] = 3
    print('Changed distractor trigger from -1 to 3')
    
    # Update the raw data with modified triggers
    stim_channel_idx = raw_modified.ch_names.index('STI')
    raw_modified._data[stim_channel_idx, :] = stim_data_modified
    
    # Now find events on the modified data
    events = find_events(raw_modified, stim_channel='STI')
    
    # Define event IDs with the new mapping
    event_id = {
        'distractor': 3,  # was -1
        'non-target': 1, 
        'target': 2
    }
    
    print(f"Found events with IDs: {np.unique(events[:, 2])}")
    
    # Create epochs
    epochs = Epochs(
        raw_modified, 
        events, 
        event_id=event_id,
        tmin=tmin, 
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=False
    )
    
    print(f"  Created {len(epochs)} epochs:")
    for event_name, _ in event_id.items():
        print(event_name)
        if event_name in epochs.event_id:
            n_events = len(epochs[event_name])
            print(f"    {event_name}: {n_events} epochs")
        else:
            print(f"    {event_name}: 0 epochs")
    
    return epochs

    
def check_trial_quality(raw, epochs, channel_name='Fz', condition='target', 
                       tmin=-1.0, tmax=2.0, show_plots=True, save_path=None):
    """
    Check trial quality statistics before filtering to identify artifacts like blinks
    
    Parameters:
    raw: Raw EEG data
    epochs: Epochs data
    channel_name: Channel to analyze
    condition: Which condition to analyze ('target', 'non-target', 'distractor')
    tmin, tmax: Time window for analysis
    show_plots: Whether to display plots
    save_path: Path to save the plots
    """
    
    if channel_name not in raw.ch_names:
        print(f"Warning: Channel {channel_name} not found. Using {raw.ch_names[0]} instead.")
        channel_name = raw.ch_names[0]
    
    # Check if we have epochs for the specified condition
    if condition not in epochs.event_id:
        print(f"No {condition} epochs found for quality check")
        if show_plots:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'No {condition} epochs available for analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title(f'Trial Quality Check - No {condition} Data')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        return None
    
    condition_epochs = epochs[condition]
    n_trials = len(condition_epochs)
    
    print(f"\nTrial Quality Analysis for {channel_name} - {condition}:")
    print(f"Number of {condition} trials: {n_trials}")
    print(f"Analysis window: {tmin} to {tmax} seconds")
    
    if n_trials == 0:
        print(f"No {condition} trials available")
        if show_plots:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'No {condition} trials available for analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title(f'Trial Quality Check - No {condition} Trials')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        return None
    
    # Extract data for the specified channel and time window
    channel_idx = condition_epochs.ch_names.index(channel_name)
    times = condition_epochs.times
    time_mask = (times >= tmin) & (times <= tmax)
    
    trial_data = condition_epochs.get_data()[:, channel_idx, time_mask]
    
    # Calculate trial statistics
    trial_stats = []
    for i in range(n_trials):
        trial = trial_data[i]
        stats = {
            'trial_index': i,
            'mean': np.mean(trial),
            'std': np.std(trial),
            'max_abs': np.max(np.abs(trial)),
            'range': np.ptp(trial),  # peak-to-peak
            'variance': np.var(trial),
            'is_outlier_max': False,
            'is_outlier_std': False,
            'is_outlier_range': False
        }
        trial_stats.append(stats)
    
    # Convert to DataFrame for easier handling
    stats_df = pd.DataFrame(trial_stats)
    
    # Define outlier thresholds (using median absolute deviation for robustness)
    def mad_based_outlier(points, thresh=3.5):
        if len(points) == 0:
            return np.zeros_like(points, dtype=bool)
        median = np.median(points)
        mad = np.median(np.abs(points - median))
        if mad == 0:
            return np.abs(points - median) > 0  # All False if no variation
        modified_z_score = 0.6745 * (points - median) / mad
        return np.abs(modified_z_score) > thresh
    
    # Identify outliers
    stats_df['is_outlier_max'] = mad_based_outlier(stats_df['max_abs'].values)
    stats_df['is_outlier_std'] = mad_based_outlier(stats_df['std'].values)
    stats_df['is_outlier_range'] = mad_based_outlier(stats_df['range'].values)
    
    n_outliers_max = stats_df['is_outlier_max'].sum()
    n_outliers_std = stats_df['is_outlier_std'].sum()
    n_outliers_range = stats_df['is_outlier_range'].sum()
    
    print(f"Trials with outlier max amplitude: {n_outliers_max}/{n_trials}")
    print(f"Trials with outlier std deviation: {n_outliers_std}/{n_trials}")
    print(f"Trials with outlier range: {n_outliers_range}/{n_trials}")
    
    if not show_plots:
        return stats_df
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Trial Quality Analysis - {condition.title()} - Channel: {channel_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Violin plot for overall statistics
    ax1 = plt.subplot(2, 3, 1)
    violin_parts = ax1.violinplot([stats_df['mean'], stats_df['std'], stats_df['max_abs'], stats_df['range']], 
                                 showmeans=True, showmedians=True)
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(['Mean', 'Std Dev', 'Max Abs', 'Range'])
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title('Trial Statistics Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Color violins
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # 2. Trial-wise maximum amplitude
    ax2 = plt.subplot(2, 3, 2)
    colors = ['red' if outlier else 'blue' for outlier in stats_df['is_outlier_max']]
    ax2.scatter(stats_df['trial_index'], stats_df['max_abs'], c=colors, alpha=0.7)
    ax2.axhline(y=np.median(stats_df['max_abs']), color='green', linestyle='--', label='Median')
    ax2.set_xlabel('Trial Index')
    ax2.set_ylabel('Max Absolute Amplitude (μV)')
    ax2.set_title('Trial-wise Max Amplitude\n(Red = Outliers)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trial-wise standard deviation
    ax3 = plt.subplot(2, 3, 3)
    colors = ['red' if outlier else 'blue' for outlier in stats_df['is_outlier_std']]
    ax3.scatter(stats_df['trial_index'], stats_df['std'], c=colors, alpha=0.7)
    ax3.axhline(y=np.median(stats_df['std']), color='green', linestyle='--', label='Median')
    ax3.set_xlabel('Trial Index')
    ax3.set_ylabel('Standard Deviation (μV)')
    ax3.set_title('Trial-wise Standard Deviation\n(Red = Outliers)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap of all trials
    ax4 = plt.subplot(2, 3, 4)
    # Normalize for better visualization
    vmax = np.percentile(np.abs(trial_data), 95)
    im = ax4.imshow(trial_data, aspect='auto', cmap='RdBu_r', 
                   extent=[tmin, tmax, n_trials, 0], 
                   vmin=-vmax, vmax=vmax)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Trial Index')
    ax4.set_title(f'{condition.title()} Trials - EEG Activity\n(Heatmap)')
    plt.colorbar(im, ax=ax4, label='Amplitude (μV)')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
    ax4.legend()
    
    # 5. Range (peak-to-peak) per trial
    ax5 = plt.subplot(2, 3, 5)
    colors = ['red' if outlier else 'blue' for outlier in stats_df['is_outlier_range']]
    ax5.scatter(stats_df['trial_index'], stats_df['range'], c=colors, alpha=0.7)
    ax5.axhline(y=np.median(stats_df['range']), color='green', linestyle='--', label='Median')
    ax5.set_xlabel('Trial Index')
    ax5.set_ylabel('Peak-to-Peak Amplitude (μV)')
    ax5.set_title('Trial-wise Range\n(Red = Outliers)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "Trial Quality Summary\n"
    summary_text += f"Condition: {condition}\n"
    summary_text += f"Channel: {channel_name}\n"
    summary_text += f"Total trials: {n_trials}\n"
    summary_text += f"Analysis window: {tmin} to {tmax}s\n\n"
    summary_text += f"Mean amplitude: {stats_df['mean'].median():.2f} μV\n"
    summary_text += f"Median std: {stats_df['std'].median():.2f} μV\n"
    summary_text += f"Median max abs: {stats_df['max_abs'].median():.2f} μV\n"
    summary_text += f"Median range: {stats_df['range'].median():.2f} μV\n\n"
    summary_text += f"Outlier trials (max): {n_outliers_max} ({n_outliers_max/n_trials*100:.1f}%)\n"
    summary_text += f"Outlier trials (std): {n_outliers_std} ({n_outliers_std/n_trials*100:.1f}%)\n"
    summary_text += f"Outlier trials (range): {n_outliers_range} ({n_outliers_range/n_trials*100:.1f}%)\n\n"
    summary_text += "Blinks typically show:\n- Range > 100μV\n- Sudden large deflections"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trial quality plot: {save_path}")
    
    plt.show()
    
    return stats_df

def get_outlier_indices(stats_df):
    """
    Extract outlier trial indices from a statistics DataFrame
    
    Parameters:
    stats_df: DataFrame with trial statistics
    
    Returns:
    list: Indices of outlier trials
    """
    if stats_df is None or len(stats_df) == 0:
        return []
    
    # Get trials that are outliers in any category
    outlier_mask = (stats_df['is_outlier_max'] | 
                   stats_df['is_outlier_std'] | 
                   stats_df['is_outlier_range'])
    
    return stats_df[outlier_mask]['trial_index'].tolist()
    
def check_all_conditions_trial_quality(raw, epochs, channel_name='Fz', 
                                      tmin=-1.0, tmax=2.0, show_plots=True, 
                                      save_dir='results', base_name='',
                                      plot_individual_outliers=True):
    """
    Check trial quality across all conditions (target, non-target, distractor)
    
    Parameters:
    raw: Raw EEG data
    epochs: Epochs data
    channel_name: Channel to analyze
    tmin, tmax: Time window for analysis
    show_plots: Whether to display plots
    save_dir: Directory to save plots
    base_name: Base filename to include in saved plot names
    plot_individual_outliers: Whether to plot individual outlier trials
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_stats = {}
    conditions = ['target', 'non-target', 'distractor']
    
    for condition in conditions:
        print(f"\n{'='*50}")
        print(f"Checking trial quality for condition: {condition}")
        print(f"{'='*50}")
        
        # Create save path with base_name included
        if base_name:
            save_path = os.path.join(save_dir, f'{base_name}_trial_quality_{condition}_{channel_name}.png')
        else:
            save_path = os.path.join(save_dir, f'trial_quality_{condition}_{channel_name}.png')
            
        stats_df = check_trial_quality(raw, epochs, 
                                     channel_name=channel_name,
                                     condition=condition,
                                     tmin=tmin, tmax=tmax, 
                                     show_plots=show_plots, 
                                     save_path=save_path)
        all_stats[condition] = stats_df
        
        # Plot individual outlier trials if requested
        if plot_individual_outliers and stats_df is not None:
            outlier_indices = get_outlier_indices(stats_df)
            if outlier_indices:
                print(f"  Plotting {len(outlier_indices)} individual outlier trials for {condition}")
                plot_outlier_trials(
                    epochs, 
                    stats_df,
                    condition=condition,
                    tmin=tmin,
                    tmax=tmax,
                    save_dir=save_dir,
                    file_prefix=f'{base_name}_'
                )
    
    return all_stats

def check_all_channels_all_conditions_trial_quality(raw, epochs, 
                                                   tmin=-1.0, tmax=2.0, 
                                                   show_plots=False, save_dir='results',
                                                   base_name=''):
    """
    Check trial quality across all channels and all conditions
    
    Parameters:
    raw: Raw EEG data
    epochs: Epochs data
    tmin, tmax: Time window for analysis
    show_plots: Whether to display plots
    save_dir: Directory to save plots
    base_name: Base filename to include in saved plot names
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_stats = {}
    conditions = ['target', 'non-target', 'distractor']
    
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Checking trial quality for condition: {condition}")
        print(f"{'='*60}")
        
        condition_stats = {}
        for channel in raw.ch_names:
            if channel == 'STI':
                continue
                
            print(f"\nChannel: {channel}")
            
            # Create save path with base_name included
            if base_name:
                save_path = os.path.join(save_dir, f'{base_name}_trial_quality_{condition}_{channel}.png')
            else:
                save_path = os.path.join(save_dir, f'trial_quality_{condition}_{channel}.png')
                
            stats_df = check_trial_quality(raw, epochs, 
                                         channel_name=channel,
                                         condition=condition,
                                         tmin=tmin, tmax=tmax, 
                                         show_plots=show_plots, 
                                         save_path=save_path)
            condition_stats[channel] = stats_df
        
        all_stats[condition] = condition_stats
    
    return all_stats

def check_trial_quality_and_plot_outliers(raw, epochs, channel_name, condition, 
                                        tmin, tmax, save_dir, base_name, 
                                        check_all_conditions=False):
    """
    Check trial quality and plot outliers for single condition or all conditions
    """
    if check_all_conditions:
        return check_all_conditions_trial_quality(
            raw, epochs, 
            channel_name=channel_name,
            tmin=tmin, 
            tmax=tmax,
            show_plots=True,
            save_dir=save_dir,
            base_name=base_name,
            plot_individual_outliers=True
        )
    else:
        # Single condition case
        trial_quality_path = os.path.join(save_dir, f'{base_name}_trial_quality_{channel_name}.png')
        stats_df = check_trial_quality(
            raw, epochs, 
            channel_name=channel_name,
            condition=condition,
            tmin=tmin, 
            tmax=tmax,
            show_plots=True,
            save_path=trial_quality_path
        )
        
        # Plot individual outlier trials
        if stats_df is not None:
            outlier_indices = get_outlier_indices(stats_df)
            if len(outlier_indices) > 0:
                plot_outlier_trials(
                    epochs, 
                    stats_df,
                    condition=condition,
                    tmin=tmin,
                    tmax=tmax,
                    save_dir=save_dir,
                    file_prefix=f'{base_name}_'
                )
        
        return {condition: stats_df}
    
def plot_outlier_trials(epochs, stats_df, condition='target', tmin=-1.0, tmax=2.0, 
                       save_dir='results', file_prefix=''):
    """
    Plot individual outlier trials with all electrodes for visual inspection
    
    Parameters:
    epochs: MNE Epochs object
    stats_df: DataFrame with trial statistics from check_trial_quality
    condition: Which condition to plot ('target', 'non-target', 'distractor')
    tmin, tmax: Time window to plot
    save_dir: Directory to save plots
    file_prefix: Prefix for saved plot files
    """
    if condition not in epochs.event_id:
        print(f"No {condition} epochs found")
        return
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get outlier trials
    outlier_trials_max = stats_df[stats_df['is_outlier_max']]['trial_index'].tolist()
    outlier_trials_std = stats_df[stats_df['is_outlier_std']]['trial_index'].tolist()
    outlier_trials_range = stats_df[stats_df['is_outlier_range']]['trial_index'].tolist()
    
    # Combine all outlier indices (remove duplicates)
    all_outlier_indices = list(set(outlier_trials_max + outlier_trials_std + outlier_trials_range))
    
    if not all_outlier_indices:
        print(f"No outlier trials found for {condition} condition")
        return
    
    print(f"\nPlotting {len(all_outlier_indices)} outlier trials for {condition} condition:")
    
    # Get the actual epoch indices in the epochs object
    condition_epochs = epochs[condition]
    
    for outlier_idx in all_outlier_indices:
        if outlier_idx >= len(condition_epochs):
            print(f"  Warning: Trial index {outlier_idx} is out of range")
            continue
            
        # Determine which types of outliers this trial is
        outlier_types = []
        if outlier_idx in outlier_trials_max:
            outlier_types.append("max_amplitude")
        if outlier_idx in outlier_trials_std:
            outlier_types.append("std_deviation")
        if outlier_idx in outlier_trials_range:
            outlier_types.append("range")
        
        # Get trial statistics
        trial_stats = stats_df[stats_df['trial_index'] == outlier_idx].iloc[0]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Outlier Trial {outlier_idx} - {condition}\nOutlier types: {", ".join(outlier_types)}', 
                    fontsize=14, fontweight='bold')
        
        # 1. Plot all channels for this trial
        ax1 = axes[0, 0]
        # Get the data for this trial (all channels, all times)
        trial_data = condition_epochs[outlier_idx].get_data()[0]  # Shape: (n_channels, n_times)
        times = condition_epochs.times
        
        # Plot all channels
        for ch_idx in range(trial_data.shape[0]):
            ax1.plot(times, trial_data[ch_idx, :], 
                    label=condition_epochs.ch_names[ch_idx], 
                    linewidth=1, alpha=0.7)
        
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.set_title(f'All Channels - Trial {outlier_idx}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, ncol=2)
        
        # 2. Plot channel with maximum amplitude (usually the most problematic)
        ax2 = axes[0, 1]
        max_amp_channel_idx = np.argmax(np.max(np.abs(trial_data), axis=1))
        max_amp_channel = condition_epochs.ch_names[max_amp_channel_idx]
        
        ax2.plot(times, trial_data[max_amp_channel_idx, :], 'r-', linewidth=2)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (μV)')
        ax2.set_title(f'Channel with Max Amplitude: {max_amp_channel}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Plot heatmap of all channels
        ax3 = axes[1, 0]
        vmax = np.percentile(np.abs(trial_data), 95)
        im = ax3.imshow(trial_data, aspect='auto', cmap='RdBu_r',
                       extent=[times[0], times[-1], trial_data.shape[0], 0],
                       vmin=-vmax, vmax=vmax)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Channel Index')
        ax3.set_title('Channel-wise Activity (Heatmap)')
        ax3.set_yticks(range(trial_data.shape[0]))
        ax3.set_yticklabels(condition_epochs.ch_names)
        plt.colorbar(im, ax=ax3, label='Amplitude (μV)')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        # 4. Display trial statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"Trial {outlier_idx} Statistics:\n\n"
        stats_text += f"Condition: {condition}\n"
        stats_text += f"Mean: {trial_stats['mean']:.2f} μV\n"
        stats_text += f"Std: {trial_stats['std']:.2f} μV\n"
        stats_text += f"Max abs: {trial_stats['max_abs']:.2f} μV\n"
        stats_text += f"Range: {trial_stats['range']:.2f} μV\n"
        stats_text += f"Variance: {trial_stats['variance']:.2f} μV²\n\n"
        stats_text += "Outlier in:\n"
        if outlier_idx in outlier_trials_max:
            stats_text += "- Max amplitude\n"
        if outlier_idx in outlier_trials_std:
            stats_text += "- Standard deviation\n"
        if outlier_idx in outlier_trials_range:
            stats_text += "- Range\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{file_prefix}{condition}_outlier_trial_{outlier_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {condition} outlier trial {outlier_idx}: {save_path}")
        plt.close(fig)  # Close the figure to free memory
    
    return all_outlier_indices

def plot_outlier_summary(epochs, stats_df, condition='target', save_dir='results', file_prefix=''):
    """
    Create a summary plot showing all outlier trials in one figure
    """
    if condition not in epochs.event_id:
        print(f"No {condition} epochs found")
        return
    
    # Get outlier trials
    outlier_trials_max = stats_df[stats_df['is_outlier_max']]['trial_index'].tolist()
    outlier_trials_std = stats_df[stats_df['is_outlier_std']]['trial_index'].tolist()
    outlier_trials_range = stats_df[stats_df['is_outlier_range']]['trial_index'].tolist()
    
    all_outlier_indices = list(set(outlier_trials_max + outlier_trials_std + outlier_trials_range))
    
    if not all_outlier_indices:
        return
    
    # Create a grid of subplots
    n_outliers = len(all_outlier_indices)
    n_cols = min(3, n_outliers)  # Maximum 3 columns
    n_rows = (n_outliers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_outliers == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    condition_epochs = epochs[condition]
    
    for idx, (outlier_idx, ax) in enumerate(zip(all_outlier_indices, axes.flat)):
        if outlier_idx >= len(condition_epochs):
            continue
            
        # Get trial data
        trial_data = condition_epochs[outlier_idx].get_data()[0]
        times = condition_epochs.times
        
        # Plot all channels for this trial
        for ch_idx in range(trial_data.shape[0]):
            ax.plot(times, trial_data[ch_idx, :], linewidth=0.5, alpha=0.7)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        
        # Determine outlier types
        outlier_types = []
        if outlier_idx in outlier_trials_max:
            outlier_types.append("max")
        if outlier_idx in outlier_trials_std:
            outlier_types.append("std")
        if outlier_idx in outlier_trials_range:
            outlier_types.append("range")
        
        ax.set_title(f'Trial {outlier_idx} ({",".join(outlier_types)})')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(all_outlier_indices), n_rows * n_cols):
        axes.flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{file_prefix}outlier_trials_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved outlier summary: {save_path}")
    plt.show()

def dedrift_signal(raw, channel_name='Fz', window_length_sec=10, polyorder=2, 
                   plot_detrending=False, title="", 
                   specific_trial=None, trial_condition='target', 
                   trial_tmin=-1.0, trial_tmax=2.0, events=None,
                   save_path=None):
    """
    Remove slow drifts from EEG data using Savitzky-Golay filter with optional trial-specific visualization
    
    Parameters:
    raw: MNE Raw object
    channel_name: Channel to visualize (if plotting)
    window_length_sec: Window length in seconds for Savitzky-Golay filter
    polyorder: Polynomial order for Savitzky-Golay filter
    plot_detrending: Whether to plot original vs detrended signal
    title: Title for plots
    specific_trial: Specific trial number to visualize in detail (None for no specific trial)
    trial_condition: Condition of the trial to visualize ('target', 'non-target', 'distractor')
    trial_tmin, trial_tmax: Time window around the trial to visualize
    events: Events array (needed if specific_trial is provided)
    save_path: Path to save the plot
    
    Returns:
    raw_detrended: Detrended Raw object
    """
    # Create a copy to avoid modifying original
    raw_detrended = raw.copy()
    
    # Calculate window length in samples (must be odd)
    window_length = int(window_length_sec * raw.info['sfreq'])
    if window_length % 2 == 0:
        window_length += 1  # Make it odd
    
    print("Applying Savitzky-Golay detrending:")
    print(f"  Window length: {window_length} samples ({window_length_sec} seconds)")
    print(f"  Polynomial order: {polyorder}")
    
    # Apply detrending to each EEG channel (excluding STI)
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw_detrended._data[ch_idx, :]
        
        # Apply Savitzky-Golay filter to estimate trend
        trend = savgol_filter(channel_data, window_length, polyorder)
        
        # Subtract trend from original signal
        raw_detrended._data[ch_idx, :] = channel_data - trend
    
    # Plot comparison if requested
    if plot_detrending:
        if specific_trial is not None and events is not None:
            # Plot focused view on specific trial
            fig = plot_trial_detrending_comparison(
                raw, raw_detrended, channel_name, title,
                specific_trial, trial_condition, trial_tmin, trial_tmax, events
            )
        else:
            # Plot overall view
            fig = plot_detrending_comparison(raw, raw_detrended, channel_name, title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved detrending plot: {save_path}")
        
        plt.show()
    
    return raw_detrended

def plot_trial_detrending_comparison(raw_original, raw_detrended, channel_name, title,
                                    specific_trial, trial_condition, trial_tmin, trial_tmax, events):
    """
    Plot detrending comparison focused on a specific trial
    """
    # Get events for the specified condition
    event_id = {'distractor': 3, 'non-target': 1, 'target': 2}
    if trial_condition not in event_id:
        print(f"Warning: Condition {trial_condition} not found. Using 'target' instead.")
        trial_condition = 'target'
    
    condition_events = events[events[:, 2] == event_id[trial_condition]]
    
    if specific_trial >= len(condition_events):
        print(f"Warning: Trial {specific_trial} not available. Only {len(condition_events)} {trial_condition} trials found.")
        # Fall back to overall view
        return plot_detrending_comparison(raw_original, raw_detrended, channel_name, title)
    
    # Get the event for the specific trial
    event = condition_events[specific_trial]
    event_time = event[0] / raw_original.info['sfreq']  # Convert sample to time
    
    # Define trial window
    start_time = event_time + trial_tmin
    end_time = event_time + trial_tmax
    
    # Extract data for this trial window
    ch_idx = raw_original.ch_names.index(channel_name)
    
    # Get original data for the trial window
    start_sample_orig = max(0, int(start_time * raw_original.info['sfreq']))
    end_sample_orig = min(len(raw_original.times), int(end_time * raw_original.info['sfreq']))
    original_data = raw_original._data[ch_idx, start_sample_orig:end_sample_orig]
    original_times = raw_original.times[start_sample_orig:end_sample_orig] - event_time
    
    # Get detrended data for the trial window
    start_sample_det = max(0, int(start_time * raw_detrended.info['sfreq']))
    end_sample_det = min(len(raw_detrended.times), int(end_time * raw_detrended.info['sfreq']))
    detrended_data = raw_detrended._data[ch_idx, start_sample_det:end_sample_det]
    detrended_times = raw_detrended.times[start_sample_det:end_sample_det] - event_time
    
    # Calculate trend (difference)
    trend_data = original_data - detrended_data
    
    # Create comprehensive figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Detrending Effect - {title}\nTrial {specific_trial} ({trial_condition}) - Channel: {channel_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Original signal (trial view)
    axes[0, 0].plot(original_times, original_data, 'b-', linewidth=1.5, alpha=0.8, label='Original')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
    axes[0, 0].set_xlabel('Time relative to stimulus (s)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_title(f'Original Signal\nTrial {specific_trial} ({trial_condition})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Detrended signal (trial view)
    axes[1, 0].plot(detrended_times, detrended_data, 'g-', linewidth=1.5, alpha=0.8, label='Detrended')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
    axes[1, 0].set_xlabel('Time relative to stimulus (s)')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].set_title(f'Detrended Signal\nTrial {specific_trial} ({trial_condition})')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 3. Trend component (trial view)
    axes[2, 0].plot(original_times, trend_data, 'r-', linewidth=1.5, alpha=0.8, label='Trend (Removed)')
    axes[2, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
    axes[2, 0].set_xlabel('Time relative to stimulus (s)')
    axes[2, 0].set_ylabel('Amplitude (μV)')
    axes[2, 0].set_title(f'Trend Component (Removed)\nTrial {specific_trial} ({trial_condition})')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # 4. Overlay comparison (trial view)
    axes[0, 1].plot(original_times, original_data, 'b-', linewidth=1.5, alpha=0.7, label='Original')
    axes[0, 1].plot(detrended_times, detrended_data, 'g-', linewidth=1.5, alpha=0.7, label='Detrended')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
    axes[0, 1].set_xlabel('Time relative to stimulus (s)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].set_title(f'Original vs Detrended Overlay\nTrial {specific_trial} ({trial_condition})')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 5. Overall signal (full recording) - Original
    full_original = raw_original._data[ch_idx, :]
    full_times = raw_original.times
    axes[1, 1].plot(full_times, full_original, 'b-', linewidth=0.5, alpha=0.7, label='Original')
    # Mark the trial window
    axes[1, 1].axvspan(start_time, end_time, alpha=0.3, color='yellow', label='Trial Window')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude (μV)')
    axes[1, 1].set_title('Full Recording - Original Signal\n(Yellow = Trial Window)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 6. Overall signal (full recording) - Detrended
    full_detrended = raw_detrended._data[ch_idx, :]
    axes[2, 1].plot(full_times, full_detrended, 'g-', linewidth=0.5, alpha=0.7, label='Detrended')
    # Mark the trial window
    axes[2, 1].axvspan(start_time, end_time, alpha=0.3, color='yellow', label='Trial Window')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Amplitude (μV)')
    axes[2, 1].set_title('Full Recording - Detrended Signal\n(Yellow = Trial Window)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    # Calculate and display statistics
    trial_stats = f"Trial {specific_trial} Statistics ({trial_condition}):\n\n"
    trial_stats += f"Original:\n"
    trial_stats += f"  Mean: {np.mean(original_data):.2f} μV\n"
    trial_stats += f"  Std: {np.std(original_data):.2f} μV\n"
    trial_stats += f"  Range: {np.ptp(original_data):.2f} μV\n\n"
    trial_stats += f"Detrended:\n"
    trial_stats += f"  Mean: {np.mean(detrended_data):.2f} μV\n"
    trial_stats += f"  Std: {np.std(detrended_data):.2f} μV\n"
    trial_stats += f"  Range: {np.ptp(detrended_data):.2f} μV\n\n"
    trial_stats += f"Trend (Removed):\n"
    trial_stats += f"  Mean: {np.mean(trend_data):.2f} μV\n"
    trial_stats += f"  Std: {np.std(trend_data):.2f} μV\n"
    trial_stats += f"  Range: {np.ptp(trend_data):.2f} μV"
    
    # Add statistics text box
    fig.text(0.02, 0.02, trial_stats, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    return fig

def plot_detrending_comparison(raw_original, raw_detrended, channel_name, title=""):
    """
    Plot comparison of original vs detrended signal (overall view)
    """
    # Get data for the specified channel
    ch_idx = raw_original.ch_names.index(channel_name)
    original_data = raw_original._data[ch_idx, :]
    detrended_data = raw_detrended._data[ch_idx, :]
    times = raw_original.times
    
    # Calculate trend (difference between original and detrended)
    trend = original_data - detrended_data
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(f'Signal Detrending - {title} - Channel: {channel_name}', fontsize=16)
    
    # 1. Original signal
    axes[0].plot(times, original_data, 'b-', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Trend component
    axes[1].plot(times, trend, 'g-', linewidth=1, alpha=0.7)
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].set_title('Estimated Trend (Removed Component)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Detrended signal
    axes[2].plot(times, detrended_data, 'r-', linewidth=1, alpha=0.7)
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Detrended Signal')
    axes[2].grid(True, alpha=0.3)
    
    # Calculate statistics
    original_stats = f"Original: mean={np.mean(original_data):.2f}μV, std={np.std(original_data):.2f}μV"
    trend_stats = f"Trend: mean={np.mean(trend):.2f}μV, std={np.std(trend):.2f}μV" 
    detrended_stats = f"Detrended: mean={np.mean(detrended_data):.2f}μV, std={np.std(detrended_data):.2f}μV"
    
    # Add statistics to plots
    axes[0].text(0.02, 0.98, original_stats, transform=axes[0].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].text(0.02, 0.98, trend_stats, transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].text(0.02, 0.98, detrended_stats, transform=axes[2].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def apply_bandpass_filter_function(raw, l_freq=1.0, h_freq=15.0, filter_type='butterworth', 
                         filter_order=4, plot_response=False):
    """
    Apply bandpass filter with configurable parameters
    
    Parameters:
    raw: MNE Raw object
    l_freq: Low frequency cutoff (Hz)
    h_freq: High frequency cutoff (Hz)
    filter_type: Type of filter ('butterworth', 'fir', 'iir')
    filter_order: Order of the filter
    plot_response: Whether to plot the filter frequency response
    
    Returns:
    raw_filtered: Filtered raw data
    fig: Figure object if plot_response is True, else None
    """
    raw_filtered = raw.copy()
    
    print(f"Applying bandpass filter:")
    print(f"  Frequency range: {l_freq}-{h_freq} Hz")
    print(f"  Filter type: {filter_type}")
    print(f"  Filter order: {filter_order}")
    
    if filter_type == 'butterworth':
        # Use Butterworth filter
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method='iir',
                           iir_params=dict(ftype='butter', order=filter_order))
    elif filter_type == 'fir':
        # Use FIR filter
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method='fir',
                           fir_design='firwin', filter_length='auto')
    else:  # iir
        # Use default IIR filter
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method='iir')
    
    fig = None
    if plot_response:
        # Plot the filter response and capture the figure
        fig = plot_bandpass_response_function(raw_filtered, l_freq, h_freq, filter_type)
    
    return raw_filtered, fig

def plot_bandpass_response_function(raw_filtered, l_freq, h_freq, filter_type):
    """
    Plot the bandpass filter frequency response
    """
    # Get a representative channel
    eeg_channels = [ch for ch in raw_filtered.ch_names if ch != 'STI']
    if not eeg_channels:
        return None
    
    channel_name = eeg_channels[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PSD
    raw_filtered.compute_psd().plot(axes=ax, show=False)
    
    # Add cutoff frequency lines
    ax.axvline(x=l_freq, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Low cutoff: {l_freq} Hz')
    ax.axvline(x=h_freq, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'High cutoff: {h_freq} Hz')
    
    ax.set_title(f'Frequency Spectrum After {l_freq}-{h_freq} Hz Bandpass Filter ({filter_type})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_filter_comparison(raw_before, raw_after, channel_name, filter_type, freq_range, title=""):
    """
    Plot comparison before and after bandpass filtering
    """
    # Get data for the specified channel
    if channel_name not in raw_before.ch_names or channel_name not in raw_after.ch_names:
        print(f"Channel {channel_name} not found in raw data")
        return None
        
    ch_idx_before = raw_before.ch_names.index(channel_name)
    ch_idx_after = raw_after.ch_names.index(channel_name)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Bandpass Filter Comparison - {title} - Channel: {channel_name}', fontsize=16)
    
    # 1. Time domain - before (first 5 seconds)
    data_before, times_before = raw_before[ch_idx_before, :]
    segment_samples = min(5000, len(data_before[0]))
    axes[0, 0].plot(times_before[:segment_samples], data_before[0][:segment_samples], 'b-', linewidth=1, alpha=0.8)
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_title('Before Bandpass Filter - Time Domain (5s)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time domain - after (first 5 seconds)
    data_after, times_after = raw_after[ch_idx_after, :]
    segment_samples = min(5000, len(data_after[0]))
    axes[1, 0].plot(times_after[:segment_samples], data_after[0][:segment_samples], 'r-', linewidth=1, alpha=0.8)
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('After Bandpass Filter - Time Domain (5s)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Frequency domain - before
    from scipy import signal
    data_before_flat = data_before.flatten()
    data_after_flat = data_after.flatten()
    
    f_before, Pxx_before = signal.welch(data_before_flat, raw_before.info['sfreq'], nperseg=1024)
    axes[0, 1].semilogy(f_before, Pxx_before, 'b-', linewidth=1.5, alpha=0.8, label='Before')
    axes[0, 1].axvline(x=freq_range[0], color='green', linestyle='--', linewidth=2, label=f'Low cut: {freq_range[0]} Hz')
    axes[0, 1].axvline(x=freq_range[1], color='green', linestyle='--', linewidth=2, label=f'High cut: {freq_range[1]} Hz')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].set_title('Before Bandpass Filter - Frequency Domain')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 30)  # Focus on 0-30 Hz range for P300
    
    # 4. Frequency domain - after
    f_after, Pxx_after = signal.welch(data_after_flat, raw_after.info['sfreq'], nperseg=1024)
    axes[1, 1].semilogy(f_after, Pxx_after, 'r-', linewidth=1.5, alpha=0.8, label='After')
    axes[1, 1].axvline(x=freq_range[0], color='green', linestyle='--', linewidth=2, label=f'Low cut: {freq_range[0]} Hz')
    axes[1, 1].axvline(x=freq_range[1], color='green', linestyle='--', linewidth=2, label=f'High cut: {freq_range[1]} Hz')
    axes[1, 1].set_ylabel('Power Spectral Density')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_title('After Bandpass Filter - Frequency Domain')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 30)  # Focus on 0-30 Hz range for P300
    
    plt.tight_layout()
    return fig

def apply_notch_filter_function(raw, freq=50., bandwidth=2.0, method='iir_simple', filter_order=2, plot_response=False):
    """
    Apply notch filter to remove line noise with robust error checking and clear visualization
    """
    raw_notched = raw.copy()
    
    # Get sampling frequency from the raw object
    sfreq = raw.info['sfreq']
    
    print(f"Applying notch filter:")
    print(f"  Center frequency: {freq} Hz")
    print(f"  Bandwidth: {bandwidth} Hz")
    print(f"  Method: {method}")
    print(f"  Filter order: {filter_order}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Nyquist frequency: {sfreq/2} Hz")
    
    # Critical safety checks
    if freq >= sfreq / 2:
        print(f"  ERROR: Notch frequency {freq} Hz is above Nyquist frequency {sfreq/2} Hz!")
        print("  This will cause serious problems. Check your sampling rate.")
        return raw_notched, None
    
    if bandwidth <= 0:
        print(f"  ERROR: Invalid bandwidth {bandwidth} Hz")
        bandwidth = 2.0  # Default fallback
    
    # Method 1: Simple, stable IIR notch (most reliable)
    if method == 'iir_simple':
        print("  Using simple IIR notch filter")
        raw_notched = apply_simple_iir_notch(raw, freq, bandwidth, sfreq)
    
    # Method 2: Traditional IIR with safety
    elif method == 'iir':
        print("  Using IIR notch filter with safety checks")
        raw_notched = apply_safe_iir_notch(raw, freq, bandwidth, filter_order, sfreq)
    
    # Method 3: FIR with proper design
    elif method == 'fir':
        print("  Using FIR notch filter")
        raw_notched = apply_safe_fir_notch(raw, freq, bandwidth, filter_order, sfreq)
    
    # Method 4: Frequency domain approach (most stable)
    elif method == 'frequency_domain':
        print("  Using frequency domain notch filter")
        raw_notched = apply_frequency_domain_notch(raw, freq, bandwidth, sfreq)
    
    else:
        print(f"  Unknown method: {method}, using simple IIR")
        raw_notched = apply_simple_iir_notch(raw, freq, bandwidth, sfreq)
    
    fig = None
    if plot_response:
        # Use the clearer visualization function and get the figure
        fig = plot_notch_filter_diagnostic_clear(raw, raw_notched, freq, bandwidth, method, sfreq)
    
    return raw_notched, fig

def apply_simple_iir_notch(raw, freq, bandwidth, sfreq):
    """
    Simple, stable IIR notch filter using 2nd order
    """
    from scipy import signal
    import copy
    
    raw_notched = copy.deepcopy(raw)
    
    # Design a simple 2nd order notch filter (most stable)
    quality_factor = freq / bandwidth
    b, a = signal.iirnotch(freq, quality_factor, sfreq)
    
    # Apply to each EEG channel
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw._data[ch_idx, :]
        
        # Check for NaN or Inf values
        if np.any(~np.isfinite(channel_data)):
            print(f"  Warning: Non-finite values in channel {raw.ch_names[ch_idx]}")
            channel_data = np.nan_to_num(channel_data)
        
        # Apply filter with zero-phase (filtfilt)
        try:
            filtered_data = signal.filtfilt(b, a, channel_data)
            raw_notched._data[ch_idx, :] = filtered_data
        except Exception as e:
            print(f"  Filter failed for channel {raw.ch_names[ch_idx]}: {e}")
            # Fallback: use simpler filtering
            sos = signal.butter(2, [freq-1, freq+1], 'bandstop', fs=sfreq, output='sos')
            filtered_data = signal.sosfiltfilt(sos, channel_data)
            raw_notched._data[ch_idx, :] = filtered_data
    
    return raw_notched

def apply_safe_iir_notch(raw, freq, bandwidth, filter_order, sfreq):
    """
    Safe IIR notch filter with extensive error checking
    """
    from scipy import signal
    import copy
    
    raw_notched = copy.deepcopy(raw)
    
    # Limit filter order for stability
    safe_order = min(filter_order, 4)  # Higher orders can be unstable
    
    # Use bandstop filter instead of notch for better stability
    low_cut = freq - bandwidth/2
    high_cut = freq + bandwidth/2
    
    # Ensure frequencies are valid
    low_cut = max(0.1, low_cut)
    high_cut = min(sfreq/2 - 0.1, high_cut)
    
    if low_cut >= high_cut:
        print("  Warning: Invalid frequency range, using default")
        low_cut = freq - 1.0
        high_cut = freq + 1.0
    
    print(f"  Using bandstop filter: {low_cut:.1f}-{high_cut:.1f} Hz")
    
    # Design stable SOS filter
    sos = signal.butter(safe_order, [low_cut, high_cut], 'bandstop', fs=sfreq, output='sos')
    
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw._data[ch_idx, :]
        
        # Apply filter
        try:
            filtered_data = signal.sosfiltfilt(sos, channel_data)
            raw_notched._data[ch_idx, :] = filtered_data
        except Exception as e:
            print(f"  SOS filter failed: {e}, using alternative")
            # Alternative: moving average as simple notch
            window_size = int(sfreq / freq)  # Number of samples per cycle
            if window_size % 2 == 0:
                window_size += 1  # Make odd
            filtered_data = signal.medfilt(channel_data, kernel_size=window_size)
            raw_notched._data[ch_idx, :] = filtered_data
    
    return raw_notched

def apply_safe_fir_notch(raw, freq, bandwidth, filter_order, sfreq):
    """
    Safe FIR notch filter design
    """
    from scipy import signal
    import copy
    
    raw_notched = copy.deepcopy(raw)
    
    # Design FIR bandstop filter
    nyquist = sfreq / 2
    low_cut = (freq - bandwidth/2) / nyquist
    high_cut = (freq + bandwidth/2) / nyquist
    
    # Ensure valid range
    low_cut = max(0.01, min(0.49, low_cut))
    high_cut = max(0.01, min(0.49, high_cut))
    
    if low_cut >= high_cut:
        low_cut = 0.49
        high_cut = 0.51
    
    # Use reasonable filter length
    filter_length = min(101, int(sfreq))  # Don't make it too long
    if filter_length % 2 == 0:
        filter_length += 1
    
    print(f"  FIR filter length: {filter_length}")
    
    # Design FIR filter
    taps = signal.firwin(filter_length, [low_cut, high_cut], pass_zero=False, fs=sfreq)
    
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw._data[ch_idx, :]
        
        try:
            filtered_data = signal.filtfilt(taps, 1.0, channel_data)
            raw_notched._data[ch_idx, :] = filtered_data
        except Exception as e:
            print(f"  FIR filter failed: {e}")
            # Fallback to simple approach
            raw_notched._data[ch_idx, :] = channel_data
    
    return raw_notched

def apply_frequency_domain_notch(raw, freq, bandwidth, sfreq):
    """
    Frequency domain notch filter - most stable approach
    """
    from scipy import fft
    import copy
    
    raw_notched = copy.deepcopy(raw)
    
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw._data[ch_idx, :]
        n = len(channel_data)
        
        # Compute FFT
        freqs = fft.fftfreq(n, 1/sfreq)
        fft_vals = fft.fft(channel_data)
        
        # Find indices to notch
        notch_indices = np.where((np.abs(freqs) >= freq - bandwidth/2) & 
                                (np.abs(freqs) <= freq + bandwidth/2))[0]
        
        # Apply gentle attenuation (not complete removal)
        attenuation = 0.1  # Reduce to 10% of original (20 dB attenuation)
        fft_vals[notch_indices] *= attenuation
        
        # Inverse FFT
        cleaned_data = np.real(fft.ifft(fft_vals))
        raw_notched._data[ch_idx, :] = cleaned_data
    
    print("  Applied frequency domain notch filter")
    return raw_notched

def plot_notch_filter_diagnostic_clear(raw_before, raw_after, freq, bandwidth, method, sfreq):
    """
    Clear diagnostic plots for notch filter with easy-to-understand results
    """
    # Get a representative channel
    eeg_channels = [ch for ch in raw_before.ch_names if ch != 'STI']
    if not eeg_channels:
        print("  No EEG channels found for notch filter diagnostic")
        return None
    
    channel_name = eeg_channels[0]
    ch_idx = raw_before.ch_names.index(channel_name)
    
    # Extract data
    data_before, times = raw_before[ch_idx, :]
    data_after, _ = raw_after[ch_idx, :]
    data_before = data_before.flatten()
    data_after = data_after.flatten()
    
    from scipy import signal
    
    # Compute power spectral density
    f_before, Pxx_before = signal.welch(data_before, sfreq, nperseg=1024)
    f_after, Pxx_after = signal.welch(data_after, sfreq, nperseg=1024)
    
    # Create clearer visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Notch Filter Results - {method} at {freq} Hz', fontsize=16, fontweight='bold')
    
    # 1. Linear scale to see actual power values
    freq_mask = (f_before >= freq - 10) & (f_before <= freq + 10)
    
    axes[0].plot(f_before[freq_mask], Pxx_before[freq_mask], 'b-', label='Before', linewidth=3, alpha=0.8)
    axes[0].plot(f_after[freq_mask], Pxx_after[freq_mask], 'r-', label='After', linewidth=3, alpha=0.8)
    axes[0].axvline(x=freq, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Target: {freq} Hz')
    axes[0].set_title('Power Spectrum - Linear Scale\n(Shows Actual Power Values)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value annotations
    target_idx = np.argmin(np.abs(f_before - freq))
    before_power = Pxx_before[target_idx]
    after_power = Pxx_after[target_idx]
    
    axes[0].annotate(f'Before: {before_power:.2e}', 
                    xy=(freq, before_power), xytext=(freq+2, before_power),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=12, color='blue')
    
    axes[0].annotate(f'After: {after_power:.2e}', 
                    xy=(freq, after_power), xytext=(freq+2, after_power*0.5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=12, color='red')
    
    # 2. Bar chart for clear comparison
    reduction_factor = before_power / after_power if after_power > 0 else float('inf')
    attenuation_db = 10 * np.log10(after_power / before_power) if before_power > 0 else 0
    
    categories = ['Before', 'After']
    power_values = [before_power, after_power]
    colors = ['blue', 'red']
    
    bars = axes[1].bar(categories, power_values, color=colors, alpha=0.7, width=0.6)
    axes[1].set_title('Power at 50 Hz - Direct Comparison')
    axes[1].set_ylabel('Power')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, power_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add statistics box
    stats_text = f"Filter Performance:\n\n"
    stats_text += f"Power Reduction: {reduction_factor:.0f}x\n"
    stats_text += f"Attenuation: {attenuation_db:.1f} dB\n\n"
    
    if attenuation_db < -40:
        stats_text += "✓ EXCELLENT\nVery strong attenuation"
    elif attenuation_db < -20:
        stats_text += "✓ VERY GOOD\nStrong attenuation"
    elif attenuation_db < -10:
        stats_text += "✓ GOOD\nModerate attenuation"
    elif attenuation_db < -3:
        stats_text += "✓ WEAK\nMild attenuation"
    else:
        stats_text += "⚠️ POOR\nLittle to no attenuation"
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    print(f"\n{'='*50}")
    print("NOTCH FILTER PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Power at {freq} Hz:")
    print(f"  Before: {before_power:.2e}")
    print(f"  After:  {after_power:.2e}")
    print(f"  Reduction: {reduction_factor:.0f} times")
    print(f"  Attenuation: {attenuation_db:.1f} dB")
    print(f"  Method: {method}")
    
    if attenuation_db < -20:
        print(f"✅ EXCELLENT: Filter is working very effectively!")
    elif attenuation_db < -10:
        print(f"✅ GOOD: Filter is working well")
    else:
        print(f"⚠️  WEAK: Filter may need adjustment")
    
    return fig

def apply_manual_notch_filter(raw, freq=50., bandwidth=2.0, filter_order=101):
    """
    Manual FIR notch filter implementation using window method
    """
    from scipy import signal
    import copy
    
    raw_notched = copy.deepcopy(raw)
    sfreq = raw.info['sfreq']
    
    # Design FIR notch filter using window method
    nyquist = sfreq / 2.0
    low_cut = (freq - bandwidth/2) / nyquist
    high_cut = (freq + bandwidth/2) / nyquist
    
    # Ensure frequencies are within valid range
    low_cut = max(0.001, min(0.499, low_cut))
    high_cut = max(0.001, min(0.499, high_cut))
    
    if low_cut >= high_cut:
        print(f"  Warning: Invalid frequency range {low_cut}-{high_cut}, using default")
        low_cut = 0.49 / nyquist
        high_cut = 0.51 / nyquist
    
    # Design bandstop filter
    taps = signal.firwin(filter_order, [low_cut, high_cut], pass_zero=False, fs=sfreq)
    
    # Apply filter to each EEG channel
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw._data[ch_idx, :]
        # Apply filter using filtfilt for zero phase distortion
        filtered_data = signal.filtfilt(taps, 1.0, channel_data)
        raw_notched._data[ch_idx, :] = filtered_data
    
    print(f"  Applied manual FIR notch filter (order: {filter_order})")
    return raw_notched

def apply_spectrum_fit_notch(raw, freq=50., bandwidth=2.0):
    """
    Advanced notch filter using spectrum fitting and subtraction
    This method estimates and subtracts the 50Hz component rather than filtering
    """
    from scipy import fft, signal
    import copy
    
    raw_notched = copy.deepcopy(raw)
    sfreq = raw.info['sfreq']
    
    eeg_channel_indices = [i for i, ch_name in enumerate(raw.ch_names) if ch_name != 'STI']
    
    for ch_idx in eeg_channel_indices:
        channel_data = raw._data[ch_idx, :]
        
        # Step 1: Compute FFT
        n = len(channel_data)
        freqs = fft.fftfreq(n, 1/sfreq)
        fft_vals = fft.fft(channel_data)
        
        # Step 2: Find 50Hz component and nearby frequencies
        target_idx = np.argmin(np.abs(freqs - freq))
        bandwidth_bins = int(bandwidth * n / sfreq)
        
        # Create indices for the frequency band to remove
        start_idx = max(0, target_idx - bandwidth_bins)
        end_idx = min(n, target_idx + bandwidth_bins)
        
        # Step 3: Create a clean spectrum by interpolating over the notch region
        clean_fft = fft_vals.copy()
        
        # Linear interpolation in frequency domain around the notch
        if start_idx > 0 and end_idx < n:
            # Values before the notch
            before_val = clean_fft[start_idx - 1]
            # Values after the notch  
            after_val = clean_fft[end_idx + 1]
            
            # Linear interpolation across the notch region
            for i in range(start_idx, end_idx + 1):
                alpha = (i - start_idx) / (end_idx - start_idx)
                clean_fft[i] = (1 - alpha) * before_val + alpha * after_val
        
        # Step 4: Inverse FFT to get cleaned signal
        cleaned_data = np.real(fft.ifft(clean_fft))
        
        raw_notched._data[ch_idx, :] = cleaned_data
    
    print("  Applied spectrum fitting notch filter")
    return raw_notched

def plot_notch_filter_verification(raw_before, raw_after, freq, bandwidth, method, sfreq):
    """
    Clearer diagnostic plots with better explanations
    """
    eeg_channels = [ch for ch in raw_before.ch_names if ch != 'STI']
    if not eeg_channels:
        return
    
    channel_name = eeg_channels[0]
    ch_idx = raw_before.ch_names.index(channel_name)
    
    # Extract data
    data_before, times = raw_before[ch_idx, :]
    data_after, _ = raw_after[ch_idx, :]
    data_before = data_before.flatten()
    data_after = data_after.flatten()
    
    from scipy import signal
    f_before, Pxx_before = signal.welch(data_before, sfreq, nperseg=1024)
    f_after, Pxx_after = signal.welch(data_after, sfreq, nperseg=1024)
    
    # Create clearer visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Notch Filter Results - {method} at {freq} Hz', fontsize=16, fontweight='bold')
    
    # 1. Linear scale to see actual power values
    freq_mask = (f_before >= freq - 10) & (f_before <= freq + 10)
    
    axes[0].plot(f_before[freq_mask], Pxx_before[freq_mask], 'b-', label='Before', linewidth=3, alpha=0.8)
    axes[0].plot(f_after[freq_mask], Pxx_after[freq_mask], 'r-', label='After', linewidth=3, alpha=0.8)
    axes[0].axvline(x=freq, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Target: {freq} Hz')
    axes[0].set_title('Power Spectrum - Linear Scale\n(Shows Actual Power Values)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value annotations
    target_idx = np.argmin(np.abs(f_before - freq))
    before_power = Pxx_before[target_idx]
    after_power = Pxx_after[target_idx]
    
    axes[0].annotate(f'Before: {before_power:.2e}', 
                    xy=(freq, before_power), xytext=(freq+2, before_power),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=12, color='blue')
    
    axes[0].annotate(f'After: {after_power:.2e}', 
                    xy=(freq, after_power), xytext=(freq+2, after_power*0.5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=12, color='red')
    
    # 2. Bar chart for clear comparison
    reduction_factor = before_power / after_power if after_power > 0 else float('inf')
    attenuation_db = 10 * np.log10(after_power / before_power) if before_power > 0 else 0
    
    categories = ['Before', 'After']
    power_values = [before_power, after_power]
    colors = ['blue', 'red']
    
    bars = axes[1].bar(categories, power_values, color=colors, alpha=0.7, width=0.6)
    axes[1].set_title('Power at 50 Hz - Direct Comparison')
    axes[1].set_ylabel('Power')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, power_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add statistics box
    stats_text = f"Filter Performance:\n\n"
    stats_text += f"Power Reduction: {reduction_factor:.0f}x\n"
    stats_text += f"Attenuation: {attenuation_db:.1f} dB\n\n"
    
    if attenuation_db < -40:
        stats_text += "✓ EXCELLENT\nVery strong attenuation"
    elif attenuation_db < -20:
        stats_text += "✓ VERY GOOD\nStrong attenuation"
    elif attenuation_db < -10:
        stats_text += "✓ GOOD\nModerate attenuation"
    elif attenuation_db < -3:
        stats_text += "✓ WEAK\nMild attenuation"
    else:
        stats_text += "⚠️ POOR\nLittle to no attenuation"
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*50}")
    print("NOTCH FILTER PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Power at {freq} Hz:")
    print(f"  Before: {before_power:.2e}")
    print(f"  After:  {after_power:.2e}")
    print(f"  Reduction: {reduction_factor:.0f} times")
    print(f"  Attenuation: {attenuation_db:.1f} dB")
    print(f"  Method: {method}")
    
    if attenuation_db < -20:
        print(f"✅ EXCELLENT: Filter is working very effectively!")
    elif attenuation_db < -10:
        print(f"✅ GOOD: Filter is working well")
    else:
        print(f"⚠️  WEAK: Filter may need adjustment")
    
    return fig

def create_default_neighbors(eeg_channels):
    """
    Create default neighbor relationships optimized for the specific electrode setup:
    Fz, C3, Cz, C4, CP1, CPz, CP2, Pz
    """
    neighbors_dict = {}
    
    # Define optimized neighbor relationships for your specific electrode setup
    standard_neighbors = {
        # Frontal electrode
        'Fz': ['C3', 'Cz', 'C4'],
        
        # Central left electrodes
        'C3': ['Fz', 'Cz', 'CP1'],
        
        # Central midline electrodes
        'Cz': ['Fz', 'C3', 'C4', 'CPz'],
        
        # Central right electrodes
        'C4': ['Fz', 'Cz', 'CP2'],
        
        # Centro-parietal left
        'CP1': ['C3', 'Cz', 'CPz', 'Pz'],
        
        # Centro-parietal midline
        'CPz': ['Cz', 'CP1', 'CP2', 'Pz'],
        
        # Centro-parietal right
        'CP2': ['C4', 'Cz', 'CPz', 'Pz'],
        
        # Parietal midline
        'Pz': ['CP1', 'CPz', 'CP2']
    }
    
    # For channels in our standard list, use the predefined neighbors
    for channel in eeg_channels:
        if channel in standard_neighbors:
            # Only include neighbors that are actually present in the data
            available_neighbors = [ch for ch in standard_neighbors[channel] if ch in eeg_channels]
            neighbors_dict[channel] = available_neighbors
        else:
            # For non-standard channels, use all other channels as potential neighbors
            potential_neighbors = [ch for ch in eeg_channels if ch != channel]
            # Use up to 4 closest neighbors
            neighbors_dict[channel] = potential_neighbors[:min(4, len(potential_neighbors))]
    
    return neighbors_dict

def apply_laplacian_reference(raw, method='large', neighbors_dict=None):
    """
    Apply Laplacian referencing to reduce volume conduction effects
    
    Parameters:
    raw: MNE Raw object
    method: 'large' for large Laplacian (subtract mean of neighbors), 
            'small' for small Laplacian (not implemented yet)
    neighbors_dict: Optional dictionary specifying neighbors for each channel.
                   If None, will use automatic neighbor detection based on electrode positions.
    
    Returns:
    raw_laplacian: Raw object with Laplacian-referenced data
    """
    raw_laplacian = raw.copy()
    
    # Get EEG channel names (exclude STI)
    eeg_channels = [ch for ch in raw.ch_names if ch != 'STI']
    
    print(f"Applying {method} Laplacian referencing:")
    print(f"  EEG channels: {eeg_channels}")
    
    if len(eeg_channels) < 2:
        print("  Warning: Not enough EEG channels for Laplacian referencing")
        return raw_laplacian
    
    # If no neighbors dictionary provided, create one based on specific electrode setup
    if neighbors_dict is None:
        neighbors_dict = create_default_neighbors(eeg_channels)
    
    print("  Neighbor relationships:")
    for target_ch, neighbors in neighbors_dict.items():
        if neighbors:  # Only print if there are neighbors
            print(f"    {target_ch} -> {neighbors}")
    
    # Apply Laplacian for each channel
    for target_ch in eeg_channels:
        if target_ch in neighbors_dict and neighbors_dict[target_ch]:
            neighbor_chs = neighbors_dict[target_ch]
            available_neighbors = [ch for ch in neighbor_chs if ch in eeg_channels]
            
            if len(available_neighbors) >= 1:
                # Get target channel data
                target_idx = raw.ch_names.index(target_ch)
                target_data = raw._data[target_idx, :]
                
                # Get neighbor data and compute mean
                neighbor_data = np.zeros_like(target_data)
                for neighbor_ch in available_neighbors:
                    neighbor_idx = raw.ch_names.index(neighbor_ch)
                    neighbor_data += raw._data[neighbor_idx, :]
                
                neighbor_mean = neighbor_data / len(available_neighbors)
                
                # Apply Laplacian: subtract mean of neighbors
                laplacian_data = target_data - neighbor_mean
                raw_laplacian._data[target_idx, :] = laplacian_data
                
                print(f"  ✓ {target_ch}: referenced to {len(available_neighbors)} neighbors")
            else:
                print(f"  ⚠️  {target_ch}: no available neighbors, skipping")
                # Keep original data if no neighbors available
        else:
            print(f"  ⚠️  {target_ch}: no neighbor information, skipping")
            # Keep original data if no neighbor information
    
    return raw_laplacian

def plot_laplacian_comparison_function(raw_before, raw_after, channel_name, title="", save_path=None):
    """
    Plot comparison before and after Laplacian referencing
    """
    if channel_name not in raw_before.ch_names or channel_name not in raw_after.ch_names:
        print(f"Channel {channel_name} not found in raw data")
        return None
        
    ch_idx_before = raw_before.ch_names.index(channel_name)
    ch_idx_after = raw_after.ch_names.index(channel_name)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Laplacian Referencing Comparison - {title} - Channel: {channel_name}', fontsize=16)
    
    # 1. Time domain - before (first 5 seconds)
    data_before, times_before = raw_before[ch_idx_before, :]
    segment_samples = min(5000, len(data_before[0]))
    axes[0, 0].plot(times_before[:segment_samples], data_before[0][:segment_samples], 'b-', linewidth=1, alpha=0.8, label='Before Laplacian')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_title('Before Laplacian - Time Domain (5s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Time domain - after (first 5 seconds)
    data_after, times_after = raw_after[ch_idx_after, :]
    segment_samples = min(5000, len(data_after[0]))
    axes[1, 0].plot(times_after[:segment_samples], data_after[0][:segment_samples], 'r-', linewidth=1, alpha=0.8, label='After Laplacian')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('After Laplacian - Time Domain (5s)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 3. Frequency domain - before
    from scipy import signal
    data_before_flat = data_before.flatten()
    data_after_flat = data_after.flatten()
    
    f_before, Pxx_before = signal.welch(data_before_flat, raw_before.info['sfreq'], nperseg=1024)
    axes[0, 1].semilogy(f_before, Pxx_before, 'b-', linewidth=1.5, alpha=0.8, label='Before Laplacian')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].set_title('Before Laplacian - Frequency Domain')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 30)  # Focus on 0-30 Hz range for P300
    
    # 4. Frequency domain - after
    f_after, Pxx_after = signal.welch(data_after_flat, raw_after.info['sfreq'], nperseg=1024)
    axes[1, 1].semilogy(f_after, Pxx_after, 'r-', linewidth=1.5, alpha=0.8, label='After Laplacian')
    axes[1, 1].set_ylabel('Power Spectral Density')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_title('After Laplacian - Frequency Domain')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 30)  # Focus on 0-30 Hz range for P300
    
    # Calculate statistics
    stats_text = f"Statistics for {channel_name}:\n\n"
    stats_text += f"Before Laplacian:\n"
    stats_text += f"  Mean: {np.mean(data_before_flat):.2f} μV\n"
    stats_text += f"  Std: {np.std(data_before_flat):.2f} μV\n"
    stats_text += f"  Range: {np.ptp(data_before_flat):.2f} μV\n\n"
    stats_text += f"After Laplacian:\n"
    stats_text += f"  Mean: {np.mean(data_after_flat):.2f} μV\n"
    stats_text += f"  Std: {np.std(data_after_flat):.2f} μV\n"
    stats_text += f"  Range: {np.ptp(data_after_flat):.2f} μV"
    
    # Add statistics text box
    fig.text(0.02, 0.02, stats_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Laplacian comparison: {save_path}")
    
    plt.show()
    
    return fig

def plot_single_trial_filter_comparison(raw_before, raw_after, epochs_before, epochs_after, 
                                       trial_index=0, condition='target', channel='Fz',
                                       tmin=-0.2, tmax=0.8, title="", save_path=None):
    """
    Plot comparison of a single trial before and after filtering
    
    Parameters:
    raw_before: Raw data before filtering
    raw_after: Raw data after filtering
    epochs_before: Epochs before filtering
    epochs_after: Epochs after filtering
    trial_index: Index of the trial to plot
    condition: Condition of the trial ('target', 'non-target', 'distractor')
    channel: Channel to plot
    tmin, tmax: Time window to plot
    title: Plot title
    save_path: Path to save the plot
    """
    # Check if the condition exists in epochs
    if condition not in epochs_before.event_id or condition not in epochs_after.event_id:
        print(f"Condition '{condition}' not found in epochs. Available: {list(epochs_before.event_id.keys())}")
        return None
    
    # Check if trial index is valid
    n_trials_before = len(epochs_before[condition])
    n_trials_after = len(epochs_after[condition])
    
    if trial_index >= n_trials_before or trial_index >= n_trials_after:
        print(f"Trial index {trial_index} is out of range. Available: 0-{min(n_trials_before, n_trials_after)-1}")
        return None
    
    # Check if channel exists
    if channel not in raw_before.ch_names:
        print(f"Channel '{channel}' not found. Available: {raw_before.ch_names}")
        return None
    
    # Get the trial data
    trial_before = epochs_before[condition][trial_index]
    trial_after = epochs_after[condition][trial_index]
    
    # Get times
    times = epochs_before.times
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Single Trial Filtering Comparison - {title}\nTrial {trial_index} ({condition}) - Channel: {channel}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Single channel - before filtering
    ch_idx = trial_before.ch_names.index(channel)
    axes[0, 0].plot(times, trial_before.get_data()[0, ch_idx, :], 'b-', linewidth=2, label='Before Filtering')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus Onset')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_title(f'Before Filtering - {channel}')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Single channel - after filtering
    axes[0, 1].plot(times, trial_after.get_data()[0, ch_idx, :], 'r-', linewidth=2, label='After Filtering')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus Onset')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].set_title(f'After Filtering - {channel}')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Overlay comparison
    axes[1, 0].plot(times, trial_before.get_data()[0, ch_idx, :], 'b-', linewidth=2, alpha=0.7, label='Before Filtering')
    axes[1, 0].plot(times, trial_after.get_data()[0, ch_idx, :], 'r-', linewidth=2, alpha=0.7, label='After Filtering')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus Onset')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].set_title(f'Overlay Comparison - {channel}')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. All channels for this trial (after filtering)
    trial_data_after = trial_after.get_data()[0]  # Shape: (n_channels, n_times)
    
    # Plot all channels
    for ch_idx_all in range(trial_data_after.shape[0]):
        ch_name = trial_after.ch_names[ch_idx_all]
        color = 'red' if ch_name == channel else 'gray'
        alpha = 1.0 if ch_name == channel else 0.3
        linewidth = 2 if ch_name == channel else 0.5
        axes[1, 1].plot(times, trial_data_after[ch_idx_all, :], 
                       color=color, alpha=alpha, linewidth=linewidth, label=ch_name if ch_name == channel else "")
    
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus Onset')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude (μV)')
    axes[1, 1].set_title(f'All Channels After Filtering\n(Highlighted: {channel})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Calculate statistics
    data_before = trial_before.get_data()[0, ch_idx, :]
    data_after = trial_after.get_data()[0, ch_idx, :]
    
    stats_text = f"Trial {trial_index} Statistics ({condition}):\n\n"
    stats_text += f"Before Filtering:\n"
    stats_text += f"  Mean: {np.mean(data_before):.2f} μV\n"
    stats_text += f"  Std: {np.std(data_before):.2f} μV\n"
    stats_text += f"  Range: {np.ptp(data_before):.2f} μV\n\n"
    stats_text += f"After Filtering:\n"
    stats_text += f"  Mean: {np.mean(data_after):.2f} μV\n"
    stats_text += f"  Std: {np.std(data_after):.2f} μV\n"
    stats_text += f"  Range: {np.ptp(data_after):.2f} μV"
    
    # Add statistics text box
    fig.text(0.02, 0.02, stats_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved single trial comparison: {save_path}")
    
    plt.show()
    
    return fig

def plot_multiple_trials_filter_comparison(epochs_before, epochs_after, 
                                          trial_indices=[0, 1, 2], condition='target', 
                                          channel='Fz', title="", save_path=None):
    """
    Plot multiple trials before and after filtering for comparison
    """
    if condition not in epochs_before.event_id:
        print(f"Condition '{condition}' not found in epochs")
        return None
    
    n_trials = min(len(epochs_before[condition]), len(epochs_after[condition]))
    valid_indices = [idx for idx in trial_indices if idx < n_trials]
    
    if not valid_indices:
        print(f"No valid trial indices. Available: 0-{n_trials-1}")
        return None
    
    n_plots = len(valid_indices)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Multiple Trials Filtering Comparison - {title}\n{condition} - Channel: {channel}', 
                 fontsize=16, fontweight='bold')
    
    times = epochs_before.times
    
    for idx, (trial_idx, ax) in enumerate(zip(valid_indices, axes.flat)):
        trial_before = epochs_before[condition][trial_idx]
        trial_after = epochs_after[condition][trial_idx]
        
        if channel in trial_before.ch_names:
            ch_idx = trial_before.ch_names.index(channel)
            data_before = trial_before.get_data()[0, ch_idx, :]
            data_after = trial_after.get_data()[0, ch_idx, :]
            
            ax.plot(times, data_before, 'b-', alpha=0.7, linewidth=1.5, label='Before')
            ax.plot(times, data_after, 'r-', alpha=0.7, linewidth=1.5, label='After')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title(f'Trial {trial_idx}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Hide unused subplots
    for idx in range(len(valid_indices), n_rows * n_cols):
        axes.flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multiple trials comparison: {save_path}")
    
    plt.show()
    
    return fig

def plot_erps(epochs, title, save_path=None):
    """
    Plot Event-Related Potentials for different conditions with GFP
    """
    conditions = ['target', 'non-target', 'distractor']
    colors = ['green', 'blue', 'red']
    
    # Create a more flexible figure layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Event-Related Potentials and Global Field Power - {title}', 
                 fontsize=16, fontweight='bold')
    
    # Create a grid specification for more control
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot individual conditions with GFP
    for i, (condition, color) in enumerate(zip(conditions, colors)):
        if condition in epochs.event_id:
            # Get the evoked response for this condition
            evoked = epochs[condition].average()
            
            # Plot ERP on the left column - use manual plotting instead of MNE's plot
            ax_erp = fig.add_subplot(gs[i, 0])
            
            # Get data and times
            data = evoked.data
            times = evoked.times
            
            # Plot all channels with transparency
            for ch_idx in range(data.shape[0]):
                ax_erp.plot(times, data[ch_idx, :], 
                           color=color, alpha=0.3, linewidth=0.5)
            
            # Plot average across channels
            mean_erp = data.mean(axis=0)
            ax_erp.plot(times, mean_erp, color='black', linewidth=2, 
                       label='Channel Average')
            
            ax_erp.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_erp.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                          label='Stimulus Onset')
            ax_erp.set_xlabel('Time (s)')
            ax_erp.set_ylabel('Amplitude (μV)')
            ax_erp.set_title(f'{condition.title()} - ERP', fontweight='bold')
            ax_erp.grid(True, alpha=0.3)
            ax_erp.legend()
            
            # Calculate and plot GFP on the right column
            ax_gfp = fig.add_subplot(gs[i, 1])
            gfp = evoked.data.std(axis=0)  # GFP = standard deviation across channels
            
            ax_gfp.plot(times, gfp, color=color, linewidth=2, label='GFP')
            ax_gfp.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_gfp.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                          label='Stimulus')
            
            # Add P300 window shading (typical P300 latency: 250-500 ms)
            ax_gfp.axvspan(0.25, 0.5, alpha=0.2, color='yellow', 
                          label='P300 Window')
            
            ax_gfp.set_xlabel('Time (s)')
            ax_gfp.set_ylabel('GFP (μV)')
            ax_gfp.set_title(f'{condition.title()} - Global Field Power', 
                           fontweight='bold')
            ax_gfp.grid(True, alpha=0.3)
            ax_gfp.legend()
            
            # Calculate and display GFP statistics in P300 window
            p300_mask = (times >= 0.25) & (times <= 0.5)
            if np.any(p300_mask):
                gfp_p300 = gfp[p300_mask]
                max_gfp = np.max(gfp_p300)
                max_time = times[p300_mask][np.argmax(gfp_p300)]
                
                stats_text = f"P300 Window (250-500 ms):\nMax GFP: {max_gfp:.2f} μV\nat {max_time:.3f} s"
                ax_gfp.text(0.02, 0.98, stats_text, transform=ax_gfp.transAxes,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Use constrained_layout instead of tight_layout for better results
    plt.tight_layout()
    
    if save_path:
        # Use bbox_inches='tight' and pad_inches to ensure everything fits
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"  Saved ERP+GFP plot: {save_path}")
    
    plt.show()
    
    # Also create a separate comparison figure with GFP for all conditions
    plot_gfp_comparison(epochs, title, save_path)
    
    return fig

def plot_gfp_comparison(epochs, title, save_path=None):
    """
    Plot GFP comparison across all conditions in a single figure
    - Places legend at bottom to avoid overlap
    """
    conditions = ['target', 'non-target', 'distractor']
    colors = ['green', 'blue', 'red']
    
    # Create figure with more height to accommodate legend at bottom
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot GFP for each condition
    for condition, color in zip(conditions, colors):
        if condition in epochs.event_id:
            evoked = epochs[condition].average()
            gfp = evoked.data.std(axis=0)
            times = evoked.times
            
            ax.plot(times, gfp, color=color, linewidth=2.5, 
                   label=condition.title(), alpha=0.8)
    
    # Add common elements
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
               linewidth=2, label='Stimulus Onset')
    
    # Create a patch for the P300 window for the legend
    import matplotlib.patches as mpatches
    p300_patch = mpatches.Patch(color='yellow', alpha=0.3, label='P300 Window')
    
    # Shade the P300 window on the plot
    ax.axvspan(0.25, 0.5, alpha=0.2, color='yellow')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Global Field Power (μV)', fontsize=12)
    ax.set_title(f'GFP Comparison - {title}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create a single combined legend and place it at the bottom
    handles, labels = ax.get_legend_handles_labels()
    handles.append(p300_patch)  # Add the P300 window patch to handles
    labels.append('P300 Window')  # Add the P300 window label
    
    # Position legend at bottom center to avoid overlap with data
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Add statistics table - position it in upper left to avoid legend
    stats_text = "GFP Peaks in P300 Window (250-500 ms):\n\n"
    stats_data = []
    
    for condition, color in zip(conditions, colors):
        if condition in epochs.event_id:
            evoked = epochs[condition].average()
            gfp = evoked.data.std(axis=0)
            times = evoked.times
            
            p300_mask = (times >= 0.25) & (times <= 0.5)
            if np.any(p300_mask):
                gfp_p300 = gfp[p300_mask]
                max_gfp = np.max(gfp_p300)
                max_time = times[p300_mask][np.argmax(gfp_p300)]
                mean_gfp = np.mean(gfp_p300)
                
                stats_data.append({
                    'condition': condition.title(),
                    'peak': max_gfp,
                    'time': max_time,
                    'mean': mean_gfp
                })
    
    # Sort by peak amplitude (descending)
    stats_data.sort(key=lambda x: x['peak'], reverse=True)
    
    for data in stats_data:
        stats_text += f"{data['condition']}:\n"
        stats_text += f"  Peak: {data['peak']:.2f} μV\n"
        stats_text += f"  Time: {data['time']:.3f} s\n"
        stats_text += f"  Mean: {data['mean']:.2f} μV\n\n"
    
    # Position stats text in upper left
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout to make room for the legend at bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space at bottom for legend
    
    if save_path:
        # Save with a different name to avoid overwriting
        gfp_save_path = save_path.replace('.png', '_gfp_comparison.png')
        # Use bbox_inches='tight' with adequate padding
        plt.savefig(gfp_save_path, dpi=300, bbox_inches='tight', 
                   pad_inches=0.8, facecolor='white')
        print(f"  Saved GFP comparison plot: {gfp_save_path}")
    
    plt.show()
    
    return fig

def create_comprehensive_topomaps(epochs_final, base_name, output_dir, times=None):
    """
    Create comprehensive topomap analysis including individual conditions and differences
    """
    if times is None:
        times = np.arange(0.2, 0.6, 0.05)  # Default P300 time window
    
    print(f"\nCreating comprehensive topomap analysis...")
    print(f"Time points: {times}")
    
    # Dictionary to store created files
    topomap_files = {
        'individual': [],
        'differences': [],
        'comparisons': []
    }
    
    try:
        # Get evoked responses for all available conditions
        evokeds = {}
        conditions_to_plot = ['target', 'non-target', 'distractor']
        
        for condition in conditions_to_plot:
            if condition in epochs_final.event_id:
                evokeds[condition] = epochs_final[condition].average()
                print(f"  Loaded {condition} condition: {len(epochs_final[condition])} epochs")
        
        # Create individual condition topomaps using MNE's built-in plotting
        for condition, evoked in evokeds.items():
            try:
                topomap_path = os.path.join(output_dir, f'{base_name}_topomaps_{condition}.png')
                fig = evoked.plot_topomap(times=times, show=False)
                fig.suptitle(f'P300 Topomaps - {base_name} - {condition.title()}', 
                            fontsize=16, fontweight='bold')
                fig.savefig(topomap_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close(fig)
                topomap_files['individual'].append(topomap_path)
                print(f"  ✓ Created individual topomap: {os.path.basename(topomap_path)}")
            except Exception as e:
                print(f"  ✗ Failed to create individual topomap for {condition}: {e}")
        
        # Create difference topomaps for available comparisons
        comparisons = [
            ('target', 'non-target', 'Target vs Non-target'),
            ('target', 'distractor', 'Target vs Distractor'),
            ('non-target', 'distractor', 'Non-target vs Distractor')
        ]
        
        for cond1, cond2, comparison_name in comparisons:
            if cond1 in evokeds and cond2 in evokeds:
                try:
                    # Calculate difference
                    evoked_diff = mne.combine_evoked([evokeds[cond1], evokeds[cond2]], weights=[1, -1])
                    
                    # Create simple difference plot using MNE's built-in plotting
                    diff_path = os.path.join(output_dir, f'{base_name}_topomaps_diff_{cond1}_vs_{cond2}.png')
                    fig_diff = evoked_diff.plot_topomap(times=times, show=False)
                    fig_diff.suptitle(f'P300 Difference - {base_name}\n{comparison_name}', 
                                    fontsize=16, fontweight='bold')
                    fig_diff.savefig(diff_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close(fig_diff)
                    topomap_files['differences'].append(diff_path)
                    print(f"  ✓ Created difference topomap: {os.path.basename(diff_path)}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to create difference topomap for {cond1} vs {cond2}: {e}")
        
        # Create manual comparison plots for key time points
        try:
            # Use key P300 time points for manual comparison
            key_times = [0.3, 0.4, 0.5]
            
            for cond1, cond2, comparison_name in comparisons:
                if cond1 in evokeds and cond2 in evokeds:
                    # Create manual comparison plot
                    comp_path = os.path.join(output_dir, f'{base_name}_topomaps_comparison_{cond1}_vs_{cond2}.png')
                    
                    # Create figure with subplots
                    fig, axes = plt.subplots(3, len(key_times), figsize=(15, 12))
                    if len(key_times) == 1:
                        axes = axes.reshape(-1, 1)
                    
                    fig.suptitle(f'P300 Comparison - {base_name}\n{comparison_name}', 
                                fontsize=16, fontweight='bold')
                    
                    # Calculate difference
                    evoked_diff = mne.combine_evoked([evokeds[cond1], evokeds[cond2]], weights=[1, -1])
                    
                    # Plot each time point manually
                    for i, time_point in enumerate(key_times):
                        # Condition 1
                        try:
                            mne.viz.plot_topomap(
                                evokeds[cond1].data[:, np.argmin(np.abs(evokeds[cond1].times - time_point))],
                                evokeds[cond1].info,
                                axes=axes[0, i],
                                show=False,
                                contours=0
                            )
                            axes[0, i].set_title(f'{cond1.title()}\n{time_point:.2f}s', fontweight='bold')
                        except Exception as e:
                            print(f"    Warning: Could not plot {cond1} at {time_point}s: {e}")
                        
                        # Condition 2
                        try:
                            mne.viz.plot_topomap(
                                evokeds[cond2].data[:, np.argmin(np.abs(evokeds[cond2].times - time_point))],
                                evokeds[cond2].info,
                                axes=axes[1, i],
                                show=False,
                                contours=0
                            )
                            axes[1, i].set_title(f'{cond2.title()}\n{time_point:.2f}s', fontweight='bold')
                        except Exception as e:
                            print(f"    Warning: Could not plot {cond2} at {time_point}s: {e}")
                        
                        # Difference
                        try:
                            mne.viz.plot_topomap(
                                evoked_diff.data[:, np.argmin(np.abs(evoked_diff.times - time_point))],
                                evoked_diff.info,
                                axes=axes[2, i],
                                show=False,
                                contours=0
                            )
                            axes[2, i].set_title(f'Difference\n{time_point:.2f}s', fontweight='bold')
                        except Exception as e:
                            print(f"    Warning: Could not plot difference at {time_point}s: {e}")
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    fig.savefig(comp_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close(fig)
                    topomap_files['comparisons'].append(comp_path)
                    print(f"  ✓ Created manual comparison plot: {os.path.basename(comp_path)}")
                    
        except Exception as e:
            print(f"  ✗ Failed to create manual comparison plots: {e}")
        
        # Create a manual summary plot
        try:
            summary_times = [0.3, 0.4, 0.5]
            summary_path = os.path.join(output_dir, f'{base_name}_topomaps_summary.png')
            
            conditions_list = list(evokeds.keys())
            n_conditions = len(conditions_list)
            n_times = len(summary_times)
            
            fig, axes = plt.subplots(n_conditions, n_times, figsize=(4*n_times, 4*n_conditions))
            if n_conditions == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'P300 Topomap Summary - {base_name}', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            for i, condition in enumerate(conditions_list):
                evoked = evokeds[condition]
                for j, time_point in enumerate(summary_times):
                    try:
                        mne.viz.plot_topomap(
                            evoked.data[:, np.argmin(np.abs(evoked.times - time_point))],
                            evoked.info,
                            axes=axes[i, j],
                            show=False,
                            contours=0
                        )
                        if j == 0:
                            axes[i, j].set_ylabel(condition.title(), fontsize=12, fontweight='bold')
                        axes[i, j].set_title(f'{time_point:.2f} s', fontsize=11, fontweight='bold')
                    except Exception as e:
                        print(f"    Warning: Could not plot {condition} at {time_point}s in summary: {e}")
            
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(summary_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close(fig)
            topomap_files['summary'] = summary_path
            print(f"  ✓ Created manual summary plot: {os.path.basename(summary_path)}")
            
        except Exception as e:
            print(f"  ✗ Failed to create manual summary plot: {e}")
        
        print(f"Topomap analysis completed: {sum(len(files) for files in topomap_files.values())} files created")
        return topomap_files
        
    except Exception as e:
        print(f"  ✗ Topomap analysis failed: {e}")
        import traceback
        print(f"  Detailed error: {traceback.format_exc()}")
        return None

def create_focused_p300_topomaps(epochs_final, base_name, output_dir):
    """
    Create focused topomaps specifically for P300 analysis at key latencies
    """
    print(f"\nCreating focused P300 topomap analysis...")
    
    # Focus on typical P300 peak latencies
    peak_times = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    focused_files = []
    
    try:
        evokeds = {}
        for condition in ['target', 'non-target', 'distractor']:
            if condition in epochs_final.event_id:
                evokeds[condition] = epochs_final[condition].average()
        
        # Create focused difference plots for key comparisons using MNE's built-in plotting
        key_comparisons = [
            ('target', 'non-target', 'Target vs Non-target (P300 Effect)'),
            ('target', 'distractor', 'Target vs Distractor')
        ]
        
        for cond1, cond2, comp_name in key_comparisons:
            if cond1 in evokeds and cond2 in evokeds:
                try:
                    evoked_diff = mne.combine_evoked([evokeds[cond1], evokeds[cond2]], weights=[1, -1])
                    
                    focused_path = os.path.join(output_dir, f'{base_name}_topomaps_p300_peak_{cond1}_vs_{cond2}.png')
                    fig = evoked_diff.plot_topomap(times=peak_times, show=False)
                    fig.suptitle(f'P300 Difference - Peak Latencies\n{base_name}\n{comp_name}', 
                                fontsize=14, fontweight='bold')
                    fig.savefig(focused_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close(fig)
                    focused_files.append(focused_path)
                    print(f"  ✓ Created focused P300 topomap: {os.path.basename(focused_path)}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to create focused P300 topomap for {cond1} vs {cond2}: {e}")
        
        return focused_files
        
    except Exception as e:
        print(f"  ✗ Focused P300 topomap analysis failed: {e}")
        return None
    
def compare_high_vs_low(results_dict):
    """
    Compare ERPs between high and low accuracy runs
    """
    high_targets = []
    low_targets = []
    high_files = []
    low_files = []
    
    for file_path, result in results_dict.items():
        if result is None:
            continue
            
        if 'high' in file_path and 'target' in result['epochs']:
            high_targets.append(result['epochs']['target'].average())
            high_files.append(os.path.basename(file_path))
        elif 'low' in file_path and 'target' in result['epochs']:
            low_targets.append(result['epochs']['target'].average())
            low_files.append(os.path.basename(file_path))
    
    if high_targets and low_targets:
        print(f"\n{'='*50}")
        print("COMPARING HIGH vs LOW ACCURACY RUNS")
        print(f"{'='*50}")
        print(f"High accuracy files: {high_files}")
        print(f"Low accuracy files: {low_files}")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        mne.viz.plot_compare_evokeds(
            {'High Accuracy': high_targets, 'Low Accuracy': low_targets},
            title='High vs Low Accuracy Runs - Target ERPs',
            axes=ax,
            show=False
        )
        
        # Save comparison plot
        comparison_path = os.path.join('results', 'high_vs_low_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {comparison_path}")
        plt.show()
        
        return True
    else:
        print("Not enough data for high vs low comparison")
        return False

def generate_pipeline_name(dedrift_data, apply_notch_filter_flag, apply_laplacian_flag, 
                          bandpass_filter_type, bandpass_l_freq, bandpass_h_freq):
    """
    Generate a descriptive folder name based on the processing pipeline configuration
    """
    pipeline_parts = []
    
    # Dedrifting
    if dedrift_data:
        pipeline_parts.append("Dedrift")
    else:
        pipeline_parts.append("noDedrift")
    
    # Notch filtering
    if apply_notch_filter_flag:
        pipeline_parts.append("Notch")
    else:
        pipeline_parts.append("noNotch")
    
    # Laplacian
    if apply_laplacian_flag:
        pipeline_parts.append("Laplac")
    else:
        pipeline_parts.append("noLaplac")
    
    # Bandpass filter
    bandpass_name = f"{bandpass_filter_type}_{bandpass_l_freq}-{bandpass_h_freq}Hz"
    pipeline_parts.append(bandpass_name)
    
    pipeline_name = "_".join(pipeline_parts)
    return pipeline_name

def save_gfp_and_trial_data(epochs_final, base_name, output_dir, gfp_format='mat'):
    """
    Save GFP and trial data for each condition in specified format
    
    Parameters:
    epochs_final: MNE Epochs object after processing
    base_name: Base filename for saving
    output_dir: Directory to save files
    gfp_format: 'mat' or 'csv' for GFP data format
    """
    import scipy.io
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Process each condition
    conditions = ['target', 'non-target', 'distractor']
    
    for condition in conditions:
        if condition in epochs_final.event_id:
            # Get epochs for this condition
            condition_epochs = epochs_final[condition]
            
            if len(condition_epochs) > 0:
                # Get trial data: shape (n_trials, n_channels, n_times)
                trial_data = condition_epochs.get_data()
                
                # Calculate GFP for each trial: std across channels -> shape (n_trials, n_times)
                gfp_data = np.std(trial_data, axis=1)
                
                # Save trial data as .mat file (n_trials, points_per_epoch, channels)
                trial_filename = os.path.join(output_dir, f'{base_name}_{condition}_trials.mat')
                trial_data_reshaped = np.transpose(trial_data, (0, 2, 1))  # (trials, times, channels)
                
                scipy.io.savemat(trial_filename, {
                    'trial_data': trial_data_reshaped,
                    'times': condition_epochs.times,
                    'channels': condition_epochs.ch_names,
                    'condition': condition,
                    'n_trials': trial_data_reshaped.shape[0],
                    'n_timepoints': trial_data_reshaped.shape[1],
                    'n_channels': trial_data_reshaped.shape[2]
                })
                
                # Save GFP data in specified format
                if gfp_format.lower() == 'mat':
                    gfp_filename = os.path.join(output_dir, f'{base_name}_{condition}_GFP.mat')
                    scipy.io.savemat(gfp_filename, {
                        'gfp_data': gfp_data,
                        'times': condition_epochs.times,
                        'condition': condition,
                        'n_trials': gfp_data.shape[0],
                        'n_timepoints': gfp_data.shape[1],
                        'channels_used': condition_epochs.ch_names
                    })
                elif gfp_format.lower() == 'csv':
                    gfp_filename = os.path.join(output_dir, f'{base_name}_{condition}_GFP.csv')
                    # Create DataFrame with trial indices and time points
                    gfp_df = pd.DataFrame(gfp_data)
                    gfp_df.columns = [f'time_{i}' for i in range(gfp_data.shape[1])]
                    gfp_df['trial_index'] = np.arange(gfp_data.shape[0])
                    gfp_df['condition'] = condition
                    # Reorder columns to have metadata first
                    cols = ['trial_index', 'condition'] + [f'time_{i}' for i in range(gfp_data.shape[1])]
                    gfp_df = gfp_df[cols]
                    gfp_df.to_csv(gfp_filename, index=False)
                
                print(f"  ✓ Saved {condition}: {len(condition_epochs)} trials")
                print(f"    GFP shape: {gfp_data.shape} (saved as {gfp_format.upper()})")
                print(f"    Trial data shape: {trial_data_reshaped.shape} (saved as MAT)")
                
                saved_files[condition] = {
                    'gfp_file': gfp_filename,
                    'trial_file': trial_filename,
                    'gfp_shape': gfp_data.shape,
                    'trial_shape': trial_data_reshaped.shape,
                    'n_trials': len(condition_epochs),
                    'gfp_format': gfp_format
                }
            else:
                print(f"  ⚠️  No trials found for condition: {condition}")
        else:
            print(f"  ⚠️  Condition not found: {condition}")
    
    return saved_files

def analyze_single_file(mat_file_path, output_dir='results', plot_raw_check=False, 
                       check_channel='Fz', save_csv=False, save_mat=False, save_gfp_format='mat', target_epoch_index=0,
                       check_trial_quality_flag=True, trial_quality_tmin=-1.0, trial_quality_tmax=2.0,
                       check_all_conditions=False,
                       dedrift_data=False, dedrift_window_sec=1.0, dedrift_polyorder=3, 
                       plot_detrending=False, detrending_trial_index=0, detrending_trial_condition='target',
                       apply_notch_filter_flag=True, notch_freq=50., notch_bandwidth=2.0, 
                       notch_method='iir_simple', notch_filter_order=2, plot_notch_response=False,
                       apply_laplacian_flag=True, laplacian_method='large', plot_laplacian_comparison=False,
                       apply_bandpass_filter_flag=True, bandpass_l_freq=1.0, bandpass_h_freq=15.0,
                       bandpass_filter_type='butterworth', bandpass_filter_order=4, plot_bandpass_response=False,
                       bandpass_trial_index=0, bandpass_trial_condition='target', plot_single_trial_comparison=True,
                       topomap_times=None):
    """
    Analyze a single .mat file and check quality at each processing stage
    """
    # Start timing
    start_time = time.time()
    
    # Generate pipeline-specific folder name
    pipeline_name = generate_pipeline_name(
        dedrift_data, 
        apply_notch_filter_flag, 
        apply_laplacian_flag,
        bandpass_filter_type, 
        bandpass_l_freq, 
        bandpass_h_freq
    )
    
    # Create pipeline-specific output directory
    pipeline_output_dir = os.path.join(output_dir, pipeline_name)
    os.makedirs(pipeline_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING PIPELINE: {pipeline_name}")
    print(f"{'='*60}")
    
    # Get base filename for saving plots
    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {base_name}")
    print(f"{'='*60}")
    
    try:
        # =========================================================================
        # STAGE 1: Load raw data and check initial quality
        # =========================================================================
        stage1_start = time.time()
        print(f"\n{'='*50}")
        print("STAGE 1: RAW DATA QUALITY CHECK")
        print(f"{'='*50}")
        
        # Load data
        raw_original = load_vtp_data(mat_file_path)
        
        # Save as CSV if requested
        csv_paths_stage1 = None
        if save_csv:
            csv_paths_stage1 = save_data_as_csv(raw_original, mat_file_path, data_type='raw', csv_dir=pipeline_output_dir)
        
        # Plot raw EEG quality check if requested
        raw_quality_stats_stage1 = None
        if plot_raw_check:
            raw_check_path_stage1 = os.path.join(pipeline_output_dir, f'{base_name}_raw_quality_stage1_raw.png')
            raw_quality_stats_stage1 = plot_raw_eeg_check(raw_original, channel_name=check_channel, 
                                                         title=f"{base_name} - Stage 1: Raw", 
                                                         save_path=raw_check_path_stage1,
                                                         target_epoch_index=target_epoch_index)
        
        # Plot trigger distribution
        trigger_plot_path = os.path.join(pipeline_output_dir, f'{base_name}_triggers.png')
        events = plot_trigger_distribution(raw_original, base_name, trigger_plot_path)
        
        # Create epochs for initial trial quality check
        epochs_stage1 = create_epochs(raw_original)
        
        # Check trial quality on raw data and get outlier indices
        trial_quality_stats_stage1 = {}
        outlier_indices_stage1 = {}
        if check_trial_quality_flag and epochs_stage1 is not None:
            print("\nChecking trial quality on RAW data:")
            trial_quality_stats_stage1 = check_trial_quality_and_plot_outliers(
                raw_original, epochs_stage1, 
                channel_name=check_channel,
                condition='target' if not check_all_conditions else None,
                tmin=trial_quality_tmin, 
                tmax=trial_quality_tmax,
                save_dir=pipeline_output_dir,
                base_name=f'{base_name}_stage1_raw',
                check_all_conditions=check_all_conditions
            )
            
            # Extract outlier indices from the results
            for condition, stats_df in trial_quality_stats_stage1.items():
                if stats_df is not None:
                    outlier_indices = get_outlier_indices(stats_df)
                    outlier_indices_stage1[condition] = outlier_indices
                    print(f"  {condition}: {len(outlier_indices)} outlier trials")
        
        stage1_time = time.time() - stage1_start
        print(f"Stage 1 completed in {stage1_time:.1f} seconds")
        
        # =========================================================================
        # STAGE 2: Apply dedrifting and check quality after dedrifting
        # =========================================================================
        stage2_start = time.time()
        raw_dedrifted = raw_original.copy()
        trial_quality_stats_stage2 = {}
        raw_quality_stats_stage2 = None
        outlier_indices_stage2 = {}
        
        if dedrift_data:
            print(f"\n{'='*50}")
            print("STAGE 2: DEDRIFTING AND QUALITY CHECK")
            print(f"{'='*50}")
            
            # Choose which trial to visualize for detrending
            detrend_trial_to_plot = detrending_trial_index
            detrend_condition_to_plot = detrending_trial_condition
            
            # If we found outliers in stage 1, use the first outlier for visualization
            if (check_trial_quality_flag and 
                trial_quality_stats_stage1 and 
                detrending_trial_condition in outlier_indices_stage1 and 
                outlier_indices_stage1[detrending_trial_condition]):
                
                detrend_trial_to_plot = outlier_indices_stage1[detrending_trial_condition][0]
                print(f"Using outlier trial {detrend_trial_to_plot} ({detrend_condition_to_plot}) for detrending visualization")
            
            # Apply dedrifting with optional trial-specific visualization
            dedrift_save_path = os.path.join(pipeline_output_dir, f'{base_name}_detrending_trial_{detrend_trial_to_plot}.png')
            
            raw_dedrifted = dedrift_signal(
                raw_original, 
                channel_name=check_channel,
                window_length_sec=dedrift_window_sec,
                polyorder=dedrift_polyorder,
                plot_detrending=plot_detrending,
                title=f"{base_name} - Stage 2: Dedrifted",
                specific_trial=detrend_trial_to_plot,
                trial_condition=detrend_condition_to_plot,
                trial_tmin=trial_quality_tmin,
                trial_tmax=trial_quality_tmax,
                events=events,
                save_path=dedrift_save_path
            )
            
            # Create epochs for dedrifted data quality check
            epochs_stage2 = create_epochs(raw_dedrifted)
            
            # Check trial quality after dedrifting
            if check_trial_quality_flag and epochs_stage2 is not None:
                print("\nChecking trial quality after DEDRIFTING:")
                trial_quality_stats_stage2 = check_trial_quality_and_plot_outliers(
                    raw_dedrifted, epochs_stage2, 
                    channel_name=check_channel,
                    condition='target' if not check_all_conditions else None,
                    tmin=trial_quality_tmin, 
                    tmax=trial_quality_tmax,
                    save_dir=pipeline_output_dir,
                    base_name=f'{base_name}_stage2_dedrifted',
                    check_all_conditions=check_all_conditions
                )
                
                # Extract outlier indices from the results
                for condition, stats_df in trial_quality_stats_stage2.items():
                    if stats_df is not None:
                        outlier_indices = get_outlier_indices(stats_df)
                        outlier_indices_stage2[condition] = outlier_indices
                        print(f"  {condition}: {len(outlier_indices)} outlier trials")
                        
            stage2_time = time.time() - stage2_start
            print(f"Stage 2 completed in {stage2_time:.1f} seconds")
        else:
            # If no dedrifting, use original data for next stage
            raw_dedrifted = raw_original.copy()
            epochs_stage2 = epochs_stage1
            stage2_time = 0.0
            print("Stage 2 skipped (dedrifting disabled)")
        
        # =========================================================================
        # STAGE 3: Apply notch filter for line noise removal
        # =========================================================================
        stage3_start = time.time()
        print(f"\n{'='*50}")
        print("STAGE 3: NOTCH FILTER FOR LINE NOISE REMOVAL")
        print(f"{'='*50}")
        
        raw_notched = raw_dedrifted.copy()
        trial_quality_stats_stage3 = {}
        raw_quality_stats_stage3 = None
        outlier_indices_stage3 = {}
        
        if apply_notch_filter_flag:
            # Apply notch filter with the specified method
            notch_plot_path = os.path.join(pipeline_output_dir, f'{base_name}_notch_filter_verification.png') if plot_notch_response else None
            
            # Apply notch filter and get the figure
            raw_notched, fig_notch = apply_notch_filter_function(
                raw_dedrifted, 
                freq=notch_freq, 
                bandwidth=notch_bandwidth,
                method=notch_method,
                filter_order=notch_filter_order,
                plot_response=plot_notch_response
            )
            
            # Save notch filter verification plot if requested
            if plot_notch_response and notch_plot_path and fig_notch is not None:
                fig_notch.savefig(notch_plot_path, dpi=300, bbox_inches='tight')
                print(f"  Saved notch filter verification: {notch_plot_path}")
                plt.close(fig_notch)
            
            # Show the figure if plot_response was True
            if plot_notch_response and fig_notch is not None:
                plt.show()
            
            # Create epochs for notch filtered data quality check
            epochs_stage3 = create_epochs(raw_notched)
            
            # Check trial quality after notch filtering
            if check_trial_quality_flag and epochs_stage3 is not None:
                print("\nChecking trial quality after NOTCH FILTERING:")
                trial_quality_stats_stage3 = check_trial_quality_and_plot_outliers(
                    raw_notched, epochs_stage3, 
                    channel_name=check_channel,
                    condition='target' if not check_all_conditions else None,
                    tmin=trial_quality_tmin, 
                    tmax=trial_quality_tmax,
                    save_dir=pipeline_output_dir,
                    base_name=f'{base_name}_stage3_notch',
                    check_all_conditions=check_all_conditions
                )
                
                # Extract outlier indices from the results
                for condition, stats_df in trial_quality_stats_stage3.items():
                    if stats_df is not None:
                        outlier_indices = get_outlier_indices(stats_df)
                        outlier_indices_stage3[condition] = outlier_indices
                        print(f"  {condition}: {len(outlier_indices)} outlier trials")
                        
            stage3_time = time.time() - stage3_start
            print(f"Stage 3 completed in {stage3_time:.1f} seconds")
        else:
            # If no notch filtering, use dedrifted data for next stage
            raw_notched = raw_dedrifted.copy()
            epochs_stage3 = epochs_stage2
            stage3_time = 0.0
            print("Stage 3 skipped (notch filtering disabled)")
            
        # =========================================================================
        # STAGE 4: Apply Laplacian referencing
        # =========================================================================
        stage4_start = time.time()
        print(f"\n{'='*50}")
        print("STAGE 4: LAPLACIAN REFERENCING")
        print(f"{'='*50}")
        
        raw_laplacian = raw_notched.copy()
        trial_quality_stats_stage4 = {}
        raw_quality_stats_stage4 = None
        outlier_indices_stage4 = {}
        
        if apply_laplacian_flag:
            # Apply Laplacian referencing with custom neighbors for your electrode setup
            raw_laplacian = apply_laplacian_reference(raw_notched, method=laplacian_method)
            
            # Plot Laplacian comparison if requested
            if plot_laplacian_comparison:
                laplacian_plot_path = os.path.join(pipeline_output_dir, f'{base_name}_laplacian_comparison.png')
                fig_laplacian = plot_laplacian_comparison_function(
                    raw_notched, raw_laplacian, 
                    check_channel, 
                    title=f"{base_name} - Stage 4: Laplacian Referenced"
                )
                if fig_laplacian is not None:
                    fig_laplacian.savefig(laplacian_plot_path, dpi=300, bbox_inches='tight')
                    print(f"  Saved Laplacian comparison: {laplacian_plot_path}")
                    plt.close(fig_laplacian)
            
            # Create epochs for Laplacian data quality check
            epochs_stage4 = create_epochs(raw_laplacian)
            
            # Check trial quality after Laplacian referencing
            if check_trial_quality_flag and epochs_stage4 is not None:
                print("\nChecking trial quality after LAPLACIAN REFERENCING:")
                trial_quality_stats_stage4 = check_trial_quality_and_plot_outliers(
                    raw_laplacian, epochs_stage4, 
                    channel_name=check_channel,
                    condition='target' if not check_all_conditions else None,
                    tmin=trial_quality_tmin, 
                    tmax=trial_quality_tmax,
                    save_dir=pipeline_output_dir,
                    base_name=f'{base_name}_stage4_laplacian',
                    check_all_conditions=check_all_conditions
                )
                
                # Extract outlier indices from the results
                for condition, stats_df in trial_quality_stats_stage4.items():
                    if stats_df is not None:
                        outlier_indices = get_outlier_indices(stats_df)
                        outlier_indices_stage4[condition] = outlier_indices
                        print(f"  {condition}: {len(outlier_indices)} outlier trials")
                        
            stage4_time = time.time() - stage4_start
            print(f"Stage 4 completed in {stage4_time:.1f} seconds")
        else:
            # If no Laplacian referencing, use notch filtered data for next stage
            raw_laplacian = raw_notched.copy()
            epochs_stage4 = epochs_stage3
            stage4_time = 0.0
            print("Stage 4 skipped (Laplacian referencing disabled)")

        # =========================================================================
        # STAGE 5: Apply bandpass filter
        # =========================================================================
        stage5_start = time.time()
        print(f"\n{'='*50}")
        print("STAGE 5: BANDPASS FILTERING")
        print(f"{'='*50}")
        
        raw_filtered = raw_laplacian.copy()
        trial_quality_stats_stage5 = {}
        raw_quality_stats_stage5 = None
        outlier_indices_stage5 = {}
        
        if apply_bandpass_filter_flag:
            # Apply bandpass filter
            bandpass_plot_path = os.path.join(pipeline_output_dir, f'{base_name}_bandpass_filter_response.png') if plot_bandpass_response else None
            
            # Apply bandpass filter and get the figure
            raw_filtered, fig_bandpass_response = apply_bandpass_filter_function(
                raw_laplacian, 
                l_freq=bandpass_l_freq, 
                h_freq=bandpass_h_freq,
                filter_type=bandpass_filter_type,
                filter_order=bandpass_filter_order,
                plot_response=plot_bandpass_response
            )
            
            # Save bandpass filter response plot if requested
            if plot_bandpass_response and bandpass_plot_path and fig_bandpass_response is not None:
                fig_bandpass_response.savefig(bandpass_plot_path, dpi=300, bbox_inches='tight')
                print(f"  Saved bandpass filter response: {bandpass_plot_path}")
                plt.close(fig_bandpass_response)
            
            # Show the figure if plot_response was True
            if plot_bandpass_response and fig_bandpass_response is not None:
                plt.show()
            
            # Plot bandpass filter comparison
            bandpass_comparison_path = os.path.join(pipeline_output_dir, f'{base_name}_bandpass_filter_comparison.png')
            fig_bandpass = plot_filter_comparison(raw_laplacian, raw_filtered, check_channel, 
                                                 bandpass_filter_type, 
                                                 [bandpass_l_freq, bandpass_h_freq],
                                                 title=f"{base_name} - Stage 5: Bandpass Filtered")
            
            if fig_bandpass is not None:
                fig_bandpass.savefig(bandpass_comparison_path, dpi=300, bbox_inches='tight')
                print(f"  Saved bandpass filter comparison: {bandpass_comparison_path}")
                plt.close(fig_bandpass)
                
            # Plot single trial comparison if requested
            if plot_single_trial_comparison:
                # Create epochs for filtered data
                epochs_filtered = create_epochs(raw_filtered)
                
                # Plot single trial comparison
                single_trial_path = os.path.join(pipeline_output_dir, f'{base_name}_bandpass_single_trial_{bandpass_trial_index}.png')
                fig_single = plot_single_trial_filter_comparison(
                    raw_laplacian, raw_filtered, 
                    epochs_stage4, epochs_filtered,
                    trial_index=bandpass_trial_index,
                    condition=bandpass_trial_condition,
                    channel=check_channel,
                    tmin=trial_quality_tmin,
                    tmax=trial_quality_tmax,
                    title=f"{base_name} - Bandpass Filter",
                    save_path=single_trial_path
                )
                
                if fig_single is not None:
                    plt.close(fig_single)
                
                # Also plot multiple trials for broader view
                multiple_trials_path = os.path.join(pipeline_output_dir, f'{base_name}_bandpass_multiple_trials.png')
                fig_multiple = plot_multiple_trials_filter_comparison(
                    epochs_stage4, epochs_filtered,
                    trial_indices=[bandpass_trial_index, bandpass_trial_index+1, bandpass_trial_index+2],
                    condition=bandpass_trial_condition,
                    channel=check_channel,
                    title=f"{base_name} - Bandpass Filter",
                    save_path=multiple_trials_path
                )
                
                if fig_multiple is not None:
                    plt.close(fig_multiple)
                    
            stage5_time = time.time() - stage5_start
            print(f"Stage 5 completed in {stage5_time:.1f} seconds")
        else:
            stage5_time = 0.0
            print("Stage 5 skipped (bandpass filtering disabled)")
        
        # Plot raw EEG quality check after all filtering
        raw_quality_stats_stage5 = None
        if plot_raw_check:
            raw_check_path_stage5 = os.path.join(pipeline_output_dir, f'{base_name}_raw_quality_stage5_bandpass.png')
            raw_quality_stats_stage5 = plot_raw_eeg_check(raw_filtered, channel_name=check_channel, 
                                                         title=f"{base_name} - Stage 5: Bandpass Filtered", 
                                                         save_path=raw_check_path_stage5,
                                                         target_epoch_index=target_epoch_index)
        
        # Create epochs for final analysis
        epochs_final = create_epochs(raw_filtered)
        
        # Check trial quality after all filtering
        if check_trial_quality_flag and epochs_final is not None:
            print(f"\nChecking trial quality after ALL FILTERING:")
            trial_quality_stats_stage5 = check_trial_quality_and_plot_outliers(
                raw_filtered, epochs_final, 
                channel_name=check_channel,
                condition='target' if not check_all_conditions else None,
                tmin=trial_quality_tmin, 
                tmax=trial_quality_tmax,
                save_dir=pipeline_output_dir,
                base_name=f'{base_name}_stage5_bandpass',
                check_all_conditions=check_all_conditions
            )
            
            # Extract outlier indices from the results
            for condition, stats_df in trial_quality_stats_stage5.items():
                if stats_df is not None:
                    outlier_indices = get_outlier_indices(stats_df)
                    outlier_indices_stage5[condition] = outlier_indices
                    print(f"  {condition}: {len(outlier_indices)} outlier trials")
        
        
        # =========================================================================
        # STAGE 6: TOPOMAP ANALYSIS
        # =========================================================================
        stage6_start = time.time()
        print(f"\n{'='*50}")
        print("STAGE 6: TOPOMAP ANALYSIS")
        print(f"{'='*50}")
        
        topomap_results = None
        focused_topomaps = None
        
        if epochs_final is not None:
            # Set default topomap times if not provided
            if topomap_times is None:
                topomap_times = np.arange(0.2, 0.6, 0.05)  # Default P300 time window
            
            # Comprehensive topomap analysis
            topomap_results = create_comprehensive_topomaps(
                epochs_final, 
                base_name, 
                pipeline_output_dir,
                times=topomap_times
            )
            
            # Focused P300 analysis at peak latencies
            focused_topomaps = create_focused_p300_topomaps(epochs_final, base_name, pipeline_output_dir)
            
            if topomap_results:
                print(f"Topomap analysis summary:")
                for plot_type, files in topomap_results.items():
                    if files:
                        print(f"  {plot_type}: {len(files)} files")
            
            if focused_topomaps:
                print(f"  Focused P300: {len(focused_topomaps)} files")
                
            stage6_time = time.time() - stage6_start
            print(f"Stage 6 completed in {stage6_time:.1f} seconds")
        else:
            stage6_time = 0.0
            print("Stage 6 skipped (no epochs available)")

        # =========================================================================
        # STAGE 7: FINAL OUTPUTS AND DATA SAVING
        # =========================================================================
        stage7_start = time.time()
        print(f"\n{'='*50}")
        print("STAGE 7: FINAL OUTPUTS AND DATA SAVING")
        print(f"{'='*50}")
        
        # Save filtered data as CSV if requested
        csv_paths_stage7 = None
        if save_csv and raw_filtered is not None:
            csv_paths_stage7 = save_data_as_csv(raw_filtered, mat_file_path, data_type='filtered', csv_dir=pipeline_output_dir)
        
        # Save GFP and trial data as MAT files if requested
        gfp_trial_files = None
        if save_mat and epochs_final is not None:
            gfp_trial_files = save_gfp_and_trial_data(epochs_final, base_name, pipeline_output_dir, save_gfp_format)
        
        # Plot and save ERPs
        if epochs_final is not None:
            erp_plot_path = os.path.join(pipeline_output_dir, f'{base_name}_erps.png')
            plot_erps(epochs_final, base_name, erp_plot_path)
            
        stage7_time = time.time() - stage7_start
        print(f"Stage 7 completed in {stage7_time:.1f} seconds")
        
        # =========================================================================
        # COMPLETE PROCESSING - Return comprehensive results
        # =========================================================================
        total_time = time.time() - start_time
        
        print(f"\n✓ Completed processing: {base_name}")
        print(f"⏱️  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📁 Results saved to: {pipeline_output_dir}")
        
        # Print data saving summary
        if save_csv:
            print(f"💾 CSV data saved: Stage 1 (raw) and Stage 7 (filtered)")
        if save_mat:
            print(f"💾 MAT files saved: GFP and trial data ({save_gfp_format.upper()} format)")
        
        # Print stage-wise timing summary
        print(f"\nStage-wise timing breakdown:")
        print(f"  Stage 1 (Raw data): {stage1_time:.1f}s")
        if dedrift_data:
            print(f"  Stage 2 (Dedrifting): {stage2_time:.1f}s")
        if apply_notch_filter_flag:
            print(f"  Stage 3 (Notch filter): {stage3_time:.1f}s")
        if apply_laplacian_flag:
            print(f"  Stage 4 (Laplacian): {stage4_time:.1f}s")
        if apply_bandpass_filter_flag:
            print(f"  Stage 5 (Bandpass filter): {stage5_time:.1f}s")
        if epochs_final is not None:
            print(f"  Stage 6 (Topomaps): {stage6_time:.1f}s")
        print(f"  Stage 7 (Final outputs): {stage7_time:.1f}s")
            
        return {
            'file_path': mat_file_path,
            'pipeline_name': pipeline_name,
            'pipeline_output_dir': pipeline_output_dir,
            'raw_original': raw_original,
            'raw_dedrifted': raw_dedrifted if dedrift_data else None,
            'raw_notched': raw_notched if apply_notch_filter_flag else None,
            'raw_laplacian': raw_laplacian if apply_laplacian_flag else None,
            'raw_filtered': raw_filtered,
            'epochs_final': epochs_final,
            'events': events,
            'n_targets': len(epochs_final['target']) if epochs_final is not None and 'target' in epochs_final.event_id else 0,
            'n_nontargets': len(epochs_final['non-target']) if epochs_final is not None and 'non-target' in epochs_final.event_id else 0,
            'n_distractors': len(epochs_final['distractor']) if epochs_final is not None and 'distractor' in epochs_final.event_id else 0,
            'total_epochs': len(epochs_final) if epochs_final is not None else 0,
            'csv_paths': {
                'stage1_raw': csv_paths_stage1,
                'stage7_filtered': csv_paths_stage7
            },
            'gfp_trial_files': gfp_trial_files,
            'raw_quality_stats': {
                'stage1_raw': raw_quality_stats_stage1,
                'stage2_dedrifted': raw_quality_stats_stage2,
                'stage3_notch': raw_quality_stats_stage3,
                'stage4_laplacian': raw_quality_stats_stage4,
                'stage5_bandpass': raw_quality_stats_stage5
            },
            'trial_quality_stats': {
                'stage1_raw': trial_quality_stats_stage1,
                'stage2_dedrifted': trial_quality_stats_stage2,
                'stage3_notch': trial_quality_stats_stage3,
                'stage4_laplacian': trial_quality_stats_stage4,
                'stage5_bandpass': trial_quality_stats_stage5
            },
            'outlier_indices': {
                'stage1_raw': outlier_indices_stage1,
                'stage2_dedrifted': outlier_indices_stage2,
                'stage3_notch': outlier_indices_stage3,
                'stage4_laplacian': outlier_indices_stage4,
                'stage5_bandpass': outlier_indices_stage5
            },
            'topomap_results': topomap_results,
            'focused_topomaps': focused_topomaps,
            'processing_times': {
                'total': total_time,
                'stage1_raw': stage1_time,
                'stage2_dedrifted': stage2_time if dedrift_data else 0,
                'stage3_notch': stage3_time if apply_notch_filter_flag else 0,
                'stage4_laplacian': stage4_time if apply_laplacian_flag else 0,
                'stage5_bandpass': stage5_time if apply_bandpass_filter_flag else 0,
                'stage6_topomaps': stage6_time,
                'stage7_outputs': stage7_time
            },
            'processing_applied': {
                'dedrifted': dedrift_data,
                'dedrift_params': {
                    'window_sec': dedrift_window_sec,
                    'polyorder': dedrift_polyorder
                },
                'notch_filtered': apply_notch_filter_flag,
                'notch_params': {
                    'method': notch_method,
                    'freq': notch_freq,
                    'bandwidth': notch_bandwidth,
                    'order': notch_filter_order
                },
                'laplacian_referenced': apply_laplacian_flag,
                'laplacian_params': {
                    'method': laplacian_method
                },
                'bandpass_filtered': apply_bandpass_filter_flag,
                'bandpass_params': {
                    'filter_type': bandpass_filter_type,
                    'l_freq': bandpass_l_freq,
                    'h_freq': bandpass_h_freq,
                    'order': bandpass_filter_order,
                    'trial_index': bandpass_trial_index,
                    'trial_condition': bandpass_trial_condition,
                    'plot_single_trial': plot_single_trial_comparison
                },
                'trial_quality_checked': check_trial_quality_flag,
                'quality_params': {
                    'tmin': trial_quality_tmin,
                    'tmax': trial_quality_tmax,
                    'check_all_conditions': check_all_conditions
                },
                'data_saving': {
                    'save_csv': save_csv,
                    'save_mat': save_mat,
                    'gfp_format': save_gfp_format
                }
            }
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"ERROR processing {mat_file_path} after {total_time:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_summary_report(results_dict, save_to_file=True, output_dir='results'):
    """
    Generate a comprehensive summary report of all processed files with pipeline information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for the report filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"processing_summary_report_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    # Group results by pipeline
    pipelines = {}
    for file_path, result in results_dict.items():
        if result is not None and 'pipeline_name' in result:
            pipeline_name = result['pipeline_name']
            if pipeline_name not in pipelines:
                pipelines[pipeline_name] = []
            pipelines[pipeline_name].append(result)
    
    # If saving to file, create a file handle and write to both console and file
    if save_to_file:
        file_handle = open(report_path, 'w', encoding='utf-8')
        
        def print_both(*args, **kwargs):
            """Print to both console and file"""
            # Print to console
            print(*args, **kwargs)
            # Write to file
            print(*args, **kwargs, file=file_handle)
    else:
        # If not saving to file, just use regular print
        def print_both(*args, **kwargs):
            print(*args, **kwargs)
        file_handle = None
    
    try:
        print_both(f"\n{'='*80}")
        print_both("COMPREHENSIVE PROCESSING SUMMARY REPORT")
        print_both(f"{'='*80}")
        print_both(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print_both(f"Report file: {report_filename}")
        print_both(f"Number of pipelines: {len(pipelines)}")
        print_both(f"{'='*80}")
        
        # Process each pipeline
        for pipeline_name, pipeline_results in pipelines.items():
            print_both(f"\n{'─'*80}")
            print_both(f"PIPELINE: {pipeline_name}")
            print_both(f"{'─'*80}")
            
            total_files = len(pipeline_results)
            processed_files = sum(1 for result in pipeline_results if result is not None)
            
            print_both(f"\nOVERVIEW:")
            print_both(f"  Files processed: {processed_files}/{total_files}")
            
            if processed_files == 0:
                print_both("No files were successfully processed in this pipeline.")
                continue
            
            # Calculate overall statistics for this pipeline
            total_targets = 0
            total_nontargets = 0
            total_distractors = 0
            total_epochs = 0
            total_processing_time = 0
            individual_times = []
            
            for result in pipeline_results:
                if result is not None:
                    total_targets += result.get('n_targets', 0)
                    total_nontargets += result.get('n_nontargets', 0)
                    total_distractors += result.get('n_distractors', 0)
                    total_epochs += result.get('total_epochs', 0)
                    
                    # Collect timing information
                    processing_times = result.get('processing_times', {})
                    total_time = processing_times.get('total', 0)
                    total_processing_time += total_time
                    individual_times.append(total_time)
            
            # Calculate timing statistics
            avg_time = total_processing_time / processed_files if processed_files > 0 else 0
            min_time = min(individual_times) if individual_times else 0
            max_time = max(individual_times) if individual_times else 0
            
            print_both(f"  Total epochs across all files:")
            print_both(f"    Targets: {total_targets}")
            print_both(f"    Non-targets: {total_nontargets}")
            print_both(f"    Distractors: {total_distractors}")
            print_both(f"    Total: {total_epochs}")
            
            print_both(f"\n  PROCESSING TIME STATISTICS:")
            print_both(f"    Total time: {total_processing_time:.1f} seconds ({total_processing_time/60:.1f} minutes)")
            print_both(f"    Average per file: {avg_time:.1f} seconds")
            print_both(f"    Range: {min_time:.1f} - {max_time:.1f} seconds")
            
            # Process each file in this pipeline
            for result in pipeline_results:
                if result is None:
                    continue
                    
                file_name = os.path.basename(result['file_path'])
                
                # Extract subject and accuracy type from filename
                subject = "Unknown"
                accuracy_type = "Unknown"
                
                if 'P1' in file_name:
                    subject = "P1"
                elif 'P2' in file_name:
                    subject = "P2"
                    
                if 'high' in file_name.lower():
                    accuracy_type = "HIGH"
                elif 'low' in file_name.lower():
                    accuracy_type = "LOW"
                
                print_both(f"\n  FILE: {file_name}")
                print_both(f"    Subject: {subject}")
                print_both(f"    Accuracy Type: {accuracy_type}")
                print_both(f"    Processing Time: {result.get('processing_times', {}).get('total', 0):.1f} seconds")
                print_both(f"    Target epochs: {result.get('n_targets', 0)}")
                print_both(f"    Output directory: {result.get('pipeline_output_dir', 'Unknown')}")
        
        print_both(f"\n{'='*80}")
        print_both("SUMMARY COMPLETE")
        print_both(f"{'='*80}")
        
        # Also save a JSON version for programmatic access
        if save_to_file:
            json_filename = f"processing_summary_report_{timestamp}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            json_data = {
                'metadata': {
                    'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_pipelines': len(pipelines)
                },
                'pipelines': {}
            }
            
            # Add pipeline details
            for pipeline_name, pipeline_results in pipelines.items():
                json_data['pipelines'][pipeline_name] = {
                    'file_count': len(pipeline_results),
                    'files': {}
                }
                
                # Add individual file details
                for result in pipeline_results:
                    if result is not None:
                        file_name = os.path.basename(result['file_path'])
                        json_data['pipelines'][pipeline_name]['files'][file_name] = {
                            'subject': 'P1' if 'P1' in file_name else 'P2' if 'P2' in file_name else 'Unknown',
                            'accuracy_type': 'HIGH' if 'high' in file_name.lower() else 'LOW' if 'low' in file_name.lower() else 'Unknown',
                            'epoch_counts': {
                                'target': result.get('n_targets', 0),
                                'non_target': result.get('n_nontargets', 0),
                                'distractor': result.get('n_distractors', 0),
                                'total': result.get('total_epochs', 0)
                            },
                            'processing_times': result.get('processing_times', {}),
                            'output_directory': result.get('pipeline_output_dir', 'Unknown')
                        }
            
            # Save JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print_both(f"\n📊 Report also saved as JSON: {json_filename}")
        
        print_both(f"\n💾 Summary report saved to: {report_path}")
        
        return {
            'total_pipelines': len(pipelines),
            'report_files': {
                'text_report': report_path if save_to_file else None,
                'json_report': json_path if save_to_file else None
            }
        }
        
    finally:
        # Always close the file handle if it was opened
        if file_handle:
            file_handle.close()


def main(plot_raw_check=True, check_channel='Fz', save_csv=False, save_mat=False, save_gfp_format='mat', target_epoch_index=0,
         check_trial_quality=True, trial_quality_tmin=-1.0, trial_quality_tmax=2.0,
         check_all_conditions=False,
         dedrift_data=True, dedrift_window_sec=1.0, dedrift_polyorder=3, 
         plot_detrending=True, detrending_trial_index=0, detrending_trial_condition='target',
         apply_notch_filter=True, notch_freq=50.0, notch_bandwidth=2.0, 
         notch_method='iir_simple', notch_filter_order=2, plot_notch_response=False,
         apply_laplacian_flag=True, laplacian_method='large', plot_laplacian_comparison=False,
         apply_bandpass_filter=True, bandpass_l_freq=1.0, bandpass_h_freq=15.0,
         bandpass_filter_type='butterworth', bandpass_filter_order=4, plot_bandpass_response=False,
         bandpass_trial_index=0, bandpass_trial_condition='target', plot_single_trial_comparison=True,
         topomap_times=None, save_summary_report=True):
    """
    Main function to process all .mat files with comprehensive 7-stage processing pipeline
    """
    # Start overall timing
    overall_start = time.time()
    
    # Find all .mat files in current directory
    mat_files = glob.glob("*.mat")
    
    if not mat_files:
        print("No .mat files found in current directory!")
        print("Please make sure your .mat files are in the same directory as this script.")
        return
    
    print(f"Found {len(mat_files)} .mat files:")
    for f in mat_files:
        print(f"  - {f}")
    
    # Generate pipeline name for this configuration
    pipeline_name = generate_pipeline_name(
        dedrift_data, 
        apply_notch_filter, 
        apply_laplacian_flag,
        bandpass_filter_type, 
        bandpass_l_freq, 
        bandpass_h_freq
    )
    
    # Print processing configuration
    print(f"\n{'='*60}")
    print(f"PROCESSING PIPELINE: {pipeline_name}")
    print(f"{'='*60}")
    
    # Print data saving configuration
    print(f"\nDATA SAVING CONFIGURATION:")
    if save_csv:
        print(f"✓ CSV export enabled - raw and filtered data will be saved")
    else:
        print(f"✗ CSV export disabled")
    
    if save_mat:
        print(f"✓ MAT export enabled - GFP and trial data will be saved")
        print(f"  GFP format: {save_gfp_format.upper()}")
    else:
        print(f"✗ MAT export disabled")
    
    if plot_raw_check:
        print(f"✓ Raw EEG quality check enabled for channel: {check_channel}")
        print(f"  Target epoch index for detailed view: {target_epoch_index}")
    else:
        print("✗ Raw EEG quality check disabled")
    
    if check_trial_quality:
        print(f"✓ Trial quality check enabled")
        print(f"  Analysis window: {trial_quality_tmin} to {trial_quality_tmax} seconds")
        if check_all_conditions:
            print("  Checking ALL conditions (target, non-target, distractor)")
        else:
            print("  Checking target condition only")
    else:
        print("✗ Trial quality check disabled")
    
    if dedrift_data:
        print(f"✓ Dedrifting enabled:")
        print(f"  Window length: {dedrift_window_sec} seconds")
        print(f"  Polynomial order: {dedrift_polyorder}")
        if plot_detrending:
            print(f"  Detrending plots enabled for trial {detrending_trial_index} ({detrending_trial_condition})")
    else:
        print("✗ Dedrifting disabled")
    
    if apply_notch_filter:
        print(f"✓ Notch filter enabled:")
        print(f"  Frequency: {notch_freq} Hz")
        print(f"  Bandwidth: {notch_bandwidth} Hz")
        print(f"  Method: {notch_method}")
        print(f"  Filter order: {notch_filter_order}")
        if plot_notch_response:
            print("  Notch filter verification plots enabled")
    else:
        print("✗ Notch filter disabled")
    
    if apply_laplacian_flag:
        print(f"✓ Laplacian referencing enabled:")
        print(f"  Method: {laplacian_method}")
        if plot_laplacian_comparison:
            print("  Laplacian comparison plots enabled")
    else:
        print("✗ Laplacian referencing disabled")
    
    if apply_bandpass_filter:
        print(f"✓ Bandpass filter enabled:")
        print(f"  Frequency range: {bandpass_l_freq}-{bandpass_h_freq} Hz")
        print(f"  Filter type: {bandpass_filter_type}")
        print(f"  Filter order: {bandpass_filter_order}")
        if plot_bandpass_response:
            print("  Bandpass filter response plots enabled")
    else:
        print("✗ Bandpass filter disabled")
    
    print(f"\nProcessing pipeline:")
    stages = [
        "1. Raw data quality check",
        "2. Dedrifting (Savitzky-Golay)" if dedrift_data else "2. Dedrifting (skipped)",
        "3. Notch filtering" if apply_notch_filter else "3. Notch filtering (skipped)",
        "4. Laplacian referencing" if apply_laplacian_flag else "4. Laplacian referencing (skipped)",
        "5. Bandpass filtering" if apply_bandpass_filter else "5. Bandpass filtering (skipped)",
        "6. Topomap analysis",
        "7. Final outputs and data saving"
    ]
    
    for stage in stages:
        print(f"  {stage}")
    
    print(f"{'='*60}")
    
    # Process each file
    results = {}
    for mat_file in mat_files:
        result = analyze_single_file(
            mat_file, 
            plot_raw_check=plot_raw_check, 
            check_channel=check_channel,
            save_csv=save_csv,
            save_mat=save_mat,
            save_gfp_format=save_gfp_format,
            target_epoch_index=target_epoch_index,
            check_trial_quality_flag=check_trial_quality,
            trial_quality_tmin=trial_quality_tmin,
            trial_quality_tmax=trial_quality_tmax,
            check_all_conditions=check_all_conditions,
            dedrift_data=dedrift_data,
            dedrift_window_sec=dedrift_window_sec,
            dedrift_polyorder=dedrift_polyorder,
            plot_detrending=plot_detrending,
            detrending_trial_index=detrending_trial_index,
            detrending_trial_condition=detrending_trial_condition,
            apply_notch_filter_flag=apply_notch_filter,
            notch_freq=notch_freq,
            notch_bandwidth=notch_bandwidth,
            notch_method=notch_method,
            notch_filter_order=notch_filter_order,
            plot_notch_response=plot_notch_response,
            apply_laplacian_flag=apply_laplacian_flag,
            laplacian_method=laplacian_method,
            plot_laplacian_comparison=plot_laplacian_comparison,
            apply_bandpass_filter_flag=apply_bandpass_filter,
            bandpass_l_freq=bandpass_l_freq,
            bandpass_h_freq=bandpass_h_freq,
            bandpass_filter_type=bandpass_filter_type,
            bandpass_filter_order=bandpass_filter_order,
            plot_bandpass_response=plot_bandpass_response,
            bandpass_trial_index=bandpass_trial_index,
            bandpass_trial_condition=bandpass_trial_condition,
            plot_single_trial_comparison=plot_single_trial_comparison,
            topomap_times=topomap_times
        )
        results[mat_file] = result
    
    # Generate comprehensive summary report
    summary_stats = generate_summary_report(results, 
                                          save_to_file=save_summary_report, 
                                          output_dir='results')
    
    # Calculate overall processing time
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    
    print(f"\n⏱️  OVERALL PROCESSING TIME:")
    print(f"  Total time: {overall_time:.1f} seconds ({overall_time/60:.1f} minutes)")
    print(f"  Average per file: {overall_time/len(mat_files):.1f} seconds")
    
    # Print data saving summary
    if save_csv or save_mat:
        print(f"\n💾 DATA SAVING SUMMARY:")
        if save_csv:
            print(f"  ✓ CSV files: Raw and filtered EEG data")
        if save_mat:
            print(f"  ✓ MAT files: GFP and trial data ({save_gfp_format.upper()} format)")
            print(f"    - GFP shape: (n_trials, points_per_epoch)")
            print(f"    - Trial data shape: (n_trials, points_per_epoch, channels)")
    
    # Print report file locations if saved
    if save_summary_report and summary_stats and 'report_files' in summary_stats:
        report_files = summary_stats['report_files']
        if report_files.get('text_report'):
            print(f"\n📄 Summary report saved to: {report_files['text_report']}")
        if report_files.get('json_report'):
            print(f"📊 JSON report saved to: {report_files['json_report']}")
    
    print(f"📁 All results saved to pipeline folder: {pipeline_name}")
    print(f"{'='*60}")
    
    return results, summary_stats


# Run the main function when the script is executed
if __name__ == "__main__":
    # =========================================================================
    # PROCESSING PARAMETERS - ADJUST THESE AS NEEDED
    # =========================================================================
    
    # Data saving parameters
    SAVE_CSV = True                    # Set to True to export raw and filtered data as CSV files
    SAVE_MAT = True                    # Set to True to export GFP and trial data as MAT files
    SAVE_GFP_FORMAT = 'mat'           # 'mat' or 'csv' for GFP data format
    
    # Basic analysis parameters
    PLOT_RAW_CHECK = True              # Set to False to skip raw EEG quality plots
    CHECK_CHANNEL = 'Fz'               # Change to any other channel name if needed
    TARGET_EPOCH_INDEX = 0             # Which target epoch to show (0 = first target, 1 = second target, etc.)
    
    # Trial quality checking parameters
    CHECK_TRIAL_QUALITY = True         # Check trial quality before filtering
    TRIAL_QUALITY_TMIN = -1.0          # Start time for trial quality analysis
    TRIAL_QUALITY_TMAX = 2.0           # End time for trial quality analysis
    CHECK_ALL_CONDITIONS = True        # Check all conditions (target, non-target, distractor)
    
    # Dedrifting parameters
    DEDRIFT_DATA = True                # Apply dedrifting using Savitzky-Golay filter
    DEDRIFT_WINDOW_SEC = 1.0           # Window length in seconds (recommended: 0.5-2 seconds)
    DEDRIFT_POLYORDER = 3              # Polynomial order (recommended: 2-3)
    PLOT_DETRENDING = True             # Plot detrending results
    DETRENDING_TRIAL_INDEX = 0         # Which trial to visualize for detrending
    DETRENDING_TRIAL_CONDITION = 'target'  # Condition of trial to visualize for detrending
    
    # Notch filter parameters 
    APPLY_NOTCH_FILTER = False         # Apply notch filter for line noise removal
    NOTCH_FREQ = 50.0                  # 50 Hz for Europe, 60.0 for US
    NOTCH_BANDWIDTH = 2.0              # Width of the notch in Hz
    NOTCH_METHOD = 'iir_simple'        # Options: 'iir_simple', 'iir', 'fir', 'frequency_domain'
    NOTCH_FILTER_ORDER = 2             # Order of the notch filter (keep low for stability)
    PLOT_NOTCH_RESPONSE = True         # Plot verification of notch filter
    
    # Laplacian parameters
    APPLY_LAPLACIAN_REFERENCE = False  # Apply Laplacian referencing
    LAPLACIAN_METHOD = 'large'         # 'large' for large Laplacian
    PLOT_LAPLACIAN_COMPARISON = True   # Plot Laplacian comparison
        
    # Bandpass filter parameters
    APPLY_BANDPASS_FILTER = True       # Apply bandpass filter
    BANDPASS_L_FREQ = 0.3              # Low frequency cutoff (Hz) - typical for P300: 0.1-1.0 Hz
    BANDPASS_H_FREQ = 30.0             # High frequency cutoff (Hz) - typical for P300: 12-20 Hz
    BANDPASS_FILTER_TYPE = 'butterworth'  # 'butterworth', 'fir', or 'iir'
    BANDPASS_FILTER_ORDER = 4          # Filter order (higher = steeper roll-off)
    PLOT_BANDPASS_RESPONSE = True      # Plot frequency response of bandpass filter
    
    # Bandpass filter visualization parameters
    BANDPASS_TRIAL_INDEX = 0           # Which trial to show for single trial comparison
    BANDPASS_TRIAL_CONDITION = 'target' # Which condition to show for single trial comparison
    PLOT_SINGLE_TRIAL_COMPARISON = True # Whether to plot single trial comparison
    
    # Topomaps times
    TOPOMAPS_TIMES = None
    
    # Reporting parameters
    SAVE_SUMMARY_REPORT = True         # Set to True to save summary report to file
    
    
    # =========================================================================
    # EXECUTE MAIN PROCESSING
    # =========================================================================
    # plot_raw_check=PLOT_RAW_CHECK;check_channel=CHECK_CHANNEL;save_csv=SAVE_CSV;target_epoch_index=TARGET_EPOCH_INDEX
    # check_trial_quality=CHECK_TRIAL_QUALITY;trial_quality_tmin=TRIAL_QUALITY_TMIN;trial_quality_tmax=TRIAL_QUALITY_TMAX
    # check_all_conditions=CHECK_ALL_CONDITIONS;dedrift_data=DEDRIFT_DATA;
    # dedrift_window_sec=DEDRIFT_WINDOW_SEC;dedrift_polyorder=DEDRIFT_POLYORDER;plot_detrending=PLOT_DETRENDING
    # detrending_trial_index=DETRENDING_TRIAL_INDEX;detrending_trial_condition=DETRENDING_TRIAL_CONDITION
    # apply_notch_filter=APPLY_NOTCH_FILTER;notch_freq=NOTCH_FREQ;notch_bandwidth=NOTCH_BANDWIDTH;
    # notch_method=NOTCH_METHOD;notch_filter_order=NOTCH_FILTER_ORDER;plot_notch_response=PLOT_NOTCH_RESPONSE
    # apply_laplacian_flag=APPLY_LAPLACIAN_REFERENCE;laplacian_method=LAPLACIAN_METHOD;
    # apply_bandpass_filter=APPLY_BANDPASS_FILTER;bandpass_l_freq=BANDPASS_L_FREQ;bandpass_h_freq=BANDPASS_H_FREQ
    # bandpass_filter_type=BANDPASS_FILTER_TYPE;bandpass_filter_order=BANDPASS_FILTER_ORDER;plot_bandpass_response=PLOT_BANDPASS_RESPONSE
    # bandpass_trial_index=BANDPASS_TRIAL_INDEX;bandpass_trial_condition=BANDPASS_TRIAL_CONDITION;plot_single_trial_comparison=PLOT_SINGLE_TRIAL_COMPARISON
    # topomap_times=TOPOMAPS_TIMES;save_summary_report=SAVE_SUMMARY_REPORT
    
    main(plot_raw_check=PLOT_RAW_CHECK, 
         check_channel=CHECK_CHANNEL,
         save_csv=SAVE_CSV,
         save_mat=SAVE_MAT,
         save_gfp_format=SAVE_GFP_FORMAT,
         target_epoch_index=TARGET_EPOCH_INDEX,
         check_trial_quality=CHECK_TRIAL_QUALITY,
         trial_quality_tmin=TRIAL_QUALITY_TMIN,
         trial_quality_tmax=TRIAL_QUALITY_TMAX,
         check_all_conditions=CHECK_ALL_CONDITIONS,
         dedrift_data=DEDRIFT_DATA,
         dedrift_window_sec=DEDRIFT_WINDOW_SEC,
         dedrift_polyorder=DEDRIFT_POLYORDER,
         plot_detrending=PLOT_DETRENDING,
         detrending_trial_index=DETRENDING_TRIAL_INDEX,
         detrending_trial_condition=DETRENDING_TRIAL_CONDITION,
         apply_notch_filter=APPLY_NOTCH_FILTER,
         notch_freq=NOTCH_FREQ,
         notch_bandwidth=NOTCH_BANDWIDTH,
         notch_method=NOTCH_METHOD,
         notch_filter_order=NOTCH_FILTER_ORDER,
         plot_notch_response=PLOT_NOTCH_RESPONSE,
         apply_laplacian_flag=APPLY_LAPLACIAN_REFERENCE,
         laplacian_method=LAPLACIAN_METHOD,
         plot_laplacian_comparison=PLOT_LAPLACIAN_COMPARISON,
         apply_bandpass_filter=APPLY_BANDPASS_FILTER,
         bandpass_l_freq=BANDPASS_L_FREQ,
         bandpass_h_freq=BANDPASS_H_FREQ,
         bandpass_filter_type=BANDPASS_FILTER_TYPE,
         bandpass_filter_order=BANDPASS_FILTER_ORDER,
         plot_bandpass_response=PLOT_BANDPASS_RESPONSE,
         topomap_times=TOPOMAPS_TIMES,
         save_summary_report=SAVE_SUMMARY_REPORT)
    
