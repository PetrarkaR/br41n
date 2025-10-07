import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.io import loadmat, savemat
import os
from datetime import datetime

"""
EEG Augmentation - Preserves exact structure, only modifies signal data
"""

class CompleteEEGAugmenter:
    def __init__(self, fs=250, output_dir='augmented_data', input_dir='input_data'):
        """
        class.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        output_dir : str
            Directory to save augmented data
        """
        self.fs = fs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.params = {
            # Time shifting
            'time_shifts_all': [-10, -5, 5, 10],  # ms for all-shift
            'time_shifts_peak': [-15, -10, 10, 15],  # ms for peak-shift
            'peak_range': (0.25, 0.65),  # seconds (for P300/ERP)
            
            # Amplitude scaling
            'amp_scales_all': [0.95, 1.05],  # ±5% for all-amp
            'amp_scales_peak': [0.90, 1.10],  # ±10% for peak-amp
            
            # Noise and artifacts
            'noise_std': 0.01,
            'jitter_std': 0.05,
        }

    def all_shift(self, X, shifts_ms=None):
        """All-Shift: Shift entire signal in time."""
        if shifts_ms is None:
            shifts_ms = self.params['time_shifts_all']
        
        X_aug = []
        for shift_ms in shifts_ms:
            shift_samples = int(shift_ms * self.fs / 1000)
            X_aug.append(np.roll(X, shift_samples, axis=0))
        return np.array(X_aug)
    
    def all_amp(self, X, scales=None):
        """All-Amp: Scale entire signal amplitude."""
        if scales is None:
            scales = self.params['amp_scales_all']
        return np.array([X * scale for scale in scales])
    
    def peak_shift(self, X, shifts_ms=None, peak_range=None):
        """Peak-Shift: Shift only peak region (250-650ms for ERP)."""
        if shifts_ms is None:
            shifts_ms = self.params['time_shifts_peak']
        if peak_range is None:
            peak_range = self.params['peak_range']
        
        start_idx = int(peak_range[0] * self.fs)
        end_idx = int(peak_range[1] * self.fs)
        
        X_aug = []
        for shift_ms in shifts_ms:
            shift_samples = int(shift_ms * self.fs / 1000)
            X_shifted = X.copy()
            if start_idx < X.shape[0] and end_idx <= X.shape[0]:
                peak = X[start_idx:end_idx, :]
                X_shifted[start_idx:end_idx, :] = np.roll(peak, shift_samples, axis=0)
            X_aug.append(X_shifted)
        return np.array(X_aug)
    
    def peak_amp(self, X, scales=None, peak_range=None):
        """Peak-Amp: Scale only peak region amplitude."""
        if scales is None:
            scales = self.params['amp_scales_peak']
        if peak_range is None:
            peak_range = self.params['peak_range']
        
        start_idx = int(peak_range[0] * self.fs)
        end_idx = int(peak_range[1] * self.fs)
        
        X_aug = []
        for scale in scales:
            X_scaled = X.copy()
            if start_idx < X.shape[0] and end_idx <= X.shape[0]:
                X_scaled[start_idx:end_idx, :] *= scale
            X_aug.append(X_scaled)
        return np.array(X_aug)

    def add_gaussian_noise(self, X, noise_std=None):
        """Add Gaussian noise: simulates sensor noise."""
        if noise_std is None:
            noise_std = self.params['noise_std']
        noise = np.random.normal(0, noise_std, X.shape)
        return X + noise
    
    def amplitude_jitter(self, X, jitter_std=None):
        """Amplitude jitter: simulates electrode contact variations."""
        if jitter_std is None:
            jitter_std = self.params['jitter_std']
        jitter = np.random.normal(1.0, jitter_std, X.shape)
        return X * jitter

    def check_quality(self, X_orig, X_aug):
        """Check augmentation quality."""
        if X_orig.shape != X_aug.shape:
            return {'status': 'shapes_differ', 'valid': False, 'acceptable': False}
        
        correlations = []
        for ch in range(X_orig.shape[1]):
            corr = np.corrcoef(X_orig[:, ch], X_aug[:, ch])[0, 1]
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        noise = X_aug - X_orig
        snr_db = 10 * np.log10(np.var(X_orig) / (np.var(noise) + 1e-10))
        rmse = np.sqrt(np.mean((X_orig - X_aug)**2))
        
        fft_orig = np.abs(np.fft.rfft(X_orig[:, 0]))
        fft_aug = np.abs(np.fft.rfft(X_aug[:, 0]))
        freq_corr = np.corrcoef(fft_orig, fft_aug)[0, 1]
        
        return {
            'status': 'valid',
            'valid': True,
            'correlation': mean_corr,
            'snr_db': snr_db,
            'rmse': rmse,
            'freq_correlation': freq_corr,
            'acceptable': mean_corr > 0.7 and snr_db > 10
        }

    def augment_dataset_by_method(self, X, y, methods='paper', quality_check=True, 
                                   save=True, original_struct_name='data', base_filename='augmented'):
        """
        Augment dataset - PRESERVES ALL STRUCTURE, only modifies signal data.
        """
        print(f"Original dataset: {len(X)} trials, {X.shape[1]} samples, {X.shape[2]} channels")
        print(f"Labels shape: {y.shape}")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"Methods: {methods}")
        print(f"{'='*70}\n")
        
        if methods == 'paper':
            method_list = ['all_shift', 'all_amp', 'peak_shift', 'peak_amp']
        elif methods == 'all':
            method_list = ['all_shift', 'all_amp', 'peak_shift', 'peak_amp', 'noise', 'jitter']
        else:
            method_list = methods if isinstance(methods, list) else [methods]
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for method_idx, method in enumerate(method_list):
            print(f"\n{'='*70}")
            print(f"Processing method {method_idx+1}/{len(method_list)}: {method}")
            print(f"{'='*70}")
            
            X_aug_list = []
            y_aug_list = []
            quality_scores = []
            
            for i, (trial, label) in enumerate(zip(X, y)):
                X_aug_list.append(trial)
                y_aug_list.append(label)
                
                if method == 'all_shift':
                    augs = self.all_shift(trial)
                elif method == 'all_amp':
                    augs = self.all_amp(trial)
                elif method == 'peak_shift':
                    augs = self.peak_shift(trial)
                elif method == 'peak_amp':
                    augs = self.peak_amp(trial)
                elif method == 'noise':
                    augs = [self.add_gaussian_noise(trial)]
                elif method == 'jitter':
                    augs = [self.amplitude_jitter(trial)]
                else:
                    continue
                
                for aug in augs:
                    if quality_check and i < 3:
                        quality = self.check_quality(trial, aug)
                        quality_scores.append(quality)
                        if not quality['acceptable']:
                            print(f"  Warning: Low quality for trial {i}")
                            continue
                    X_aug_list.append(aug)
                    y_aug_list.append(label)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(X)} trials...")
            
            X_aug = np.array(X_aug_list)
            y_aug = np.array(y_aug_list)
            
            # Preserve original label shape
            if len(y.shape) == 2 and y.shape[1] == 1:
                y_aug = y_aug.reshape(-1, 1)
            
            unique_classes = np.unique(y)
            final_class_dist = {int(cls): int(np.sum(y_aug.flatten() == cls)) for cls in unique_classes}
            
            report = {
                'method': method,
                'original_size': len(X),
                'augmented_size': len(X_aug),
                'expansion_factor': len(X_aug) / len(X),
                'augmented_class_distribution': final_class_dist,
                'quality_scores': quality_scores if quality_check else None,
                'timestamp': timestamp
            }
            
            print(f"\nMethod '{method}' complete:")
            print(f"  Generated: {len(X_aug)} trials ({report['expansion_factor']:.2f}x expansion)")
            print(f"  Class distribution: {final_class_dist}")
            
            if quality_check and quality_scores:
                valid_scores = [q for q in quality_scores if q['status'] == 'valid']
                if valid_scores:
                    mean_corr = np.mean([q['correlation'] for q in valid_scores])
                    mean_snr = np.mean([q['snr_db'] for q in valid_scores])
                    print(f"  Quality: corr={mean_corr:.3f}, SNR={mean_snr:.2f} dB")
            
            if save:
                filename = os.path.join(self.output_dir, f'{base_filename}_{method}_{timestamp}.mat')
                
                data_struct = np.empty((1,), dtype=[('X', 'O'), ('y', 'O'), ('fs', 'O')])
                data_struct['X'][0] = X_aug
                data_struct['y'][0] = y_aug
                data_struct['fs'][0] = np.array([[self.fs]])
                
                savemat(filename, {original_struct_name: data_struct})
                print(f"  Saved → {filename}")
            
            results[method] = {'X': X_aug, 'y': y_aug, 'report': report}
        
        print(f"\n{'='*70}")
        print(f"ALL METHODS COMPLETE")
        print(f"{'='*70}")
        print(f"Generated {len(method_list)} augmented files")
        
        return results


if __name__ == "__main__":
    import glob
    
    input_folder = "sophie"
    mat_files = glob.glob(os.path.join(input_folder, "*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {input_folder}/")
        exit(1)
    
    print(f"Found {len(mat_files)} .mat files to process")
    print(f"{'='*70}\n")
    
    def unwrap(v):
        if np.isscalar(v):
            return v
        if isinstance(v, np.ndarray) and v.dtype == object:
            if v.size == 1:
                return unwrap(v.reshape(-1)[0])
            else:
                return [unwrap(x) for x in v.reshape(-1)]
        if isinstance(v, np.void) or (isinstance(v, np.ndarray) and v.dtype.names is not None):
            return v
        return v
    
    def get_field(ds, name):
        if getattr(ds, 'dtype', None) and ds.dtype.names and name in ds.dtype.names:
            return ds[name]
        lname = name.lower()
        for fn in ds.dtype.names:
            if fn.lower() == lname:
                return ds[fn]
        return None
    
    for file_idx, mat_file in enumerate(mat_files):
        print(f"\n{'#'*70}")
        print(f"Processing file {file_idx+1}/{len(mat_files)}: {os.path.basename(mat_file)}")
        print(f"{'#'*70}\n")
        
        try:
            data = loadmat(mat_file)
            
            struct_name = None
            for key in data.keys():
                if not key.startswith('__'):
                    struct_name = key
                    break
            
            if struct_name is None:
                print(f"  ERROR: Could not find data structure in {mat_file}")
                continue
            
            print(f"  Found data structure: '{struct_name}'")
            mat_struct = data[struct_name]
            ds = unwrap(mat_struct)
            
            X = None
            y = None
            fs = 250
            
            if hasattr(ds, 'dtype') and ds.dtype.names:
                X = unwrap(get_field(ds, 'X'))
                y = unwrap(get_field(ds, 'y'))
                fs = unwrap(get_field(ds, 'fs')) or 250
            else:
                if isinstance(ds, np.ndarray):
                    X = ds
                    print(f"  Data is direct array format (shape: {X.shape})")
                else:
                    print(f"  ERROR: Unknown data format in {mat_file}")
                    continue
            
            if X is None:
                print(f"  ERROR: No data found in {mat_file}")
                continue
            
            # Handle labels - preserve as-is or create if missing
            if y is None:
                print(f"  WARNING: No labels found, creating zeros")
                y = np.zeros((X.shape[0], 1))
            else:
                y = np.array(y)
                if len(y.shape) == 1:
                    y = y.reshape(-1, 1)
            
            # Ensure X is 3D: (trials, samples, channels)
            if len(X.shape) == 2:
                print(f"  WARNING: X is 2D, assuming single trial")
                X = X[np.newaxis, :, :]
            
            fs_val = float(np.array(fs).reshape(-1)[0]) if fs is not None else 250
            
            print(f"  Data shape: {X.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Using ALL {X.shape[0]} trials")
            
            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            output_dir = os.path.join('augmented_data', base_name)
            
            augmenter = CompleteEEGAugmenter(fs=fs_val, output_dir=output_dir)
            
            results = augmenter.augment_dataset_by_method(
                X, y,
                methods='paper',
                quality_check=True,
                save=True,
                original_struct_name=struct_name,
                base_filename=base_name
            )
            
            print(f"  ✓ Successfully processed {mat_file}")
            
        except Exception as e:
            print(f"  ERROR processing {mat_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"ALL FILES PROCESSED")
    print(f"{'='*70}")