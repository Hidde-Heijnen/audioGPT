#!/usr/bin/env python3
"""
Compare resampling quality between different backends.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import resampling functions
import sys
import os
sys.path.append(os.path.dirname(__file__))
from resample_parquet_column import (
    _resample_vector_np, _resample_vector_poly, validate_alias_rejection,
    _has_scipy, _has_soxr
)

if _has_soxr:
    from resample_parquet_column import _resample_vector_soxr

def compare_backends(token_name="the", parquet_path="data/camstories/audio_embed_full.parquet"):
    """Compare resampling quality for a specific token."""
    
    df = pd.read_parquet(parquet_path)
    token_row = df[df['token'] == token_name]
    
    if token_row.empty:
        print(f"Token '{token_name}' not found!")
        return
    
    # Get original 24kHz data
    original = np.array(token_row.iloc[0]['24000_tight'], dtype=np.float32)
    
    print(f"Comparing resampling quality for token: '{token_name}'")
    print(f"Original length: {len(original)} samples")
    
    # Test different backends
    backends = [("NumPy Linear", _resample_vector_np)]
    if _has_scipy:
        backends.append(("SciPy Polyphase", _resample_vector_poly))
    if _has_soxr:
        backends.append(("SoXR", _resample_vector_soxr))
    
    results = {}
    
    for name, func in backends:
        try:
            resampled = func(original, 24000, 8000)
            validation = validate_alias_rejection(original, resampled, 24000, 8000)
            results[name] = {
                'data': resampled,
                'validation': validation
            }
            if validation['alias_db'] is not None:
                print(f"{name:16}: {validation['alias_db']:6.1f} dB alias rejection ({validation['status']})")
            else:
                print(f"{name:16}: {validation['status']}")
        except Exception as e:
            print(f"{name:16}: Error - {e}")
    
    # Plot comparison
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 4*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, result) in enumerate(results.items()):
        data = result['data']
        
        # Time domain
        time = np.arange(len(data)) / 8000
        axes[i, 0].plot(time, data, 'b-', linewidth=0.8)
        axes[i, 0].set_title(f'{name} - Waveform')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Frequency domain
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1/8000)
        power_db = 20 * np.log10(np.abs(fft) + 1e-12)
        
        axes[i, 1].plot(freqs, power_db, 'g-', linewidth=1)
        axes[i, 1].set_xlim(0, 4000)
        axes[i, 1].set_ylim(power_db.max() - 100, power_db.max() + 5)
        
        # Add validation info to title
        val = result['validation']
        if val['alias_db'] is not None:
            title = f'{name} - Spectrum ({val["alias_db"]:.1f} dB rejection)'
        else:
            title = f'{name} - Spectrum ({val["status"]})'
        axes[i, 1].set_title(title)
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (dB)')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    compare_backends() 