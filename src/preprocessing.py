"""
ECG Signal Preprocessing Module for CardioInsight.

Provides bandpass filtering, lead-wise z-score normalization, and signal alignment
for 12-lead ECG signals (EchoNext standard: 12 leads, 250 Hz, 10 seconds = 2500 samples).
"""

from typing import Union, Tuple
import numpy as np
from scipy.signal import butter, filtfilt


def create_bandpass_filter(
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 250.0,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a digital Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz (removes baseline wander, default 0.5 Hz).
    highcut : float
        Upper cutoff frequency in Hz (removes high-frequency EMG & powerline noise, default 40.0 Hz).
    fs : float
        Sampling frequency in Hz (default 250.0 Hz).
    order : int
        Filter order (default 4).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(
    ecg: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 250.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter across all ECG leads.

    Parameters
    ----------
    ecg : np.ndarray of shape (12, T)
        Raw multi-lead ECG signal.
    """
    b, a = create_bandpass_filter(lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    filtered = np.zeros_like(ecg, dtype=np.float32)
    for lead_idx in range(ecg.shape[0]):
        filtered[lead_idx] = filtfilt(b, a, ecg[lead_idx])
    return filtered


def normalize_ecg(ecg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Perform lead-wise z-score normalization along the temporal axis.

    (x - mean) / (std + eps)
    """
    mean = np.mean(ecg, axis=-1, keepdims=True)
    std = np.std(ecg, axis=-1, keepdims=True)
    return (ecg - mean) / (std + eps)


def preprocess_ecg(
    ecg: Union[np.ndarray, list],
    fs: float = 250.0,
    apply_filter: bool = True,
    target_length: int = 2500,
) -> np.ndarray:
    """
    Complete standard ECG preprocessing pipeline.

    Parameters
    ----------
    ecg : array-like
        Raw ECG array of shape (12, T) or (T, 12).
    fs : float
        Sampling rate in Hz (default 250 Hz).
    apply_filter : bool
        Whether to apply 0.5 - 40 Hz Butterworth filtering.
    target_length : int
        Expected temporal sample count (default 2500 for 10s @ 250 Hz).

    Returns
    -------
    processed_ecg : np.ndarray of shape (12, target_length)
        Standardized, filtered, and normalized 12-lead ECG.
    """
    data = np.asarray(ecg, dtype=np.float32)

    # Ensure shape is (leads, time)
    if data.ndim == 2:
        if data.shape[0] != 12 and data.shape[1] == 12:
            data = data.T
        if data.shape[0] != 12:
            raise ValueError(f"Expected 12 leads, got shape {data.shape}")
    else:
        raise ValueError(f"Expected 2D ECG array (12, T), got {data.ndim}D array")

    # Align length (pad or truncate to target_length)
    current_len = data.shape[1]
    if current_len < target_length:
        pad_width = target_length - current_len
        data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")
    elif current_len > target_length:
        data = data[:, :target_length]

    # Optional bandpass filter
    if apply_filter:
        data = bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=fs, order=4)

    # Lead-wise z-score normalization
    data = normalize_ecg(data)

    return data.astype(np.float32)


if __name__ == "__main__":
    raw_synthetic = np.random.randn(12, 2500)
    proc = preprocess_ecg(raw_synthetic, fs=250.0)
    print("Preprocessing successful! Output shape:", proc.shape)
    print(f"Mean across leads: {np.mean(proc, axis=1)[:3]}")
    print(f"Std across leads: {np.std(proc, axis=1)[:3]}")
