"""沿用原项目的 PPG/ECG 峰检测和 ABP 表型计算。"""

import neurokit2 as nk
import numpy as np


def calculate_heart_rate(signal, fs, modality):
    """PPG 用 Bishop 峰，ECG 用 NeuroKit R 峰；HR = 60 * fs / 平均峰间隔。"""
    signal = np.asarray(signal)
    if signal.ndim != 1 or len(signal) < 4 or not np.isfinite(signal).all() or np.ptp(signal) == 0:
        raise ValueError(f'{modality} 波形必须是一维有限数值序列，且不能为常数。')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('采样率必须为正数。')
    if modality not in ('PPG', 'ECG'):
        raise ValueError('心率提取仅支持 PPG 或 ECG。')
    method = 'bishop' if modality == 'PPG' else 'neurokit'
    detector = nk.ppg_peaks if modality == 'PPG' else nk.ecg_peaks
    try:
        _, info = detector(signal, sampling_rate=fs, method=method, show=False)
    except (ValueError, IndexError) as exc:
        raise ValueError(f'{modality} 峰检测失败：{exc}') from exc
    peaks = np.asarray(info['PPG_Peaks' if modality == 'PPG' else 'ECG_R_Peaks'], dtype=int)
    if len(peaks) < 2:
        raise ValueError(f'{modality} 至少需要检测到两个心搏峰才能计算心率。')
    interval = float(np.mean(np.diff(peaks)) / fs)
    return {'bpm': 60 / interval, 'mean_interval_ms': interval * 1000,
            'method': method, 'n_peaks': len(peaks), 'peak_indices': peaks.tolist()}


def caculate_abp_phenotype(signal, fs):
    """输入为已还原单位的单条 ABP（mmHg），返回 SBP、DBP 和近似 MAP。"""
    signal = np.asarray(signal)  # 保留原项目波形的 float32 精度。
    if signal.ndim != 1 or len(signal) < 4 or not np.isfinite(signal).all():
        raise ValueError('ABP 必须是一维有限数值波形，且至少包含 4 个采样点。')
    if not np.isfinite(fs) or fs <= 0 or np.ptp(signal) == 0:
        raise ValueError('ABP 波形不能为常数，采样率必须为正数。')
    _, info = nk.ppg_peaks(signal, sampling_rate=fs, method='bishop', show=False)
    # NeuroKit2 0.2.9 的公共 API 只返回峰；反相信号的 Bishop 峰即原信号的谷。
    _, onset_info = nk.ppg_peaks(-signal, sampling_rate=fs, method='bishop', show=False)
    peaks, onsets = info['PPG_Peaks'], onset_info['PPG_Peaks']
    if len(peaks) == 0 or len(onsets) == 0:
        raise ValueError('未检测到可用的 ABP 峰谷，无法计算血压。')
    sbp = float(np.mean(signal[peaks]))
    dbp = float(np.mean(signal[onsets]))
    return {'SBP': sbp, 'DBP': dbp, 'MAP': dbp + (sbp - dbp) / 3}
