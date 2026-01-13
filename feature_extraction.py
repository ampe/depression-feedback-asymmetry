"""
Извлечение полного набора фич для EEG emotion recognition:
- Differential Entropy (DE)
- Power Spectral Density (PSD)
- Statistical features
- Wavelet Transform features
- Connectivity features
- ERP features
"""
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from typing import Dict, List, Tuple
from config import P300_CHANNELS, P300_WINDOW, BIN_LABELS


# =============================================================================
# 1. DIFFERENTIAL ENTROPY (DE) - самая популярная фича в литературе
# =============================================================================

def compute_differential_entropy(data: np.ndarray, srate: float, band: Tuple[float, float]) -> float:
    """
    Вычисляет Differential Entropy для частотного диапазона.
    DE = 0.5 * log(2 * pi * e * variance)
    
    Для EEG сигнала, отфильтрованного в заданном диапазоне.
    """
    # Фильтруем сигнал в нужном диапазоне
    nyq = srate / 2
    low = band[0] / nyq
    high = band[1] / nyq
    
    # Проверка границ
    if high >= 1:
        high = 0.99
    if low <= 0:
        low = 0.01
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        # Differential Entropy
        variance = np.var(filtered)
        if variance > 0:
            de = 0.5 * np.log(2 * np.pi * np.e * variance)
        else:
            de = 0
        return de
    except:
        return 0


# =============================================================================
# 2. POWER SPECTRAL DENSITY (PSD)
# =============================================================================

def compute_psd_welch(data: np.ndarray, srate: float, band: Tuple[float, float]) -> Tuple[float, float]:
    """
    Вычисляет PSD методом Welch.
    
    Returns:
        mean_power: средняя мощность в диапазоне
        peak_power: пиковая мощность
    """
    nperseg = min(len(data), int(srate * 0.5))
    if nperseg < 4:
        return 0.0, 0.0
    
    freqs, psd = signal.welch(data, srate, nperseg=nperseg)
    
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if band_mask.sum() == 0:
        return 0.0, 0.0
    
    band_psd = psd[band_mask]
    mean_power = np.mean(band_psd)
    peak_power = np.max(band_psd)
    
    return mean_power, peak_power


def compute_relative_psd(data: np.ndarray, srate: float, band: Tuple[float, float]) -> float:
    """
    Относительная мощность = мощность в диапазоне / общая мощность
    """
    nperseg = min(len(data), int(srate * 0.5))
    if nperseg < 4:
        return 0.0
    
    freqs, psd = signal.welch(data, srate, nperseg=nperseg)
    
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0
    
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.sum(psd[band_mask])
    
    return band_power / total_power


# =============================================================================
# 3. STATISTICAL FEATURES
# =============================================================================

def compute_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """
    Статистические фичи временного ряда
    """
    features = {}
    
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    features['range'] = np.ptp(data)  # max - min
    features['median'] = np.median(data)
    
    # Моменты высших порядков
    features['skewness'] = stats.skew(data)
    features['kurtosis'] = stats.kurtosis(data)
    
    # Энергия и мощность
    features['energy'] = np.sum(data ** 2)
    features['rms'] = np.sqrt(np.mean(data ** 2))  # Root Mean Square
    
    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    features['zcr'] = zero_crossings / len(data)
    
    # Hjorth parameters
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    
    var0 = np.var(data)
    var1 = np.var(diff1)
    var2 = np.var(diff2)
    
    features['hjorth_activity'] = var0
    features['hjorth_mobility'] = np.sqrt(var1 / var0) if var0 > 0 else 0
    features['hjorth_complexity'] = (np.sqrt(var2 / var1) / features['hjorth_mobility']) if var1 > 0 and features['hjorth_mobility'] > 0 else 0
    
    return features


# =============================================================================
# 4. WAVELET TRANSFORM FEATURES
# =============================================================================

def compute_wavelet_features(data: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict[str, float]:
    """
    Дискретное вейвлет-преобразование (DWT) фичи
    """
    features = {}
    
    try:
        # Многоуровневое разложение
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # coeffs[0] = approximation (lowest frequencies)
        # coeffs[1:] = details (high to low frequencies)
        
        for i, coeff in enumerate(coeffs):
            level_name = f'approx' if i == 0 else f'detail_{level - i + 1}'
            
            # Статистики коэффициентов
            features[f'dwt_{level_name}_mean'] = np.mean(coeff)
            features[f'dwt_{level_name}_std'] = np.std(coeff)
            features[f'dwt_{level_name}_energy'] = np.sum(coeff ** 2)
            features[f'dwt_{level_name}_entropy'] = -np.sum((coeff ** 2) * np.log(coeff ** 2 + 1e-10))
        
        # Общая энергия по уровням
        total_energy = sum(np.sum(c ** 2) for c in coeffs)
        for i, coeff in enumerate(coeffs):
            level_name = f'approx' if i == 0 else f'detail_{level - i + 1}'
            level_energy = np.sum(coeff ** 2)
            features[f'dwt_{level_name}_rel_energy'] = level_energy / (total_energy + 1e-10)
            
    except Exception as e:
        # Если вейвлет не работает, возвращаем нули
        for i in range(level + 1):
            level_name = f'approx' if i == 0 else f'detail_{level - i + 1}'
            features[f'dwt_{level_name}_mean'] = 0
            features[f'dwt_{level_name}_std'] = 0
            features[f'dwt_{level_name}_energy'] = 0
            features[f'dwt_{level_name}_entropy'] = 0
            features[f'dwt_{level_name}_rel_energy'] = 0
    
    return features


# =============================================================================
# 5. CONNECTIVITY FEATURES
# =============================================================================

def compute_connectivity_features(trial_data: np.ndarray, channels: List[str]) -> Dict[str, float]:
    """
    Connectivity фичи между каналами
    """
    features = {}
    n_channels = trial_data.shape[0]
    
    # Корреляционная матрица
    corr_matrix = np.corrcoef(trial_data)
    
    # Убираем NaN
    corr_matrix = np.nan_to_num(corr_matrix, nan=0)
    
    # Глобальные connectivity метрики
    upper_tri = corr_matrix[np.triu_indices(n_channels, k=1)]
    
    features['conn_mean'] = np.mean(upper_tri)
    features['conn_std'] = np.std(upper_tri)
    features['conn_max'] = np.max(upper_tri)
    features['conn_min'] = np.min(upper_tri)
    
    # Специфические пары (если каналы есть)
    pairs = [
        ('F3', 'F4', 'frontal'),
        ('C3', 'C4', 'central'),
        ('P3', 'P4', 'parietal'),
        ('Fz', 'Pz', 'anterior_posterior'),
        ('F3', 'P3', 'left_ap'),
        ('F4', 'P4', 'right_ap'),
    ]
    
    for ch1, ch2, name in pairs:
        if ch1 in channels and ch2 in channels:
            idx1 = channels.index(ch1)
            idx2 = channels.index(ch2)
            features[f'conn_{name}'] = corr_matrix[idx1, idx2]
    
    # Coherence-like metric (упрощённая версия)
    # Межполушарная связность
    left_chs = [channels.index(ch) for ch in ['F3', 'C3', 'P3'] if ch in channels]
    right_chs = [channels.index(ch) for ch in ['F4', 'C4', 'P4'] if ch in channels]
    
    if left_chs and right_chs:
        inter_corr = []
        for l_idx in left_chs:
            for r_idx in right_chs:
                inter_corr.append(corr_matrix[l_idx, r_idx])
        features['conn_interhemispheric'] = np.mean(inter_corr)
    
    return features


def compute_phase_connectivity(trial_data: np.ndarray, srate: float, channels: List[str], 
                                band: Tuple[float, float] = (8, 13)) -> Dict[str, float]:
    """
    Phase-based connectivity (PLV - Phase Locking Value)
    """
    features = {}
    
    n_channels = trial_data.shape[0]
    
    # Фильтруем в alpha диапазоне
    nyq = srate / 2
    low = max(band[0] / nyq, 0.01)
    high = min(band[1] / nyq, 0.99)
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = np.array([signal.filtfilt(b, a, trial_data[i, :]) for i in range(n_channels)])
        
        # Hilbert transform для фазы
        analytic = signal.hilbert(filtered, axis=1)
        phase = np.angle(analytic)
        
        # PLV матрица
        plv_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phase[i, :] - phase[j, :]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
        
        # Средний PLV
        upper_tri = plv_matrix[np.triu_indices(n_channels, k=1)]
        features['plv_mean'] = np.mean(upper_tri)
        features['plv_std'] = np.std(upper_tri)
        
        # Специфические пары
        if 'F3' in channels and 'F4' in channels:
            features['plv_frontal'] = plv_matrix[channels.index('F3'), channels.index('F4')]
        if 'P3' in channels and 'P4' in channels:
            features['plv_parietal'] = plv_matrix[channels.index('P3'), channels.index('P4')]
            
    except:
        features['plv_mean'] = 0
        features['plv_std'] = 0
        features['plv_frontal'] = 0
        features['plv_parietal'] = 0
    
    return features


# =============================================================================
# 6. ASYMMETRY FEATURES
# =============================================================================

def compute_asymmetry_features(trial_data: np.ndarray, channels: List[str], srate: float) -> Dict[str, float]:
    """
    Asymmetry фичи (ключевые для эмоций)
    """
    features = {}
    
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }
    
    pairs = [
        ('F3', 'F4', 'frontal'),
        ('C3', 'C4', 'central'),
        ('P3', 'P4', 'parietal'),
    ]
    
    for ch_left, ch_right, region in pairs:
        if ch_left not in channels or ch_right not in channels:
            continue
        
        left_idx = channels.index(ch_left)
        right_idx = channels.index(ch_right)
        
        for band_name, band_range in bands.items():
            # DE asymmetry: ln(Right) - ln(Left)
            de_left = compute_differential_entropy(trial_data[left_idx, :], srate, band_range)
            de_right = compute_differential_entropy(trial_data[right_idx, :], srate, band_range)
            features[f'de_asym_{region}_{band_name}'] = de_right - de_left
            
            # Power asymmetry
            psd_left, _ = compute_psd_welch(trial_data[left_idx, :], srate, band_range)
            psd_right, _ = compute_psd_welch(trial_data[right_idx, :], srate, band_range)
            
            log_left = np.log(psd_left + 1e-10)
            log_right = np.log(psd_right + 1e-10)
            features[f'psd_asym_{region}_{band_name}'] = log_right - log_left
    
    return features


# =============================================================================
# MAIN EXTRACTION FUNCTIONS
# =============================================================================

def extract_all_features_single_trial(trial_data: np.ndarray, channels: List[str], srate: float) -> Dict[str, float]:
    """
    Извлекает ВСЕ фичи для одного триала
    """
    features = {}
    
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }
    
    key_channels = ['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    
    # === 1. DE и PSD для каждого канала и диапазона ===
    for ch_name in key_channels:
        if ch_name not in channels:
            continue
        ch_idx = channels.index(ch_name)
        ch_data = trial_data[ch_idx, :]
        
        for band_name, band_range in bands.items():
            # Differential Entropy
            de = compute_differential_entropy(ch_data, srate, band_range)
            features[f'{ch_name}_de_{band_name}'] = de
            
            # PSD
            psd_mean, psd_peak = compute_psd_welch(ch_data, srate, band_range)
            features[f'{ch_name}_psd_{band_name}'] = psd_mean
            features[f'{ch_name}_psd_peak_{band_name}'] = psd_peak
            
            # Relative PSD
            rel_psd = compute_relative_psd(ch_data, srate, band_range)
            features[f'{ch_name}_rel_psd_{band_name}'] = rel_psd
    
    # === 2. Statistical features для ключевых каналов ===
    for ch_name in ['Fz', 'Cz', 'Pz']:
        if ch_name not in channels:
            continue
        ch_idx = channels.index(ch_name)
        ch_data = trial_data[ch_idx, :]
        
        stat_features = compute_statistical_features(ch_data)
        for key, value in stat_features.items():
            features[f'{ch_name}_stat_{key}'] = value
    
    # === 3. Wavelet features ===
    for ch_name in ['Fz', 'Cz', 'Pz']:
        if ch_name not in channels:
            continue
        ch_idx = channels.index(ch_name)
        ch_data = trial_data[ch_idx, :]
        
        wavelet_features = compute_wavelet_features(ch_data)
        for key, value in wavelet_features.items():
            features[f'{ch_name}_{key}'] = value
    
    # === 4. Connectivity features ===
    conn_features = compute_connectivity_features(trial_data, channels)
    features.update(conn_features)
    
    # Phase connectivity
    phase_conn = compute_phase_connectivity(trial_data, srate, channels)
    features.update(phase_conn)
    
    # === 5. Asymmetry features ===
    asym_features = compute_asymmetry_features(trial_data, channels, srate)
    features.update(asym_features)
    
    # === 6. Band ratios ===
    for ch_name in ['Fz', 'Cz', 'Pz']:
        if ch_name not in channels:
            continue
        ch_idx = channels.index(ch_name)
        ch_data = trial_data[ch_idx, :]
        
        theta_psd, _ = compute_psd_welch(ch_data, srate, bands['theta'])
        alpha_psd, _ = compute_psd_welch(ch_data, srate, bands['alpha'])
        beta_psd, _ = compute_psd_welch(ch_data, srate, bands['beta'])
        
        features[f'{ch_name}_theta_alpha_ratio'] = theta_psd / (alpha_psd + 1e-10)
        features[f'{ch_name}_theta_beta_ratio'] = theta_psd / (beta_psd + 1e-10)
        features[f'{ch_name}_alpha_beta_ratio'] = alpha_psd / (beta_psd + 1e-10)
    
    return features


def extract_trial_features(subject_data: Dict, bins_of_interest: List[int] = [3, 6]):
    """
    Извлекает полный набор фич для каждого триала
    """
    data = subject_data['data']
    times = subject_data['times']
    channels = subject_data['channels']
    epoch_bins = subject_data['epoch_bins']
    srate = subject_data['srate']
    
    all_features = []
    all_bins = []
    all_indices = []
    feature_names = None
    
    for trial_idx in range(data.shape[2]):
        bin_num = epoch_bins[trial_idx]
        
        if bin_num not in bins_of_interest:
            continue
        
        trial_data = data[:, :, trial_idx]
        
        # Извлекаем все фичи
        features = extract_all_features_single_trial(trial_data, channels, srate)
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        feature_vector = [features.get(fn, 0) for fn in feature_names]
        
        all_features.append(feature_vector)
        all_bins.append(bin_num)
        all_indices.append(trial_idx)
    
    return np.array(all_features), np.array(all_bins), np.array(all_indices), feature_names


def extract_erp_features(subject_data: Dict, bins_of_interest: List[int] = [3, 6]) -> Dict:
    """
    Извлекает фичи для субъекта (усреднённые по триалам)
    """
    data = subject_data['data']
    channels = subject_data['channels']
    epoch_bins = subject_data['epoch_bins']
    srate = subject_data['srate']
    
    features = {}
    
    for bin_num in bins_of_interest:
        bin_name = BIN_LABELS.get(bin_num, f'Bin{bin_num}')
        bin_mask = epoch_bins == bin_num
        n_trials = bin_mask.sum()
        
        if n_trials == 0:
            continue
        
        bin_data = data[:, :, bin_mask]
        
        # Усредняем фичи по триалам
        trial_features_list = []
        for trial_idx in range(bin_data.shape[2]):
            trial_data = bin_data[:, :, trial_idx]
            trial_features = extract_all_features_single_trial(trial_data, channels, srate)
            trial_features_list.append(trial_features)
        
        # Усредняем
        if trial_features_list:
            for key in trial_features_list[0].keys():
                values = [tf[key] for tf in trial_features_list]
                features[f'{bin_name}_{key}'] = np.mean(values)
        
        features[f'{bin_name}_n_trials'] = n_trials
    
    return features


def extract_group_features(group_data: Dict[str, Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Извлекает фичи для всей группы (subject-level)
    """
    all_features = []
    subject_ids = []
    feature_names = None
    
    for subj_id, subj_data in group_data.items():
        try:
            features = extract_erp_features(subj_data)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            feature_vector = [features.get(fn, np.nan) for fn in feature_names]
            all_features.append(feature_vector)
            subject_ids.append(subj_id)
            
        except Exception as e:
            print(f"  [ERROR] {subj_id}: {e}")
    
    X = np.array(all_features)
    
    return X, feature_names, subject_ids


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    from data_loader import load_all_groups
    
    print("=" * 60)
    print("TESTING COMPREHENSIVE FEATURE EXTRACTION")
    print("=" * 60)
    
    # Тест на 1 субъекте (полная экстракция занимает время)
    data = load_all_groups(max_subjects_per_group=1)
    
    print("\n" + "=" * 60)
    print("TRIAL-LEVEL FEATURES")
    print("=" * 60)
    
    subj_data = list(data['mdd_negative'].values())[0]
    X_trials, bins, indices, feature_names = extract_trial_features(subj_data)
    
    print(f"\nShape: {X_trials.shape}")
    print(f"Total features: {len(feature_names)}")
    
    # Категории фич
    de_features = [f for f in feature_names if '_de_' in f]
    psd_features = [f for f in feature_names if '_psd_' in f]
    stat_features = [f for f in feature_names if '_stat_' in f]
    wavelet_features = [f for f in feature_names if 'dwt_' in f]
    conn_features = [f for f in feature_names if 'conn_' in f or 'plv_' in f]
    asym_features = [f for f in feature_names if '_asym_' in f]
    ratio_features = [f for f in feature_names if '_ratio' in f]
    
    print(f"\nFeature breakdown:")
    print(f"  Differential Entropy: {len(de_features)}")
    print(f"  PSD: {len(psd_features)}")
    print(f"  Statistical: {len(stat_features)}")
    print(f"  Wavelet: {len(wavelet_features)}")
    print(f"  Connectivity: {len(conn_features)}")
    print(f"  Asymmetry: {len(asym_features)}")
    print(f"  Band ratios: {len(ratio_features)}")
    
    print(f"\nSample DE features:")
    for fn in de_features[:5]:
        idx = feature_names.index(fn)
        print(f"  {fn}: {X_trials[:, idx].mean():.4f} ± {X_trials[:, idx].std():.4f}")
    
    print(f"\nSample Asymmetry features:")
    for fn in asym_features[:5]:
        idx = feature_names.index(fn)
        print(f"  {fn}: {X_trials[:, idx].mean():.4f} ± {X_trials[:, idx].std():.4f}")