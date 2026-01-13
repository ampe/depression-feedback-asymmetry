"""
Загрузчик EEG данных из EEGLAB .set/.fdt файлов
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from config import GROUPS, EPOCHS_PATTERN, SRATE


def find_epochs_files(group_path: Path) -> List[Tuple[str, Path, Path]]:
    """
    Находит все пары .set + .fdt файлов для epoched данных
    
    Returns:
        List of (subject_id, set_path, fdt_path)
    """
    results = []
    
    for subj_folder in sorted(group_path.iterdir()):
        if not subj_folder.is_dir():
            continue
        
        subj_id = subj_folder.name
        
        # Ищем .set файл
        set_files = list(subj_folder.glob(EPOCHS_PATTERN))
        
        if not set_files:
            print(f"  [WARN] {subj_id}: no epochs .set file found")
            continue
        
        set_path = set_files[0]
        
        # Ищем соответствующий .fdt файл
        # Может быть с тем же именем или с длинным именем
        fdt_path = set_path.with_suffix('.fdt')
        
        if not fdt_path.exists():
            # Пробуем длинное имя
            long_fdt_name = set_path.stem + '_ICA_ev_ICA_elist_hpfilt_ref_bins_epochs.fdt'
            fdt_path = subj_folder / long_fdt_name
            
            if not fdt_path.exists():
                # Ищем любой .fdt с похожим именем
                fdt_candidates = list(subj_folder.glob(f'{subj_id}*epochs*.fdt'))
                if fdt_candidates:
                    fdt_path = fdt_candidates[0]
                else:
                    print(f"  [WARN] {subj_id}: no .fdt file found for {set_path.name}")
                    continue
        
        results.append((subj_id, set_path, fdt_path))
    
    return results


def load_eeglab_epochs(set_path: Path, fdt_path: Path) -> Dict:
    """
    Загружает epoched данные из EEGLAB .set + .fdt файлов
    
    Returns:
        dict with keys: 'data', 'times', 'channels', 'epoch_bins', 'srate', etc.
    """
    # Загружаем метаданные из .set
    mat = sio.loadmat(str(set_path), squeeze_me=True, struct_as_record=False)
    eeg = mat['EEG']
    
    n_channels = int(eeg.nbchan)
    n_points = int(eeg.pnts)
    n_trials = int(eeg.trials)
    srate = float(eeg.srate)
    
    # Загружаем данные из .fdt
    data = np.fromfile(str(fdt_path), dtype=np.float32)
    
    expected_size = n_channels * n_points * n_trials
    if len(data) != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data)}")
    
    # Reshape: (channels, points, trials)
    data = data.reshape((n_channels, n_points, n_trials), order='F')
    
    # Временная ось
    times = np.linspace(eeg.xmin, eeg.xmax, n_points)
    
    # Названия каналов
    channels = [getattr(ch, 'labels', f'Ch{i}') for i, ch in enumerate(eeg.chanlocs)]
    
    # Извлекаем bin для каждой эпохи
    epoch_bins = []
    for ep in eeg.epoch:
        bini = getattr(ep, 'eventbini', 0)
        if hasattr(bini, '__len__'):
            # Несколько событий в эпохе — берём первый ненулевой bin
            bini = next((b for b in bini if b > 0), 0)
        epoch_bins.append(int(bini))
    
    epoch_bins = np.array(epoch_bins)
    
    return {
        'data': data,              # (channels, points, trials)
        'times': times,            # (points,) in seconds
        'channels': channels,      # list of channel names
        'epoch_bins': epoch_bins,  # (trials,) bin number for each epoch
        'srate': srate,
        'n_channels': n_channels,
        'n_points': n_points,
        'n_trials': n_trials,
    }


def load_group(group_name: str, max_subjects: Optional[int] = None) -> Dict[str, Dict]:
    """
    Загружает всех субъектов из группы
    
    Args:
        group_name: ключ из GROUPS dict
        max_subjects: ограничить количество (для тестирования)
    
    Returns:
        dict: {subject_id: subject_data_dict}
    """
    if group_name not in GROUPS:
        raise ValueError(f"Unknown group: {group_name}. Available: {list(GROUPS.keys())}")
    
    group_path = GROUPS[group_name]
    print(f"\nLoading group: {group_name}")
    print(f"Path: {group_path}")
    
    # Находим файлы
    files = find_epochs_files(group_path)
    print(f"Found {len(files)} subjects with epochs data")
    
    if max_subjects:
        files = files[:max_subjects]
        print(f"Limited to {max_subjects} subjects for testing")
    
    # Загружаем данные
    group_data = {}
    
    for subj_id, set_path, fdt_path in files:
        try:
            subj_data = load_eeglab_epochs(set_path, fdt_path)
            group_data[subj_id] = subj_data
            print(f"  ✓ {subj_id}: {subj_data['n_trials']} epochs, {subj_data['n_channels']} channels")
        except Exception as e:
            print(f"  ✗ {subj_id}: {e}")
    
    print(f"Successfully loaded: {len(group_data)}/{len(files)} subjects")
    
    return group_data


def load_all_groups(max_subjects_per_group: Optional[int] = None) -> Dict[str, Dict[str, Dict]]:
    """
    Загружает все группы
    
    Returns:
        dict: {group_name: {subject_id: subject_data}}
    """
    all_data = {}
    
    for group_name in GROUPS:
        all_data[group_name] = load_group(group_name, max_subjects_per_group)
    
    return all_data


# === ТЕСТ ===
if __name__ == '__main__':
    # Тестируем на 2 субъектах из каждой группы
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    data = load_all_groups(max_subjects_per_group=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for group_name, group_data in data.items():
        print(f"\n{group_name}:")
        for subj_id, subj in group_data.items():
            print(f"  {subj_id}: data shape = {subj['data'].shape}")
            print(f"          channels = {subj['channels'][:5]}...")
            print(f"          bins distribution = {np.bincount(subj['epoch_bins'])[1:]}")