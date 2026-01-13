"""
Конфигурация путей к данным
"""
from pathlib import Path

# Корневая папка с данными
DATA_ROOT = Path(r"E:\depression-feedback-asymmetry-data\initial data, ICA done")

# Папки групп
GROUPS = {
    'mdd_negative': DATA_ROOT / 'F32_group_negative',
    'mdd_positive': DATA_ROOT / 'F32_group_positive',
    'healthy_negative': DATA_ROOT / 'Health_group_negative',
    # 'healthy_positive': DATA_ROOT / 'Health_group_positive',  # только сырые данные
}

# Паттерн для финальных файлов
EPOCHS_PATTERN = '*_ICA_ev_ICA_elist_hpfilt_ref_bins_epochs.set'

# EEG параметры
SRATE = 500  # Hz
EPOCH_TMIN = -0.2  # seconds
EPOCH_TMAX = 0.798  # seconds

# Bins (события)
BIN_LABELS = {
    1: 'Benefits',
    2: 'Hazards',
    3: 'Feedback2',
    4: 'Choice',
    5: 'Start',
    6: 'Feedback1'
}

# Каналы для анализа P300
P300_CHANNELS = ['Pz', 'Cz', 'Fz', 'CPz']
P300_WINDOW = (0.200, 0.500)  # seconds