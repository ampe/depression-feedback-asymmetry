"""
Основной анализ: загрузка всех данных, статистика, визуализация
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from data_loader import load_all_groups
from feature_extraction import extract_group_features


def run_full_analysis():
    """Запуск полного анализа на всех субъектах"""
    
    print("=" * 60)
    print("LOADING ALL SUBJECTS")
    print("=" * 60)
    
    # Загружаем ВСЕ данные (без ограничения)
    data = load_all_groups(max_subjects_per_group=None)
    
    print("\n" + "=" * 60)
    print("EXTRACTING FEATURES")
    print("=" * 60)
    
    # Извлекаем фичи для каждой группы
    results = {}
    for group_name, group_data in data.items():
        print(f"\nProcessing {group_name}...")
        X, feature_names, subject_ids = extract_group_features(group_data)
        results[group_name] = {
            'X': X,
            'feature_names': feature_names,
            'subject_ids': subject_ids
        }
        print(f"  ✓ {X.shape[0]} subjects, {X.shape[1]} features")
    
    # === СТАТИСТИЧЕСКИЙ АНАЛИЗ ===
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    feature_names = results['mdd_negative']['feature_names']
    
    # 1. MDD: Positive vs Negative (within-subject comparison)
    print("\n--- MDD: Positive vs Negative Feedback ---")
    
    X_mdd_neg = results['mdd_negative']['X']
    X_mdd_pos = results['mdd_positive']['X']
    
    # Находим парные субъекты (S1021 ↔ S1022, etc.)
    neg_ids = results['mdd_negative']['subject_ids']
    pos_ids = results['mdd_positive']['subject_ids']
    
    # Создаём маппинг (убираем последнюю цифру для matching)
    neg_base = {s[:-1]: i for i, s in enumerate(neg_ids)}
    pos_base = {s[:-1]: i for i, s in enumerate(pos_ids)}
    
    # Находим общих
    common_base = set(neg_base.keys()) & set(pos_base.keys())
    print(f"  Paired subjects: {len(common_base)}")
    
    paired_neg_idx = [neg_base[b] for b in sorted(common_base)]
    paired_pos_idx = [pos_base[b] for b in sorted(common_base)]
    
    X_neg_paired = X_mdd_neg[paired_neg_idx]
    X_pos_paired = X_mdd_pos[paired_pos_idx]
    
    print("\n  Key features (paired t-test):")
    key_features = [fn for fn in feature_names if 'asymmetry' in fn or 'p300_mean' in fn]
    
    significant_features = []
    for fn in key_features:
        idx = feature_names.index(fn)
        neg_vals = X_neg_paired[:, idx]
        pos_vals = X_pos_paired[:, idx]
        
        # Убираем NaN
        mask = ~(np.isnan(neg_vals) | np.isnan(pos_vals))
        if mask.sum() < 5:
            continue
            
        t_stat, p_val = stats.ttest_rel(neg_vals[mask], pos_vals[mask])
        
        if p_val < 0.1:  # показываем тренды тоже
            sig = "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    {fn}:")
            print(f"      Negative: {neg_vals[mask].mean():.2f} ± {neg_vals[mask].std():.2f}")
            print(f"      Positive: {pos_vals[mask].mean():.2f} ± {pos_vals[mask].std():.2f}")
            print(f"      t={t_stat:.2f}, p={p_val:.4f} {sig}")
            significant_features.append((fn, p_val, t_stat))
    
    # 2. MDD vs Healthy (в negative condition)
    print("\n--- MDD vs Healthy (Negative condition) ---")
    
    X_healthy_neg = results['healthy_negative']['X']
    
    print(f"  MDD: n={X_mdd_neg.shape[0]}, Healthy: n={X_healthy_neg.shape[0]}")
    print("\n  Key features (independent t-test):")
    
    for fn in key_features:
        idx = feature_names.index(fn)
        mdd_vals = X_mdd_neg[:, idx]
        healthy_vals = X_healthy_neg[:, idx]
        
        # Убираем NaN
        mdd_clean = mdd_vals[~np.isnan(mdd_vals)]
        healthy_clean = healthy_vals[~np.isnan(healthy_vals)]
        
        if len(mdd_clean) < 5 or len(healthy_clean) < 5:
            continue
        
        t_stat, p_val = stats.ttest_ind(mdd_clean, healthy_clean)
        
        if p_val < 0.1:
            sig = "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    {fn}:")
            print(f"      MDD:     {mdd_clean.mean():.2f} ± {mdd_clean.std():.2f}")
            print(f"      Healthy: {healthy_clean.mean():.2f} ± {healthy_clean.std():.2f}")
            print(f"      t={t_stat:.2f}, p={p_val:.4f} {sig}")
    
    # === ВИЗУАЛИЗАЦИЯ ===
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    create_visualizations(results, feature_names)
    
    return results


def create_visualizations(results, feature_names):
    """Создаёт графики"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Фронтальная асимметрия: MDD pos vs neg
    ax = axes[0, 0]
    
    fn = 'Feedback2_frontal_asymmetry'
    if fn in feature_names:
        idx = feature_names.index(fn)
        
        mdd_neg = results['mdd_negative']['X'][:, idx]
        mdd_pos = results['mdd_positive']['X'][:, idx]
        healthy_neg = results['healthy_negative']['X'][:, idx]
        
        data_to_plot = [
            mdd_neg[~np.isnan(mdd_neg)],
            mdd_pos[~np.isnan(mdd_pos)],
            healthy_neg[~np.isnan(healthy_neg)]
        ]
        
        ax.boxplot(data_to_plot, labels=['MDD\nNegative', 'MDD\nPositive', 'Healthy\nNegative'])
        ax.set_ylabel('Frontal Asymmetry (F4-F3) µV')
        ax.set_title('Frontal Asymmetry at Feedback2')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 2. P300 amplitude comparison
    ax = axes[0, 1]
    
    fn = 'Feedback2_Pz_p300_mean'
    if fn in feature_names:
        idx = feature_names.index(fn)
        
        mdd_neg = results['mdd_negative']['X'][:, idx]
        mdd_pos = results['mdd_positive']['X'][:, idx]
        healthy_neg = results['healthy_negative']['X'][:, idx]
        
        data_to_plot = [
            mdd_neg[~np.isnan(mdd_neg)],
            mdd_pos[~np.isnan(mdd_pos)],
            healthy_neg[~np.isnan(healthy_neg)]
        ]
        
        ax.boxplot(data_to_plot, labels=['MDD\nNegative', 'MDD\nPositive', 'Healthy\nNegative'])
        ax.set_ylabel('P300 Amplitude (µV)')
        ax.set_title('P300 at Pz (Feedback2)')
    
    # 3. Scatter: Frontal asymmetry Feedback1 vs Feedback2
    ax = axes[1, 0]
    
    fn1 = 'Feedback1_frontal_asymmetry'
    fn2 = 'Feedback2_frontal_asymmetry'
    
    if fn1 in feature_names and fn2 in feature_names:
        idx1 = feature_names.index(fn1)
        idx2 = feature_names.index(fn2)
        
        # MDD negative
        x = results['mdd_negative']['X'][:, idx1]
        y = results['mdd_negative']['X'][:, idx2]
        mask = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[mask], y[mask], alpha=0.6, label='MDD Negative', c='red')
        
        # MDD positive
        x = results['mdd_positive']['X'][:, idx1]
        y = results['mdd_positive']['X'][:, idx2]
        mask = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[mask], y[mask], alpha=0.6, label='MDD Positive', c='blue')
        
        # Healthy
        x = results['healthy_negative']['X'][:, idx1]
        y = results['healthy_negative']['X'][:, idx2]
        mask = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[mask], y[mask], alpha=0.6, label='Healthy Negative', c='green')
        
        ax.set_xlabel('Frontal Asymmetry - Feedback1')
        ax.set_ylabel('Frontal Asymmetry - Feedback2')
        ax.set_title('Frontal Asymmetry: Feedback1 vs Feedback2')
        ax.legend()
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
    # 4. Bar plot: all asymmetry measures
    ax = axes[1, 1]
    
    asym_features = [fn for fn in feature_names if 'asymmetry' in fn]
    
    x = np.arange(len(asym_features))
    width = 0.25
    
    means_mdd_neg = []
    means_mdd_pos = []
    means_healthy = []
    
    for fn in asym_features:
        idx = feature_names.index(fn)
        means_mdd_neg.append(np.nanmean(results['mdd_negative']['X'][:, idx]))
        means_mdd_pos.append(np.nanmean(results['mdd_positive']['X'][:, idx]))
        means_healthy.append(np.nanmean(results['healthy_negative']['X'][:, idx]))
    
    ax.bar(x - width, means_mdd_neg, width, label='MDD Negative', color='red', alpha=0.7)
    ax.bar(x, means_mdd_pos, width, label='MDD Positive', color='blue', alpha=0.7)
    ax.bar(x + width, means_healthy, width, label='Healthy Negative', color='green', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([fn.replace('_asymmetry', '').replace('_', '\n') for fn in asym_features], 
                       fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Asymmetry (R-L) µV')
    ax.set_title('Brain Asymmetry by Condition')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Сохраняем
    save_path = 'analysis_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    try:
        results = run_full_analysis()
    except Exception as e:
        import traceback
        print("ERROR:")
        traceback.print_exc()
    input("Press Enter to exit...")  # чтобы окно не закрылось