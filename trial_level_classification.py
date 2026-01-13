"""
Trial-level классификация: используем каждый триал как отдельный sample
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


from data_loader import load_all_groups
from feature_extraction import extract_trial_features
from config import P300_CHANNELS, P300_WINDOW, BIN_LABELS

def prepare_trial_dataset(data, task='feedback_type'):
    """
    Готовит датасет на уровне триалов
    
    Args:
        task: 'feedback_type' (Feedback1 vs Feedback2) или 'group' (MDD vs Healthy)
    """
    all_X = []
    all_y = []
    all_subjects = []  # для GroupKFold
    all_groups = []    # MDD/Healthy
    all_conditions = [] # Positive/Negative
    all_trial_idx = []
    
    feature_names = None
    
    for group_name, group_data in data.items():
        # Определяем группу и условие
        if 'mdd' in group_name:
            group_label = 'MDD'
        else:
            group_label = 'Healthy'
        
        if 'negative' in group_name:
            condition = 'Negative'
        else:
            condition = 'Positive'
        
        for subj_id, subj_data in group_data.items():
            X, bins, indices, fn = extract_trial_features(subj_data)
            
            if feature_names is None:
                feature_names = fn
            
            all_X.append(X)
            all_subjects.extend([subj_id] * len(X))
            all_groups.extend([group_label] * len(X))
            all_conditions.extend([condition] * len(X))
            all_trial_idx.extend(indices.tolist())
            
            # Labels зависят от задачи
            if task == 'feedback_type':
                # Bin 6 = Feedback1, Bin 3 = Feedback2
                y = (bins == 6).astype(int)  # 1 = Feedback1, 0 = Feedback2
            else:
                y = bins
            
            all_y.append(y)
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    return {
        'X': X,
        'y': y,
        'subjects': np.array(all_subjects),
        'groups': np.array(all_groups),
        'conditions': np.array(all_conditions),
        'trial_indices': np.array(all_trial_idx),
        'feature_names': feature_names
    }


def evaluate_with_group_cv(X, y, groups, task_name):
    """
    Оценка с GroupKFold (субъекты не смешиваются между train/test)
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Очистка NaN
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    groups_clean = groups[mask]
    
    print(f"After cleaning: {len(y_clean)} samples")
    
    # Уникальные группы для GroupKFold
    unique_groups = np.unique(groups_clean)
    group_ids = np.array([np.where(unique_groups == g)[0][0] for g in groups_clean])
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
    }
    
    # GroupKFold - субъекты не смешиваются
    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    
    print(f"\n{n_splits}-Fold GroupKFold Results (subjects don't leak):")
    print("-" * 50)
    
    results = {}
    
    for name, model in models.items():
        scores = []
        auc_scores = []
        
        for train_idx, test_idx in gkf.split(X_clean, y_clean, group_ids):
            model.fit(X_clean[train_idx], y_clean[train_idx])
            
            score = model.score(X_clean[test_idx], y_clean[test_idx])
            scores.append(score)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_clean[test_idx])[:, 1]
                auc = roc_auc_score(y_clean[test_idx], y_proba)
                auc_scores.append(auc)
        
        acc_mean, acc_std = np.mean(scores), np.std(scores)
        
        if auc_scores:
            auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
            print(f"{name:25s}: Acc={acc_mean:.3f}±{acc_std:.3f}, AUC={auc_mean:.3f}±{auc_std:.3f}")
        else:
            auc_mean, auc_std = None, None
            print(f"{name:25s}: Acc={acc_mean:.3f}±{acc_std:.3f}")
        
        results[name] = {
            'accuracy': acc_mean,
            'accuracy_std': acc_std,
            'auc': auc_mean,
            'auc_std': auc_std
        }
    
    return results, X_clean, y_clean, groups_clean


def analyze_trial_dynamics(dataset, model_name='Logistic Regression'):
    """
    Анализирует как меняется предсказание по ходу эксперимента
    """
    X = dataset['X']
    y = dataset['y']
    trial_indices = dataset['trial_indices']
    subjects = dataset['subjects']
    groups = dataset['groups']
    conditions = dataset['conditions']
    
    # Очистка
    mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
    X = X[mask]
    y = y[mask]
    trial_indices = trial_indices[mask]
    subjects = subjects[mask]
    groups = groups[mask]
    conditions = conditions[mask]
    
    # Обучаем модель
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X, y)
    
    # Получаем вероятности для каждого триала
    probas = model.predict_proba(X)[:, 1]
    
    # Группируем по субъектам и смотрим динамику
    print("\n" + "="*60)
    print("TRIAL DYNAMICS ANALYSIS")
    print("="*60)
    
    # Разделяем по группам
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (group, condition) in enumerate([
        ('MDD', 'Negative'), ('MDD', 'Positive'), 
        ('Healthy', 'Negative')
    ]):
        if idx >= 4:
            break
            
        ax = axes[idx // 2, idx % 2]
        
        mask_gc = (groups == group) & (conditions == condition)
        
        if mask_gc.sum() == 0:
            continue
        
        # Собираем данные по триалам
        trial_data = []
        for subj in np.unique(subjects[mask_gc]):
            subj_mask = mask_gc & (subjects == subj)
            subj_trials = trial_indices[subj_mask]
            subj_probas = probas[subj_mask]
            
            # Сортируем по индексу триала
            sort_idx = np.argsort(subj_trials)
            trial_data.append(subj_probas[sort_idx])
        
        # Выравниваем длину (берём минимум)
        min_len = min(len(t) for t in trial_data)
        trial_matrix = np.array([t[:min_len] for t in trial_data])
        
        # Среднее и SEM
        mean_prob = trial_matrix.mean(axis=0)
        sem_prob = trial_matrix.std(axis=0) / np.sqrt(len(trial_data))
        
        trials = np.arange(min_len)
        ax.plot(trials, mean_prob, linewidth=2, label=f'{group} {condition}')
        ax.fill_between(trials, mean_prob - sem_prob, mean_prob + sem_prob, alpha=0.3)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('P(Feedback1)')
        ax.set_title(f'{group} - {condition} (n={len(trial_data)})')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
    
    # 4-й subplot: все группы вместе
    ax = axes[1, 1]
    
    for group, condition, color in [
        ('MDD', 'Negative', 'red'),
        ('MDD', 'Positive', 'blue'),
        ('Healthy', 'Negative', 'green')
    ]:
        mask_gc = (groups == group) & (conditions == condition)
        
        if mask_gc.sum() == 0:
            continue
        
        trial_data = []
        for subj in np.unique(subjects[mask_gc]):
            subj_mask = mask_gc & (subjects == subj)
            subj_trials = trial_indices[subj_mask]
            subj_probas = probas[subj_mask]
            sort_idx = np.argsort(subj_trials)
            trial_data.append(subj_probas[sort_idx])
        
        min_len = min(len(t) for t in trial_data)
        trial_matrix = np.array([t[:min_len] for t in trial_data])
        mean_prob = trial_matrix.mean(axis=0)
        
        ax.plot(np.arange(min_len), mean_prob, linewidth=2, 
                label=f'{group} {condition}', color=color)
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('P(Feedback1)')
    ax.set_title('All Groups Comparison')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('trial_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: trial_dynamics.png")
    
    plt.show()
    
    return probas, trial_indices, subjects, groups, conditions


def main():
    print("="*60)
    print("TRIAL-LEVEL CLASSIFICATION")
    print("="*60)
    
    # Загрузка данных
    print("\nLoading data...")
    data = load_all_groups(max_subjects_per_group=None)
    
    # Подготовка trial-level датасета
    print("\nPreparing trial-level dataset...")
    dataset = prepare_trial_dataset(data, task='feedback_type')
    
    print(f"\nDataset shape: {dataset['X'].shape}")
    print(f"Features: {len(dataset['feature_names'])}")
    print(f"Feature names: {dataset['feature_names'][:10]}...")
    
    # === Task 1: Feedback1 vs Feedback2 (все данные) ===
    results1, X1, y1, groups1 = evaluate_with_group_cv(
        dataset['X'], 
        dataset['y'],
        dataset['subjects'],
        "Feedback1 vs Feedback2 (all data)"
    )
    
    # === Task 2: То же, но только MDD Negative ===
    mask_mdd_neg = (dataset['groups'] == 'MDD') & (dataset['conditions'] == 'Negative')
    results2, _, _, _ = evaluate_with_group_cv(
        dataset['X'][mask_mdd_neg],
        dataset['y'][mask_mdd_neg],
        dataset['subjects'][mask_mdd_neg],
        "Feedback1 vs Feedback2 (MDD Negative only)"
    )
    
    # === Task 3: MDD vs Healthy (только Feedback2 триалы в Negative condition) ===
    mask_neg = dataset['conditions'] == 'Negative'
    mask_fb2 = dataset['y'] == 0  # Feedback2
    mask_task3 = mask_neg & mask_fb2
    
    y_group = (dataset['groups'][mask_task3] == 'MDD').astype(int)
    
    results3, _, _, _ = evaluate_with_group_cv(
        dataset['X'][mask_task3],
        y_group,
        dataset['subjects'][mask_task3],
        "MDD vs Healthy (Feedback2, Negative condition)"
    )
    
    # === Анализ динамики ===
    print("\n" + "="*60)
    print("ANALYZING TRIAL DYNAMICS")
    print("="*60)
    
    analyze_trial_dynamics(dataset)
    
    return dataset, results1, results2, results3


if __name__ == '__main__':
    dataset, r1, r2, r3 = main()
    input("\nPress Enter to exit...")