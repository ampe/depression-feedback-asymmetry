"""
ML классификация: MDD vs Healthy, Positive vs Negative
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_groups
from feature_extraction import extract_group_features

def main():
    import sys
    print("=" * 60)
    print("ML CLASSIFICATION PIPELINE")
    print("=" * 60)
    sys.stdout.flush()  # <-- добавь эту строку

def prepare_datasets(results):
    """
    Готовит датасеты для классификации
    
    Returns:
        dict с разными задачами классификации
    """
    datasets = {}
    
    feature_names = results['mdd_negative']['feature_names']
    
    # === Task 1: MDD vs Healthy (в negative condition) ===
    X_mdd = results['mdd_negative']['X']
    X_healthy = results['healthy_negative']['X']
    
    X_task1 = np.vstack([X_mdd, X_healthy])
    y_task1 = np.array([1] * len(X_mdd) + [0] * len(X_healthy))  # 1=MDD, 0=Healthy
    
    datasets['mdd_vs_healthy'] = {
        'X': X_task1,
        'y': y_task1,
        'description': 'MDD vs Healthy (Negative condition)',
        'classes': ['Healthy', 'MDD']
    }
    
    # === Task 2: Positive vs Negative feedback (только MDD) ===
    X_neg = results['mdd_negative']['X']
    X_pos = results['mdd_positive']['X']
    
    X_task2 = np.vstack([X_neg, X_pos])
    y_task2 = np.array([0] * len(X_neg) + [1] * len(X_pos))  # 0=Negative, 1=Positive
    
    datasets['pos_vs_neg_mdd'] = {
        'X': X_task2,
        'y': y_task2,
        'description': 'Positive vs Negative feedback (MDD only)',
        'classes': ['Negative', 'Positive']
    }
    
    # === Task 3: Все три группы ===
    X_task3 = np.vstack([
        results['mdd_negative']['X'],
        results['mdd_positive']['X'],
        results['healthy_negative']['X']
    ])
    y_task3 = np.array(
        [0] * len(results['mdd_negative']['X']) +      # MDD Negative
        [1] * len(results['mdd_positive']['X']) +      # MDD Positive
        [2] * len(results['healthy_negative']['X'])    # Healthy Negative
    )
    
    datasets['three_groups'] = {
        'X': X_task3,
        'y': y_task3,
        'description': '3-class: MDD-Neg vs MDD-Pos vs Healthy-Neg',
        'classes': ['MDD Negative', 'MDD Positive', 'Healthy Negative']
    }
    
    return datasets, feature_names


def clean_data(X, y):
    """Убирает NaN и inf"""
    # Заменяем inf на nan
    X = np.where(np.isinf(X), np.nan, X)
    
    # Убираем строки с NaN
    mask = ~np.any(np.isnan(X), axis=1)
    
    return X[mask], y[mask]


def evaluate_models(X, y, task_name, classes):
    """
    Оценивает несколько моделей с cross-validation
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Class distribution: {dict(zip(classes, np.bincount(y)))}")
    
    # Очистка данных
    X_clean, y_clean = clean_data(X, y)
    print(f"After cleaning: {len(y_clean)} samples")
    
    # Модели для тестирования
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        'Logistic + Feature Selection': Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=10)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    print(f"\n5-Fold Cross-Validation Results:")
    print("-" * 50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='accuracy')
        
        # AUC для бинарной классификации
        if len(classes) == 2:
            auc_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='roc_auc')
            auc_mean = auc_scores.mean()
            auc_std = auc_scores.std()
        else:
            auc_mean = None
            auc_std = None
        
        results[name] = {
            'accuracy': scores.mean(),
            'accuracy_std': scores.std(),
            'auc': auc_mean,
            'auc_std': auc_std
        }
        
        if auc_mean:
            print(f"{name:30s}: Acc={scores.mean():.3f}±{scores.std():.3f}, AUC={auc_mean:.3f}±{auc_std:.3f}")
        else:
            print(f"{name:30s}: Acc={scores.mean():.3f}±{scores.std():.3f}")
    
    return results, X_clean, y_clean


def analyze_feature_importance(X, y, feature_names, task_name):
    """
    Анализирует важность фич
    """
    X_clean, y_clean = clean_data(X, y)
    
    # Обучаем Random Forest на всех данных для feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_clean)
    
    # Feature importance
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print(f"\nTop 10 Important Features ({task_name}):")
    print("-" * 50)
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]:40s}: {importance[idx]:.4f}")
    
    return importance, indices


def plot_results(all_results, datasets):
    """
    Визуализация результатов
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Accuracy comparison across tasks
    ax = axes[0, 0]
    
    tasks = list(all_results.keys())
    models = list(all_results[tasks[0]].keys())
    
    x = np.arange(len(tasks))
    width = 0.15
    
    for i, model in enumerate(models):
        accs = [all_results[task][model]['accuracy'] for task in tasks]
        stds = [all_results[task][model]['accuracy_std'] for task in tasks]
        ax.bar(x + i*width, accs, width, label=model, yerr=stds, capsize=3)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison Across Tasks')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(['MDD vs\nHealthy', 'Pos vs Neg\n(MDD)', '3-class'], fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylim(0.3, 1.0)
    
    # 2. ROC curve для лучшей модели (MDD vs Healthy)
    ax = axes[0, 1]
    
    task = 'mdd_vs_healthy'
    X, y = datasets[task]['X'], datasets[task]['y']
    X_clean, y_clean = clean_data(X, y)
    
    # Обучаем лучшую модель
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # LOO для ROC (более честная оценка на малых выборках)
    y_proba = np.zeros(len(y_clean))
    loo = LeaveOneOut()
    
    for train_idx, test_idx in loo.split(X_clean):
        model.fit(X_clean[train_idx], y_clean[train_idx])
        y_proba[test_idx] = model.predict_proba(X_clean[test_idx])[:, 1]
    
    fpr, tpr, _ = roc_curve(y_clean, y_proba)
    auc = roc_auc_score(y_clean, y_proba)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: MDD vs Healthy')
    ax.legend()
    
    # 3. Confusion matrix
    ax = axes[1, 0]
    
    model.fit(X_clean, y_clean)
    y_pred = model.predict(X_clean)
    cm = confusion_matrix(y_clean, y_pred)
    
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Healthy', 'MDD'])
    ax.set_yticklabels(['Healthy', 'MDD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (MDD vs Healthy)')
    
    # Добавляем числа
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16)
    
    plt.colorbar(im, ax=ax)
    
    # 4. Feature importance
    ax = axes[1, 1]
    
    # Загружаем feature_names из results
    from data_loader import load_all_groups
    from feature_extraction import extract_group_features
    
    # Используем уже загруженные данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_clean)
    
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1][:10]
    
    # Получаем имена фич
    data = load_all_groups(max_subjects_per_group=1)
    _, feature_names, _ = extract_group_features(data['mdd_negative'])
    
    top_features = [feature_names[i] for i in indices]
    top_importance = [importance[i] for i in indices]
    
    ax.barh(range(10), top_importance[::-1])
    ax.set_yticks(range(10))
    ax.set_yticklabels([fn.replace('_', '\n') for fn in top_features[::-1]], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Features (Random Forest)')
    
    plt.tight_layout()
    plt.savefig('ml_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: ml_results.png")
    
    plt.show()


def main():
    print("=" * 60)
    print("ML CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # Загрузка данных
    print("\nLoading data...")
    data = load_all_groups(max_subjects_per_group=None)
    
    # Извлечение фич
    print("\nExtracting features...")
    results = {}
    for group_name, group_data in data.items():
        X, feature_names, subject_ids = extract_group_features(group_data)
        results[group_name] = {
            'X': X,
            'feature_names': feature_names,
            'subject_ids': subject_ids
        }
    
    # Подготовка датасетов
    datasets, feature_names = prepare_datasets(results)
    
    # Оценка моделей для каждой задачи
    all_results = {}
    
    for task_name, task_data in datasets.items():
        task_results, X_clean, y_clean = evaluate_models(
            task_data['X'], 
            task_data['y'],
            task_data['description'],
            task_data['classes']
        )
        all_results[task_name] = task_results
        
        # Feature importance для бинарных задач
        if len(task_data['classes']) == 2:
            analyze_feature_importance(
                task_data['X'],
                task_data['y'],
                feature_names,
                task_data['description']
            )
    
    # Визуализация
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_results(all_results, datasets)
    
    return all_results, datasets


if __name__ == '__main__':
    results, datasets = main()
    input("\nPress Enter to exit...")