# ============================================
# HYPERPARAMETER TUNING - COMPLETE GUIDE
# ============================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import time

# ============================================
# 1. CHUẨN BỊ DỮ LIỆU
# ============================================

print("="*60)
print("1. DATA PREPARATION")
print("="*60)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================
# 2. BASELINE MODEL (không tune)
# ============================================

print("\n" + "="*60)
print("2. BASELINE SVM (default params)")
print("="*60)

svm_baseline = SVC(random_state=42)
svm_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = svm_baseline.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_baseline, average='weighted'):.4f}")

# ============================================
# 3. GRID SEARCH CV
# ============================================

print("\n" + "="*60)
print("3. GRID SEARCH CV")
print("="*60)

# Định nghĩa grid parameters
param_grid_svm = {
    'C': [0.1, 1, 10, 100],              # Regularization
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
    'kernel': ['rbf', 'linear']          # Kernel type
}

print(f"Total combinations: {4 * 5 * 2} = {4*5*2}")
print(f"With 5-fold CV: {4*5*2*5} = {4*5*2*5} models to train\n")

# Grid Search
start_time = time.time()

grid_search_svm = GridSearchCV(
    SVC(random_state=42),
    param_grid_svm,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',      # Metric để tối ưu
    n_jobs=-1,               # Dùng tất cả CPU cores
    verbose=1                # Hiển thị progress
)

grid_search_svm.fit(X_train_scaled, y_train)

elapsed_time = time.time() - start_time

print(f"\nTime elapsed: {elapsed_time:.2f}s")
print(f"\nBest params: {grid_search_svm.best_params_}")
print(f"Best CV score: {grid_search_svm.best_score_:.4f}")

# Test trên test set
y_pred_grid = grid_search_svm.predict(X_test_scaled)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"Test F1-score: {f1_score(y_test, y_pred_grid, average='weighted'):.4f}")

# Xem top 10 best configurations
results_df = pd.DataFrame(grid_search_svm.cv_results_)
important_cols = ['param_C', 'param_gamma', 'param_kernel', 
                  'mean_test_score', 'std_test_score', 'rank_test_score']
results_summary = results_df[important_cols].sort_values('rank_test_score')

print("\nTop 5 best configurations:")
print(results_summary.head(5).to_string(index=False))

# ============================================
# 4. RANDOMIZED SEARCH CV
# ============================================

print("\n" + "="*60)
print("4. RANDOMIZED SEARCH CV")
print("="*60)

from scipy.stats import loguniform

# Định nghĩa phân phối parameters (rộng hơn Grid Search)
param_distributions_svm = {
    'C': loguniform(0.01, 100),          # Log-uniform distribution
    'gamma': loguniform(0.0001, 1),      # Log-uniform distribution
    'kernel': ['rbf', 'linear', 'poly']  # Discrete choices
}

print(f"Will try 20 random combinations (instead of all)\n")

# Randomized Search
start_time = time.time()

random_search_svm = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions_svm,
    n_iter=20,               # Chỉ thử 20 combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_svm.fit(X_train_scaled, y_train)

elapsed_time = time.time() - start_time

print(f"\nTime elapsed: {elapsed_time:.2f}s")
print(f"\nBest params: {random_search_svm.best_params_}")
print(f"Best CV score: {random_search_svm.best_score_:.4f}")

# Test trên test set
y_pred_random = random_search_svm.predict(X_test_scaled)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred_random):.4f}")
print(f"Test F1-score: {f1_score(y_test, y_pred_random, average='weighted'):.4f}")

# ============================================
# 5. VÍ DỤ VỚI RANDOM FOREST
# ============================================

print("\n" + "="*60)
print("5. GRID SEARCH - RANDOM FOREST")
print("="*60)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"Total combinations: {3 * 3 * 3 * 3} = {3*3*3*3}\n")

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train_scaled, y_train)

print(f"\nBest params: {grid_search_rf.best_params_}")
print(f"Best CV score: {grid_search_rf.best_score_:.4f}")

y_pred_rf = grid_search_rf.predict(X_test_scaled)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# ============================================
# 6. SO SÁNH TẤT CẢ CÁC PHƯƠNG PHÁP
# ============================================

print("\n" + "="*60)
print("6. COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Method': ['Baseline SVM', 'Grid Search SVM', 'Random Search SVM', 'Grid Search RF'],
    'Test Accuracy': [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_grid),
        accuracy_score(y_test, y_pred_random),
        accuracy_score(y_test, y_pred_rf)
    ],
    'Test F1': [
        f1_score(y_test, y_pred_baseline, average='weighted'),
        f1_score(y_test, y_pred_grid, average='weighted'),
        f1_score(y_test, y_pred_random, average='weighted'),
        f1_score(y_test, y_pred_rf, average='weighted')
    ]
})

print(comparison.to_string(index=False))
print(f"\nBest method: {comparison.loc[comparison['Test Accuracy'].idxmax(), 'Method']}")

# ============================================
# 7. TIPS & BEST PRACTICES
# ============================================

print("\n" + "="*60)
print("7. TIPS & BEST PRACTICES")
print("="*60)

print("""
Khi nào dùng Grid Search?
- Dataset nhỏ/trung bình
- Ít hyperparameters (< 5)
- Cần tìm params TỐI ƯU tuyệt đối
- Có thời gian chờ

Khi nào dùng Random Search?
- Dataset lớn
- Nhiều hyperparameters (> 5)
- Cần kết quả NHANH
- Không cần tối ưu tuyệt đối

Scoring metrics:
- 'accuracy': Accuracy
- 'f1': F1-score (binary)
- 'f1_weighted': F1-score (multi-class)
- 'roc_auc': ROC-AUC
- 'precision', 'recall': Precision/Recall
""")
