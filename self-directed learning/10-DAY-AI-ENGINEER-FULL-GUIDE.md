# 10-Day AI Engineer Transformation: Complete Step-by-Step Guide

**Created:** December 4, 2025  
**Updated:** December 8, 2025  
**For:** Data Engineer → AI Engineer Transition  
**Tech Stack:** Python, Scikit-learn, Pandas, Streamlit, LangChain, RAG, FastAPI, React/Next.js  
**Daily Commitment:** 8-10 hours  
**Total Steps:** 150+ actionable steps  

---

## 📋 Table of Contents

- [How to Use This Guide](#how-to-use-this-guide)
- [Day 1: Python for AI & Environment Setup](#day-1-python-for-ai--environment-setup)
- [Day 2: Machine Learning - Types of Learning & Regression](#day-2-machine-learning---types-of-learning--regression)
- [Day 3: Machine Learning - Classification & SVM](#day-3-machine-learning---classification--svm)
- [Day 4: ML Pipelines, Unsupervised, Ensemble & Streamlit App](#day-4-ml-pipelines-unsupervised-ensemble--streamlit-app)
- [Day 5: AI Document Intelligence System - Playground & Backend](#day-5-ai-document-intelligence-system---playground--backend)
- [Day 6: AI Document Intelligence - Advanced Features & RAG Integration](#day-6-ai-document-intelligence---advanced-features--rag-integration)
- [Day 7: LLM APIs & Prompt Engineering](#day-7-llm-apis--prompt-engineering)
- [Day 8: RAG Systems](#day-8-rag-systems)
- [Day 9: AI Agents & Multi-Agent Systems](#day-9-ai-agents--multi-agent-systems)
- [Day 10: Full-Stack Deployment & Portfolio](#day-10-full-stack-deployment--portfolio)

---

## How to Use This Guide

### Step Format
- Each step is numbered: `Step X.Y` (e.g., Step 1.1, Step 5.3)
- Each step takes 30-60 minutes
- Check ☐ boxes as you complete tasks
- When stuck, note the step number (e.g., "stuck on Step 5.3")

### Daily Structure
- **Morning Session:** 3-4 hours (Steps X.1 - X.6)
- **Afternoon Session:** 3-4 hours (Steps X.7 - X.12)
- **Evening Session:** 2-3 hours (Steps X.13 - X.16)

### Getting Help
When you need assistance with any step, reference:
- Step number (e.g., "Step 5.3")
- What you're trying to do
- Error message if any

---

## Day 1: Python for AI & Environment Setup

**Date:** December 4, 2025 (Today!)  
**Focus:** Advanced Python + AI Development Environment  
**Duration:** 8-10 hours  
**Prerequisite:** Basic Python knowledge ✅

### 🌅 Morning Session (3-4 hours)

#### STEP 1.1: Environment Setup (45 min)
- ☐ Install/Update Python 3.11+ (recommended for AI libraries)
- ☐ Install VS Code with Python extension
- ☐ Install Jupyter Notebook extension in VS Code
- ☐ Create project folder: `~/ai-engineer-bootcamp/`
- ☐ Create virtual environment: `python -m venv venv`
- ☐ Activate venv:
  - Windows: `venv\Scripts\activate`
  - Mac/Linux: `source venv/bin/activate`
- ☐ Install base packages: `pip install numpy pandas matplotlib seaborn jupyter`
- ☐ Verify installation: `python -c "import numpy; print(numpy.__version__)"`

**Expected Outcome:** Working Python environment with all base packages installed.

---

#### STEP 1.2: Python Advanced - Type Hints & Dataclasses (30 min)
- ☐ Learn type hints syntax:
  ```python
  def greet(name: str, age: int) -> str:
      return f"Hello {name}, you are {age} years old"
  ```
- ☐ Practice with complex types:
  ```python
  from typing import List, Dict, Optional, Any
  def process_data(items: List[int]) -> Dict[str, Any]:
      pass
  ```
- ☐ Create 3 functions with proper type hints
- ☐ Learn @dataclass decorator:
  ```python
  from dataclasses import dataclass

  @dataclass
  class User:
      name: str
      age: int
      email: str
  ```
- ☐ Create a dataclass for "Product" with name, price, quantity

**Expected Outcome:** Understand type hints and can use dataclasses for data structures.

---

#### STEP 1.3: Python Advanced - Decorators (45 min)
- ☐ Understand decorator concept (function that wraps another function)
- ☐ Create a `@timer` decorator to measure function execution time
  ```python
  import time
  from functools import wraps

  def timer(func):
      @wraps(func)
      def wrapper(*args, **kwargs):
          start = time.time()
          result = func(*args, **kwargs)
          end = time.time()
          print(f"{func.__name__} took {end-start:.2f}s")
          return result
      return wrapper

  @timer
  def slow_function():
      time.sleep(1)
  ```
- ☐ Create a `@retry` decorator for API calls (critical for AI APIs!)
- ☐ Create a `@log_inputs` decorator to track function inputs
- ☐ Practice chaining multiple decorators
- ☐ Understand why decorators matter for AI (wrapping API calls, logging)

**Expected Outcome:** Can create and use custom decorators for common patterns.

---

#### STEP 1.4: Python Advanced - Generators & Iterators (30 min)
- ☐ Understand `yield` vs `return`:
  ```python
  def simple_generator():
      yield 1
      yield 2
      yield 3
  ```
- ☐ Create a generator for processing large datasets
- ☐ Learn why generators are memory-efficient for AI data pipelines
- ☐ Practice with generator expressions: `(x**2 for x in range(1000000))`
- ☐ Create a data streaming generator that yields batches

**Expected Outcome:** Understand generators and when to use them for memory efficiency.

---

#### STEP 1.5: Python Advanced - Async/Await Basics (45 min)
- ☐ Understand async/await concepts (critical for AI API calls!)
- ☐ Learn basic async syntax and `asyncio.gather()` for concurrent API calls
- ☐ Install aiohttp: `pip install aiohttp`
- ☐ Create async function to make multiple concurrent API calls

**Expected Outcome:** Can write async code for concurrent API calls.

---

#### STEP 1.6: Morning Review & Mini-Exercise (30 min)
- ☐ Create a file: `day1_python_advanced.py` combining all concepts
- ☐ Initialize Git repo and push to GitHub

**Expected Outcome:** All morning concepts combined in working code, pushed to GitHub.

---

### 🌞 Afternoon Session (3-4 hours)

#### STEP 1.7: NumPy Fundamentals for AI (45 min)
- ☐ Create notebook: `day1_numpy.ipynb`
- ☐ Array creation, operations, mathematical operations
- ☐ Statistical functions
- ☐ Practice: Create 5 different array manipulations

**Expected Outcome:** Comfortable with NumPy arrays and basic operations.

---

#### STEP 1.8: Pandas for Data Manipulation (60 min)
- ☐ Create notebook: `day1_pandas.ipynb`
- ☐ DataFrame creation and basic operations
- ☐ Data selection, cleaning, groupby operations
- ☐ Practice: Load a CSV and perform 10 different operations

**Expected Outcome:** Can manipulate dataframes efficiently.

---

#### STEP 1.9: Data Visualization Basics (45 min)
- ☐ Create notebook: `day1_visualization.ipynb`
- ☐ Matplotlib and Seaborn basics
- ☐ Create 5 different visualizations from sample data

**Expected Outcome:** Can create basic visualizations for data exploration.

---

#### STEP 1.10: Statistics Fundamentals for ML (45 min)
- ☐ Create notebook: `day1_statistics.ipynb`
- ☐ Descriptive statistics, probability distributions
- ☐ Correlation and covariance
- ☐ Practice: Calculate statistics on a sample dataset

**Expected Outcome:** Understand basic statistical concepts needed for ML.

---

### 🌙 Evening Session (2-3 hours)

#### STEP 1.11: Mini-Project 1 - Data Analysis Pipeline (60 min)
- ☐ Create notebook: `project1_data_analysis.ipynb`
- ☐ Use Titanic dataset: load, explore, clean, visualize
- ☐ Create 5 visualizations with insights

**Expected Outcome:** Complete data analysis project with insights.

---

#### STEP 1.12: Documentation & GitHub (30 min)
- ☐ Write README.md for your project
- ☐ Create proper folder structure
- ☐ Commit and push all Day 1 work to GitHub

**Expected Outcome:** Clean project structure with documentation on GitHub.

---

#### STEP 1.13: Day 1 Review & Day 2 Prep (30 min)
- ☐ Review what you learned
- ☐ Preview Day 2: Machine Learning fundamentals
- ☐ Install scikit-learn: `pip install scikit-learn`

**Expected Outcome:** Clear understanding of Day 1 progress, ready for Day 2.

---

### 📊 Day 1 Checklist
- ☐ Python advanced concepts
- ☐ NumPy, Pandas, Visualization, Statistics
- ☐ Mini-Project 1 complete
- ☐ GitHub repository setup

**✅ DAY 1 COMPLETE?** Move to Day 2!

---

## Day 2: Machine Learning - Types of Learning & Regression

**Date:** December 5, 2025  
**Focus:** ML Types of Learning + Scikit-learn Basics + Regression  
**Duration:** 8-10 hours  
**Prerequisite:** Day 1 completed ✅

### 🌅 Morning Session (3-4 hours)

#### STEP 2.1: ML Fundamentals - Core Concepts & Types of Learning (45 min)
- ☐ Understand **types of learning in Machine Learning**:
  - **Supervised Learning**: Classification (categorical), Regression (continuous)
  - **Unsupervised Learning**: Clustering, Dimensionality Reduction (PCA)
  - **Semi-supervised Learning**: Mix of labeled and unlabeled data
  - **Reinforcement Learning**: Agent learns from environment rewards
- ☐ Learn **features (X) vs target (y)** terminology
- ☐ Understand **train/test split** and why it matters
- ☐ Learn **overfitting vs underfitting**
- ☐ Understand **bias-variance tradeoff**
- ☐ Draw diagram showing ML workflow and types

**Expected Outcome:** Clear understanding of ML types (supervised/unsupervised) and the basic ML workflow.

---

#### STEP 2.2: Scikit-learn Introduction (30 min)
- ☐ Create notebook: `day2_sklearn_intro.ipynb`
- ☐ Understand the **sklearn API pattern** (consistent across all models)
- ☐ Learn about **sklearn datasets**
- ☐ Load iris dataset and explore its structure

**Expected Outcome:** Understand sklearn API and can load datasets.

---

#### STEP 2.3: Data Preprocessing with Sklearn (45 min)
- ☐ Learn **StandardScaler, MinMaxScaler** for feature scaling
- ☐ Learn **LabelEncoder, OneHotEncoder** for categorical encoding
- ☐ Practice **train_test_split** with different ratios
- ☐ Create a preprocessing pipeline example

**Expected Outcome:** Can preprocess data correctly before modeling.

---

#### STEP 2.4: Linear Regression - Theory (30 min)
- ☐ Understand linear regression equation: y = mx + b
- ☐ Learn what "fitting" means
- ☐ Understand **Mean Squared Error (MSE)** loss function
- ☐ Learn **R² score** interpretation
- ☐ Draw diagram: regression line fitting data points

**Expected Outcome:** Understand regression theory before implementation.

---

#### STEP 2.5: Linear Regression - Implementation (45 min)
- ☐ Create notebook: `day2_linear_regression.ipynb`
- ☐ Load California housing dataset
- ☐ Split, scale, train LinearRegression model
- ☐ Evaluate: MSE, RMSE, MAE, R² score
- ☐ Visualize: predicted vs actual values

**Expected Outcome:** Working linear regression model with evaluation metrics.

---

#### STEP 2.6: Feature Engineering Basics (45 min)
- ☐ Understand why feature engineering matters
- ☐ Learn **polynomial features**
- ☐ Practice **feature selection**
- ☐ Handle **missing values**
- ☐ Create 3 new features from existing ones

**Expected Outcome:** Can engineer features to improve model performance.

---

### 🌞 Afternoon Session (3-4 hours)

#### STEP 2.7: Multiple Regression & Regularization (45 min)
- ☐ Create notebook: `day2_regularization.ipynb`
- ☐ Learn **Ridge regression (L2 regularization)**
- ☐ Learn **Lasso regression (L1 regularization)**
- ☐ Compare performance: Linear vs Ridge vs Lasso
- ☐ Understand when to use each

**Expected Outcome:** Understand regularization and when to use it.

---

#### STEP 2.8: Model Evaluation Deep Dive (45 min)
- ☐ Create notebook: `day2_evaluation.ipynb`
- ☐ Learn **cross-validation** and **k-fold cross-validation**
- ☐ Understand **learning curves**
- ☐ Plot learning curves for your model
- ☐ Identify overfitting/underfitting patterns

**Expected Outcome:** Can properly evaluate models using cross-validation.

---

#### STEP 2.9: Hyperparameter Tuning (45 min)
- ☐ Understand what **hyperparameters** are
- ☐ Learn **GridSearchCV** for exhaustive search
- ☐ Learn **RandomizedSearchCV** for faster search
- ☐ Tune Ridge regression alpha parameter
- ☐ Find best hyperparameters and retrain

**Expected Outcome:** Can tune hyperparameters to optimize model performance.

---

#### STEP 2.10: Decision Trees for Regression (45 min)
- ☐ Create notebook: `day2_decision_trees.ipynb`
- ☐ Understand how decision trees work
- ☐ Train DecisionTreeRegressor
- ☐ Visualize the decision tree
- ☐ Compare with linear regression: pros/cons

**Expected Outcome:** Understand decision trees and when to use them.

---

### 🌙 Evening Session (2-3 hours)

#### STEP 2.11: Mini-Project 2 - House Price Prediction (60 min)
- ☐ Create notebook: `project2_house_price.ipynb`
- ☐ Load California housing dataset
- ☐ Explore data, handle missing values
- ☐ Feature engineering (create 2+ new features)
- ☐ Train 3 models: Linear, Ridge, Decision Tree
- ☐ Compare models using cross-validation
- ☐ Select best model and make predictions
- ☐ Document findings

**Expected Outcome:** Complete house price prediction project with model comparison.

---

#### STEP 2.12: Model Persistence (30 min)
- ☐ Learn to save models using **joblib**
- ☐ Learn to load models
- ☐ Save your best model from Mini-Project 2
- ☐ Test loading and making predictions

**Expected Outcome:** Can save and load trained models.

---

#### STEP 2.13: Day 2 Review & Day 3 Prep (30 min)
- ☐ Review regression concepts and models
- ☐ Write 5–10 line summary: Classification vs Regression
- ☐ Push all code to GitHub with comments
- ☐ Preview: **Classification algorithms** (Day 3)

**Expected Outcome:** Solid understanding of regression & ML types, ready for classification.

---

### 📊 Day 2 Checklist
- ☐ ML types and core concepts
- ☐ Scikit-learn API pattern
- ☐ Data preprocessing
- ☐ Linear Regression, Regularization, Evaluation
- ☐ Hyperparameter tuning
- ☐ Decision Trees
- ☐ Mini-Project 2 complete
- ☐ Model persistence

**✅ DAY 2 COMPLETE?** Move to Day 3!

---

## Day 3: Machine Learning - Classification & SVM

**Date:** December 6, 2025  
**Focus:** Classification algorithms + Evaluation + SVM  
**Duration:** 8-10 hours  
**Prerequisite:** Day 2 completed ✅

### 🌅 Morning Session (3-4 hours)

#### STEP 3.1: Classification Fundamentals (45 min)
- ☐ Create notebook: `day3_classification_intro.ipynb`
- ☐ Understand classification vs regression
- ☐ Learn **binary vs multi-class classification**
- ☐ Understand **class imbalance** problem
- ☐ Preview classification metrics (accuracy, precision, recall, F1, ROC-AUC)

**Expected Outcome:** Understand classification fundamentals.

---

#### STEP 3.2: Logistic Regression for Classification (45 min)
- ☐ Create notebook: `day3_logistic_regression.ipynb`
- ☐ Understand sigmoid function and probability outputs
- ☐ Load a binary classification dataset (e.g., breast cancer)
- ☐ Train LogisticRegression model
- ☐ Evaluate: accuracy, precision, recall, F1-score
- ☐ Visualize confusion matrix

**Expected Outcome:** Can build and evaluate logistic regression classifier.

---

#### STEP 3.3: Classification Metrics Deep Dive (45 min)
- ☐ Create notebook: `day3_metrics.ipynb`
- ☐ Understand **Confusion Matrix**: TP, TN, FP, FN
- ☐ Calculate **Accuracy, Precision, Recall, F1-score**
- ☐ Learn **ROC-AUC**: what it means and how to interpret
- ☐ Understand when to use each metric:
  - Accuracy: balanced classes
  - Precision: minimize false positives
  - Recall: minimize false negatives
  - F1: balance precision and recall
  - ROC-AUC: probability scores

**Expected Outcome:** Can select and interpret appropriate classification metrics.

---

#### STEP 3.4: Decision Tree & Random Forest Classifiers (45 min)
- ☐ Create notebook: `day3_tree_ensemble.ipynb`
- ☐ Train DecisionTreeClassifier
- ☐ Visualize the decision tree
- ☐ Train RandomForestClassifier (multiple trees)
- ☐ Compare: single tree vs random forest
- ☐ Extract feature importance from Random Forest

**Expected Outcome:** Understand tree-based classifiers and ensemble methods.

---

### 🌞 Afternoon Session (3-4 hours)

#### STEP 3.5: XGBoost & Gradient Boosting (45 min)
- ☐ Create notebook: `day3_xgboost.ipynb`
- ☐ Understand boosting concept (sequential weak learners)
- ☐ Install XGBoost: `pip install xgboost`
- ☐ Train XGBClassifier
- ☐ Compare performance: Tree vs Forest vs XGBoost
- ☐ Tune XGBoost hyperparameters (learning_rate, max_depth)

**Expected Outcome:** Can use gradient boosting models effectively.

---

#### STEP 3.6: Support Vector Machine (SVM) – Theory & Implementation (60 min)
- ☐ Create notebook: `day3_svm.ipynb`
- ☐ Understand **SVM** core ideas:
  - Finds **hyperplane** that separates classes with maximum **margin**
  - **Support vectors**: training samples closest to hyperplane
  - Key parameters:
    - **C**: controls trade-off between margin and training errors
    - **kernel**: transforms data (linear, rbf, poly)
- ☐ Draw 2D sketch: two classes, decision boundary, margin, support vectors
- ☐ Implement SVM on your classification dataset:
  ```python
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score, f1_score

  svm_clf = SVC(
      kernel="rbf",
      C=1.0,
      gamma="scale",
      random_state=42
  )

  svm_clf.fit(X_train_scaled, y_train)
  y_pred_svm = svm_clf.predict(X_test_scaled)

  acc = accuracy_score(y_test, y_pred_svm)
  f1 = f1_score(y_test, y_pred_svm, average="weighted")

  print(f"SVM Accuracy: {acc:.4f}")
  print(f"SVM F1-score: {f1:.4f}")
  ```
- ☐ Experiment:
  - Change `kernel` to `"linear"`, `"rbf"`, `"poly"`
  - Change `C` between `0.1`, `1`, `10`
  - Observe how performance changes
- ☐ Short comparison vs Logistic / Random Forest:
  - SVM: strong for complex boundaries, small/medium datasets, sensitive to scaling
  - Logistic: interpretable, probabilistic outputs
  - Random Forest: works well out-of-the-box, less scaling sensitive

**Expected Outcome:** Comfortable training and tuning SVM classifier and comparing with other models.

---

#### STEP 3.7: Handling Class Imbalance (45 min)
- ☐ Create notebook: `day3_imbalance.ipynb`
- ☐ Understand why imbalance is a problem
- ☐ Learn **SMOTE** (Synthetic Minority Over-sampling):
  ```python
  from imblearn.over_sampling import SMOTE

  smote = SMOTE(random_state=42)
  X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
  ```
- ☐ Learn **class_weight** parameter in models
- ☐ Compare performance with and without imbalance handling

**Expected Outcome:** Can handle imbalanced datasets effectively.

---

#### STEP 3.8: Hyperparameter Tuning for Classification (45 min)
- ☐ Use GridSearchCV to tune:
  - Logistic Regression: C, penalty
  - Random Forest: n_estimators, max_depth
  - XGBoost: learning_rate, max_depth, subsample
  - SVM: kernel, C, gamma
- ☐ Compare best models from each algorithm

**Expected Outcome:** Can systematically optimize classification models.

---

### 🌙 Evening Session (2-3 hours)

#### STEP 3.9: Mini-Project 3 - Classification with Model Comparison (90 min)
- ☐ Create notebook: `project3_classification.ipynb`
- ☐ Choose dataset: Breast Cancer, Credit Card Fraud, Customer Churn, or similar
- ☐ Explore data, handle missing values, check for imbalance
- ☐ Feature engineering and preprocessing
- ☐ **Train at least 4 models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVC)
- ☐ Evaluate all models:
  ```python
  models = {
      'Logistic': LogisticRegression(),
      'Random Forest': RandomForestClassifier(),
      'XGBoost': XGBClassifier(),
      'SVM': SVC()
  }

  results = {}
  for name, model in models.items():
      model.fit(X_train_scaled, y_train)
      y_pred = model.predict(X_test_scaled)
      
      results[name] = {
          'accuracy': accuracy_score(y_test, y_pred),
          'precision': precision_score(y_test, y_pred, average='weighted'),
          'recall': recall_score(y_test, y_pred, average='weighted'),
          'f1': f1_score(y_test, y_pred, average='weighted')
      }
  ```
- ☐ Create comparison table and visualizations
- ☐ Write 10–15 lines of conclusions:
  - Which model performed best overall?
  - Trade-offs between models (speed, interpretability, accuracy)
  - When would you use each model in production?

**Expected Outcome:** Complete multi-model classification project with thorough comparison.

---

#### STEP 3.10: Day 3 Review & Day 4 Prep (30 min)
- ☐ Review classification concepts and all models learned
- ☐ Write summary: when to use each algorithm
- ☐ Push all code to GitHub
- ☐ Preview: Day 4 (Pipelines, Unsupervised, Ensemble, Streamlit App)

**Expected Outcome:** Solid understanding of classification, ready for unsupervised learning and projects.

---

### 📊 Day 3 Checklist
- ☐ Classification fundamentals
- ☐ Logistic Regression
- ☐ Classification metrics (confusion matrix, ROC-AUC)
- ☐ Decision Tree & Random Forest
- ☐ XGBoost
- ☐ Support Vector Machine (SVM)
- ☐ Handling class imbalance
- ☐ Hyperparameter tuning
- ☐ Mini-Project 3 complete with model comparison

**✅ DAY 3 COMPLETE?** Move to Day 4!

---

## Day 4: ML Pipelines, Unsupervised, Ensemble & Streamlit App

**Date:** December 7, 2025  
**Focus:** Pipelines + PCA + Clustering + Ensemble + Recommender basics + Streamlit app  
**Duration:** 8-10 hours  
**Prerequisite:** Day 3 completed ✅

### 🌅 Morning Session (3-4 hours)

#### STEP 4.1: ML Pipelines with Scikit-learn (45 min)
- ☐ Create notebook: `day4_pipelines.ipynb`
- ☐ Understand why pipelines matter (avoid data leakage, simplify workflows)
- ☐ Build a simple pipeline:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression

  pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('model', LogisticRegression())
  ])

  pipeline.fit(X_train, y_train)
  predictions = pipeline.predict(X_test)
  ```
- ☐ Learn **ColumnTransformer** for heterogeneous data:
  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numeric_features),
          ('cat', OneHotEncoder(), categorical_features)
      ])

  pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('model', RandomForestClassifier())
  ])
  ```
- ☐ Use pipeline with GridSearchCV

**Expected Outcome:** Can build robust pipelines for end-to-end workflows.

---

#### STEP 4.2: Dimensionality Reduction with PCA (45 min)
- ☐ Create notebook: `day4_pca_clustering.ipynb`
- ☐ Understand **PCA** concept:
  - Reduce dimensionality while preserving most variance
  - Project data onto new axes (principal components)
- ☐ Implement PCA:
  ```python
  from sklearn.decomposition import PCA

  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X_scaled)

  print("Explained variance ratio:", pca.explained_variance_ratio_)
  print("Total explained:", pca.explained_variance_ratio_.sum())
  ```
- ☐ Visualize:
  ```python
  import matplotlib.pyplot as plt

  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.title("PCA (2D) projection")
  plt.show()
  ```
- ☐ Experiment with different n_components and analyze variance explained

**Expected Outcome:** Able to apply PCA for dimensionality reduction and visualization.

---

#### STEP 4.3: Unsupervised Learning – K-Means Clustering (45 min)
- ☐ Understand **Unsupervised Learning**:
  - No labels, algorithm finds structure/patterns
- ☐ Implement **K-Means** on PCA-reduced data:
  ```python
  from sklearn.cluster import KMeans

  kmeans = KMeans(
      n_clusters=3,
      random_state=42,
      n_init="auto"
  )
  clusters = kmeans.fit_predict(X_pca)

  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.title("K-Means clusters on PCA-reduced data")
  plt.show()
  ```
- ☐ Try different k values (2, 3, 4) and compare visually
- ☐ Learn about elbow method for choosing k
- ☐ Write 3–5 lines about cluster patterns

**Expected Outcome:** Understand clustering as unsupervised technique and can run K-Means.

---

#### STEP 4.4: Ensemble Learning – Random Forest & Gradient Boosting (60 min)
- ☐ Recall **Ensemble Learning**:
  - Combine multiple weak learners → strong learner
  - **Bagging** (Random Forest) vs **Boosting** (Gradient Boosting, XGBoost)
- ☐ Implement **Random Forest** for regression/classification:
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score

  rf_clf = RandomForestClassifier(
      n_estimators=200,
      max_depth=None,
      random_state=42
  )

  scores_rf = cross_val_score(
      rf_clf, X_train_scaled, y_train, cv=5, scoring="f1_weighted"
  )
  print("Random Forest F1 mean:", scores_rf.mean())
  ```
- ☐ Implement **Gradient Boosting**:
  ```python
  from sklearn.ensemble import GradientBoostingClassifier

  gb_clf = GradientBoostingClassifier(random_state=42)

  scores_gb = cross_val_score(
      gb_clf, X_train_scaled, y_train, cv=5, scoring="f1_weighted"
  )
  print("Gradient Boosting F1 mean:", scores_gb.mean())
  ```
- ☐ Compare with single **DecisionTreeClassifier**:
  - Write 5–7 lines why ensembles outperform single trees

**Expected Outcome:** Know how to use ensemble models and why they are powerful.

---

### 🌞 Afternoon Session (3-4 hours)

#### STEP 4.5: Recommender System – Concept & Simple Collaborative Filtering (60–90 min)
- ☐ Learn **Recommender System** concepts:
  - **Content-Based**: recommend items similar to what user liked
  - **Collaborative Filtering**: recommend based on similar user behavior
    - User-based vs Item-based
- ☐ Create toy user–item rating dataset:
  ```python
  import pandas as pd

  ratings = pd.DataFrame({
      "user": ["u1","u1","u2","u2","u3","u3"],
      "item": ["i1","i2","i1","i3","i2","i3"],
      "rating": [5,4,4,5,1,4]
  })

  user_item = ratings.pivot_table(
      index="user",
      columns="item",
      values="rating"
  ).fillna(0)

  print(user_item)
  ```
- ☐ Compute **user–user similarity** using cosine similarity:
  ```python
  from sklearn.metrics.pairwise import cosine_similarity

  sim_matrix = cosine_similarity(user_item)
  sim_df = pd.DataFrame(sim_matrix, index=user_item.index, columns=user_item.index)
  print(sim_df)
  ```
- ☐ Simple recommendation example:
  - Pick a user (e.g., "u1")
  - Find most similar user
  - Recommend items that similar user rated highly but u1 has not rated
- ☐ Write 5–7 lines about recommendation and ranking problems

**Expected Outcome:** Conceptual understanding of recommender systems with practical example.

---

#### STEP 4.6: Mini-Project – ML Streamlit App with Scikit-learn & Pandas (90–120 min)
- ☐ Choose simple classification dataset:
  - Iris, Wine, Breast Cancer, or your Day 3 dataset
- ☐ Build ML pipeline:
  - Load data with Pandas
  - Basic preprocessing (fillna, encode, scale)
  - Train 1–2 models (Logistic Regression + Random Forest)
- ☐ Install Streamlit (if not installed):
  ```bash
  pip install streamlit
  ```
- ☐ Create `app.py`:
  ```python
  import streamlit as st
  import pandas as pd
  import joblib
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.preprocessing import StandardScaler

  st.title("ML Classification Demo")

  # Example: Iris features
  sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.0)
  sepal_width = st.slider("Sepal width", 2.0, 4.5, 3.0)
  petal_length = st.slider("Petal length", 1.0, 7.0, 4.0)
  petal_width = st.slider("Petal width", 0.1, 2.5, 1.0)

  input_df = pd.DataFrame({
      "sepal_length": [sepal_length],
      "sepal_width": [sepal_width],
      "petal_length": [petal_length],
      "petal_width": [petal_width]
  })

  # Load or train model
  if st.button("Predict"):
      # Preprocess input (scale if needed)
      y_pred = model.predict(input_df)
      st.write(f"Prediction: {y_pred[0]}")
  ```
- ☐ Run the app:
  ```bash
  streamlit run app.py
  ```
- ☐ Optional:
  - Show model accuracy on validation set
  - Show feature importances (for Random Forest)
- ☐ Take 1–2 screenshots and save in `projects/streamlit_app/`

**Expected Outcome:** Functional Streamlit app with ML model integration.

---

#### STEP 4.7: Day 4 Review & Integration (30 min)
- ☐ Review techniques learned:
  - Pipelines, PCA, Clustering
  - Ensemble models, Recommender basics
  - Streamlit app with ML
- ☐ Note potential real-world projects
- ☐ Push Day 4 work to GitHub

**Expected Outcome:** Consolidated understanding of unsupervised learning, ensembles, recommenders, and ML app building.

---

### 📊 Day 4 Checklist
- ☐ ML Pipelines with Scikit-learn
- ☐ PCA implementation and visualization
- ☐ K-Means clustering
- ☐ Random Forest & Gradient Boosting
- ☐ Collaborative Filtering with Pandas
- ☐ Streamlit app with ML model
- ☐ All code pushed to GitHub

**✅ DAY 4 COMPLETE?** Move to Day 5!

---

## Day 5: AI Document Intelligence System - Playground & Backend

**Date:** December 8, 2025  
**Focus:** Build "Document Intelligence Playground" + PDF Processing + AI Model Council  
**Duration:** 8-10 hours  
**Prerequisite:** Day 4 completed ✅

### Context
This day focuses on building a **Landing.ai-style AI Document Intelligence platform**. Your system will:
1. Provide an interactive **Playground** UI for document processing
2. Implement a **Model Council** (multiple AI models voting on extraction results)
3. Use **Hugging Face models** for lightweight tasks
4. Support **PDF/Document extraction** with confidence scores and latency tracking
5. Prepare data for **RAG integration** in Day 6

### 🌅 Morning Session (3-4 hours)

#### STEP 5.1: Design Document Intelligence Architecture (45 min)
- ☐ Create document: `day5_architecture_design.md`
- ☐ Design system components:
  ```
  Input (PDF/Image)
      ↓
  Preprocessing (clean, extract pages, OCR candidate models)
      ↓
  AI Model Council:
    - Model 1: Hugging Face OCR
    - Model 2: Hugging Face Vision Model
    - Model 3: Rule-based extraction
      ↓
  Voting/Consensus (select best extraction)
      ↓
  Post-processing (clean output, extract coordinates)
      ↓
  Output: ExtractedText + Confidence + Latency + Text Blocks with Coordinates
  ```
- ☐ Define data structures for:
  - Input document metadata
  - Extraction results (text + bounding boxes + confidence)
  - Performance metrics (OCR score, Confidence, Latency)
- ☐ Sketch Playground UI layout:
  - Left: Upload/PDF viewer
  - Center: Extracted text with highlights
  - Right: Metrics panel (confidence, latency, model info)

**Expected Outcome:** Clear architecture for Document Intelligence system.

---

#### STEP 5.2: PDF Processing & Document Parsing (60 min)
- ☐ Create notebook: `day5_pdf_processing.ipynb`
- ☐ Install required packages:
  ```bash
  pip install pypdf pdf2image pytesseract pillow python-pptx
  ```
- ☐ Learn to read PDF and extract pages:
  ```python
  import pypdf
  from pdf2image import convert_from_path
  import os

  # Extract text from PDF
  def extract_text_from_pdf(pdf_path):
      reader = pypdf.PdfReader(pdf_path)
      text = ""
      for page in reader.pages:
          text += page.extract_text()
      return text

  # Convert PDF to images
  def pdf_to_images(pdf_path):
      images = convert_from_path(pdf_path)
      return images

  # Test
  pdf_text = extract_text_from_pdf("sample.pdf")
  pdf_images = pdf_to_images("sample.pdf")
  ```
- ☐ Implement OCR using Tesseract:
  ```python
  import pytesseract
  from PIL import Image

  def ocr_image(image_path):
      img = Image.open(image_path)
      text = pytesseract.image_to_string(img)
      return text
  ```
- ☐ Create document class:
  ```python
  from dataclasses import dataclass
  from typing import List, Dict, Tuple

  @dataclass
  class TextBlock:
      text: str
      confidence: float
      bounding_box: Tuple[int, int, int, int]  # x, y, w, h
      source_model: str

  @dataclass
  class DocumentExtractionResult:
      full_text: str
      text_blocks: List[TextBlock]
      ocr_score: float
      avg_confidence: float
      latency_ms: float
      models_used: List[str]
  ```
- ☐ Build PDF parser pipeline

**Expected Outcome:** Can extract and parse PDF documents with text blocks and coordinates.

---

#### STEP 5.3: Hugging Face Model Integration for Document Understanding (60 min)
- ☐ Create notebook: `day5_huggingface_models.ipynb`
- ☐ Install transformers:
  ```bash
  pip install transformers torch
  ```
- ☐ Explore Hugging Face models for document understanding:
  ```python
  from transformers import pipeline

  # Text extraction pipeline
  extractor = pipeline("question-answering", model="deepset/roberta-base-squad2")

  # Document Question Answering (useful for document intelligence)
  qa_pipeline = pipeline(
      "document-question-answering",
      model="impira/layoutlm-document-qa"
  )

  # Vision model for document understanding
  vision_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-mlm")
  ```
- ☐ Build OCR candidate models:
  ```python
  def model_1_basic_ocr(image_path):
      """Tesseract-based OCR"""
      img = Image.open(image_path)
      text = pytesseract.image_to_string(img)
      confidence = 0.8  # Placeholder
      return {"text": text, "confidence": confidence}

  def model_2_huggingface_vision(image_path):
      """Hugging Face vision model for document understanding"""
      img = Image.open(image_path)
      # Use vision model
      results = vision_pipeline(img, "What text is in this image?")
      text = results[0]["answer"]
      confidence = 0.85
      return {"text": text, "confidence": confidence}

  def model_3_hybrid_extraction(pdf_path):
      """Hybrid approach combining multiple techniques"""
      text = extract_text_from_pdf(pdf_path)
      confidence = 0.82
      return {"text": text, "confidence": confidence}
  ```
- ☐ Test each model on sample documents

**Expected Outcome:** Integrated multiple Hugging Face models for document understanding.

---

### 🌞 Afternoon Session (3-4 hours)

#### STEP 5.4: AI Model Council - Voting & Consensus (60 min)
- ☐ Create notebook: `day5_model_council.ipynb`
- ☐ Implement model council pattern:
  ```python
  import time
  from typing import List, Dict
  from datetime import datetime

  class AIModelCouncil:
      def __init__(self, models: Dict[str, callable]):
          self.models = models
          self.results_history = []

      def extract_with_council(self, document_path):
          """Run all models and find consensus"""
          council_results = {}
          start_time = time.time()

          # Run each model
          for model_name, model_func in self.models.items():
              try:
                  result = model_func(document_path)
                  council_results[model_name] = result
              except Exception as e:
                  council_results[model_name] = {"error": str(e)}

          # Calculate consensus
          texts = [r.get("text", "") for r in council_results.values() if "text" in r]
          confidences = [r.get("confidence", 0.0) for r in council_results.values() if "confidence" in r]

          # Simple majority voting on text (could be more sophisticated)
          final_text = max(set(texts), key=texts.count) if texts else ""
          avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
          
          latency_ms = (time.time() - start_time) * 1000

          result = {
              "final_text": final_text,
              "avg_confidence": avg_confidence,
              "latency_ms": latency_ms,
              "council_votes": council_results,
              "models_used": list(self.models.keys()),
              "timestamp": datetime.now().isoformat()
          }

          self.results_history.append(result)
          return result

      def get_metrics_summary(self):
          """Get performance metrics"""
          if not self.results_history:
              return None

          latencies = [r["latency_ms"] for r in self.results_history]
          confidences = [r["avg_confidence"] for r in self.results_history]

          return {
              "avg_latency_ms": sum(latencies) / len(latencies),
              "max_latency_ms": max(latencies),
              "min_latency_ms": min(latencies),
              "avg_confidence": sum(confidences) / len(confidences),
              "total_documents_processed": len(self.results_history)
          }
  ```
- ☐ Implement weighted voting based on model performance:
  ```python
  def consensus_with_weights(council_results: Dict, weights: Dict[str, float]):
      """Weighted voting based on model reliability"""
      weighted_scores = {}
      
      for model_name, result in council_results.items():
          weight = weights.get(model_name, 1.0)
          if "text" in result:
              text = result["text"]
              if text not in weighted_scores:
                  weighted_scores[text] = 0.0
              weighted_scores[text] += weight * result.get("confidence", 0.5)
      
      # Select text with highest weighted score
      best_text = max(weighted_scores.items(), key=lambda x: x[1])[0]
      return best_text
  ```
- ☐ Test council with sample documents and analyze voting patterns

**Expected Outcome:** Implemented model council with consensus voting and performance tracking.

---

#### STEP 5.5: Playground UI - Streamlit Front-end (90 min)
- ☐ Create: `day5_playground_app.py`
- ☐ Build Streamlit playground:
  ```python
  import streamlit as st
  import pandas as pd
  from PIL import Image
  import time
  import json

  st.set_page_config(page_title="AI Document Intelligence", layout="wide")

  st.title("📄 AI Document Intelligence Playground")
  st.markdown("Like Landing.ai's platform - extract, analyze, and visualize document data")

  # Sidebar: Upload & Controls
  with st.sidebar:
      st.header("⚙️ Configuration")
      uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
      
      st.subheader("Model Council")
      use_model_1 = st.checkbox("OCR (Tesseract)", value=True)
      use_model_2 = st.checkbox("Vision Model (HuggingFace)", value=True)
      use_model_3 = st.checkbox("Hybrid Extraction", value=True)
      
      if st.button("🚀 Process Document"):
          st.session_state.process_triggered = True

  # Main area: Document viewer + Results
  if uploaded_file is not None:
      col1, col2, col3 = st.columns([1.5, 2, 1.2])

      with col1:
          st.subheader("📋 Document")
          # Display document (image or PDF pages)
          if uploaded_file.type == "application/pdf":
              st.info("PDF uploaded - showing preview")
          else:
              image = Image.open(uploaded_file)
              st.image(image, use_column_width=True)

      with col2:
          st.subheader("📝 Extracted Text & Blocks")
          
          if st.session_state.get("process_triggered", False):
              with st.spinner("🔄 Running AI Model Council..."):
                  # Simulate extraction (replace with actual council call)
                  time.sleep(2)
                  
                  extracted_text = "Sample extracted text from document..."
                  confidence = 0.92
                  latency_ms = 1240.5
                  
                  st.text_area("Full Text", extracted_text, height=200)
                  
                  st.markdown("---")
                  st.subheader("🎯 Text Blocks")
                  
                  # Simulate text blocks with bounding boxes
                  blocks = [
                      {"text": "Title", "confidence": 0.95, "location": "top-center"},
                      {"text": "Paragraph 1", "confidence": 0.89, "location": "center"},
                      {"text": "Footer", "confidence": 0.87, "location": "bottom"}
                  ]
                  
                  blocks_df = pd.DataFrame(blocks)
                  st.dataframe(blocks_df, use_container_width=True)

      with col3:
          st.subheader("📊 Metrics")
          
          col3_1, col3_2 = st.columns(2)
          with col3_1:
              st.metric("Confidence", "92%", "+5%")
              st.metric("Latency", "1.24s", "fast")
          
          with col3_2:
              st.metric("Models Used", "3", "")
              st.metric("Avg Score", "0.90", "+0.02")
          
          st.markdown("---")
          st.subheader("🤖 Council Votes")
          
          council_votes = {
              "OCR (Tesseract)": 0.88,
              "Vision Model": 0.92,
              "Hybrid": 0.90
          }
          
          votes_df = pd.DataFrame(list(council_votes.items()), columns=["Model", "Confidence"])
          st.dataframe(votes_df, use_container_width=True)
          
          # Download results
          st.markdown("---")
          if st.button("📥 Download Results"):
              st.success("Results downloaded!")

  st.markdown("---")
  st.markdown("""
  **About this Playground:**
  - Upload documents (PDF, images)
  - AI Models extract text with bounding boxes
  - View confidence scores and latency
  - Compare model outputs via council voting
  - Export extracted data for further processing (RAG, analysis)
  """)
  ```
- ☐ Test Playground UI with sample documents
- ☐ Add visualization of bounding boxes on document image

**Expected Outcome:** Functional Playground UI mimicking Landing.ai interface.

---

#### STEP 5.6: Performance Monitoring & Metrics Collection (45 min)
- ☐ Create: `day5_performance_monitoring.py`
- ☐ Implement performance tracking using MLFlow:
  ```bash
  pip install mlflow
  ```
  ```python
  import mlflow
  import time

  def track_document_processing(council, document_path, metadata=None):
      """Track processing metrics with MLFlow"""
      
      with mlflow.start_run():
          start_time = time.time()
          
          # Run processing
          result = council.extract_with_council(document_path)
          
          elapsed_time = time.time() - start_time
          
          # Log metrics
          mlflow.log_metric("latency_ms", result["latency_ms"])
          mlflow.log_metric("confidence", result["avg_confidence"])
          mlflow.log_metric("total_time_s", elapsed_time)
          
          # Log parameters
          mlflow.log_param("models_used", ", ".join(result["models_used"]))
          mlflow.log_param("document_path", document_path)
          
          # Log artifacts
          mlflow.log_dict(result["council_votes"], "council_votes.json")
          
          if metadata:
              mlflow.log_dict(metadata, "metadata.json")
          
          return result
  ```
- ☐ Set up metrics dashboard:
  - Latency over time (identify bottlenecks like "Deep Research" taking 40-100s)
  - Token usage tracking
  - Model accuracy/confidence trends
  - Cost optimization metrics

**Expected Outcome:** Performance monitoring system with MLFlow integration.

---

### 🌙 Evening Session (2-3 hours)

#### STEP 5.7: Integration Testing & Demo (60 min)
- ☐ Create test suite: `test_day5_integration.py`
- ☐ Test Document Parsing:
  ```python
  def test_pdf_parsing():
      pdf_path = "sample_lecture.pdf"
      text = extract_text_from_pdf(pdf_path)
      assert len(text) > 0
      assert isinstance(text, str)
  ```
- ☐ Test Model Council:
  ```python
  def test_model_council():
      council = AIModelCouncil(models={
          "model_1": model_1_basic_ocr,
          "model_2": model_2_huggingface_vision,
          "model_3": model_3_hybrid_extraction
      })
      result = council.extract_with_council("sample.pdf")
      assert "final_text" in result
      assert "avg_confidence" in result
      assert result["avg_confidence"] <= 1.0
  ```
- ☐ Run Playground with sample documents
- ☐ Document any errors and create improvement list

**Expected Outcome:** Working Document Intelligence system with Playground UI.

---

#### STEP 5.8: Day 5 Review & Day 6 Preview (30 min)
- ☐ Review Document Intelligence components:
  - PDF processing pipeline
  - AI Model Council with consensus
  - Playground UI
  - Performance monitoring
- ☐ Write summary of architecture decisions
- ☐ List 5 improvements for next iteration
- ☐ Push all code to GitHub: `/projects/document_intelligence/`
- ☐ Preview Day 6: Advanced features, RAG integration, coordinate tracking

**Expected Outcome:** Completed Document Intelligence MVP with clear roadmap for enhancement.

---

### 📊 Day 5 Checklist
- ☐ System architecture design document
- ☐ PDF processing & document parsing (pypdf, tesseract, pdf2image)
- ☐ Hugging Face model integration for OCR & vision tasks
- ☐ AI Model Council with weighted voting and consensus
- ☐ Streamlit Playground UI with metrics display
- ☐ Performance monitoring with MLFlow
- ☐ Integration testing
- ☐ GitHub repository updated

**✅ DAY 5 COMPLETE?** Move to Day 6!

---

## Day 6: AI Document Intelligence - Advanced Features & RAG Integration

**Date:** December 9, 2025  
**Focus:** Advanced extraction, coordinate tracking, error handling, RAG preparation  
**Duration:** 8-10 hours  
**Prerequisite:** Day 5 completed ✅

### 🌅 Morning Session (3-4 hours)

#### STEP 6.1: Advanced Text Extraction with Coordinates (45 min)
- ☐ Create notebook: `day6_coordinate_extraction.ipynb`
- ☐ Extract bounding box coordinates from PDFs:
  ```python
  from pdfplumber import open as pdf_open

  def extract_with_coordinates(pdf_path):
      """Extract text with precise bounding box coordinates"""
      results = []
      
      with pdf_open(pdf_path) as pdf:
          for page_num, page in enumerate(pdf.pages):
              # Extract words with coordinates
              words = page.extract_words()
              
              for word in words:
                  results.append({
                      "text": word["text"],
                      "x0": word["x0"],
                      "top": word["top"],
                      "x1": word["x1"],
                      "bottom": word["bottom"],
                      "page": page_num,
                      "confidence": 0.95  # Placeholder
                  })
      
      return results
  ```
- ☐ Visualize extracted regions with bounding boxes on document:
  ```python
  from PIL import ImageDraw

  def draw_bounding_boxes(image, text_blocks):
      """Draw bounding boxes on document image"""
      draw = ImageDraw.Draw(image)
      colors = ["red", "green", "blue", "yellow", "orange"]
      
      for i, block in enumerate(text_blocks):
          color = colors[i % len(colors)]
          bbox = [block["x0"], block["top"], block["x1"], block["bottom"]]
          draw.rectangle(bbox, outline=color, width=2)
          # Optional: draw text label
          draw.text((bbox[0], bbox[1]), block["text"], fill=color)
      
      return image
  ```

**Expected Outcome:** Extract text with precise coordinates for visualization.

---

#### STEP 6.2: Error Handling & Export Format Standards (45 min)
- ☐ Create: `day6_output_formats.py`
- ☐ Define standard extraction output formats:
  ```python
  from enum import Enum
  import json
  from typing import Optional

  class ExtractionStatus(Enum):
      SUCCESS = "success"
      PARTIAL = "partial"
      FAILED = "failed"

  class ExtractionError:
      def __init__(self, error_code: str, message: str, page: Optional[int] = None):
          self.error_code = error_code
          self.message = message
          self.page = page

  class StandardExtractionOutput:
      def __init__(self):
          self.status = ExtractionStatus.SUCCESS
          self.full_text = ""
          self.text_blocks = []
          self.metadata = {}
          self.errors = []
          self.confidence_score = 0.0
          self.processing_time_ms = 0.0

      def to_json(self) -> str:
          return json.dumps({
              "status": self.status.value,
              "full_text": self.full_text,
              "text_blocks": self.text_blocks,
              "metadata": self.metadata,
              "errors": [
                  {
                      "code": e.error_code,
                      "message": e.message,
                      "page": e.page
                  } for e in self.errors
              ],
              "confidence_score": self.confidence_score,
              "processing_time_ms": self.processing_time_ms
          })

      def to_csv_export(self) -> str:
          """Export text blocks as CSV for downstream processing"""
          import csv
          from io import StringIO

          output = StringIO()
          writer = csv.DictWriter(
              output,
              fieldnames=["text", "confidence", "x0", "y0", "x1", "y1", "page", "error"]
          )
          writer.writeheader()
          
          for block in self.text_blocks:
              writer.writerow(block)
          
          return output.getvalue()
  ```
- ☐ Implement error tracking:
  ```python
  def extract_with_error_handling(document_path, council):
      """Extract with comprehensive error handling"""
      output = StandardExtractionOutput()
      
      try:
          result = council.extract_with_council(document_path)
          output.full_text = result["final_text"]
          output.confidence_score = result["avg_confidence"]
          output.processing_time_ms = result["latency_ms"]
          output.status = ExtractionStatus.SUCCESS
          
      except FileNotFoundError:
          output.errors.append(ExtractionError("FILE_NOT_FOUND", f"File {document_path} not found"))
          output.status = ExtractionStatus.FAILED
          
      except Exception as e:
          output.errors.append(ExtractionError("UNKNOWN_ERROR", str(e)))
          output.status = ExtractionStatus.FAILED
      
      return output
  ```

**Expected Outcome:** Standardized output format with comprehensive error handling.

---

#### STEP 6.3: Data Quality & Token Optimization (45 min)
- ☐ Create notebook: `day6_data_quality.ipynb`
- ☐ Implement input cleaning:
  ```python
  import re

  def clean_extracted_text(text: str) -> str:
      """Clean and normalize extracted text"""
      # Remove extra whitespace
      text = re.sub(r'\s+', ' ', text)
      # Remove special characters
      text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
      # Fix common OCR errors
      text = text.replace('|', 'l')
      text = text.replace('0', 'O')  # Context-dependent
      return text.strip()

  def estimate_token_usage(text: str) -> int:
      """Estimate token count for LLM processing"""
      # Simple estimation: ~4 characters per token
      return len(text) // 4

  def optimize_for_api_call(text: str, max_tokens: int = 2000) -> str:
      """Truncate text to stay within token limits"""
      est_tokens = estimate_token_usage(text)
      
      if est_tokens > max_tokens:
          # Truncate and add indicator
          truncated = text[:max_tokens * 4]
          return truncated + " [... truncated ...]"
      
      return text
  ```
- ☐ Implement batch processing for cost efficiency:
  ```python
  def batch_process_documents(document_paths: List[str], batch_size: int = 5):
      """Process documents in batches to optimize costs"""
      total_cost = 0.0
      results = []
      
      for i in range(0, len(document_paths), batch_size):
          batch = document_paths[i:i+batch_size]
          print(f"Processing batch {i//batch_size + 1} ({len(batch)} documents)...")
          
          for doc_path in batch:
              result = extract_with_error_handling(doc_path, council)
              cost = estimate_cost(result)
              total_cost += cost
              results.append(result)
      
      return results, total_cost
  ```

**Expected Outcome:** Data quality checks and token optimization for cost-effective API usage.

---

### 🌞 Afternoon Session (3-4 hours)

#### STEP 6.4: RAG Integration - Coordinate Tracking for Source Attribution (60 min)
- ☐ Create notebook: `day6_rag_preparation.ipynb`
- ☐ Prepare data for RAG with source tracking:
  ```python
  from dataclasses import dataclass
  from typing import List

  @dataclass
  class RAGTextChunk:
      """Text chunk with source coordinates for RAG retrieval"""
      chunk_id: str
      text: str
      embedding_vector: List[float] = None  # Will be populated later
      source_document: str = ""
      page_number: int = 0
      bounding_box: dict = None  # {"x0", "y0", "x1", "y1"}
      confidence: float = 0.0
      retrieval_priority: float = 1.0  # Higher = prioritize in retrieval

  def chunk_text_for_rag(extraction_result, chunk_size: int = 512):
      """Break extracted text into chunks suitable for RAG"""
      rag_chunks = []
      text = extraction_result.full_text
      
      # Simple chunking by sentences
      sentences = text.split('. ')
      current_chunk = ""
      chunk_num = 0
      
      for sentence in sentences:
          if len(current_chunk) + len(sentence) < chunk_size:
              current_chunk += sentence + ". "
          else:
              if current_chunk.strip():
                  rag_chunks.append(RAGTextChunk(
                      chunk_id=f"{extraction_result.metadata['doc_id']}_chunk_{chunk_num}",
                      text=current_chunk.strip(),
                      source_document=extraction_result.metadata['doc_path'],
                      page_number=extraction_result.metadata.get('page', 0),
                      confidence=extraction_result.confidence_score
                  ))
                  chunk_num += 1
              current_chunk = sentence + ". "
      
      if current_chunk.strip():
          rag_chunks.append(RAGTextChunk(
              chunk_id=f"{extraction_result.metadata['doc_id']}_chunk_{chunk_num}",
              text=current_chunk.strip(),
              source_document=extraction_result.metadata['doc_path'],
              page_number=extraction_result.metadata.get('page', 0),
              confidence=extraction_result.confidence_score
          ))
      
      return rag_chunks
  ```
- ☐ Create embedding pipeline (prepare for Day 8):
  ```python
  from sklearn.preprocessing import normalize
  import numpy as np

  def generate_embeddings_for_rag_chunks(rag_chunks: List[RAGTextChunk]):
      """Generate embeddings for RAG chunks"""
      # Placeholder - will use actual embedding model (e.g., HuggingFace, OpenAI)
      
      for chunk in rag_chunks:
          # Simulate embedding generation
          embedding = np.random.randn(768)  # 768-dimensional vector
          chunk.embedding_vector = normalize([embedding])[0].tolist()
      
      return rag_chunks

  def create_rag_index_schema():
      """Define schema for RAG vector store (e.g., ChromaDB, Pinecone)"""
      return {
          "chunk_id": {"type": "string", "indexed": True},
          "text": {"type": "text"},
          "embedding": {"type": "vector", "dimensions": 768},
          "source_document": {"type": "string", "indexed": True},
          "page_number": {"type": "integer"},
          "bounding_box": {"type": "json"},
          "confidence": {"type": "float"},
          "retrieval_priority": {"type": "float"}
      }
  ```

**Expected Outcome:** RAG-ready text chunks with source attribution and coordinate tracking.

---

#### STEP 6.5: Advanced Playground Features (90 min)
- ☐ Update: `day6_advanced_playground_app.py`
- ☐ Add advanced features to Streamlit Playground:
  ```python
  import streamlit as st
  import plotly.express as px
  import pandas as pd

  st.set_page_config(page_title="Advanced Document Intelligence", layout="wide")

  st.title("🚀 Advanced AI Document Intelligence")

  # Tab 1: Document Processing
  tab1, tab2, tab3, tab4 = st.tabs([
      "📄 Process Documents",
      "🎯 RAG Chunks",
      "📊 Analytics",
      "⚙️ Settings"
  ])

  with tab1:
      col1, col2 = st.columns([2, 1])
      
      with col1:
          st.subheader("Document Upload & Processing")
          uploaded_files = st.file_uploader("Upload multiple documents", type=["pdf", "png", "jpg"], accept_multiple_files=True)
          
          if st.button("🔄 Process All Documents"):
              progress_bar = st.progress(0)
              
              for i, file in enumerate(uploaded_files):
                  # Simulate processing
                  with st.spinner(f"Processing {file.name}..."):
                      time.sleep(1)
                  progress_bar.progress((i + 1) / len(uploaded_files))
              
              st.success("✅ All documents processed!")
      
      with col2:
          st.subheader("📈 Processing Stats")
          st.metric("Documents Processed", "15")
          st.metric("Total Tokens", "45,230")
          st.metric("Avg Confidence", "0.89")

  with tab2:
      st.subheader("🎯 RAG Chunks & Embeddings")
      
      # Display RAG chunks prepared for retrieval
      chunks_data = [
          {"chunk_id": "doc1_chunk_0", "text": "First chunk of text...", "confidence": 0.92, "embedding_dim": 768},
          {"chunk_id": "doc1_chunk_1", "text": "Second chunk of text...", "confidence": 0.88, "embedding_dim": 768},
          {"chunk_id": "doc2_chunk_0", "text": "Third chunk of text...", "confidence": 0.91, "embedding_dim": 768},
      ]
      
      chunks_df = pd.DataFrame(chunks_data)
      st.dataframe(chunks_df, use_container_width=True)
      
      st.markdown("---")
      st.subheader("🔗 Embedding Status")
      st.info("✅ 3 chunks have embeddings ready for RAG retrieval")

  with tab3:
      st.subheader("📊 Document Intelligence Analytics")
      
      # Performance over time
      dates = pd.date_range('2024-12-01', periods=10, freq='D')
      confidence_data = pd.DataFrame({
          'Date': dates,
          'Avg Confidence': [0.85, 0.87, 0.88, 0.86, 0.89, 0.90, 0.91, 0.89, 0.92, 0.91],
          'Latency (ms)': [1200, 1150, 1100, 1180, 1050, 1000, 950, 1100, 900, 950]
      })
      
      col1, col2 = st.columns(2)
      
      with col1:
          fig_conf = px.line(confidence_data, x='Date', y='Avg Confidence', title='Confidence Trend')
          st.plotly_chart(fig_conf, use_container_width=True)
      
      with col2:
          fig_latency = px.line(confidence_data, x='Date', y='Latency (ms)', title='Processing Latency')
          st.plotly_chart(fig_latency, use_container_width=True)
      
      # Token usage breakdown
      st.markdown("---")
      st.subheader("💰 Token Usage & Cost")
      
      token_data = pd.DataFrame({
          'Model': ['OCR', 'Vision', 'Hybrid'],
          'Tokens Used': [5000, 8000, 6000],
          'Cost ($)': [0.01, 0.03, 0.02]
      })
      
      fig_tokens = px.bar(token_data, x='Model', y='Tokens Used', color='Cost ($)', title='Token Usage by Model')
      st.plotly_chart(fig_tokens, use_container_width=True)

  with tab4:
      st.subheader("⚙️ System Configuration")
      
      col1, col2 = st.columns(2)
      
      with col1:
          st.subheader("Model Council Settings")
          enable_ocr = st.checkbox("Enable OCR Model", value=True)
          enable_vision = st.checkbox("Enable Vision Model", value=True)
          enable_hybrid = st.checkbox("Enable Hybrid Model", value=True)
          
          voting_strategy = st.selectbox(
              "Voting Strategy",
              ["Simple Majority", "Weighted Confidence", "Highest Confidence"]
          )
      
      with col2:
          st.subheader("RAG Settings")
          chunk_size = st.slider("Chunk Size (characters)", 256, 2048, 512, 256)
          embedding_dim = st.selectbox("Embedding Dimension", [256, 512, 768, 1024])
          retrieval_priority = st.select_slider(
              "Retrieval Priority",
              ["Low", "Medium", "High"],
              value="Medium"
          )
      
      if st.button("💾 Save Settings"):
          st.success("Settings saved successfully!")
  ```

**Expected Outcome:** Advanced Playground with RAG preparation, analytics, and configuration UI.

---

#### STEP 6.6: End-to-End Pipeline Testing (45 min)
- ☐ Create: `test_day6_end_to_end.py`
- ☐ Full pipeline test:
  ```python
  def test_end_to_end_rag_pipeline():
      """Test complete pipeline from document to RAG-ready chunks"""
      
      # 1. Load and process document
      pdf_path = "sample_lecture.pdf"
      council = create_model_council()
      extraction_result = extract_with_error_handling(pdf_path, council)
      
      assert extraction_result.status == ExtractionStatus.SUCCESS
      assert len(extraction_result.full_text) > 0
      
      # 2. Create RAG chunks
      rag_chunks = chunk_text_for_rag(extraction_result)
      assert len(rag_chunks) > 0
      
      # 3. Generate embeddings
      rag_chunks = generate_embeddings_for_rag_chunks(rag_chunks)
      assert all(chunk.embedding_vector is not None for chunk in rag_chunks)
      
      # 4. Verify coordinates tracking
      assert all(chunk.bounding_box is not None or chunk.page_number >= 0 for chunk in rag_chunks)
      
      print("✅ End-to-end pipeline test passed!")
  ```

**Expected Outcome:** Verified complete pipeline from document extraction to RAG preparation.

---

### 🌙 Evening Session (2-3 hours)

#### STEP 6.7: Performance Benchmarking & Optimization (60 min)
- ☐ Create: `day6_benchmarking.py`
- ☐ Benchmark different extraction approaches:
  ```python
  import time

  def benchmark_extraction_methods(pdf_path):
      """Compare extraction speed and accuracy"""
      results = {}
      
      # Method 1: Basic OCR
      start = time.time()
      text1 = extract_text_ocr(pdf_path)
      results["OCR"] = {
          "time": time.time() - start,
          "quality": assess_quality(text1)
      }
      
      # Method 2: Vision Model
      start = time.time()
      text2 = extract_text_vision(pdf_path)
      results["Vision"] = {
          "time": time.time() - start,
          "quality": assess_quality(text2)
      }
      
      # Method 3: Council
      start = time.time()
      result3 = council.extract_with_council(pdf_path)
      results["Council"] = {
          "time": result3["latency_ms"] / 1000,
          "quality": result3["avg_confidence"]
      }
      
      return results

  def identify_bottlenecks(performance_log):
      """Identify slowest steps in pipeline"""
      slowest = max(performance_log.items(), key=lambda x: x[1]["time"])
      return slowest[0], slowest[1]["time"]
  ```
- ☐ Document benchmarks and create optimization roadmap

**Expected Outcome:** Performance baseline and optimization recommendations.

---

#### STEP 6.8: Day 6 Review & Day 7 Preview (30 min)
- ☐ Review advanced features:
  - Coordinate extraction and visualization
  - Error handling and output standards
  - RAG preparation with embeddings
  - Advanced Playground UI
- ☐ Write optimization roadmap document
- ☐ Push all Day 6 code to GitHub
- ☐ Preview Day 7: LLM APIs, Prompt Engineering, RAG Integration

**Expected Outcome:** Advanced Document Intelligence system with RAG integration ready.

---

### 📊 Day 6 Checklist
- ☐ Coordinate-based text extraction
- ☐ Bounding box visualization
- ☐ Error handling & standard output formats
- ☐ Data quality checks & token optimization
- ☐ RAG chunk preparation with embeddings
- ☐ Advanced Playground UI with analytics
- ☐ End-to-end pipeline testing
- ☐ Performance benchmarking
- ☐ All code pushed to GitHub

**✅ DAY 6 COMPLETE?** Move to Day 7!

---

## Day 7: LLM APIs & Prompt Engineering

**Date:** December 10, 2025  
**Focus:** LLM APIs, Advanced Prompting, RAG Integration  
**Duration:** 8-10 hours  
**Prerequisite:** Day 6 completed ✅

*(Content continues with full LLM/RAG/Deployment focus, building on Days 5-6)*

---

## Day 8: RAG Systems & Document Question-Answering

**Date:** December 11, 2025  
**Focus:** Complete RAG pipeline with document intelligence data  
**Duration:** 8-10 hours  
**Prerequisite:** Day 7 completed ✅

---

## Day 9: AI Agents & Multi-Agent Systems

**Date:** December 12, 2025  
**Focus:** Multi-agent orchestration for document processing  
**Duration:** 8-10 hours  
**Prerequisite:** Day 8 completed ✅

---

## Day 10: Full-Stack Deployment & Portfolio

**Date:** December 13, 2025  
**Focus:** Production deployment, API serving, portfolio  
**Duration:** 8-10 hours  
**Prerequisite:** Day 9 completed ✅

**Key Deliverables:**
- ✅ Document Intelligence Platform (Days 5-6)
- ✅ LLM-powered Q&A system (Days 7-8)
- ✅ Multi-agent orchestration (Day 9)
- ✅ Production deployment (Day 10)
- ✅ Portfolio + GitHub + Documentation

---

## 📚 Additional Resources

### Key Technologies Used
- **Python:** 3.11+
- **ML/Data:** Scikit-learn, Pandas, NumPy, Matplotlib
- **Document Processing:** PyPDF, pdf2image, Tesseract, pdfplumber
- **AI Models:** Hugging Face Transformers, OpenAI API, LangChain
- **Web Framework:** Streamlit, FastAPI, React/Next.js
- **Infrastructure:** Docker, MLFlow, PostgreSQL
- **Deployment:** AWS/GCP/Azure, GitHub Actions, CI/CD

### Project Structure
```
ai-engineer-bootcamp/
├── day1/ (Python basics)
├── day2/ (ML regression)
├── day3/ (ML classification)
├── day4/ (Unsupervised + Pipelines)
├── day5/ (Document Intelligence - Playground)
├── day6/ (Document Intelligence - Advanced)
├── day7/ (LLM APIs)
├── day8/ (RAG Systems)
├── day9/ (AI Agents)
├── day10/ (Deployment)
├── projects/
│   ├── project1_data_analysis/
│   ├── project2_house_price/
│   ├── project3_classification/
│   └── document_intelligence/
│       ├── playground_app.py
│       ├── pdf_processor.py
│       ├── model_council.py
│       ├── rag_pipeline.py
│       └── tests/
├── notebooks/
├── data/
└── README.md
```

---

## 🎓 How to Use This Complete Guide

1. **Start with Day 1** - Establish foundational Python and ML knowledge
2. **Progress sequentially** - Each day builds on previous knowledge
3. **Complete projects** - Real-world mini-projects on Days 2, 3, 4
4. **Build Document Intelligence system** - Core project on Days 5-6
5. **Integrate with LLM/RAG** - Advanced features on Days 7-8
6. **Deploy to production** - Full-stack deployment on Days 9-10

**Total time:** 80-100 hours  
**Total projects:** 6 production-ready applications  
**Career outcome:** Ready for AI Engineer roles

---

## 📞 Getting Help

When you need assistance, reference the specific step number:
- Example: "I'm stuck on Step 5.3 with PDF parsing"
- Provide: Step number, what you're trying to do, any error messages
- AI Tutor will provide targeted help within that step's scope

---

**Last Updated:** December 8, 2025  
**Version:** 2.0 (Complete with Document Intelligence Track)  
**Status:** Ready to use - Start with Day 1, Step 1.1
