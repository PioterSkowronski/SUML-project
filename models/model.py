import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    make_scorer,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# LOAD DATA
df = pd.read_csv('weatherMelbourne.csv')
df.columns = df.columns.str.strip()

# TARGET (0 / 1)
df = df.dropna(subset=['RainTomorrow'])
y = df['RainTomorrow'].astype(str).str.strip().map({'No': 0, 'Yes': 1})

mask_ok = ~y.isna()
df = df.loc[mask_ok].copy()
y = y.loc[mask_ok].astype(int)

# FEATURES
X = df.drop(columns=['RainTomorrow'])

# usuń Date (zalecane)
if 'Date' in X.columns:
    X = X.drop(columns=['Date'])

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CLASS IMBALANCE
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# PREPROCESSING (w pipeline)
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

# PIPELINE: preprocessor + model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(
        objective='binary',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    ))
])

# GRID SEARCH
# (celowy "BŁĄD": cv=100 - jak w zadaniu)
custom_f1_scorer = make_scorer(f1_score, pos_label=1, average='binary')

param_grid = {
    'model__n_estimators': [100, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__num_leaves': [20],
    'model__max_depth': [7, -1],
    'model__min_child_samples': [20]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=custom_f1_scorer,
    cv=100,
    verbose=1,
    n_jobs=-1
)

print("Starting GridSearchCV fit...")
grid_search.fit(X_train, y_train)   # UWAGA: SUROWE X_train, pipeline sam zrobi preprocessing
print("GridSearchCV fit complete.")

print("\nBest parameters found:")
print(grid_search.best_params_)
print("Best F1-score (positive class):", grid_search.best_score_)

# EVALUATION (na surowym X_test)
best_pipeline = grid_search.best_estimator_

proba = best_pipeline.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("ROC AUC:", round(roc_auc_score(y_test, proba), 4))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# SAVE PIPELINE (model + preprocessing)
pipeline_filename = 'best_rain_pipeline.joblib'
joblib.dump(best_pipeline, pipeline_filename)

print(f"\nBest pipeline saved successfully to {pipeline_filename}")