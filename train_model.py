import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

DATA_PATH = "heart_disease_prediction_2026.csv"
MODEL_PATH = "heart_disease_model.joblib"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()

X = df.drop(columns=["heart_disease"])
y = df["heart_disease"]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]

print("Best parameters:", search.best_params_)
print("CV ROC-AUC:", round(search.best_score_, 4))
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print("Test precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Test recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("Test F1:", round(f1_score(y_test, pred, zero_division=0), 4))
print("Test ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(best_model, MODEL_PATH)
print(f"Saved trained model to {MODEL_PATH}")
