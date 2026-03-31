import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
import json
import os
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple

warnings.filterwarnings('ignore')

df: pd.DataFrame = pd.read_csv("ressources/dataset.csv")

feature_cols: List[str] = [f"c{i}_{p}" for i in range(9) for p in ["x", "O"]]
target_cols: List[str] = ["x_wins", "is_draw"]

X: pd.DataFrame = df[feature_cols]
y_xwins: pd.Series = df["x_wins"]
y_draw: pd.Series = df["is_draw"]

X_train, X_test, y_xwins_train, y_xwins_test = train_test_split(
    X, y_xwins, test_size=0.2, random_state=42, stratify=y_xwins)

_, _, y_draw_train, y_draw_test = train_test_split(
    X, y_draw, test_size=0.2, random_state=42, stratify=y_draw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_xwins = LogisticRegression(max_iter=1000, random_state=42)
lr_xwins.fit(X_train_scaled, y_xwins_train)

lr_draw = LogisticRegression(max_iter=1000, random_state=42)
lr_draw.fit(X_train_scaled, y_draw_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost (GBM)": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                                     early_stopping=True, validation_fraction=0.1),
}

results_xwins = []
results_draw = []

for name, model in models.items():
    model_xw = model.__class__(**model.get_params())
    model_xw.fit(X_train_scaled, y_xwins_train)
    y_pred = model_xw.predict(X_test_scaled)
    y_proba = model_xw.predict_proba(X_test_scaled)[:, 1] if hasattr(model_xw, 'predict_proba') else None
    acc = accuracy_score(y_xwins_test, y_pred)
    f1 = f1_score(y_xwins_test, y_pred)
    auc = roc_auc_score(y_xwins_test, y_proba) if y_proba is not None else "N/A"
    cv_scores = cross_val_score(model_xw, X_train_scaled, y_xwins_train, cv=5, scoring='f1')
    results_xwins.append({
        "Modèle": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC-ROC": auc if isinstance(auc, str) else f"{auc:.4f}",
        "CV F1 (mean±std)": f"{cv_scores.mean():.4f}±{cv_scores.std():.4f}"
    })

    model_dr = model.__class__(**model.get_params())
    model_dr.fit(X_train_scaled, y_draw_train)
    y_pred_d = model_dr.predict(X_test_scaled)
    y_proba_d = model_dr.predict_proba(X_test_scaled)[:, 1] if hasattr(model_dr, 'predict_proba') else None
    acc_d = accuracy_score(y_draw_test, y_pred_d)
    f1_d = f1_score(y_draw_test, y_pred_d)
    auc_d = roc_auc_score(y_draw_test, y_proba_d) if y_proba_d is not None else "N/A"
    cv_scores_d = cross_val_score(model_dr, X_train_scaled, y_draw_train, cv=5, scoring='f1')
    results_draw.append({
        "Modèle": name,
        "Accuracy": acc_d,
        "F1-Score": f1_d,
        "AUC-ROC": auc_d if isinstance(auc_d, str) else f"{auc_d:.4f}",
        "CV F1 (mean±std)": f"{cv_scores_d.mean():.4f}±{cv_scores_d.std():.4f}"
    })

scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X)

lr_xwins_final = LogisticRegression(max_iter=1000, random_state=42)
lr_xwins_final.fit(X_all_scaled, y_xwins)

lr_draw_final = LogisticRegression(max_iter=1000, random_state=42)
lr_draw_final.fit(X_all_scaled, y_draw)

rf_xwins_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_xwins_final.fit(X_all_scaled, y_xwins)

rf_draw_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_draw_final.fit(X_all_scaled, y_draw)

models_json = {
    "scaler": {
        "mean": scaler_final.mean_.tolist(),
        "scale": scaler_final.scale_.tolist()
    },
    "lr_xwins": {
        "coef": lr_xwins_final.coef_[0].tolist(),
        "intercept": float(lr_xwins_final.intercept_[0])
    },
    "lr_draw": {
        "coef": lr_draw_final.coef_[0].tolist(),
        "intercept": float(lr_draw_final.intercept_[0])
    }
}

output_path = os.path.join("interfaces", "public", "models.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(models_json, f)

for name, model_obj in [("lr_xwins", lr_xwins_final), ("lr_draw", lr_draw_final),
                          ("rf_xwins", rf_xwins_final), ("rf_draw", rf_draw_final)]:
    pkl_path = os.path.join("ressources", f"{name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_obj, f)

with open("ressources/scaler.pkl", "wb") as f:
    pickle.dump(scaler_final, f)
