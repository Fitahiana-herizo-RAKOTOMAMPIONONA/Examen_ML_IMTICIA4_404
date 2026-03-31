#!/usr/bin/env python3
"""
Notebook source — EDA, Baseline (Régression Logistique), et Modèles Avancés
pour le projet Morpion ML.

À convertir en notebook avec: jupyter nbconvert --to notebook --execute notebook_source.py
Ou utiliser directement comme script.
"""

# %% [markdown]
# # 🎮 Morpion ML — Analyse Exploratoire et Modélisation
# ## Pipeline complet : EDA → Baseline → Modèles Avancés

# %% Imports
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid", palette="viridis")

# %% Chargement du dataset
print("=" * 60)
print("📊 ÉTAPE 1 — ANALYSE EXPLORATOIRE (EDA)")
print("=" * 60)

df = pd.read_csv("ressources/dataset.csv")
print(f"\n📁 Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\nPremières lignes :")
print(df.head(10))
print(f"\nDescription statistique :")
print(df.describe())
print(f"\nTypes de données :")
print(df.dtypes)
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")

# %% Features et Targets
feature_cols = [f"c{i}_{p}" for i in range(9) for p in ["x", "o"]]
target_cols = ["x_wins", "is_draw"]

X = df[feature_cols]
y_xwins = df["x_wins"]
y_draw = df["is_draw"]

# Créer un label multi-classe pour l'analyse
df["result"] = "O wins"
df.loc[df["x_wins"] == 1, "result"] = "X wins"
df.loc[df["is_draw"] == 1, "result"] = "Draw"

# %% Distribution des classes
print("\n" + "─" * 40)
print("📊 Distribution des classes :")
print("─" * 40)
result_counts = df["result"].value_counts()
print(result_counts)
print(f"\nProportions :")
print(result_counts / len(df) * 100)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Bar plot
colors = ["#2ecc71", "#3498db", "#e74c3c"]
result_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black", alpha=0.85)
axes[0].set_title("Distribution des résultats", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Nombre d'états")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
for i, v in enumerate(result_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops=dict(edgecolor='black'))
axes[1].set_title("Répartition (%)", fontsize=14, fontweight="bold")

# x_wins vs is_draw distribution
for target, color, label in [("x_wins", "#2ecc71", "x_wins"), ("is_draw", "#3498db", "is_draw")]:
    counts = df[target].value_counts().sort_index()
    axes[2].bar([f"{label}=0", f"{label}=1"],
                counts.values, alpha=0.7, color=color, edgecolor="black", label=label)
axes[2].set_title("Distribution binaire des targets", fontsize=14, fontweight="bold")
axes[2].legend()

plt.tight_layout()
plt.savefig("ressources/eda_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/eda_distribution.png")

# %% Importance des cases (occupation par X et O)
print("\n" + "─" * 40)
print("📊 Importance des cases :")
print("─" * 40)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Taux d'occupation par X pour chaque case
x_occupation = np.array([df[f"c{i}_x"].mean() for i in range(9)]).reshape(3, 3)
o_occupation = np.array([df[f"c{i}_o"].mean() for i in range(9)]).reshape(3, 3)

sns.heatmap(x_occupation, annot=True, fmt=".3f", cmap="Greens", ax=axes[0, 0],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[0, 0].set_title("Taux occupation X", fontsize=12, fontweight="bold")

sns.heatmap(o_occupation, annot=True, fmt=".3f", cmap="Reds", ax=axes[0, 1],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[0, 1].set_title("Taux occupation O", fontsize=12, fontweight="bold")

# Taux d'occupation quand X gagne
x_wins_df = df[df["x_wins"] == 1]
x_occ_wins = np.array([x_wins_df[f"c{i}_x"].mean() for i in range(9)]).reshape(3, 3)
sns.heatmap(x_occ_wins, annot=True, fmt=".3f", cmap="YlGn", ax=axes[0, 2],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[0, 2].set_title("Occupation X (quand X gagne)", fontsize=12, fontweight="bold")

# Taux d'occupation quand O gagne
o_wins_df = df[df["result"] == "O wins"]
x_occ_loses = np.array([o_wins_df[f"c{i}_x"].mean() for i in range(9)]).reshape(3, 3)
sns.heatmap(x_occ_loses, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1, 0],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[1, 0].set_title("Occupation X (quand O gagne)", fontsize=12, fontweight="bold")

# Taux d'occupation quand match nul
draw_df = df[df["is_draw"] == 1]
x_occ_draw = np.array([draw_df[f"c{i}_x"].mean() for i in range(9)]).reshape(3, 3)
sns.heatmap(x_occ_draw, annot=True, fmt=".3f", cmap="Blues", ax=axes[1, 1],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[1, 1].set_title("Occupation X (match nul)", fontsize=12, fontweight="bold")

# Différence X occupation: gagne vs perd
diff = x_occ_wins - x_occ_loses
sns.heatmap(diff, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=axes[1, 2],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[1, 2].set_title("Diff. X (gagne - perd)", fontsize=12, fontweight="bold")

plt.suptitle("Analyse de l'importance des cases", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("ressources/eda_cases_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/eda_cases_importance.png")

# %% Heatmap de corrélation
print("\n" + "─" * 40)
print("📊 Heatmap de corrélation :")
print("─" * 40)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Corrélation features ↔ x_wins
corr_xwins = df[feature_cols + ["x_wins"]].corr()["x_wins"][:-1]
corr_matrix = corr_xwins.values.reshape(9, 2)
corr_df = pd.DataFrame(corr_matrix,
                        index=[f"Case {i}" for i in range(9)],
                        columns=["ci_x", "ci_o"])
sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=axes[0])
axes[0].set_title("Corrélation features ↔ x_wins", fontsize=14, fontweight="bold")

# Corrélation features ↔ is_draw
corr_draw = df[feature_cols + ["is_draw"]].corr()["is_draw"][:-1]
corr_matrix2 = corr_draw.values.reshape(9, 2)
corr_df2 = pd.DataFrame(corr_matrix2,
                         index=[f"Case {i}" for i in range(9)],
                         columns=["ci_x", "ci_o"])
sns.heatmap(corr_df2, annot=True, fmt=".3f", cmap="RdYlBu", center=0, ax=axes[1])
axes[1].set_title("Corrélation features ↔ is_draw", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("ressources/eda_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/eda_correlation.png")

# Matrice de corrélation complète
plt.figure(figsize=(14, 12))
full_corr = df[feature_cols + target_cols].corr()
mask = np.triu(np.ones_like(full_corr, dtype=bool))
sns.heatmap(full_corr, mask=mask, annot=False, cmap="coolwarm", center=0,
            linewidths=0.5, square=True)
plt.title("Matrice de corrélation complète", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("ressources/eda_full_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/eda_full_correlation.png")


# %%
print("\n" + "=" * 60)
print("📊 ÉTAPE 2 — BASELINE : RÉGRESSION LOGISTIQUE")
print("=" * 60)

# Split train/test
X_train, X_test, y_xwins_train, y_xwins_test = train_test_split(
    X, y_xwins, test_size=0.2, random_state=42, stratify=y_xwins)

_, _, y_draw_train, y_draw_test = train_test_split(
    X, y_draw, test_size=0.2, random_state=42, stratify=y_draw)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Modèle 1 : Régression Logistique pour x_wins
print("\n" + "─" * 40)
print("🔹 Régression Logistique — x_wins")
print("─" * 40)

lr_xwins = LogisticRegression(max_iter=1000, random_state=42)
lr_xwins.fit(X_train_scaled, y_xwins_train)
y_pred_xwins = lr_xwins.predict(X_test_scaled)

acc_xwins = accuracy_score(y_xwins_test, y_pred_xwins)
f1_xwins = f1_score(y_xwins_test, y_pred_xwins)
cm_xwins = confusion_matrix(y_xwins_test, y_pred_xwins)

print(f"  Accuracy : {acc_xwins:.4f}")
print(f"  F1-Score : {f1_xwins:.4f}")
print(f"\n  Rapport de classification :")
print(classification_report(y_xwins_test, y_pred_xwins, target_names=["Not X wins", "X wins"]))

# %% Modèle 2 : Régression Logistique pour is_draw
print("─" * 40)
print("🔹 Régression Logistique — is_draw")
print("─" * 40)

lr_draw = LogisticRegression(max_iter=1000, random_state=42)
lr_draw.fit(X_train_scaled, y_draw_train)
y_pred_draw = lr_draw.predict(X_test_scaled)

acc_draw = accuracy_score(y_draw_test, y_pred_draw)
f1_draw = f1_score(y_draw_test, y_pred_draw)
cm_draw = confusion_matrix(y_draw_test, y_pred_draw)

print(f"  Accuracy : {acc_draw:.4f}")
print(f"  F1-Score : {f1_draw:.4f}")
print(f"\n  Rapport de classification :")
print(classification_report(y_draw_test, y_pred_draw, target_names=["Not Draw", "Draw"]))

# %% Matrices de confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_xwins, annot=True, fmt="d", cmap="Greens", ax=axes[0],
            xticklabels=["Not X wins", "X wins"],
            yticklabels=["Not X wins", "X wins"])
axes[0].set_title("Confusion Matrix — x_wins", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Prédit")
axes[0].set_ylabel("Réel")

sns.heatmap(cm_draw, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Not Draw", "Draw"],
            yticklabels=["Not Draw", "Draw"])
axes[1].set_title("Confusion Matrix — is_draw", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Prédit")
axes[1].set_ylabel("Réel")

plt.tight_layout()
plt.savefig("ressources/baseline_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/baseline_confusion_matrix.png")

# %% Analyse des coefficients
print("\n" + "─" * 40)
print("📊 Analyse des coefficients — Plateau 3×3")
print("─" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Coefficients x_wins pour X
coefs_xwins_x = np.array([lr_xwins.coef_[0][i*2] for i in range(9)]).reshape(3, 3)
coefs_xwins_o = np.array([lr_xwins.coef_[0][i*2+1] for i in range(9)]).reshape(3, 3)

sns.heatmap(coefs_xwins_x, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=axes[0, 0],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[0, 0].set_title("Coeff. LR x_wins — Pièces X", fontsize=12, fontweight="bold")

sns.heatmap(coefs_xwins_o, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=axes[0, 1],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[0, 1].set_title("Coeff. LR x_wins — Pièces O", fontsize=12, fontweight="bold")

# Coefficients is_draw pour X
coefs_draw_x = np.array([lr_draw.coef_[0][i*2] for i in range(9)]).reshape(3, 3)
coefs_draw_o = np.array([lr_draw.coef_[0][i*2+1] for i in range(9)]).reshape(3, 3)

sns.heatmap(coefs_draw_x, annot=True, fmt=".3f", cmap="RdYlBu", center=0, ax=axes[1, 0],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[1, 0].set_title("Coeff. LR is_draw — Pièces X", fontsize=12, fontweight="bold")

sns.heatmap(coefs_draw_o, annot=True, fmt=".3f", cmap="RdYlBu", center=0, ax=axes[1, 1],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[1, 1].set_title("Coeff. LR is_draw — Pièces O", fontsize=12, fontweight="bold")

plt.suptitle("Coefficients Régression Logistique — Mapping 3×3", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("ressources/baseline_coefficients.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/baseline_coefficients.png")

print("\n  Coefficients x_wins (pièces X sur plateau 3×3):")
print(pd.DataFrame(coefs_xwins_x, columns=["Col 0", "Col 1", "Col 2"],
                    index=["Lig 0", "Lig 1", "Lig 2"]).to_string())
print("\n  Coefficients x_wins (pièces O sur plateau 3×3):")
print(pd.DataFrame(coefs_xwins_o, columns=["Col 0", "Col 1", "Col 2"],
                    index=["Lig 0", "Lig 1", "Lig 2"]).to_string())

# %%
print("\n" + "=" * 60)
print("📊 ÉTAPE 3 — MODÈLES AVANCÉS")
print("=" * 60)

# Modèles à tester
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
    print(f"\n  🔸 {name}")

    # x_wins
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
    print(f"    x_wins — Acc: {acc:.4f}, F1: {f1:.4f}")

    # is_draw
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
    print(f"    is_draw — Acc: {acc_d:.4f}, F1: {f1_d:.4f}")

# %% Tableau comparatif
print("\n" + "─" * 40)
print("📊 Tableau comparatif — x_wins")
print("─" * 40)
df_results_xwins = pd.DataFrame(results_xwins)
print(df_results_xwins.to_string(index=False))

print("\n" + "─" * 40)
print("📊 Tableau comparatif — is_draw")
print("─" * 40)
df_results_draw = pd.DataFrame(results_draw)
print(df_results_draw.to_string(index=False))

# %% Graphiques comparatifs
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# x_wins comparison
model_names = [r["Modèle"] for r in results_xwins]
x_pos = np.arange(len(model_names))
width = 0.35

axes[0].bar(x_pos - width/2, [r["Accuracy"] for r in results_xwins], width,
            label="Accuracy", color="#2ecc71", alpha=0.8, edgecolor="black")
axes[0].bar(x_pos + width/2, [r["F1-Score"] for r in results_xwins], width,
            label="F1-Score", color="#3498db", alpha=0.8, edgecolor="black")
axes[0].set_title("Comparaison Modèles — x_wins", fontsize=14, fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, rotation=30, ha="right")
axes[0].legend()
axes[0].set_ylim(0.5, 1.05)

# is_draw comparison
axes[1].bar(x_pos - width/2, [r["Accuracy"] for r in results_draw], width,
            label="Accuracy", color="#2ecc71", alpha=0.8, edgecolor="black")
axes[1].bar(x_pos + width/2, [r["F1-Score"] for r in results_draw], width,
            label="F1-Score", color="#e74c3c", alpha=0.8, edgecolor="black")
axes[1].set_title("Comparaison Modèles — is_draw", fontsize=14, fontweight="bold")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names, rotation=30, ha="right")
axes[1].legend()
axes[1].set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig("ressources/models_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Figure sauvegardée : ressources/models_comparison.png")

# %% Confusion matrices pour les meilleurs modèles
fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 8))

for idx, (name, model) in enumerate(models.items()):
    # x_wins
    m = model.__class__(**model.get_params())
    m.fit(X_train_scaled, y_xwins_train)
    cm = confusion_matrix(y_xwins_test, m.predict(X_test_scaled))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[0, idx],
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    axes[0, idx].set_title(f"{name}\nx_wins", fontsize=10, fontweight="bold")

    # is_draw
    m2 = model.__class__(**model.get_params())
    m2.fit(X_train_scaled, y_draw_train)
    cm2 = confusion_matrix(y_draw_test, m2.predict(X_test_scaled))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=axes[1, idx],
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    axes[1, idx].set_title(f"{name}\nis_draw", fontsize=10, fontweight="bold")

plt.suptitle("Matrices de confusion — Tous les modèles", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("ressources/all_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/all_confusion_matrices.png")

# %% Feature importance (Random Forest)
print("\n" + "─" * 40)
print("📊 Feature Importance — Random Forest")
print("─" * 40)

rf_xwins = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_xwins.fit(X_train_scaled, y_xwins_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Importance pour x_wins
imp_xwins = rf_xwins.feature_importances_
imp_3x3_x = np.array([imp_xwins[i*2] for i in range(9)]).reshape(3, 3)
imp_3x3_o = np.array([imp_xwins[i*2+1] for i in range(9)]).reshape(3, 3)

sns.heatmap(imp_3x3_x + imp_3x3_o, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[0],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[0].set_title("Importance RF — x_wins (X+O)", fontsize=12, fontweight="bold")

rf_draw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_draw.fit(X_train_scaled, y_draw_train)
imp_draw = rf_draw.feature_importances_
imp_draw_3x3 = np.array([imp_draw[i*2] + imp_draw[i*2+1] for i in range(9)]).reshape(3, 3)

sns.heatmap(imp_draw_3x3, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1],
            xticklabels=["Col 0", "Col 1", "Col 2"],
            yticklabels=["Lig 0", "Lig 1", "Lig 2"])
axes[1].set_title("Importance RF — is_draw (X+O)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("ressources/feature_importance_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : ressources/feature_importance_rf.png")

# %% Sauvegarde des modèles pour l'interface
print("\n" + "=" * 60)
print("💾 EXPORT DES MODÈLES POUR L'INTERFACE")
print("=" * 60)

# Entraîner les meilleurs modèles sur tout le dataset
scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X)

# On utilise la Régression Logistique pour l'export (poids simples en JSON)
lr_xwins_final = LogisticRegression(max_iter=1000, random_state=42)
lr_xwins_final.fit(X_all_scaled, y_xwins)

lr_draw_final = LogisticRegression(max_iter=1000, random_state=42)
lr_draw_final.fit(X_all_scaled, y_draw)

# Aussi entraîner un Random Forest final
rf_xwins_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_xwins_final.fit(X_all_scaled, y_xwins)

rf_draw_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_draw_final.fit(X_all_scaled, y_draw)

# Export JSON des coefficients LR pour le frontend
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

# Aussi exporter un lookup table complet via minimax pour le mode hybride
# (on sauvegarde une évaluation pour chaque état possible)
output_path = os.path.join("interfaces", "public", "models.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(models_json, f)

print(f"✅ Modèles exportés : {output_path}")

# Sauvegarder les modèles pickle
for name, model_obj in [("lr_xwins", lr_xwins_final), ("lr_draw", lr_draw_final),
                          ("rf_xwins", rf_xwins_final), ("rf_draw", rf_draw_final)]:
    pkl_path = os.path.join("ressources", f"{name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_obj, f)
    print(f"✅ Modèle sauvegardé : {pkl_path}")

# Sauvegarder le scaler
with open("ressources/scaler.pkl", "wb") as f:
    pickle.dump(scaler_final, f)
print("✅ Scaler sauvegardé : ressources/scaler.pkl")

# %% Résumé final
print("\n" + "=" * 60)
print("📊 RÉSUMÉ FINAL")
print("=" * 60)
print(f"\n  Dataset : {len(df)} états")
print(f"  Features : 18 (9 cases × 2 joueurs)")
print(f"  Targets : x_wins, is_draw")
print(f"\n  Meilleurs résultats x_wins :")
best_xw = max(results_xwins, key=lambda r: r["F1-Score"])
print(f"    {best_xw['Modèle']} — F1: {best_xw['F1-Score']:.4f}, Acc: {best_xw['Accuracy']:.4f}")
print(f"\n  Meilleurs résultats is_draw :")
best_dr = max(results_draw, key=lambda r: r["F1-Score"])
print(f"    {best_dr['Modèle']} — F1: {best_dr['F1-Score']:.4f}, Acc: {best_dr['Accuracy']:.4f}")

print("\n✅ Pipeline ML terminé avec succès !")
