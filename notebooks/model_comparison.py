"""
model_comparison.py - Comparaison de plusieurs modeles (Sprint bonus)
=====================================================================

Ce module entraine et compare plusieurs classifieurs sur les memes
donnees, avec la meme validation par sujet (LOSO). Il correspond a la
fonctionnalite F8 du cahier des charges.

Modeles compares :
  - Random Forest
  - SVM (noyau RBF)
  - XGBoost
  - Regression logistique (repere simple)

Tous sont evalues equitablement : meme pipeline (normalisation incluse),
meme protocole de validation par sujet, memes metriques.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

try:
    from . import config
except ImportError:
    import config


def get_candidate_models(random_state=None):
    """
    Renvoie un dictionnaire {nom: pipeline} des modeles a comparer.

    Chaque pipeline inclut la normalisation (StandardScaler) pour une
    comparaison equitable et sans fuite de donnees.
    """
    if random_state is None:
        random_state = config.RANDOM_SEED

    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=random_state,
                n_jobs=-1, class_weight="balanced")),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=False,
                        random_state=random_state, class_weight="balanced")),
        ]),
        "Régression logistique": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=random_state,
                class_weight="balanced")),
        ]),
    }

    # XGBoost est optionnel : on ne l'ajoute que s'il est installe
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, random_state=random_state,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0)),
        ])
    except ImportError:
        pass

    return models


def _encode_labels_if_needed(y, pipeline_name):
    """
    XGBoost exige des etiquettes numeriques. Pour les autres modeles,
    on garde les etiquettes texte. Renvoie (y_pret, decodeur ou None).
    """
    if pipeline_name == "XGBoost":
        classes = sorted(np.unique(y))
        mapping = {c: i for i, c in enumerate(classes)}
        y_enc = np.array([mapping[v] for v in y])
        return y_enc, classes
    return y, None


def compare_models(X, y, groups, models=None, verbose=True):
    """
    Compare plusieurs modeles en validation par sujet (LOSO).

    Pour chaque modele : on agrege les predictions sur tous les plis
    (un sujet en test a chaque tour), puis on calcule les metriques.

    Retour
    ------
    pandas.DataFrame trie par macro-F1 decroissant, colonnes :
        Modèle, Macro-F1, Exactitude équilibrée, Exactitude, Écart-type F1
    """
    if models is None:
        models = get_candidate_models()

    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
        raise ValueError("Au moins 2 sujets sont necessaires pour la validation LOSO.")

    logo = LeaveOneGroupOut()
    labels = sorted(np.unique(y))
    rows = []

    for name, pipeline in models.items():
        if verbose:
            print(f"Evaluation : {name} ...")

        y_model, decoder = _encode_labels_if_needed(y, name)
        labels_model = sorted(np.unique(y_model))

        y_true_all, y_pred_all = [], []
        per_subject_f1 = []

        for train_idx, test_idx in logo.split(X, y_model, groups):
            pipe = clone(pipeline)
            pipe.fit(X[train_idx], y_model[train_idx])
            y_pred = pipe.predict(X[test_idx])
            y_true = y_model[test_idx]

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            per_subject_f1.append(
                f1_score(y_true, y_pred, average="macro",
                         labels=labels_model, zero_division=0))

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        rows.append({
            "Modèle": name,
            "Macro-F1": round(f1_score(y_true_all, y_pred_all, average="macro",
                                       labels=labels_model, zero_division=0), 3),
            "Exactitude équilibrée": round(
                balanced_accuracy_score(y_true_all, y_pred_all), 3),
            "Exactitude": round(accuracy_score(y_true_all, y_pred_all), 3),
            "Écart-type F1": round(float(np.std(per_subject_f1)), 3),
        })

    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os, joblib
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import model as model_mod

    feat_path = config.PROCESSED_DIR / "features.csv"
    if not os.path.exists(feat_path):
        print("Lancez d'abord le Sprint 3 pour creer features.csv")
    else:
        X, y, groups, names = model_mod.load_features()
        print(f"Donnees : {X.shape[0]} epochs, {len(np.unique(groups))} sujets\n")
        result = compare_models(X, y, groups)
        print("\n=== Comparaison des modeles (validation par sujet) ===\n")
        print(result.to_string(index=False))
        print(f"\nMeilleur modele : {result.iloc[0]['Modèle']} "
              f"(macro-F1 = {result.iloc[0]['Macro-F1']})")
