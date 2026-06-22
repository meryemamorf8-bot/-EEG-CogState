"""
explainability.py - Explicabilite IA (XAI) avec SHAP
====================================================

Ce module explique POURQUOI le modele prend ses decisions, au lieu de
se contenter de predire. Il utilise SHAP (SHapley Additive exPlanations),
la methode de reference en IA explicable.

Principe de SHAP
----------------
Pour une prediction donnee, SHAP attribue a chaque feature une "valeur"
qui mesure sa contribution : combien cette feature a pousse la decision
vers une classe plutot qu'une autre. La somme des contributions explique
l'ecart entre la prediction et la moyenne.

Le module fournit :
  - les contributions SHAP globales (quelles features comptent en general)
  - les contributions pour une prediction precise (pourquoi CE resultat)
  - un regroupement par bande de frequence et par canal (interpretation neuro)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def compute_shap_values(model, X, feature_names, max_background=100, max_explain=200):
    """
    Calcule les valeurs SHAP pour un modele a base d'arbres (Random Forest).

    On utilise TreeExplainer, optimise et exact pour les forets d'arbres.
    Pour rester rapide, on echantillonne si X est gros.

    Parametres
    ----------
    model : pipeline sklearn (scaler + RandomForest) OU le classifieur seul
    X : np.ndarray (n_epochs, n_features) — features (AVANT normalisation
        si on passe le pipeline complet)
    feature_names : list[str]
    max_background, max_explain : limites d'echantillonnage pour la vitesse

    Retour
    ------
    dict avec :
      shap_matrix : np.ndarray (n_samples, n_features) — contributions moyennes
                    en valeur absolue, agregees sur les classes
      feature_names : list[str]
      X_used : les echantillons effectivement expliques
    """
    import shap

    # Si on a un pipeline, on applique le scaler puis on explique le classifieur
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")
        clf = model.named_steps.get("clf")
        X_scaled = scaler.transform(X) if scaler is not None else X
    else:
        clf = model
        X_scaled = X

    # Echantillonnage pour la vitesse
    if X_scaled.shape[0] > max_explain:
        idx = np.random.RandomState(config.RANDOM_SEED).choice(
            X_scaled.shape[0], max_explain, replace=False)
        X_used = X_scaled[idx]
    else:
        X_used = X_scaled

    explainer = shap.TreeExplainer(clf)
    shap_out = explainer.shap_values(X_used)

    # shap_values renvoie une liste (une matrice par classe) pour la classif.
    # On agrege en valeur absolue moyenne sur les classes.
    if isinstance(shap_out, list):
        shap_matrix = np.mean([np.abs(s) for s in shap_out], axis=0)
    else:
        arr = np.abs(shap_out)
        # Cas 3D (n_samples, n_features, n_classes) -> moyenne sur les classes
        if arr.ndim == 3:
            shap_matrix = arr.mean(axis=2)
        else:
            shap_matrix = arr

    return {
        "shap_matrix": shap_matrix,
        "feature_names": feature_names,
        "X_used": X_used,
    }


def global_feature_importance(shap_result, top=15):
    """
    Importance globale des features = moyenne des |valeurs SHAP| sur les echantillons.

    Retour
    ------
    pandas.DataFrame (feature, importance) trie, limite a `top`.
    """
    mean_abs = shap_result["shap_matrix"].mean(axis=0)
    df = pd.DataFrame({
        "feature": shap_result["feature_names"],
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df.head(top)


def importance_by_band(shap_result):
    """
    Regroupe l'importance SHAP par bande de frequence (interpretation neuro).

    On somme les contributions de toutes les features dont le nom contient
    le nom d'une bande (ex. "O1_alpha_power", "F3_alpha_rel" -> bande alpha).

    Retour
    ------
    pandas.DataFrame (band, importance) trie.
    """
    mean_abs = shap_result["shap_matrix"].mean(axis=0)
    names = shap_result["feature_names"]
    band_imp = {b: 0.0 for b in config.FREQ_BANDS}
    for name, imp in zip(names, mean_abs):
        for b in config.FREQ_BANDS:
            if f"_{b}_" in name or name.endswith(f"_{b}"):
                band_imp[b] += imp
    df = pd.DataFrame({"band": list(band_imp.keys()),
                       "importance": list(band_imp.values())})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def importance_by_channel(shap_result, top=10):
    """
    Regroupe l'importance SHAP par canal EEG (quelle electrode compte le plus).

    Retour
    ------
    pandas.DataFrame (channel, importance) trie, limite a `top`.
    """
    mean_abs = shap_result["shap_matrix"].mean(axis=0)
    names = shap_result["feature_names"]
    chan_imp = {ch: 0.0 for ch in config.CHANNEL_NAMES}
    for name, imp in zip(names, mean_abs):
        # Le nom commence par le canal : "O1_alpha_power" -> "O1"
        ch = name.split("_")[0]
        if ch in chan_imp:
            chan_imp[ch] += imp
    df = pd.DataFrame({"channel": list(chan_imp.keys()),
                       "importance": list(chan_imp.values())})
    return df.sort_values("importance", ascending=False).reset_index(drop=True).head(top)


def explain_single_prediction(model, x_single, feature_names, top=8):
    """
    Explique UNE prediction : quelles features ont le plus pese, et dans quel sens.

    Contrairement aux fonctions globales, on garde ici le SIGNE de la
    contribution (positive = pousse vers la classe predite).

    Parametres
    ----------
    x_single : np.ndarray (1, n_features) ou (n_features,)
    Retour
    ------
    pandas.DataFrame (feature, contribution) triee par |contribution|.
    """
    import shap

    x = np.atleast_2d(x_single)
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")
        clf = model.named_steps.get("clf")
        x_scaled = scaler.transform(x) if scaler is not None else x
    else:
        clf = model
        x_scaled = x

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(x_scaled)

    # Classe predite et son index dans l'ordre des classes du modele
    predicted = clf.predict(x_scaled)[0]
    classes = list(clf.classes_)
    ci = classes.index(predicted)

    # Recuperer le vecteur de contributions pour la classe predite
    if isinstance(sv, list):
        contrib = sv[ci][0]
    elif np.asarray(sv).ndim == 3:
        contrib = np.asarray(sv)[0, :, ci]
    else:
        contrib = np.asarray(sv)[0]

    df = pd.DataFrame({"feature": feature_names, "contribution": contrib})
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    return df.head(top)


if __name__ == "__main__":
    import os, joblib
    feat_path = config.PROCESSED_DIR / "features.csv"
    model_path = config.MODELS_DIR / "model.joblib"
    if not (os.path.exists(feat_path) and os.path.exists(model_path)):
        print("Il faut d'abord features.csv (Sprint 3) et model.joblib (Sprint 4).")
    else:
        df = pd.read_csv(feat_path)
        meta = ["subject", "label"]
        names = [c for c in df.columns if c not in meta]
        X = df[names].values
        model = joblib.load(model_path)

        print("Calcul des valeurs SHAP (peut prendre un moment)...\n")
        res = compute_shap_values(model, X, names)

        print("Top 10 features (importance globale SHAP) :")
        print(global_feature_importance(res, top=10).to_string(index=False))

        print("\nImportance par bande de frequence :")
        print(importance_by_band(res).to_string(index=False))

        print("\nImportance par canal (top 5) :")
        print(importance_by_channel(res, top=5).to_string(index=False))
