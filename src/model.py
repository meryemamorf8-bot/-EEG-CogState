"""
model.py - Machine learning et evaluation (Sprint 4)
====================================================

Ce module entraine un modele de classification des etats cognitifs et
l'evalue de maniere rigoureuse. Il correspond a la fonctionnalite F4 du
cahier des charges, et constitue le coeur scientifique du projet.

POINT METHODOLOGIQUE CENTRAL — la validation par sujet
------------------------------------------------------
Plusieurs epochs proviennent du meme sujet et partagent sa signature
cerebrale. Si on melangeait aleatoirement les epochs entre entrainement
et test, des fragments d'un meme sujet se retrouveraient des deux cotes :
le modele "reconnaitrait" le sujet au lieu d'apprendre l'etat cognitif,
ce qui gonflerait artificiellement les scores.

On utilise donc une validation Leave-One-Subject-Out (LOSO) / par groupes :
a chaque tour, un sujet entier sert de test, les autres servent a entrainer.
C'est la seule facon de mesurer la vraie capacite a generaliser a un
nouvel utilisateur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, classification_report,
)

try:
    from . import config
except ImportError:
    import config


# ----------------------------------------------------------------------
# 1. CHARGEMENT DU TABLEAU DE FEATURES
# ----------------------------------------------------------------------
def load_features(path=None):
    """
    Charge le tableau de features produit au Sprint 3.

    Retour
    ------
    (X, y, groups, feature_names)
      X : np.ndarray (n_epochs, n_features) — les features
      y : np.ndarray (n_epochs,)            — les etiquettes (label)
      groups : np.ndarray (n_epochs,)       — le sujet de chaque epoch
      feature_names : list[str]             — noms des colonnes de features
    """
    if path is None:
        path = config.PROCESSED_DIR / "features.csv"

    df = pd.read_csv(path)

    meta_cols = ["subject", "label"]
    feature_names = [c for c in df.columns if c not in meta_cols]

    X = df[feature_names].values
    y = df["label"].values
    groups = df["subject"].values
    return X, y, groups, feature_names


# ----------------------------------------------------------------------
# 2. CONSTRUCTION DU PIPELINE
# ----------------------------------------------------------------------
def make_pipeline(n_estimators=300, max_depth=None, random_state=None):
    """
    Construit le pipeline de classification : normalisation + Random Forest.

    Le pipeline encapsule la normalisation (StandardScaler) AVEC le modele.
    C'est essentiel : la normalisation est ainsi apprise uniquement sur les
    donnees d'entrainement de chaque pli, jamais sur le test (sinon fuite).

    Parametres
    ----------
    n_estimators : nombre d'arbres de la foret
    max_depth : profondeur maximale (None = illimitee)
    random_state : graine (defaut : config.RANDOM_SEED)

    Retour
    ------
    sklearn Pipeline pret a etre entraine.
    """
    if random_state is None:
        random_state = config.RANDOM_SEED

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,                  # utilise tous les coeurs CPU
            class_weight="balanced",    # gere un eventuel desequilibre des classes
        )),
    ])


# ----------------------------------------------------------------------
# 3. EVALUATION PAR SUJET (LOSO)
# ----------------------------------------------------------------------
def evaluate_loso(X, y, groups, pipeline=None, verbose=True):
    """
    Evalue le modele en validation Leave-One-Subject-Out.

    A chaque tour : un sujet sert de test, les autres d'entrainement.
    On agrege les predictions de tous les tours pour calculer les metriques
    globales, et on garde le score par sujet pour mesurer la stabilite.

    NB : s'il n'y a qu'un seul sujet, LOSO est impossible — la fonction
    le signale et renvoie None (il faut au moins 2 sujets).

    Retour
    ------
    dict avec :
      y_true, y_pred : etiquettes reelles et predites (agregees)
      macro_f1, balanced_acc, accuracy : metriques globales
      per_subject : DataFrame (un score par sujet de test)
      labels : ordre des classes pour la matrice de confusion
    """
    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
        if verbose:
            print("ATTENTION : il faut au moins 2 sujets pour la validation LOSO.")
            print(f"Sujets disponibles : {n_subjects}.")
        return None

    if pipeline is None:
        pipeline = make_pipeline()

    logo = LeaveOneGroupOut()
    labels = sorted(np.unique(y))

    y_true_all, y_pred_all = [], []
    per_subject_rows = []

    if verbose:
        print(f"Validation Leave-One-Subject-Out sur {n_subjects} sujets :\n")

    for train_idx, test_idx in logo.split(X, y, groups):
        test_subject = groups[test_idx][0]

        # On clone le pipeline a chaque pli pour repartir propre
        from sklearn.base import clone
        pipe = clone(pipeline)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        y_true = y[test_idx]

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        per_subject_rows.append({
            "subject": test_subject,
            "n_epochs": len(test_idx),
            "macro_f1": round(f1, 3),
            "accuracy": round(acc, 3),
        })
        if verbose:
            print(f"  Sujet {test_subject:>3} en test : "
                  f"macro-F1 = {f1:.3f} | accuracy = {acc:.3f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    results = {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "macro_f1": f1_score(y_true_all, y_pred_all, average="macro",
                             labels=labels, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true_all, y_pred_all),
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "per_subject": pd.DataFrame(per_subject_rows),
        "labels": labels,
    }
    return results


def print_evaluation_report(results):
    """Affiche un rapport de synthese a partir du dict d'evaluate_loso."""
    if results is None:
        return
    print("\n" + "=" * 52)
    print("RESULTATS GLOBAUX (validation par sujet)")
    print("=" * 52)
    print(f"  Macro-F1            : {results['macro_f1']:.3f}")
    print(f"  Exactitude equilibree: {results['balanced_acc']:.3f}")
    print(f"  Exactitude          : {results['accuracy']:.3f}")

    ps = results["per_subject"]
    print(f"\n  Stabilite entre sujets (macro-F1) :")
    print(f"     moyenne = {ps['macro_f1'].mean():.3f} | "
          f"ecart-type = {ps['macro_f1'].std():.3f}")
    print(f"     min = {ps['macro_f1'].min():.3f} | "
          f"max = {ps['macro_f1'].max():.3f}")

    print("\n  Matrice de confusion :")
    cm = confusion_matrix(results["y_true"], results["y_pred"],
                          labels=results["labels"])
    header = "            " + "  ".join(f"{l[:8]:>8}" for l in results["labels"])
    print("  (lignes = reel, colonnes = predit)")
    print(header)
    for i, lab in enumerate(results["labels"]):
        row = "  ".join(f"{v:>8}" for v in cm[i])
        print(f"  {lab[:10]:>10}  {row}")


# ----------------------------------------------------------------------
# 4. COMPARAISON DE LA REFERENCE ALEATOIRE
# ----------------------------------------------------------------------
def baseline_score(y):
    """
    Renvoie le macro-F1 d'un classifieur naif (predit toujours la classe
    majoritaire). Sert de point de comparaison : le modele doit faire mieux.
    """
    labels = sorted(np.unique(y))
    values, counts = np.unique(y, return_counts=True)
    majority = values[np.argmax(counts)]
    y_pred = np.full(len(y), majority)
    return f1_score(y, y_pred, average="macro", labels=labels, zero_division=0)


# ----------------------------------------------------------------------
# 5. ENTRAINEMENT FINAL & SAUVEGARDE
# ----------------------------------------------------------------------
def train_final_model(X, y, pipeline=None, save=True, path=None):
    """
    Entraine le modele final sur TOUTES les donnees disponibles et le sauvegarde.

    Ce modele est celui que l'application web (Sprint 5) utilisera pour
    predire l'etat d'un nouveau fichier. L'evaluation honnete a deja ete
    faite par LOSO ; ici on exploite toutes les donnees pour le modele
    de production.

    Retour
    ------
    (pipeline entraine, chemin de sauvegarde ou None)
    """
    if pipeline is None:
        pipeline = make_pipeline()
    pipeline.fit(X, y)

    saved_path = None
    if save:
        if path is None:
            path = config.MODELS_DIR / "model.joblib"
        joblib.dump(pipeline, path)
        saved_path = path
    return pipeline, saved_path


def top_feature_importances(pipeline, feature_names, top=15):
    """
    Renvoie les features les plus importantes du Random Forest entraine.

    Utile pour interpreter : quelles caracteristiques le modele utilise-t-il
    le plus pour distinguer les etats ?

    Retour
    ------
    pandas.DataFrame (feature, importance) trie par importance decroissante.
    """
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


if __name__ == "__main__":
    import os
    feat_path = config.PROCESSED_DIR / "features.csv"
    if not os.path.exists(feat_path):
        print("Tableau de features introuvable.")
        print("-> Lancez d'abord le Sprint 3 (notebook 03 ou python src/features.py)")
    else:
        X, y, groups, names = load_features()
        print(f"Donnees : {X.shape[0]} epochs, {X.shape[1]} features, "
              f"{len(np.unique(groups))} sujets\n")

        # Reference naive
        print(f"Reference (classe majoritaire) : macro-F1 = {baseline_score(y):.3f}\n")

        # Evaluation par sujet
        results = evaluate_loso(X, y, groups)
        print_evaluation_report(results)

        # Modele final
        print("\nEntrainement du modele final sur toutes les donnees...")
        model, path = train_final_model(X, y)
        print(f"Modele sauvegarde : {path}")

        print("\nFeatures les plus importantes :")
        print(top_feature_importances(model, names, top=10).to_string(index=False))
