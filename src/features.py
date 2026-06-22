"""
features.py - Extraction de caracteristiques EEG (Sprint 3)
===========================================================

Ce module transforme chaque epoch (une fenetre de signal) en un vecteur
de nombres exploitable par un modele de machine learning. Il correspond
a la fonctionnalite F3 du cahier des charges.

Pour chaque epoch et chaque canal, on calcule :
  1. La puissance de chaque bande de frequence (delta -> gamma)
  2. Des ratios inter-bandes (beta/alpha, theta/beta, ...)
  3. Des indicateurs statistiques (variance, Hjorth, entropie spectrale)

La sortie finale est un tableau (DataFrame) : une ligne par epoch,
une colonne par feature, plus les colonnes 'label' et 'subject'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

try:
    from . import config, io_eeg, preprocessing
except ImportError:
    import config
    import io_eeg
    import preprocessing


# Compatibilite NumPy : trapz renomme en trapezoid dans les versions recentes
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ----------------------------------------------------------------------
# 1. PUISSANCE DES BANDES DE FREQUENCE
# ----------------------------------------------------------------------
def band_powers(signal_1d, fs=None):
    """
    Calcule la puissance de chaque bande de frequence d'un signal 1D.

    On estime la densite spectrale de puissance (PSD) par la methode de
    Welch, puis on integre la PSD sur chaque bande (aire sous la courbe).

    Parametres
    ----------
    signal_1d : np.ndarray 1D (un canal, un epoch)
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)

    Retour
    ------
    dict {nom_bande: puissance_absolue} + {nom_bande + '_rel': puissance_relative}
    La puissance relative = part de la bande dans la puissance totale (somme = 1).
    """
    if fs is None:
        fs = config.SAMPLING_RATE

    # nperseg adapte a la longueur de l'epoch (mais <= longueur du signal)
    nperseg = min(len(signal_1d), fs * 2)
    freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg)

    out = {}
    total = 0.0
    abs_powers = {}
    for band, (lo, hi) in config.FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        # Integration de la PSD sur la bande (methode des trapezes)
        power = _trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
        abs_powers[band] = power
        out[f"{band}_power"] = power
        total += power

    # Puissances relatives (eviter la division par zero)
    for band in config.FREQ_BANDS:
        out[f"{band}_rel"] = abs_powers[band] / total if total > 0 else 0.0

    return out


# ----------------------------------------------------------------------
# 2. RATIOS INTER-BANDES
# ----------------------------------------------------------------------
def band_ratios(powers):
    """
    Calcule les ratios inter-bandes a partir des puissances absolues.

    Ces ratios sont souvent plus discriminants que les puissances brutes :
      - beta/alpha : indice de vigilance / concentration
      - theta/beta : indice de somnolence
      - (theta+alpha)/beta : charge mentale
      - alpha/delta : eveil

    Parametres
    ----------
    powers : dict issu de band_powers (doit contenir les cles '<bande>_power')

    Retour
    ------
    dict {nom_ratio: valeur}
    """
    eps = 1e-10  # evite la division par zero
    p = {b: powers[f"{b}_power"] for b in config.FREQ_BANDS}

    return {
        "ratio_beta_alpha": p["beta"] / (p["alpha"] + eps),
        "ratio_theta_beta": p["theta"] / (p["beta"] + eps),
        "ratio_theta_alpha_beta": (p["theta"] + p["alpha"]) / (p["beta"] + eps),
        "ratio_alpha_delta": p["alpha"] / (p["delta"] + eps),
    }


# ----------------------------------------------------------------------
# 3. INDICATEURS STATISTIQUES & DE COMPLEXITE
# ----------------------------------------------------------------------
def hjorth_parameters(signal_1d):
    """
    Calcule les trois parametres de Hjorth (mesures de la forme du signal).

    - Activite  : variance du signal (puissance)
    - Mobilite  : frequence moyenne (variance de la derivee / variance)
    - Complexite: variation de frequence (regularite du signal)

    Retour
    ------
    dict {hjorth_activity, hjorth_mobility, hjorth_complexity}
    """
    eps = 1e-10
    x = signal_1d
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + eps))
    complexity = np.sqrt(var_ddx / (var_dx + eps)) / (mobility + eps)

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def spectral_entropy(signal_1d, fs=None):
    """
    Calcule l'entropie spectrale : mesure de la "complexite" du spectre.

    Une entropie elevee = energie repartie sur beaucoup de frequences
    (signal complexe / bruite) ; une entropie faible = energie concentree
    sur peu de frequences (signal regulier, ex. forte onde alpha).

    Retour
    ------
    float (entropie normalisee entre 0 et 1).
    """
    if fs is None:
        fs = config.SAMPLING_RATE
    eps = 1e-10

    nperseg = min(len(signal_1d), fs * 2)
    _, psd = welch(signal_1d, fs=fs, nperseg=nperseg)

    psd_norm = psd / (psd.sum() + eps)          # distribution de probabilite
    psd_norm = psd_norm[psd_norm > 0]            # on garde les valeurs > 0
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    # Normalisation par l'entropie maximale (log2 du nombre de points)
    return entropy / np.log2(len(psd_norm)) if len(psd_norm) > 1 else 0.0


def statistical_features(signal_1d, fs=None):
    """
    Regroupe les indicateurs statistiques et de complexite d'un signal 1D.

    Retour
    ------
    dict : variance, ecart-type, Hjorth (x3), entropie spectrale,
           taux de passage par zero.
    """
    out = {
        "variance": np.var(signal_1d),
        "std": np.std(signal_1d),
        "zero_crossing": int(np.sum(np.diff(np.sign(signal_1d)) != 0)),
    }
    out.update(hjorth_parameters(signal_1d))
    out["spectral_entropy"] = spectral_entropy(signal_1d, fs=fs)
    return out


# ----------------------------------------------------------------------
# 4. FEATURES D'UN EPOCH (TOUS CANAUX)
# ----------------------------------------------------------------------
def extract_epoch_features(epoch, fs=None, channels=None):
    """
    Extrait toutes les features d'un epoch (toutes bandes, ratios, stats),
    pour chaque canal.

    Parametres
    ----------
    epoch : np.ndarray de forme (n_echantillons, n_canaux)
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)
    channels : noms des canaux (defaut : config.CHANNEL_NAMES)

    Retour
    ------
    dict {nom_feature: valeur}. Les noms sont prefixes par le canal,
    ex. "O1_alpha_power", "F3_ratio_beta_alpha", "T7_variance".
    """
    if channels is None:
        channels = config.CHANNEL_NAMES

    features = {}
    for ci, ch in enumerate(channels):
        sig = epoch[:, ci]

        powers = band_powers(sig, fs=fs)
        ratios = band_ratios(powers)
        stats = statistical_features(sig, fs=fs)

        for k, v in {**powers, **ratios, **stats}.items():
            features[f"{ch}_{k}"] = v

    return features


# ----------------------------------------------------------------------
# 5. CONSTRUCTION DU TABLEAU DE FEATURES (TOUS FICHIERS)
# ----------------------------------------------------------------------
def build_feature_table(catalogue=None, fs=None, verbose=True):
    """
    Construit le tableau complet de features a partir de tous les fichiers.

    Pour chaque fichier : pretraitement (Sprint 2) -> epochs, puis extraction
    des features de chaque epoch. Chaque ligne du tableau = un epoch.

    Parametres
    ----------
    catalogue : DataFrame de io_eeg.list_recordings() (defaut : auto)
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)
    verbose : afficher la progression

    Retour
    ------
    pandas.DataFrame : colonnes de features + 'label' + 'subject'.
    Une ligne par epoch.
    """
    if catalogue is None:
        catalogue = io_eeg.list_recordings()

    if catalogue.empty:
        return pd.DataFrame()

    rows = []
    for _, rec in catalogue.iterrows():
        result = preprocessing.preprocess_file(rec["path"], fs=fs)
        epochs = result["epochs"]

        if verbose:
            print(f"  {rec['filename']:14s} [{result['label']:14s}] "
                  f"-> {result['n_epochs']} epochs")

        for i in range(epochs.shape[0]):
            feats = extract_epoch_features(epochs[i], fs=fs)
            feats["label"] = result["label"]
            feats["subject"] = result["subject"]
            rows.append(feats)

    table = pd.DataFrame(rows)

    # On place 'label' et 'subject' en premieres colonnes pour la lisibilite
    meta_cols = ["subject", "label"]
    feat_cols = [c for c in table.columns if c not in meta_cols]
    return table[meta_cols + feat_cols]


def save_feature_table(table, path=None):
    """
    Sauvegarde le tableau de features en CSV dans data/processed/.

    Retour
    ------
    Path du fichier ecrit.
    """
    if path is None:
        path = config.PROCESSED_DIR / "features.csv"
    table.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    catalogue = io_eeg.list_recordings()
    if catalogue.empty:
        print("Aucun fichier dans data/raw/. Placez-y des fichiers STEW.")
    else:
        print("Extraction des features sur tous les fichiers :\n")
        table = build_feature_table(catalogue)

        print(f"\nTableau de features : {table.shape[0]} epochs x "
              f"{table.shape[1]} colonnes")
        print(f"  ({table.shape[1] - 2} features + 2 colonnes meta)")
        print(f"\nRepartition des classes :")
        print(table["label"].value_counts().to_string())

        out = save_feature_table(table)
        print(f"\nTableau sauvegarde : {out}")
