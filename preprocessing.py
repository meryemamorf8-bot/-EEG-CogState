"""
preprocessing.py - Pretraitement des signaux EEG (Sprint 2)
===========================================================

Ce module nettoie le signal brut et le decoupe en fenetres (epochs).
Il correspond a la fonctionnalite F2 du cahier des charges.

Chaine de traitement (dans l'ordre) :
  1. Filtre passe-bande (1-45 Hz)  -> enleve derives lentes + hautes freqs
  2. Filtre notch (50 Hz)          -> enleve le bruit du secteur electrique
  3. Segmentation en epochs        -> decoupe en fenetres de 2 s

Toutes les operations travaillent sur des tableaux numpy de forme
(n_echantillons, n_canaux), coherents avec la sortie de io_eeg.load_recording().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

try:
    from . import config, io_eeg
except ImportError:
    import config
    import io_eeg


# ----------------------------------------------------------------------
# 1. FILTRE PASSE-BANDE
# ----------------------------------------------------------------------
def bandpass_filter(data, low=None, high=None, fs=None, order=4):
    """
    Applique un filtre passe-bande de Butterworth (sans dephasage).

    On utilise sosfiltfilt : representation en sections du second ordre (sos),
    plus stable numeriquement, et filtrage aller-retour (zero-phase) qui
    ne decale pas le signal dans le temps.

    Parametres
    ----------
    data : np.ndarray de forme (n_echantillons, n_canaux) ou (n_echantillons,)
    low, high : frequences de coupure basse/haute en Hz
                (defaut : config.BANDPASS_LOW / config.BANDPASS_HIGH)
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)
    order : ordre du filtre (4 par defaut, bon compromis)

    Retour
    ------
    np.ndarray filtre, meme forme que l'entree.
    """
    if low is None:
        low = config.BANDPASS_LOW
    if high is None:
        high = config.BANDPASS_HIGH
    if fs is None:
        fs = config.SAMPLING_RATE

    nyq = 0.5 * fs  # frequence de Nyquist
    # Securite : la borne haute ne peut pas atteindre/depasser Nyquist
    high = min(high, nyq - 1.0)

    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    # axis=0 : on filtre le long du temps, canal par canal
    return sosfiltfilt(sos, data, axis=0)


# ----------------------------------------------------------------------
# 2. FILTRE NOTCH (COUPE-BANDE ETROIT)
# ----------------------------------------------------------------------
def notch_filter(data, freq=None, fs=None, quality=30.0):
    """
    Supprime une frequence precise (le secteur : 50 Hz en Europe).

    Parametres
    ----------
    data : np.ndarray (n_echantillons, n_canaux) ou (n_echantillons,)
    freq : frequence a supprimer en Hz (defaut : config.NOTCH_FREQ)
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)
    quality : facteur de qualite Q (plus grand = encoche plus etroite)

    Retour
    ------
    np.ndarray filtre. Si freq >= Nyquist, le signal est renvoye tel quel
    (le secteur n'est pas representable, donc rien a filtrer).
    """
    if freq is None:
        freq = config.NOTCH_FREQ
    if fs is None:
        fs = config.SAMPLING_RATE

    nyq = 0.5 * fs
    if freq >= nyq:
        # A 128 Hz, Nyquist = 64 Hz : 50 Hz est bien filtrable.
        # Cette garde protege si on change la frequence d'echantillonnage.
        return np.asarray(data)

    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, data, axis=0)


# ----------------------------------------------------------------------
# 3. CHAINE COMPLETE DE NETTOYAGE
# ----------------------------------------------------------------------
def clean_signal(data, fs=None):
    """
    Applique la chaine complete : passe-bande puis notch.

    Parametres
    ----------
    data : np.ndarray (n_echantillons, n_canaux) ou pandas.DataFrame
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)

    Retour
    ------
    np.ndarray nettoye, de forme (n_echantillons, n_canaux).
    """
    # Accepte un DataFrame (sortie de io_eeg) ou un tableau numpy
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=float)

    filtered = bandpass_filter(data, fs=fs)
    filtered = notch_filter(filtered, fs=fs)
    return filtered


# ----------------------------------------------------------------------
# 4. SEGMENTATION EN EPOCHS
# ----------------------------------------------------------------------
def segment_epochs(data, fs=None, epoch_seconds=None, overlap=None):
    """
    Decoupe le signal en fenetres (epochs) de duree fixe, avec recouvrement.

    Exemple : 150 s a 128 Hz, fenetres de 2 s sans recouvrement -> 75 epochs.
    Avec 50 % de recouvrement, on obtient ~149 epochs (plus d'exemples).

    Parametres
    ----------
    data : np.ndarray (n_echantillons, n_canaux) ou DataFrame
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)
    epoch_seconds : duree d'une fenetre en s (defaut : config.EPOCH_SECONDS)
    overlap : recouvrement entre 0 et 1 (defaut : config.EPOCH_OVERLAP)

    Retour
    ------
    np.ndarray de forme (n_epochs, n_echantillons_par_epoch, n_canaux).
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if fs is None:
        fs = config.SAMPLING_RATE
    if epoch_seconds is None:
        epoch_seconds = config.EPOCH_SECONDS
    if overlap is None:
        overlap = config.EPOCH_OVERLAP

    win = int(round(epoch_seconds * fs))       # taille d'une fenetre en echantillons
    step = int(round(win * (1.0 - overlap)))   # pas entre deux debuts de fenetre
    step = max(step, 1)                         # securite : pas au moins 1

    n_samples = data.shape[0]
    if n_samples < win:
        # Signal trop court pour meme une fenetre
        return np.empty((0, win, data.shape[1]))

    starts = range(0, n_samples - win + 1, step)
    epochs = np.stack([data[s:s + win] for s in starts], axis=0)
    return epochs


# ----------------------------------------------------------------------
# 5. PIPELINE COMPLET POUR UN FICHIER
# ----------------------------------------------------------------------
def preprocess_file(path, fs=None):
    """
    Charge un fichier, le nettoie, et le segmente en epochs.

    C'est la fonction "tout-en-un" qu'on appellera au Sprint 3.

    Parametres
    ----------
    path : chemin d'un fichier EEG STEW
    fs : frequence d'echantillonnage (defaut : config.SAMPLING_RATE)

    Retour
    ------
    dict avec :
        epochs : np.ndarray (n_epochs, n_samples_par_epoch, n_canaux)
        label  : etat cognitif (str)
        subject: numero de sujet (int)
        n_epochs : nombre d'epochs produits (int)
    """
    df = io_eeg.load_recording(path)
    meta = io_eeg.parse_filename(path)

    cleaned = clean_signal(df, fs=fs)
    epochs = segment_epochs(cleaned, fs=fs)

    return {
        "epochs": epochs,
        "label": meta["label"],
        "subject": meta["subject"],
        "n_epochs": epochs.shape[0],
    }


if __name__ == "__main__":
    # Demonstration sur le premier fichier disponible
    catalogue = io_eeg.list_recordings()
    if catalogue.empty:
        print("Aucun fichier dans data/raw/. Placez-y des fichiers STEW.")
    else:
        path = catalogue.iloc[0]["path"]
        print(f"Pretraitement de : {catalogue.iloc[0]['filename']}")

        df = io_eeg.load_recording(path)
        print(f"  Signal brut    : {df.shape} (echantillons x canaux)")

        cleaned = clean_signal(df)
        print(f"  Apres nettoyage: {cleaned.shape}")

        epochs = segment_epochs(cleaned)
        print(f"  Apres epoching : {epochs.shape} (epochs x echantillons x canaux)")
        print(f"  -> {epochs.shape[0]} epochs de {epochs.shape[1]} echantillons "
              f"({config.EPOCH_SECONDS}s) sur {epochs.shape[2]} canaux")
