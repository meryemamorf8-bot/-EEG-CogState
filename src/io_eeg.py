"""
io_eeg.py - Lecture des fichiers EEG du dataset STEW
====================================================

Ce module gere l'importation des fichiers EEG bruts (F1 du cahier des
charges). Il sait :
  - lire un fichier STEW .txt en DataFrame (14 colonnes nommees)
  - deduire l'etiquette (relaxation / concentration) depuis le nom du fichier
  - lister tous les enregistrements presents dans data/raw/
  - construire le vecteur temps associe a un enregistrement

Format STEW attendu :
  - fichier texte, valeurs separees par des espaces
  - chaque ligne = un echantillon ; chaque colonne = un canal
  - 14 colonnes dans l'ordre de config.CHANNEL_NAMES
  - nom de fichier : subXX_lo.txt (repos) ou subXX_hi.txt (multitache)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Import des parametres centralises. Fonctionne que le module soit lance
# depuis la racine ou importe comme package src.io_eeg.
try:
    from . import config
except ImportError:  # execution directe (python src/io_eeg.py)
    import config


# Expression reguliere pour extraire le numero de sujet et la condition
# depuis un nom comme "sub07_hi.txt" -> sujet=7, suffixe="hi"
_FILENAME_RE = re.compile(r"sub(?P<subject>\d+)_(?P<suffix>lo|hi)", re.IGNORECASE)


def parse_filename(path: str | Path) -> dict:
    """
    Extrait les metadonnees encodees dans le nom d'un fichier STEW.

    Parametres
    ----------
    path : chemin ou nom du fichier (ex. "sub23_hi.txt")

    Retour
    ------
    dict avec les cles : subject (int), suffix (str), label (str)

    Leve
    ----
    ValueError si le nom ne correspond pas au format STEW attendu.
    """
    name = Path(path).name
    match = _FILENAME_RE.search(name)
    if match is None:
        raise ValueError(
            f"Nom de fichier non reconnu : '{name}'. "
            f"Format attendu : subXX_lo.txt ou subXX_hi.txt"
        )
    suffix = match.group("suffix").lower()
    return {
        "filename": name,
        "subject": int(match.group("subject")),
        "suffix": suffix,
        "label": config.LABEL_FROM_SUFFIX[suffix],
    }


def load_recording(path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier EEG STEW en DataFrame.

    Parametres
    ----------
    path : chemin vers le fichier .txt

    Retour
    ------
    pandas.DataFrame de forme (n_echantillons, 14), colonnes = noms des canaux.

    Leve
    ----
    FileNotFoundError si le fichier n'existe pas.
    ValueError si le nombre de colonnes ne correspond pas a 14 canaux.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    # Les fichiers STEW sont separes par des espaces (un ou plusieurs).
    # sep=r"\s+" gere les espaces multiples ; header=None car pas d'en-tete.
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # Controle d'integrite : on doit retrouver exactement 14 canaux.
    if df.shape[1] != config.N_CHANNELS:
        raise ValueError(
            f"{path.name} : {df.shape[1]} colonnes trouvees, "
            f"{config.N_CHANNELS} attendues. Fichier peut-etre corrompu "
            f"ou format inattendu."
        )

    df.columns = config.CHANNEL_NAMES
    return df


def get_time_vector(n_samples: int, sampling_rate: int | None = None) -> np.ndarray:
    """
    Construit le vecteur temps (en secondes) pour un enregistrement.

    Parametres
    ----------
    n_samples : nombre d'echantillons
    sampling_rate : frequence en Hz (par defaut config.SAMPLING_RATE)

    Retour
    ------
    numpy.ndarray de longueur n_samples, de 0 a (n_samples-1)/fs.
    """
    if sampling_rate is None:
        sampling_rate = config.SAMPLING_RATE
    return np.arange(n_samples) / sampling_rate


def list_recordings(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Recense tous les fichiers EEG presents dans le dossier brut.

    Parametres
    ----------
    raw_dir : dossier a scanner (par defaut config.RAW_DIR)

    Retour
    ------
    pandas.DataFrame avec une ligne par fichier valide et les colonnes :
        path, filename, subject, suffix, label
    Trie par sujet puis condition. DataFrame vide si aucun fichier.
    """
    if raw_dir is None:
        raw_dir = config.RAW_DIR
    raw_dir = Path(raw_dir)

    rows = []
    # On accepte .txt majuscule ou minuscule
    for path in sorted(raw_dir.glob("*.txt")) + sorted(raw_dir.glob("*.TXT")):
        try:
            meta = parse_filename(path)
        except ValueError:
            # Fichier txt qui ne suit pas la convention STEW : on l'ignore
            continue
        rows.append({
            "path": str(path),
            "filename": path.name,
            "subject": meta["subject"],
            "suffix": meta["suffix"],
            "label": meta["label"],
        })

    df = pd.DataFrame(rows, columns=["path", "filename", "subject", "suffix", "label"])
    if not df.empty:
        df = df.sort_values(["subject", "suffix"]).reset_index(drop=True)
    return df


def describe_recording(path: str | Path) -> dict:
    """
    Renvoie un petit resume d'un enregistrement (utile pour l'exploration).

    Retour
    ------
    dict : filename, subject, label, n_samples, duration_sec, n_channels
    """
    meta = parse_filename(path)
    df = load_recording(path)
    n_samples = len(df)
    return {
        "filename": Path(path).name,
        "subject": meta["subject"],
        "label": meta["label"],
        "n_samples": n_samples,
        "duration_sec": round(n_samples / config.SAMPLING_RATE, 2),
        "n_channels": df.shape[1],
    }


if __name__ == "__main__":
    # Demonstration : recense les fichiers et decrit le premier trouve.
    print("Recherche de fichiers STEW dans :", config.RAW_DIR)
    catalogue = list_recordings()

    if catalogue.empty:
        print("\nAucun fichier STEW trouve.")
        print("-> Telechargez le dataset et placez les .txt dans data/raw/")
    else:
        print(f"\n{len(catalogue)} fichier(s) trouve(s) :\n")
        print(catalogue.to_string(index=False))

        first = catalogue.iloc[0]["path"]
        print("\nResume du premier enregistrement :")
        for k, v in describe_recording(first).items():
            print(f"  {k:14s}: {v}")
