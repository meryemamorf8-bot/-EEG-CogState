"""
config.py - Configuration centrale du projet EEG-CogState
==========================================================

Tous les paramètres du projet sont centralisés ici afin de garantir
la reproductibilite : les memes constantes sont utilisees partout
(lecture, pretraitement, features, modele).

Dataset cible : STEW (Simultaneous Task EEG Workload)
  - 48 sujets
  - Appareil Emotiv EPOC : 14 canaux, 128 Hz
  - 2.5 minutes d'enregistrement par cas
  - Fichiers .txt : subXX_lo.txt (repos) / subXX_hi.txt (multitache)
"""

from pathlib import Path

# ----------------------------------------------------------------------
# 1. CHEMINS DU PROJET
# ----------------------------------------------------------------------
# Racine du projet (dossier EEG-CogState/), calculee automatiquement
# a partir de l'emplacement de ce fichier (src/config.py -> parent.parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # fichiers STEW bruts (.txt)
PROCESSED_DIR = DATA_DIR / "processed"  # donnees nettoyees / features
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Cree les dossiers de sortie s'ils n'existent pas (sans erreur si deja la)
for _d in (PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. PARAMETRES D'ACQUISITION (specifiques a STEW / Emotiv EPOC)
# ----------------------------------------------------------------------
SAMPLING_RATE = 128  # Hz - frequence d'echantillonnage de l'Emotiv EPOC

# Les 14 canaux de l'Emotiv EPOC, dans l'ordre des colonnes des fichiers STEW
CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
N_CHANNELS = len(CHANNEL_NAMES)  # 14

# Positions 2D approximatives des electrodes (vue de dessus du scalp),
# coordonnees normalisees dans [-1, 1] : x = gauche(-)/droite(+),
# y = arriere(-)/avant(+). Sert a dessiner la topographie (topomap).
# Ces positions suivent le systeme international 10-20 pour le casque Emotiv EPOC.
CHANNEL_POSITIONS = {
    "AF3": (-0.25, 0.83), "AF4": (0.25, 0.83),
    "F7": (-0.71, 0.59),  "F3": (-0.38, 0.60),
    "F4": (0.38, 0.60),   "F8": (0.71, 0.59),
    "FC5": (-0.58, 0.30), "FC6": (0.58, 0.30),
    "T7": (-0.84, 0.0),   "T8": (0.84, 0.0),
    "P7": (-0.71, -0.59), "P8": (0.71, -0.59),
    "O1": (-0.25, -0.83), "O2": (0.25, -0.83),
}

# ----------------------------------------------------------------------
# 3. ETIQUETTES (labels) DES ETATS
# ----------------------------------------------------------------------
# Dans STEW, le suffixe du fichier indique la condition :
#   "lo" = repos (low workload)  -> on l'associe a la relaxation
#   "hi" = multitache (high workload) -> on l'associe a la concentration
LABEL_FROM_SUFFIX = {
    "lo": "relaxation",
    "hi": "concentration",
}
# Liste ordonnee des classes (utile plus tard pour le modele)
CLASSES = ["relaxation", "concentration"]

# ----------------------------------------------------------------------
# 4. BANDES DE FREQUENCE EEG (bornes en Hz)
# ----------------------------------------------------------------------
# IMPORTANT : il n'existe pas de consensus strict sur ces bornes.
# Celles-ci sont une convention repandue ; on les declare ici une fois
# pour toutes et on ne les change plus entre entrainement et inference.
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),  # plafond a 45 Hz : au-dela, signal souvent bruite
}

# ----------------------------------------------------------------------
# 5. PRETRAITEMENT (utilise au Sprint 2 - defini ici par anticipation)
# ----------------------------------------------------------------------
BANDPASS_LOW = 1.0    # Hz - frequence de coupure basse du filtre passe-bande
BANDPASS_HIGH = 45.0  # Hz - frequence de coupure haute
NOTCH_FREQ = 50.0     # Hz - filtre coupe-bande (secteur europeen ; 60 Hz aux USA)

EPOCH_SECONDS = 2.0   # duree d'une fenetre (epoch) en secondes
EPOCH_OVERLAP = 0.5   # recouvrement entre epochs (0.5 = 50 %)

# ----------------------------------------------------------------------
# 6. REPRODUCTIBILITE
# ----------------------------------------------------------------------
RANDOM_SEED = 42  # graine fixee partout (numpy, sklearn) pour des resultats stables


if __name__ == "__main__":
    # Petit affichage de controle quand on lance directement ce fichier
    print("Configuration EEG-CogState")
    print("-" * 40)
    print(f"Racine projet : {PROJECT_ROOT}")
    print(f"Dossier brut  : {RAW_DIR}")
    print(f"Frequence     : {SAMPLING_RATE} Hz")
    print(f"Canaux ({N_CHANNELS})   : {', '.join(CHANNEL_NAMES)}")
    print(f"Classes       : {', '.join(CLASSES)}")
    print(f"Bandes        : {list(FREQ_BANDS.keys())}")
