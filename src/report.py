"""
report.py - Generation de rapport PDF (Sprint bonus)
====================================================

Ce module genere un rapport PDF synthetique apres une analyse, contenant
le resultat, les probabilites, le spectre et la repartition des bandes.
Il correspond a la fonctionnalite F9 du cahier des charges.

On utilise matplotlib (deja installe) avec son backend PDF : pas de
dependance supplementaire. Le PDF est cree en memoire (BytesIO) pour
pouvoir etre propose au telechargement dans l'application web.
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sans affichage (pour serveur)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch

try:
    from . import config, preprocessing, features
except ImportError:
    import config
    import preprocessing
    import features


def generate_report(df, result, filename, channel="O1"):
    """
    Genere un rapport PDF de l'analyse et le renvoie sous forme d'octets.

    Parametres
    ----------
    df : DataFrame du signal EEG (echantillons x canaux)
    result : dict renvoye par la prediction (predicted_label, probabilities,
             n_epochs)
    filename : nom du fichier analyse (pour l'afficher dans le rapport)
    channel : canal a illustrer dans les graphiques

    Retour
    ------
    bytes : le contenu du PDF (a ecrire dans un fichier ou proposer au
    telechargement).
    """
    fs = config.SAMPLING_RATE
    pred = result["predicted_label"]
    conf = result["probabilities"][pred] * 100

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---------- PAGE 1 : synthese ----------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.suptitle("EEG-CogState — Rapport d'analyse", fontsize=18,
                     fontweight="bold", y=0.97, color="#1F3864")

        # Bloc d'informations (texte)
        info_lines = [
            f"Fichier analysé : {filename}",
            f"Date du rapport : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Durée du signal : {len(df) / fs:.0f} s   |   "
            f"Canaux : {df.shape[1]}   |   Fréquence : {fs} Hz",
            f"Fenêtres analysées (epochs) : {result['n_epochs']}",
        ]
        fig.text(0.08, 0.90, "\n".join(info_lines), fontsize=10.5,
                 va="top", color="#333333", linespacing=1.8)

        # Resultat principal (encadre)
        fig.text(0.08, 0.79, "Résultat", fontsize=13, fontweight="bold",
                 color="#2E5496")
        fig.text(0.08, 0.755, f"État cognitif estimé :  {pred.upper()}",
                 fontsize=14, fontweight="bold", color="#0F6E56")
        fig.text(0.08, 0.725, f"Confiance :  {conf:.1f} %", fontsize=12,
                 color="#333333")

        # Graphe des probabilites
        ax_prob = fig.add_axes([0.08, 0.50, 0.84, 0.16])
        classes = list(result["probabilities"].keys())
        values = [result["probabilities"][c] * 100 for c in classes]
        colors = ["#1D9E75" if c == pred else "#9DB8DC" for c in classes]
        ax_prob.barh(classes, values, color=colors)
        ax_prob.set_xlim(0, 100)
        ax_prob.set_xlabel("Probabilité (%)")
        ax_prob.set_title("Probabilités par état", fontsize=11, loc="left")
        for i, v in enumerate(values):
            ax_prob.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

        # Repartition des bandes (sur le canal choisi)
        cleaned = preprocessing.clean_signal(df)
        epochs = preprocessing.segment_epochs(cleaned)
        ci = config.CHANNEL_NAMES.index(channel)
        band_means = {}
        for b in config.FREQ_BANDS:
            rels = [features.band_powers(epochs[i, :, ci])[f"{b}_rel"]
                    for i in range(epochs.shape[0])]
            band_means[b] = np.mean(rels) * 100

        ax_band = fig.add_axes([0.08, 0.27, 0.84, 0.16])
        band_colors = {"delta": "#85B7EB", "theta": "#5DCAA5", "alpha": "#1D9E75",
                       "beta": "#EF9F27", "gamma": "#E24B4A"}
        ax_band.bar(list(band_means.keys()), list(band_means.values()),
                    color=[band_colors[b] for b in band_means])
        ax_band.set_ylabel("Puissance relative (%)")
        ax_band.set_title(f"Répartition des bandes — canal {channel}",
                          fontsize=11, loc="left")

        # Avertissement en pied de page
        fig.text(0.08, 0.06,
                 "Outil pédagogique et exploratoire. Ne constitue pas un "
                 "dispositif médical.\nLes résultats dépendent de la qualité "
                 "du signal et du modèle entraîné.",
                 fontsize=8, color="#888888", va="bottom", style="italic")

        pdf.savefig(fig)
        plt.close(fig)

        # ---------- PAGE 2 : spectre ----------
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.suptitle("Analyse spectrale", fontsize=15, fontweight="bold",
                      y=0.96, color="#1F3864")

        sig = cleaned[:, ci]
        f, psd = welch(sig, fs=fs, nperseg=min(512, len(sig)))
        ax_psd = fig2.add_axes([0.10, 0.55, 0.82, 0.32])
        ax_psd.semilogy(f, psd, color="#1F3864", linewidth=1.3)
        for band, (lo, hi) in config.FREQ_BANDS.items():
            ax_psd.axvspan(lo, hi, alpha=0.12, color=band_colors[band])
        ax_psd.set_xlim(0, 45)
        ax_psd.set_xlabel("Fréquence (Hz)")
        ax_psd.set_ylabel("PSD (µV²/Hz)")
        ax_psd.set_title(f"Densité spectrale de puissance — canal {channel}",
                         fontsize=11, loc="left")

        # Signal temporel (extrait)
        n = min(int(5 * fs), len(df))
        t = np.arange(n) / fs
        ax_sig = fig2.add_axes([0.10, 0.13, 0.82, 0.30])
        ax_sig.plot(t, df[channel].values[:n], color="#2E5496", linewidth=0.6)
        ax_sig.set_xlabel("Temps (s)")
        ax_sig.set_ylabel("Amplitude (µV)")
        ax_sig.set_title(f"Signal brut (5 s) — canal {channel}",
                         fontsize=11, loc="left")

        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    import os, joblib
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    feat_path = config.PROCESSED_DIR / "features.csv"
    model_path = config.MODELS_DIR / "model.joblib"
    sample = config.RAW_DIR
    files = list(config.RAW_DIR.glob("*.txt"))

    if not (model_path.exists() and files):
        print("Il faut un modele entraine et des fichiers dans data/raw/.")
    else:
        import pandas as pd
        from src import io_eeg
        path = files[0]
        df = io_eeg.load_recording(path)
        model = joblib.load(model_path)

        # Prediction rapide
        cleaned = preprocessing.clean_signal(df)
        epochs = preprocessing.segment_epochs(cleaned)
        feat_rows = [features.extract_epoch_features(epochs[i])
                     for i in range(epochs.shape[0])]
        X = pd.DataFrame(feat_rows).values
        proba = model.predict_proba(X).mean(axis=0)
        classes = list(model.classes_)
        result = {
            "predicted_label": classes[int(np.argmax(proba))],
            "probabilities": {classes[i]: float(proba[i]) for i in range(len(classes))},
            "n_epochs": epochs.shape[0],
        }

        pdf_bytes = generate_report(df, result, path.name)
        out = config.RESULTS_DIR / "rapport_demo.pdf"
        out.write_bytes(pdf_bytes)
        print(f"Rapport PDF genere : {out} ({len(pdf_bytes)} octets)")
