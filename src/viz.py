"""
viz.py - Visualisation des signaux EEG (Sprint 1)
=================================================

Fonctions de trace pour explorer le signal brut. Au Sprint 1, on se
limite a une visualisation simple : signaux temporels par canal et
spectre rapide pour verifier la qualite des donnees.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config, io_eeg
except ImportError:
    import config
    import io_eeg


def plot_raw_signals(path, n_seconds=5, n_channels=6, save_to=None):
    """
    Trace les premieres secondes de plusieurs canaux EEG (signal brut).

    Parametres
    ----------
    path : chemin du fichier EEG
    n_seconds : duree a afficher (secondes)
    n_channels : nombre de canaux a tracer (les premiers)
    save_to : si fourni, sauvegarde la figure a ce chemin
    """
    df = io_eeg.load_recording(path)
    meta = io_eeg.parse_filename(path)
    fs = config.SAMPLING_RATE

    n = min(int(n_seconds * fs), len(df))
    t = io_eeg.get_time_vector(n)
    channels = config.CHANNEL_NAMES[:n_channels]

    fig, axes = plt.subplots(n_channels, 1, figsize=(11, 1.6 * n_channels),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ax.plot(t, df[ch].values[:n], linewidth=0.6, color="#2E5496")
        ax.set_ylabel(ch, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle(
        f"Signal EEG brut - {meta['filename']} "
        f"(etat : {meta['label']}, {n_seconds}s, {fs} Hz)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
        print(f"Figure sauvegardee : {save_to}")
    return fig


def plot_psd(path, channel="O1", save_to=None):
    """
    Trace la densite spectrale de puissance (PSD) d'un canal via Welch.

    Permet de visualiser quelles bandes de frequence dominent : pic vers
    10 Hz (alpha) = repos/relaxation ; energie vers 20 Hz (beta) = effort.

    Parametres
    ----------
    path : chemin du fichier EEG
    channel : nom du canal a analyser
    save_to : si fourni, sauvegarde la figure
    """
    from scipy.signal import welch

    df = io_eeg.load_recording(path)
    meta = io_eeg.parse_filename(path)
    fs = config.SAMPLING_RATE

    freqs, psd = welch(df[channel].values, fs=fs, nperseg=min(256, len(df)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(freqs, psd, color="#1F3864", linewidth=1.2)

    # Colorer les bandes de frequence pour les reperer visuellement
    band_colors = {
        "delta": "#E8EEF7", "theta": "#D5E0F0", "alpha": "#B9CDE8",
        "beta": "#9DB8DC", "gamma": "#E8EEF7",
    }
    for band, (lo, hi) in config.FREQ_BANDS.items():
        ax.axvspan(lo, hi, alpha=0.4, color=band_colors.get(band, "#EEE"),
                   label=f"{band} ({lo}-{hi} Hz)")

    ax.set_xlim(0, 45)
    ax.set_xlabel("Frequence (Hz)")
    ax.set_ylabel("Densite spectrale de puissance (uV2/Hz)")
    ax.set_title(
        f"Spectre (PSD) - {meta['filename']} - canal {channel} "
        f"(etat : {meta['label']})", fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
        print(f"Figure sauvegardee : {save_to}")
    return fig


def plot_before_after(path, channel="O1", n_seconds=4, save_to=None):
    """
    Compare le signal brut et le signal nettoye (Sprint 2).

    Affiche deux lignes : en haut le signal temporel avant/apres,
    en bas les spectres (PSD) avant/apres. On voit alors disparaitre
    le bruit secteur (pic a 50 Hz) et la derive basse frequence.

    Necessite preprocessing (importe localement pour ne pas alourdir
    le module quand on n'utilise que les fonctions du Sprint 1).
    """
    from scipy.signal import welch
    try:
        from . import preprocessing
    except ImportError:
        import preprocessing

    df = io_eeg.load_recording(path)
    meta = io_eeg.parse_filename(path)
    fs = config.SAMPLING_RATE

    raw = df[channel].values
    cleaned = preprocessing.clean_signal(df)[:, config.CHANNEL_NAMES.index(channel)]

    n = min(int(n_seconds * fs), len(raw))
    t = io_eeg.get_time_vector(n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # Ligne 1 : signal temporel
    axes[0, 0].plot(t, raw[:n], linewidth=0.6, color="#B23A3A")
    axes[0, 0].set_title(f"Signal BRUT - {channel}")
    axes[0, 1].plot(t, cleaned[:n], linewidth=0.6, color="#2E7D4F")
    axes[0, 1].set_title(f"Signal NETTOYE - {channel}")
    for ax in axes[0]:
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

    # Ligne 2 : spectres
    f_raw, p_raw = welch(raw, fs=fs, nperseg=min(512, len(raw)))
    f_cl, p_cl = welch(cleaned, fs=fs, nperseg=min(512, len(cleaned)))
    axes[1, 0].semilogy(f_raw, p_raw, color="#B23A3A", linewidth=1.1)
    axes[1, 0].axvline(50, color="#999", linestyle="--", linewidth=1,
                       label="50 Hz (secteur)")
    axes[1, 0].set_title("Spectre BRUT")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].semilogy(f_cl, p_cl, color="#2E7D4F", linewidth=1.1)
    axes[1, 1].set_title("Spectre NETTOYE")
    for ax in axes[1]:
        ax.set_xlim(0, 64)
        ax.set_xlabel("Frequence (Hz)")
        ax.set_ylabel("PSD (uV2/Hz)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Avant / Apres pretraitement - {meta['filename']} (etat : {meta['label']})",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
        print(f"Figure sauvegardee : {save_to}")
    return fig


def plot_epochs_overview(path, channel="O1", n_show=4, save_to=None):
    """
    Affiche quelques epochs successifs d'un canal apres segmentation (Sprint 2).

    Permet de visualiser le decoupage du signal en fenetres de duree fixe.
    """
    try:
        from . import preprocessing
    except ImportError:
        import preprocessing

    df = io_eeg.load_recording(path)
    meta = io_eeg.parse_filename(path)
    cleaned = preprocessing.clean_signal(df)
    epochs = preprocessing.segment_epochs(cleaned)

    ch_idx = config.CHANNEL_NAMES.index(channel)
    n_show = min(n_show, epochs.shape[0])
    win = epochs.shape[1]
    t = io_eeg.get_time_vector(win)

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 1.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, epochs[i, :, ch_idx], linewidth=0.7, color="#2E5496")
        ax.set_ylabel(f"Epoch {i+1}", rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
    axes[-1].set_xlabel("Temps dans l'epoch (s)")
    fig.suptitle(
        f"Epochs apres segmentation - {meta['filename']} - canal {channel} "
        f"({epochs.shape[0]} epochs au total)", fontsize=11, y=1.0,
    )
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
        print(f"Figure sauvegardee : {save_to}")
    return fig


if __name__ == "__main__":
    catalogue = io_eeg.list_recordings()
    if catalogue.empty:
        print("Aucun fichier dans data/raw/. Placez-y des fichiers STEW.")
    else:
        first = catalogue.iloc[0]["path"]
        print(f"Visualisation de : {Path(first).name}")
        plot_raw_signals(first, save_to=config.RESULTS_DIR / "sprint1_signals.png")
        plot_psd(first, channel="O1", save_to=config.RESULTS_DIR / "sprint1_psd.png")
        print("Termine. Figures dans le dossier results/.")
