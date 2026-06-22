"""
test_pipeline.py - Tests unitaires du pipeline EEG-CogState
===========================================================

Ces tests verifient que chaque brique du projet fonctionne correctement,
sans dependre du dataset reel : on genere un petit signal synthetique.

Lancement (depuis la racine du projet) :
    pytest
ou, pour plus de details :
    pytest -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Rend le dossier src/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, preprocessing, features, history


# ----------------------------------------------------------------------
#  FIXTURES (donnees de test partagees)
# ----------------------------------------------------------------------
@pytest.fixture
def fake_signal():
    """Genere un faux signal EEG (10 s, 14 canaux) avec un rythme alpha."""
    fs = config.SAMPLING_RATE
    n = fs * 10
    t = np.arange(n) / fs
    rng = np.random.RandomState(0)
    data = np.zeros((n, config.N_CHANNELS))
    for c in range(config.N_CHANNELS):
        data[:, c] = rng.randn(n) * 5 + 15 * np.sin(2 * np.pi * 10 * t)
    return pd.DataFrame(data, columns=config.CHANNEL_NAMES)


# ----------------------------------------------------------------------
#  TESTS : CONFIGURATION
# ----------------------------------------------------------------------
def test_config_has_14_channels():
    assert config.N_CHANNELS == 14
    assert len(config.CHANNEL_NAMES) == 14


def test_config_channel_positions_complete():
    # Chaque canal doit avoir une position pour la topographie
    assert set(config.CHANNEL_POSITIONS.keys()) == set(config.CHANNEL_NAMES)


def test_config_bands_defined():
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        assert band in config.FREQ_BANDS
        lo, hi = config.FREQ_BANDS[band]
        assert lo < hi


# ----------------------------------------------------------------------
#  TESTS : PRETRAITEMENT
# ----------------------------------------------------------------------
def test_bandpass_preserves_shape(fake_signal):
    out = preprocessing.bandpass_filter(fake_signal.values)
    assert out.shape == fake_signal.values.shape


def test_notch_removes_50hz():
    # Signal a 50 Hz pur -> doit etre fortement attenue par le notch
    fs = config.SAMPLING_RATE
    t = np.arange(fs * 5) / fs
    sig = np.sin(2 * np.pi * 50 * t).reshape(-1, 1)
    filtered = preprocessing.notch_filter(sig)
    assert np.var(filtered) < np.var(sig) * 0.5


def test_clean_signal_runs(fake_signal):
    out = preprocessing.clean_signal(fake_signal)
    assert out.shape == fake_signal.values.shape
    assert not np.isnan(out).any()


def test_segment_epochs_shape(fake_signal):
    cleaned = preprocessing.clean_signal(fake_signal)
    epochs = preprocessing.segment_epochs(cleaned)
    # Chaque epoch : (echantillons, canaux)
    assert epochs.ndim == 3
    assert epochs.shape[2] == config.N_CHANNELS
    expected_win = int(config.EPOCH_SECONDS * config.SAMPLING_RATE)
    assert epochs.shape[1] == expected_win


def test_segment_epochs_too_short():
    # Signal plus court qu'un epoch -> aucun epoch produit
    short = np.zeros((10, config.N_CHANNELS))
    epochs = preprocessing.segment_epochs(short)
    assert epochs.shape[0] == 0


# ----------------------------------------------------------------------
#  TESTS : FEATURES
# ----------------------------------------------------------------------
def test_band_powers_keys():
    sig = np.random.RandomState(1).randn(config.SAMPLING_RATE * 2)
    powers = features.band_powers(sig)
    for band in config.FREQ_BANDS:
        assert f"{band}_power" in powers
        assert f"{band}_rel" in powers


def test_relative_powers_sum_to_one():
    sig = np.random.RandomState(2).randn(config.SAMPLING_RATE * 2)
    powers = features.band_powers(sig)
    total_rel = sum(powers[f"{b}_rel"] for b in config.FREQ_BANDS)
    assert total_rel == pytest.approx(1.0, abs=1e-6)


def test_extract_epoch_features_count(fake_signal):
    cleaned = preprocessing.clean_signal(fake_signal)
    epochs = preprocessing.segment_epochs(cleaned)
    feats = features.extract_epoch_features(epochs[0])
    # Doit produire plusieurs features par canal
    assert len(feats) > config.N_CHANNELS
    # Toutes les valeurs sont finies
    assert all(np.isfinite(v) for v in feats.values())


def test_alpha_detected_in_alpha_signal(fake_signal):
    # Le faux signal est a 10 Hz (alpha) : la puissance alpha doit dominer
    cleaned = preprocessing.clean_signal(fake_signal)
    epochs = preprocessing.segment_epochs(cleaned)
    powers = features.band_powers(epochs[0, :, 0])
    rels = {b: powers[f"{b}_rel"] for b in config.FREQ_BANDS}
    assert max(rels, key=rels.get) == "alpha"


# ----------------------------------------------------------------------
#  TESTS : HISTORIQUE (SQLite)
# ----------------------------------------------------------------------
def test_history_add_and_get(tmp_path):
    db = tmp_path / "test.db"
    history.add_analysis("f.txt", "concentration", 80.0, 100,
                         {"concentration": 0.8, "relaxation": 0.2}, db_path=db)
    records = history.get_history(db_path=db)
    assert len(records) == 1
    assert records[0]["predicted"] == "concentration"
    assert records[0]["probabilities"]["concentration"] == 0.8


def test_history_count_and_clear(tmp_path):
    db = tmp_path / "test.db"
    for i in range(3):
        history.add_analysis(f"f{i}.txt", "relaxation", 75.0, 100,
                             {"relaxation": 0.75, "concentration": 0.25}, db_path=db)
    assert history.count_analyses(db_path=db) == 3
    history.clear_history(db_path=db)
    assert history.count_analyses(db_path=db) == 0
