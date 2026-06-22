"""
history.py - Historique des analyses (base SQLite)
==================================================

Ce module enregistre chaque analyse effectuee dans une petite base de
donnees SQLite locale, et permet de consulter l'historique. Il correspond
a la fonctionnalite F6 du cahier des charges (sauvegarde des analyses).

SQLite est inclus dans Python (aucune installation), et stocke tout dans
un simple fichier (data/history.db). Parfait pour une application locale.

Chaque entree d'historique contient :
  - un identifiant et un horodatage
  - le nom du fichier analyse
  - l'etat predit et la confiance
  - les probabilites par classe (stockees en JSON)
  - le nombre d'epochs analyses
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from . import config
except ImportError:
    import config


# Emplacement de la base : data/history.db
DB_PATH = config.DATA_DIR / "history.db"


def _connect(db_path=None):
    """Ouvre une connexion SQLite (cree le fichier s'il n'existe pas)."""
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(str(db_path))


def init_db(db_path=None):
    """
    Cree la table d'historique si elle n'existe pas encore.

    A appeler une fois au demarrage. Sans danger si la table existe deja
    (CREATE TABLE IF NOT EXISTS).
    """
    conn = _connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                filename     TEXT    NOT NULL,
                predicted    TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                n_epochs     INTEGER NOT NULL,
                probabilities TEXT   NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def add_analysis(filename, predicted, confidence, n_epochs, probabilities,
                 db_path=None):
    """
    Enregistre une analyse dans l'historique.

    Parametres
    ----------
    filename : nom du fichier analyse
    predicted : etat cognitif predit (str)
    confidence : confiance en % (float)
    n_epochs : nombre d'epochs analyses (int)
    probabilities : dict {classe: probabilite}
    Retour
    ------
    int : l'identifiant de la ligne inseree.
    """
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO analyses "
            "(timestamp, filename, predicted, confidence, n_epochs, probabilities) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                predicted,
                float(confidence),
                int(n_epochs),
                json.dumps(probabilities),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_history(limit=50, db_path=None):
    """
    Recupere les analyses, de la plus recente a la plus ancienne.

    Retour
    ------
    list[dict] : une entree par analyse, avec les probabilites deja
    decodees depuis le JSON.
    """
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["probabilities"] = json.loads(d["probabilities"])
            result.append(d)
        return result
    finally:
        conn.close()


def count_analyses(db_path=None):
    """Renvoie le nombre total d'analyses enregistrees."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    finally:
        conn.close()


def clear_history(db_path=None):
    """Vide entierement l'historique (supprime toutes les lignes)."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM analyses")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # Petite demonstration / test du module
    test_db = config.DATA_DIR / "history_demo.db"
    if test_db.exists():
        test_db.unlink()

    print("Initialisation de la base...")
    init_db(test_db)

    print("Ajout de 3 analyses de demonstration...")
    add_analysis("sub01_hi.txt", "concentration", 87.3, 149,
                 {"concentration": 0.873, "relaxation": 0.127}, db_path=test_db)
    add_analysis("sub02_lo.txt", "relaxation", 91.0, 149,
                 {"concentration": 0.090, "relaxation": 0.910}, db_path=test_db)
    add_analysis("sub03_hi.txt", "concentration", 78.5, 149,
                 {"concentration": 0.785, "relaxation": 0.215}, db_path=test_db)

    print(f"\nNombre total d'analyses : {count_analyses(test_db)}\n")
    print("Historique (du plus recent au plus ancien) :")
    for a in get_history(db_path=test_db):
        print(f"  #{a['id']} | {a['timestamp']} | {a['filename']:14s} "
              f"-> {a['predicted']:14s} ({a['confidence']:.1f}%)")

    # Nettoyage du fichier de demo
    test_db.unlink()
    print("\nTest reussi (base de demo supprimee).")
