# 🧠 EEG-CogState

> Plateforme intelligente d'analyse cognitive EEG — classification des états mentaux, IA explicable et visualisations neurophysiologiques.
>
> ## 📸 Aperçu

### Résultat de l'analyse — état cognitif, confiance et topographie cérébrale

<img width="1911" height="867" alt="Capture d&#39;écran 2026-06-27 191001" src="https://github.com/user-attachments/assets/c365d5e0-8646-4617-99a6-15efc2a32152" />

### Explicabilité IA (SHAP) et tableau de bord

<img width="1918" height="845" alt="Capture d&#39;écran 2026-06-27 191241" src="https://github.com/user-attachments/assets/adbb2e9b-5698-4f8a-b030-edf742aefa4a" />

---

EEG-CogState analyse des signaux électroencéphalographiques (EEG) et estime l'état cognitif d'un utilisateur (concentration, relaxation) à l'aide du traitement du signal et du machine learning. Le projet couvre toute la chaîne : importation → prétraitement → extraction de features → classification → visualisation → explicabilité.

---

## ✨ Fonctionnalités

- **Pipeline EEG complet** : lecture, filtrage (passe-bande + notch 50 Hz), segmentation en epochs
- **Extraction de features** : puissances des 5 bandes (δ, θ, α, β, γ), ratios, paramètres de Hjorth, entropie
- **Classification** : Random Forest dans un pipeline scikit-learn
- **Évaluation rigoureuse** : validation **par sujet** (Leave-One-Subject-Out) — pas de fuite de données
- **Application web** (Streamlit) : thème clair/sombre, jauge de confiance, topographie du scalp, graphes interactifs
- **IA explicable (SHAP)** : pourquoi le modèle décide — importance par bande, par canal, par feature
- **Historique** : sauvegarde des analyses en base SQLite
- **Tests automatisés** : 14 tests unitaires (pytest)

---

## 📊 Dataset

[STEW — Simultaneous Task EEG Workload](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
48 sujets · 14 canaux (Emotiv EPOC) · 128 Hz.

Convention de nommage : `subXX_lo.txt` (repos → relaxation), `subXX_hi.txt` (multitâche → concentration).

---

## 🚀 Installation

```bash
cd EEG-CogState

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Placez ensuite les fichiers `.txt` STEW dans `data/raw/`.

---

## 🎯 Utilisation

### Application web

```bash
streamlit run app/streamlit_app.py
```

Chargez un fichier EEG → l'app prédit l'état, affiche les visualisations et explique la décision.

### Notebooks (analyse étape par étape)

```bash
jupyter notebook notebooks/01_exploration.ipynb     # exploration du signal
jupyter notebook notebooks/02_preprocessing.ipynb   # prétraitement
jupyter notebook notebooks/03_features.ipynb        # extraction de features
jupyter notebook notebooks/04_model.ipynb           # machine learning
jupyter notebook notebooks/05_explainability.ipynb  # explicabilité SHAP
```

### Tests

```bash
pytest -v
```

---

## 📁 Structure du projet

```
EEG-CogState/
├── app/
│   └── streamlit_app.py       # application web
├── src/
│   ├── config.py              # paramètres centralisés
│   ├── io_eeg.py              # lecture des fichiers EEG
│   ├── preprocessing.py       # filtrage & segmentation
│   ├── features.py            # extraction de caractéristiques
│   ├── model.py               # entraînement & évaluation (LOSO)
│   ├── explainability.py      # IA explicable (SHAP)
│   ├── history.py             # historique SQLite
│   └── viz.py                 # visualisations
├── notebooks/                 # 5 notebooks d'analyse
├── tests/
│   └── test_pipeline.py       # 14 tests unitaires
├── data/                      # données (raw / processed) — non versionnées
├── models/                    # modèles entraînés
├── results/                   # figures générées
├── requirements.txt
└── README.md
```

---

## 🔬 Note méthodologique

L'évaluation se fait **par sujet** (Leave-One-Subject-Out), jamais par segment. Plusieurs epochs proviennent du même individu et partagent sa signature ; les répartir au hasard entre entraînement et test créerait une fuite de données qui gonflerait artificiellement les scores. La validation par sujet mesure la vraie capacité à généraliser à un **nouvel** utilisateur.

Sur de vraies données EEG, un macro-F1 entre 0,6 et 0,8 est un bon résultat. Un score quasi parfait doit éveiller la méfiance (sur-apprentissage ou fuite).

---

## 🛣️ Feuille de route

| Étape | Objectif | Statut |
|-------|----------|--------|
| Sprint 1 | Socle EEG : chargement + visualisation | ✅ |
| Sprint 2 | Prétraitement : filtrage, notch, epoching | ✅ |
| Sprint 3 | Features : bandes, ratios, entropie | ✅ |
| Sprint 4 | Machine learning + validation par sujet | ✅ |
| Sprint 5 | Application web Streamlit | ✅ |
| XAI | Explicabilité IA (SHAP) | ✅ |
| Historique | Sauvegarde SQLite des analyses | ✅ |
| Tests | Suite de tests automatisés (pytest) | ✅ |

### Perspectives

- Topomaps multi-bandes (α / β / θ côte à côte)
- Comparateur de modèles (RF / SVM / XGBoost)
- Rapport PDF automatique
- Analyse temps réel simulée
- Deep learning (EEGNet)

---

## ⚠️ Avertissement

EEG-CogState est un outil **pédagogique et exploratoire**. Il ne constitue pas un dispositif médical et ne doit pas être utilisé à des fins de diagnostic ou de décision clinique.

---

## 🛠️ Stack technique

Python · NumPy · Pandas · SciPy · scikit-learn · SHAP · Streamlit · Plotly · Matplotlib · pytest · SQLite
