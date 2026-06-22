"""
streamlit_app.py - Application web EEG-CogState (Sprint 5, version pro)
======================================================================

Interface web complete qui assemble toute la chaine du projet (F5 du
cahier des charges). Design "cockpit" sombre/clair, avec :
  - jauge de confiance circulaire
  - topographie du scalp (topomap)
  - barres de probabilites par etat
  - visualisations : signal, spectre PSD, bandes, votes des epochs

Lancement (depuis la racine du projet) :
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import welch
from scipy.interpolate import griddata

from src import config, preprocessing, features

# ======================================================================
#  CONFIGURATION DE LA PAGE
# ======================================================================
st.set_page_config(page_title="EEG-CogState", page_icon="🧠", layout="wide")

MODEL_PATH = config.MODELS_DIR / "model.joblib"

# --- Palette : deux themes (sombre par defaut) ---
THEMES = {
    "Sombre": {
        "bg": "#14171F", "panel": "#1A1E29", "border": "#2C3142",
        "text": "#E4E7EE", "muted": "#8A92A6", "accent": "#5DCAA5",
        "accent2": "#85B7EB", "grid": "#2C3142", "plot_bg": "#1A1E29",
    },
    "Clair": {
        "bg": "#F4F6FA", "panel": "#FFFFFF", "border": "#D9E2F3",
        "text": "#1F2733", "muted": "#5F6B7A", "accent": "#1D9E75",
        "accent2": "#2E75B6", "grid": "#E4EAF2", "plot_bg": "#FFFFFF",
    },
}


# ======================================================================
#  FONCTIONS UTILITAIRES
# ======================================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def read_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != config.N_CHANNELS:
        raise ValueError(
            f"{df.shape[1]} colonnes trouvees, {config.N_CHANNELS} attendues. "
            f"Verifiez qu'il s'agit bien d'un fichier STEW (14 canaux)."
        )
    df.columns = config.CHANNEL_NAMES
    return df


def predict_state(df, model):
    cleaned = preprocessing.clean_signal(df)
    epochs = preprocessing.segment_epochs(cleaned)
    if epochs.shape[0] == 0:
        return None
    feat_rows = [features.extract_epoch_features(epochs[i]) for i in range(epochs.shape[0])]
    X = pd.DataFrame(feat_rows).values
    preds = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    mean_proba = proba.mean(axis=0)
    probabilities = {classes[i]: float(mean_proba[i]) for i in range(len(classes))}
    predicted_label = classes[int(np.argmax(mean_proba))]
    return {
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "n_epochs": epochs.shape[0],
        "per_epoch_predictions": preds,
    }


def mean_band_power_per_channel(df, band="alpha"):
    """Puissance relative moyenne d'une bande, pour chaque canal (pour la topomap)."""
    cleaned = preprocessing.clean_signal(df)
    epochs = preprocessing.segment_epochs(cleaned)
    values = []
    for ci in range(config.N_CHANNELS):
        rels = [features.band_powers(epochs[i, :, ci])[f"{band}_rel"]
                for i in range(epochs.shape[0])]
        values.append(np.mean(rels))
    return np.array(values)


# ======================================================================
#  FONCTIONS DE TRACE
# ======================================================================
def style_fig(fig, theme, height=300, title=None):
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["text"], size=12),
        margin=dict(l=45, r=20, t=45 if title else 20, b=40),
        xaxis=dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
        yaxis=dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
    )
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig


def plot_signal(df, channel, theme, n_seconds=5):
    fs = config.SAMPLING_RATE
    n = min(int(n_seconds * fs), len(df))
    t = np.arange(n) / fs
    fig = go.Figure(go.Scatter(x=t, y=df[channel].values[:n], mode="lines",
                               line=dict(color=theme["accent2"], width=1)))
    fig.update_xaxes(title="Temps (s)")
    fig.update_yaxes(title="Amplitude (µV)")
    return style_fig(fig, theme, title=f"Signal brut — {channel}")


def plot_psd(df, channel, theme):
    fs = config.SAMPLING_RATE
    cleaned = preprocessing.clean_signal(df)[:, config.CHANNEL_NAMES.index(channel)]
    f, psd = welch(cleaned, fs=fs, nperseg=min(512, len(cleaned)))
    fig = go.Figure(go.Scatter(x=f, y=psd, mode="lines",
                               line=dict(color=theme["accent"], width=2)))
    band_fill = {"delta": "rgba(133,183,235,0.10)", "theta": "rgba(93,202,165,0.10)",
                 "alpha": "rgba(93,202,165,0.16)", "beta": "rgba(239,159,39,0.12)",
                 "gamma": "rgba(133,183,235,0.08)"}
    for band, (lo, hi) in config.FREQ_BANDS.items():
        fig.add_vrect(x0=lo, x1=hi, fillcolor=band_fill.get(band, "rgba(0,0,0,0)"),
                      line_width=0, annotation_text=band,
                      annotation_position="top",
                      annotation_font_size=10, annotation_font_color=theme["muted"])
    fig.update_xaxes(title="Fréquence (Hz)", range=[0, 45])
    fig.update_yaxes(title="PSD (µV²/Hz)")
    return style_fig(fig, theme, title=f"Spectre de puissance — {channel}")


def plot_band_powers(df, channel, theme):
    cleaned = preprocessing.clean_signal(df)
    epochs = preprocessing.segment_epochs(cleaned)
    ci = config.CHANNEL_NAMES.index(channel)
    means = {}
    for b in config.FREQ_BANDS:
        rels = [features.band_powers(epochs[i, :, ci])[f"{b}_rel"]
                for i in range(epochs.shape[0])]
        means[b] = np.mean(rels) * 100
    fig = go.Figure(go.Bar(x=list(means.keys()), y=list(means.values()),
                           marker_color=theme["accent"],
                           text=[f"{v:.1f}%" for v in means.values()],
                           textposition="auto"))
    fig.update_xaxes(title="Bande")
    fig.update_yaxes(title="Puissance relative (%)")
    return style_fig(fig, theme, title=f"Répartition des bandes — {channel}")


def plot_topomap(df, theme, band="alpha"):
    """Topographie du scalp : interpole la puissance d'une bande sur une tete vue de dessus."""
    powers = mean_band_power_per_channel(df, band=band)
    pos = np.array([config.CHANNEL_POSITIONS[ch] for ch in config.CHANNEL_NAMES])
    xs, ys = pos[:, 0], pos[:, 1]

    # Grille d'interpolation circulaire
    grid_n = 100
    gx, gy = np.meshgrid(np.linspace(-1, 1, grid_n), np.linspace(-1, 1, grid_n))
    grid_z = griddata((xs, ys), powers, (gx, gy), method="cubic")
    # Masque circulaire (hors du crane = NaN)
    mask = gx**2 + gy**2 > 1.0
    grid_z[mask] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(-1, 1, grid_n), y=np.linspace(-1, 1, grid_n), z=grid_z,
        colorscale="YlOrRd", showscale=True, line_width=0,
        colorbar=dict(title="%", thickness=12, len=0.8),
    ))
    # Points des electrodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=config.CHANNEL_NAMES,
        textposition="top center", textfont=dict(size=9, color=theme["text"]),
        marker=dict(size=7, color=theme["text"], line=dict(width=1, color=theme["panel"])),
        hoverinfo="text",
    ))
    # Contour de la tete + nez
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                             line=dict(color=theme["muted"], width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[-0.12, 0, 0.12], y=[1.0, 1.13, 1.0], mode="lines",
                             line=dict(color=theme["muted"], width=2), hoverinfo="skip"))
    fig.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False, range=[-1.25, 1.25], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-1.25, 1.3]),
        font=dict(color=theme["text"]),
    )
    return fig


def plot_probabilities(probabilities, theme):
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    best = max(probabilities, key=probabilities.get)
    colors = [theme["accent"] if c == best else theme["accent2"] for c in classes]
    fig = go.Figure(go.Bar(x=values, y=classes, orientation="h", marker_color=colors,
                           text=[f"{v:.1f}%" for v in values], textposition="auto"))
    fig.update_xaxes(title="Probabilité (%)", range=[0, 100])
    return style_fig(fig, theme, height=240, title="Probabilités par état")


def gauge(confidence, theme):
    """Jauge circulaire de confiance."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=confidence,
        number={"suffix": "%", "font": {"size": 30, "color": theme["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": theme["muted"]},
            "bar": {"color": theme["accent"]},
            "bgcolor": theme["panel"], "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": theme["grid"]}],
        },
    ))
    fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=theme["text"]), margin=dict(l=20, r=20, t=20, b=10))
    return fig


# ======================================================================
#  INTERFACE
# ======================================================================
def css(theme):
    st.markdown(f"""
    <style>
      .stApp {{ background: {theme['bg']}; }}
      .block-container {{ padding-top: 2rem; max-width: 1100px; }}
      .ecs-header {{ display:flex; align-items:center; gap:12px; margin-bottom:0.3rem; }}
      .ecs-logo {{ width:42px; height:42px; border-radius:10px; background:{theme['panel']};
                   display:flex; align-items:center; justify-content:center; font-size:22px;
                   border:1px solid {theme['border']}; }}
      .ecs-title {{ font-size:22px; font-weight:600; color:{theme['text']}; }}
      .ecs-sub {{ font-size:13px; color:{theme['muted']}; }}
      .ecs-panel {{ background:{theme['panel']}; border:1px solid {theme['border']};
                    border-radius:12px; padding:1.1rem 1.2rem; margin-bottom:0.8rem; }}
      .ecs-label {{ font-size:12px; color:{theme['muted']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:0.5rem; }}
      .ecs-metric {{ font-size:24px; font-weight:600; color:{theme['text']}; }}
      .ecs-metric-l {{ font-size:12px; color:{theme['muted']}; }}
      .ecs-state {{ font-size:24px; font-weight:600; color:{theme['accent']}; }}
      [data-testid="stMetricValue"] {{ color:{theme['text']}; }}
      [data-testid="stMetricLabel"] {{ color:{theme['muted']}; }}
    </style>
    """, unsafe_allow_html=True)


# --- Sidebar : theme + infos ---
model = load_model()

with st.sidebar:
    st.markdown("### Réglages")
    theme_name = st.radio("Thème", list(THEMES.keys()), index=0, horizontal=True)
    theme = THEMES[theme_name]
    st.markdown("---")
    st.markdown("#### À propos")
    st.markdown("Analyse un enregistrement EEG (STEW, 14 canaux, 128 Hz) "
                "et estime l'état cognitif associé.")
    if model is not None:
        st.markdown("**États reconnus :**")
        for cls in model.classes_:
            st.markdown(f"- {cls}")
    st.markdown("---")
    st.caption("⚠️ Outil pédagogique. Ne constitue pas un dispositif médical.")

css(theme)

# --- En-tete ---
st.markdown(f"""
<div class="ecs-header">
  <div class="ecs-logo">🧠</div>
  <div>
    <div class="ecs-title">EEG-CogState</div>
    <div class="ecs-sub">Classification des états cognitifs · v2</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Aucun modèle entraîné trouvé. Lancez d'abord le Sprint 4 "
             "(notebook 04_model.ipynb) pour créer models/model.joblib.")
    st.stop()

# --- Upload ---
st.markdown('<div class="ecs-label">1 · Charger un fichier EEG</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Fichier .txt STEW (14 colonnes séparées par des espaces)",
                            type=["txt"], label_visibility="collapsed")

if uploaded is None:
    st.info("👆 Chargez un fichier EEG pour lancer l'analyse.")
    st.stop()

try:
    df = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Impossible de lire le fichier : {e}")
    st.stop()

duration = len(df) / config.SAMPLING_RATE
c1, c2, c3, c4 = st.columns(4)
c1.metric("Canaux", df.shape[1])
c2.metric("Durée", f"{duration:.0f} s")
c3.metric("Échantillons", f"{len(df):,}".replace(",", " "))
c4.metric("Fréquence", f"{config.SAMPLING_RATE} Hz")

# --- Prediction ---
with st.spinner("Analyse en cours..."):
    result = predict_state(df, model)

if result is None:
    st.error("Signal trop court pour être analysé.")
    st.stop()

pred = result["predicted_label"]
conf = result["probabilities"][pred] * 100

st.markdown('<div class="ecs-label">2 · Résultat de l\'analyse</div>', unsafe_allow_html=True)
r1, r2 = st.columns([1, 1])
with r1:
    st.markdown(f'<div class="ecs-panel"><div class="ecs-label">État estimé</div>'
                f'<div class="ecs-state">{pred}</div>'
                f'<div class="ecs-metric-l">Analyse sur {result["n_epochs"]} epochs de 2 s</div></div>',
                unsafe_allow_html=True)
    st.plotly_chart(gauge(conf, theme), use_container_width=True)
with r2:
    st.plotly_chart(plot_probabilities(result["probabilities"], theme),
                    use_container_width=True)
    st.plotly_chart(plot_topomap(df, theme, band="alpha"), use_container_width=True)

# --- Visualisations detaillees ---
st.markdown('<div class="ecs-label">3 · Visualisations détaillées</div>', unsafe_allow_html=True)
channel = st.selectbox("Canal", config.CHANNEL_NAMES,
                       index=config.CHANNEL_NAMES.index("O1"))

tab1, tab2, tab3, tab4 = st.tabs(["Signal", "Spectre", "Bandes", "Votes des epochs"])
with tab1:
    st.plotly_chart(plot_signal(df, channel, theme), use_container_width=True)
with tab2:
    st.plotly_chart(plot_psd(df, channel, theme), use_container_width=True)
with tab3:
    st.plotly_chart(plot_band_powers(df, channel, theme), use_container_width=True)
with tab4:
    preds = result["per_epoch_predictions"]
    unique, counts = np.unique(preds, return_counts=True)
    fig_ep = go.Figure(go.Bar(x=list(unique), y=list(counts),
                              marker_color=theme["accent2"],
                              text=list(counts), textposition="auto"))
    fig_ep.update_xaxes(title="État")
    fig_ep.update_yaxes(title="Nombre d'epochs")
    st.plotly_chart(style_fig(fig_ep, theme, title="Votes des epochs par état"),
                    use_container_width=True)

st.markdown("---")
st.caption("EEG-CogState · Sprint 5 · Application web de démonstration")
