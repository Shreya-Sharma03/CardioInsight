"""
CardioInsight — ECG-based Heart Failure Risk Analysis
Streamlit Product-Style Clinical Research Demo
"""

import sys
import textwrap
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from config.config import DEFAULT_PATH_CONFIG, DEFAULT_EVAL_CONFIG
from evaluation.evaluator import load_fusion_checkpoint


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioInsight — Heart Failure Risk Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# High-Contrast Clinical Research Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Force light theme colors universally */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F9F8F6 !important;
        color: #1A1C1E !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Force all text elements to have readable, dark colors */
    p, span, label, h1, h2, h3, h4, h5, h6, div, .stMarkdown {
        color: #1A1C1E !important;
    }

    /* Style bordered containers as clean, elegant white product cards */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E0D8 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        padding: 18px 20px !important;
    }

    /* Top App Header */
    .app-header {
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #E5E0D8;
        margin-bottom: 20px;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
    }
    .app-title {
        font-size: 26px;
        font-weight: 700;
        color: #1A1C1E !important;
        letter-spacing: -0.5px;
        margin: 0;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .app-subtitle {
        font-size: 14px;
        color: #5A6065 !important;
        margin: 2px 0 0 0;
        font-weight: 400;
    }
    .badge-status {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background-color: #EAE6DF;
        color: #4A4E51 !important;
        padding: 4px 10px;
        border-radius: 12px;
    }

    /* Card Titles */
    .card-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #5A6065 !important;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Risk Assessment Badges */
    .risk-badge-low {
        display: inline-block;
        background-color: #EDF4F0 !important;
        color: #3D7055 !important;
        border: 1px solid #C8DEC5;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-badge-high {
        display: inline-block;
        background-color: #FDF2EE !important;
        color: #C05638 !important;
        border: 1px solid #F6D3C8;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Large Probability Display */
    .prob-val {
        font-size: 46px;
        font-weight: 700;
        line-height: 1.1;
        margin: 12px 0 4px 0;
    }
    .prob-sub {
        font-size: 13px;
        color: #5A6065 !important;
        margin-bottom: 14px;
    }

    /* Illustrative Label Pill */
    .illustrative-pill {
        display: inline-block;
        background-color: #F4F0EA;
        color: #726A5E !important;
        border: 1px solid #E2DCCF;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 500;
    }

    /* Reliability Horizontal Bars */
    .bar-row {
        margin-bottom: 14px;
    }
    .bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 500;
        color: #1A1C1E !important;
        margin-bottom: 5px;
    }
    .bar-track {
        height: 10px;
        background-color: #EFECE6;
        border-radius: 5px;
        overflow: hidden;
    }
    .bar-fill-sage {
        height: 100%;
        background-color: #4A7C59;
        border-radius: 5px;
    }
    .bar-fill-slate {
        height: 100%;
        background-color: #6C7A89;
        border-radius: 5px;
    }

    /* Benchmark Metric Tiles */
    .metric-tile {
        background-color: #FBF9F6;
        border: 1px solid #E5E0D8;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-num {
        font-size: 20px;
        font-weight: 700;
        color: #1A1C1E !important;
    }
    .metric-lbl {
        font-size: 11px;
        color: #5A6065 !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 2px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4A7C59 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 6px 20px !important;
        transition: opacity 0.15s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    """Load the verified Epoch 7 fusion model checkpoint."""
    ckpt_path = BASE_DIR / DEFAULT_PATH_CONFIG.best_fusion_checkpoint
    if not ckpt_path.exists():
        ckpt_path = BASE_DIR / "best_fusion_model.zip"
    model = load_fusion_checkpoint(str(ckpt_path))
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def get_validation_features():
    """Load cached precomputed validation feature arrays."""
    val_res_path = BASE_DIR / DEFAULT_PATH_CONFIG.resnet_features_val
    val_gru_path = BASE_DIR / DEFAULT_PATH_CONFIG.bigru_features_val

    if not val_res_path.exists():
        val_res_path = BASE_DIR / "resnet_features_val.npy"
    if not val_gru_path.exists():
        val_gru_path = BASE_DIR / "bigru_features_val.npy"

    resnet_val = np.load(str(val_res_path))
    bigru_val = np.load(str(val_gru_path))
    return resnet_val, bigru_val


# ---------------------------------------------------------------------------
# Waveform Generator (Illustrative ECG preview)
# ---------------------------------------------------------------------------
def generate_illustrative_rhythm(sample_idx: int):
    """
    Generate an illustrative synthetic Lead II ECG rhythm strip.
    Explicitly labeled as an illustrative preview, NOT model input.
    """
    duration = 2.6
    fs = 250
    t = np.linspace(0, duration, int(duration * fs))

    # Realistic rhythm variance depending on record index
    heart_rate = 68 + (sample_idx % 20)
    period = 60.0 / heart_rate

    ecg = np.zeros_like(t)
    for beat_time in np.arange(0.25, duration, period):
        # P wave
        ecg += 0.14 * np.exp(-((t - (beat_time - 0.16)) / 0.038)**2)
        # Q wave
        ecg -= 0.10 * np.exp(-((t - (beat_time - 0.04)) / 0.014)**2)
        # R wave
        ecg += 1.22 * np.exp(-((t - beat_time) / 0.022)**2)
        # S wave
        ecg -= 0.32 * np.exp(-((t - (beat_time + 0.04)) / 0.018)**2)
        # T wave
        ecg += 0.28 * np.exp(-((t - (beat_time + 0.22)) / 0.065)**2)

    # Mild physiological baseline wander
    ecg += 0.02 * np.sin(2 * np.pi * 0.45 * t)
    return t, ecg, heart_rate


def plot_ecg_rhythm(t, ecg, heart_rate: int):
    """Render a clean clinical ECG strip with standard grid markings."""
    fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FCFAF7")

    # Standard ECG Grid lines (minor: 0.04s, major: 0.2s)
    minor_x = np.arange(0, t[-1] + 0.04, 0.04)
    major_x = np.arange(0, t[-1] + 0.20, 0.20)
    minor_y = np.arange(-0.6, 1.6, 0.1)
    major_y = np.arange(-0.6, 1.6, 0.5)

    for x in minor_x:
        ax.axvline(x, color="#F0E8E2", linewidth=0.5, zorder=1)
    for x in major_x:
        ax.axvline(x, color="#E2D2C8", linewidth=0.8, zorder=2)
    for y in minor_y:
        ax.axhline(y, color="#F0E8E2", linewidth=0.5, zorder=1)
    for y in major_y:
        ax.axhline(y, color="#E2D2C8", linewidth=0.8, zorder=2)

    # Lead II waveform trace
    ax.plot(t, ecg, color="#1D2A22", linewidth=1.35, zorder=4)

    # Axis boundaries and typography
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Time (seconds)", fontsize=8.5, color="#5A6065", labelpad=3)
    ax.set_ylabel("Amplitude (mV)", fontsize=8.5, color="#5A6065", labelpad=3)
    ax.tick_params(axis="both", labelsize=8, colors="#5A6065")

    for spine in ax.spines.values():
        spine.set_color("#DDD7CC")
        spine.set_linewidth(0.8)

    ax.text(
        0.02, 0.89, f"Lead II — Simulated Rate: {heart_rate} bpm",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="600",
        color="#3D453E",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#E5E0D8", alpha=0.9),
        zorder=5,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Real Model Inference
# ---------------------------------------------------------------------------
def run_real_inference(model, resnet_feat_vec, bigru_feat_vec):
    """
    Execute genuine forward pass through ECGFusionClassifier.
    Returns:
        prob (float): Heart failure risk probability (0.0 to 1.0)
        res_weight (float): Reliability weight for ResNet34-1D branch
        gru_weight (float): Reliability weight for BiGRU branch
    """
    res_tensor = torch.tensor(resnet_feat_vec, dtype=torch.float32).unsqueeze(0)
    gru_tensor = torch.tensor(bigru_feat_vec, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, fused, alpha_res, alpha_gru = model(res_tensor, gru_tensor)
        prob = float(torch.sigmoid(logits).squeeze().item())
        res_weight = float(alpha_res.squeeze().item())
        gru_weight = float(alpha_gru.squeeze().item())

    return prob, res_weight, gru_weight


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # Load model & precomputed validation features
    model = get_model()
    resnet_val, bigru_val = get_validation_features()
    total_val_samples = len(resnet_val)

    # --- TOP APP BAR ---
    st.markdown(
        textwrap.dedent("""
        <div class="app-header">
            <div>
                <h1 class="app-title">CardioInsight</h1>
                <p class="app-subtitle">ECG-based Heart Failure Risk Analysis</p>
            </div>
            <div class="badge-status">Research Prototype</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # --- CONTROL & SAMPLE SELECTION TOOLBAR ---
    with st.container(border=True):
        col_sel, col_custom, col_btn = st.columns([3, 2, 2])

        presets = {
            "Record #0042": 42,
            "Record #0100": 100,
            "Record #0500": 500,
            "Record #0000": 0,
            "Record #1000": 1000,
            "Custom Record ID...": -1,
        }

        with col_sel:
            selected_preset = st.selectbox(
                "Select Patient Record",
                options=list(presets.keys()),
                index=0,
                label_visibility="collapsed",
            )

        preset_idx = presets[selected_preset]
        with col_custom:
            if preset_idx == -1:
                sample_idx = st.number_input(
                    "Record Index",
                    min_value=0,
                    max_value=total_val_samples - 1,
                    value=15,
                    step=1,
                    label_visibility="collapsed",
                )
            else:
                sample_idx = preset_idx
                st.markdown(
                    f"<div style='padding-top: 8px; font-size: 13px; color: #5A6065;'>Cohort Index: <b>#{sample_idx}</b> of {total_val_samples}</div>",
                    unsafe_allow_html=True,
                )

        with col_btn:
            analyze_clicked = st.button("Analyze Sample", use_container_width=True)

    # State management for selected sample
    if "current_sample" not in st.session_state:
        st.session_state.current_sample = sample_idx

    if analyze_clicked or st.session_state.current_sample != sample_idx:
        st.session_state.current_sample = int(sample_idx)

    active_idx = st.session_state.current_sample

    # Run Real Model Inference
    res_vec = resnet_val[active_idx]
    gru_vec = bigru_val[active_idx]
    prob, w_resnet, w_bigru = run_real_inference(model, res_vec, gru_vec)

    # Decision Logic (Saved 0.41 cutoff)
    decision_threshold = DEFAULT_EVAL_CONFIG.saved_threshold  # 0.41
    is_high_risk = prob >= decision_threshold

    # --- MAIN ANALYSIS AREA ---
    col_left, col_right = st.columns([1.15, 1.0], gap="large")

    # === LEFT PANEL: ECG PREVIEW ===
    with col_left:
        with st.container(border=True):
            st.markdown(
                textwrap.dedent("""
                <div class="card-title">
                    <span>ECG Waveform Preview</span>
                    <span class="illustrative-pill">Illustrative ECG waveform — not the model input</span>
                </div>
                """),
                unsafe_allow_html=True,
            )

            # Illustrative Rhythm Plot
            t_axis, ecg_signal, bpm = generate_illustrative_rhythm(active_idx)
            fig = plot_ecg_rhythm(t_axis, ecg_signal, bpm)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Feature Vector Confirmation
            st.markdown(
                textwrap.dedent(f"""
                <div style="margin-top: 6px; padding: 12px 14px; background-color: #F8F6F1; border-radius: 6px; border: 1px solid #EAE4D8;">
                    <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #5A6065; letter-spacing: 0.5px; margin-bottom: 4px;">
                        Active Feature Embeddings (Model Input)
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #1A1C1E;">
                        <span><b>ResNet34-1D:</b> 512 dimensions (Norm: {np.linalg.norm(res_vec):.2f})</span>
                        <span><b>BiGRU:</b> 256 dimensions (Norm: {np.linalg.norm(gru_vec):.2f})</span>
                    </div>
                    <div style="font-size: 11px; color: #726A5E; margin-top: 5px;">
                        Note: Prediction is generated using the sample's verified precomputed feature vectors.
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

    # === RIGHT PANEL: PREDICTION RESULT & RELIABILITY ===
    with col_right:
        with st.container(border=True):
            if is_high_risk:
                badge_html = '<div class="risk-badge-high">HIGH RISK</div>'
                prob_color = "#C05638"
            else:
                badge_html = '<div class="risk-badge-low">LOW RISK</div>'
                prob_color = "#3D7055"

            fill_width = min(100.0, max(0.0, prob * 100))
            gauge_color = "#C05638" if is_high_risk else "#4A7C59"
            w_res_pct = w_resnet * 100
            w_gru_pct = w_bigru * 100

            right_panel_content = textwrap.dedent(f"""
            <div class="card-title">
                <span>Prediction Result</span>
                <span>Decision Threshold: 0.41</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                {badge_html}
                <span style="font-size: 12px; color: #5A6065; font-weight: 500;">Record #{active_idx}</span>
            </div>
            <div class="prob-val" style="color: {prob_color} !important;">
                {prob * 100:.1f}%
            </div>
            <div class="prob-sub">
                Calculated probability score of heart failure risk
            </div>
            <div style="position: relative; height: 8px; background-color: #EFECE6; border-radius: 4px; margin-bottom: 22px;">
                <div style="position: absolute; left: 0; top: 0; height: 100%; width: {fill_width}%; background-color: {gauge_color}; border-radius: 4px;"></div>
                <div style="position: absolute; left: 41%; top: -3px; height: 14px; width: 2px; background-color: #1A1C1E; z-index: 5;" title="Threshold: 0.41"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #726A5E; margin-top: -16px; margin-bottom: 18px;">
                <span>0%</span>
                <span style="padding-left: 10px;">▲ Cutoff (41%)</span>
                <span>100%</span>
            </div>
            <div style="border-top: 1px solid #EBE6DC; padding-top: 14px; margin-top: 4px;">
                <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #5A6065; letter-spacing: 0.5px; margin-bottom: 12px;">
                    Reliability Weight Contribution
                </div>
                <div class="bar-row">
                    <div class="bar-header">
                        <span>ResNet34-1D Branch</span>
                        <span><b>{w_res_pct:.1f}%</b></span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill-sage" style="width: {w_res_pct}%;"></div>
                    </div>
                </div>
                <div class="bar-row" style="margin-bottom: 8px;">
                    <div class="bar-header">
                        <span>BiGRU Branch</span>
                        <span><b>{w_gru_pct:.1f}%</b></span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill-slate" style="width: {w_gru_pct}%;"></div>
                    </div>
                </div>
                <div style="font-size: 11px; color: #726A5E; margin-top: 10px;">
                    Per-sample reliability weights computed by the fusion model gating layer (Sum: {w_res_pct + w_gru_pct:.1f}%).
                </div>
            </div>
            """)
            st.markdown(right_panel_content, unsafe_allow_html=True)

    # --- COMPACT BENCHMARKS & MODEL PERFORMANCE ---
    with st.container(border=True):
        st.markdown(
            textwrap.dedent("""
            <div class="card-title" style="margin-bottom: 12px;">
                <span>Model Performance</span>
                <span style="font-size: 11px; font-weight: 500; color: #726A5E;">Validation Split · Epoch 7 Best Checkpoint · Threshold: 0.41</span>
            </div>
            """),
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                textwrap.dedent("""
                <div class="metric-tile">
                    <div class="metric-num">88.92%</div>
                    <div class="metric-lbl">ROC-AUC</div>
                </div>
                """),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                textwrap.dedent("""
                <div class="metric-tile">
                    <div class="metric-num">82.35%</div>
                    <div class="metric-lbl">Balanced Accuracy</div>
                </div>
                """),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                textwrap.dedent("""
                <div class="metric-tile">
                    <div class="metric-num">77.94%</div>
                    <div class="metric-lbl">Recall (Sensitivity)</div>
                </div>
                """),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                textwrap.dedent("""
                <div class="metric-tile">
                    <div class="metric-num">86.76%</div>
                    <div class="metric-lbl">Specificity</div>
                </div>
                """),
                unsafe_allow_html=True,
            )

    # --- COMPACT EXPANDER: RESEARCH LIMITATIONS ---
    with st.expander("About this model & research considerations"):
        st.markdown(
            """
            * **Input Representation:** CardioInsight evaluates precomputed 512-dimensional morphological embeddings from ResNet34-1D and 256-dimensional temporal representations from BiGRU. It does not ingest raw multichannel waveform arrays directly.
            * **Decision Threshold:** The 0.41 cutoff was inherited from the original exploratory notebook (Cell 58). Its selection rationale is undocumented, and it was not established as the F1-optimal threshold.
            * **Feature Alignment:** ResNet and BiGRU feature matrices match row-for-row ($N=1,827$ val, $N=3,654$ test). Upstream recording manifests and patient-level identifiers were not bundled with the local cache.
            * **Intended Use:** Investigational decision-support demo intended for research evaluation, not for autonomous or unassisted clinical diagnostic deployment.
            """
        )


if __name__ == "__main__":
    main()
