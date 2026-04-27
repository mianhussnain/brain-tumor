"""
Brain Tumor Detection System - Modern UI (v1.2)
"""
import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os
import config
from predict import BrainTumorPredictor, get_confidence_level

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Main background ── */
.stApp { background: var(--background-color); }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.hero-banner h1 {
    color: #fff !important;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.hero-banner p {
    color: rgba(255,255,255,0.88);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* ── Stat cards ── */
.stat-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.08));
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.stat-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #667eea;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Result cards ── */
.result-tumor {
    background: linear-gradient(135deg, rgba(220,53,69,0.12), rgba(220,53,69,0.04));
    border: 2px solid rgba(220,53,69,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-notumor {
    background: linear-gradient(135deg, rgba(40,167,69,0.12), rgba(40,167,69,0.04));
    border: 2px solid rgba(40,167,69,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-subtitle {
    font-size: 0.92rem;
    opacity: 0.8;
}
.conf-badge {
    display: inline-block;
    background: rgba(102,126,234,0.18);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #667eea;
    margin-top: 0.5rem;
}

/* ── Warning box ── */
.warn-box {
    background: rgba(255,193,7,0.1);
    border: 1px solid rgba(255,193,7,0.4);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    margin-top: 1rem;
}

/* ── Model badge ── */
.model-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 1rem;
}

/* ── Sidebar tweaks ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(102,126,234,0.07) 0%, rgba(118,75,162,0.05) 100%);
    border-right: 1px solid rgba(102,126,234,0.15);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #667eea;
    font-weight: 700;
}

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #667eea !important;
    color: #667eea !important;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(102,126,234,0.35);
    border-radius: 14px;
    padding: 1rem;
    background: rgba(102,126,234,0.04);
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.8rem !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(102,126,234,0.45) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    border-radius: 99px !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(102,126,234,0.07);
    border: 1px solid rgba(102,126,234,0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_4class_model():
    if not os.path.exists(config.FINAL_MODEL_PATH):
        return None
    try:
        return BrainTumorPredictor(config.FINAL_MODEL_PATH, mode="4class")
    except Exception as e:
        st.error(f"Error loading 4-class model: {e}")
        return None

@st.cache_resource
def load_binary_model():
    if not os.path.exists(config.BINARY_MODEL_PATH):
        return None
    try:
        return BrainTumorPredictor(config.BINARY_MODEL_PATH, mode="binary")
    except Exception as e:
        st.error(f"Error loading binary model: {e}")
        return None


def chart_colors():
    return {"primary": "#667eea", "accent": "#764ba2", "danger": "#f64f59", "success": "#43e97b"}

def create_prediction_chart(predictions, key_suffix=""):
    classes     = list(predictions.keys())
    confs       = [predictions[c] * 100 for c in classes]
    max_c       = max(confs)
    colors      = [chart_colors()["danger"] if c == max_c else chart_colors()["primary"] for c in confs]
    fig = go.Figure(go.Bar(
        y=classes, x=confs, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"<b>{c:.1f}%</b>" for c in confs],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        xaxis=dict(range=[0, 120], showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False, title="Confidence (%)"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=60, t=10, b=10),
        height=max(280, len(classes) * 70),
        showlegend=False,
    )
    return fig

def create_binary_gauge(p_tumor):
    color = "#f64f59" if p_tumor >= 0.5 else "#43e97b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p_tumor * 100, 1),
        number=dict(suffix="%", font=dict(size=36, family="Inter", color=color)),
        title=dict(text="Tumor Probability", font=dict(size=14, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=11)),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(128,128,128,0.08)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50],   color="rgba(67,233,123,0.12)"),
                dict(range=[50, 100], color="rgba(246,79,89,0.12)"),
            ],
            threshold=dict(line=dict(color="rgba(128,128,128,0.5)", width=2), thickness=0.75, value=50),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, b=10, l=20, r=20), height=260,
    )
    return fig


def display_4class_results(result):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 Predicted Class", result["predicted_class"].upper())
    with c2:
        st.metric("📊 Confidence", f"{result['confidence']:.1%}", delta=get_confidence_level(result["confidence"]))
    with c3:
        status = "High Confidence ✅" if result["is_confident"] else "Low Confidence ⚠️"
        st.metric("🔍 Status", "Valid" if result["is_confident"] else "Review", delta=status)
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(create_prediction_chart(result["all_predictions"]), use_container_width=True, key="single_4class")
    if not result["is_confident"]:
        st.markdown(f'<div class="warn-box">⚠️ <strong>Low Confidence</strong>: {result["confidence"]:.1%} — please verify with a medical professional.</div>', unsafe_allow_html=True)


def display_binary_results(result):
    p_tumor  = result["all_predictions"]["Tumor"]
    is_tumor = result["predicted_class"] == "Tumor"
    conf     = result["confidence"]
    c1, c2   = st.columns([1.1, 0.9])
    with c1:
        if is_tumor:
            st.markdown(f"""
            <div class="result-tumor">
                <div class="result-title">🔴 Tumor Detected</div>
                <div class="result-subtitle">Signs of a brain tumor were identified in this MRI scan.</div>
                <div class="conf-badge">Confidence: {conf:.1%}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-notumor">
                <div class="result-title">🟢 No Tumor Detected</div>
                <div class="result-subtitle">No signs of a brain tumor were found in this MRI scan.</div>
                <div class="conf-badge">Confidence: {conf:.1%}</div>
            </div>""", unsafe_allow_html=True)
        st.plotly_chart(create_prediction_chart(result["all_predictions"]), use_container_width=True, key="single_binary_bar")
    with c2:
        st.plotly_chart(create_binary_gauge(p_tumor), use_container_width=True, key="single_binary_gauge")
    if not result["is_confident"]:
        st.markdown(f'<div class="warn-box">⚠️ <strong>Low Confidence</strong>: {conf:.1%} — consult a radiologist.</div>', unsafe_allow_html=True)


def main():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1>🧠 Brain Tumor Detection System</h1>
        <p>AI-Powered Medical Imaging Analysis — EfficientNetB4 Transfer Learning</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤖 Model Selection")
        model_choice = st.radio(
            "Choose detection model:",
            options=["🧠 4-Class Classifier", "✅ Binary Detector"],
            index=0,
        )
        is_binary = model_choice.startswith("✅")

        st.divider()
        if is_binary:
            st.markdown("#### Binary Detector")
            st.markdown("Answers: **Tumor** or **No Tumor**")
            st.markdown("- Architecture: EfficientNetB4\n- Input: 224×224 px\n- Dataset: 253 images")
            if not os.path.exists(config.BINARY_MODEL_PATH):
                st.warning("Model not trained yet.\n```\npython train_binary.py\n```")
        else:
            st.markdown("#### 4-Class Classifier")
            st.markdown("Classifies: Glioma · Meningioma · No Tumor · Pituitary")
            st.markdown("- Architecture: EfficientNetB4\n- Input: 224×224 px\n- Dataset: 7,200+ images")

        st.divider()
        st.markdown("#### Confidence Guide")
        st.markdown("🟢 High ≥ 90%\n🟡 Moderate 60–89%\n🔴 Low < 60%")
        st.caption("⚠️ For research assistance only. Not a diagnostic tool.")

    # ── Session state init ────────────────────────────────────────────────────
    if "batch_uploader_key" not in st.session_state:
        st.session_state.batch_uploader_key = 0

    # ── Load model ────────────────────────────────────────────────────────────
    predictor = load_binary_model() if is_binary else load_4class_model()
    if predictor is None:
        cmd = "train_binary.py" if is_binary else "train.py"
        st.error(f"❌ Model not found. Run: `python {cmd}`")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "ℹ️ About"])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f'<span class="model-badge">Active: {model_choice}</span>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a brain MRI image (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_uploader",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_col, res_col = st.columns([1, 1.2])
            with img_col:
                st.image(image, caption="Uploaded MRI", use_container_width=True)
            with res_col:
                with st.spinner("🔬 Analysing…"):
                    result = predictor.predict(np.array(image))
                st.success("Analysis complete!", icon="✅")
                st.markdown("<br>", unsafe_allow_html=True)
                if is_binary:
                    p_tumor = result["all_predictions"]["Tumor"]
                    is_tum  = result["predicted_class"] == "Tumor"
                    st.plotly_chart(create_binary_gauge(p_tumor), use_container_width=True, key="res_gauge")
                    label = "🔴 Tumor Detected" if is_tum else "🟢 No Tumor"
                    color = "#f64f59" if is_tum else "#43e97b"
                    st.markdown(f'<div style="text-align:center;font-size:1.2rem;font-weight:700;color:{color}">{label}</div>', unsafe_allow_html=True)
                else:
                    st.metric("Predicted", result["predicted_class"].upper())
                    st.metric("Confidence", f"{result['confidence']:.1%}", delta=get_confidence_level(result["confidence"]))

            st.divider()
            if is_binary:
                display_binary_results(result)
            else:
                display_4class_results(result)

            with st.expander("📋 Full Confidence Breakdown"):
                for cls, conf in result["sorted_predictions"]:
                    pct = conf * 100
                    st.markdown(f"**{cls}** — {pct:.2f}%")
                    st.progress(conf)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown(f'<span class="model-badge">Active: {model_choice}</span>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload multiple brain MRI images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key=f"batch_uploader_{st.session_state.batch_uploader_key}",
        )
        if uploaded_files:
            btn_col, clear_col = st.columns([3, 1])
            with btn_col:
                st.info(f"📸 {len(uploaded_files)} image(s) ready for analysis")
            with clear_col:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.batch_uploader_key += 1
                    st.rerun()
            if st.button("🚀 Process All Images", type="primary"):
                bar = st.progress(0, text="Processing…")
                results_list = []
                for i, f in enumerate(uploaded_files):
                    bar.progress((i + 1) / len(uploaded_files), text=f"Processing {f.name}…")
                    res = predictor.predict(np.array(Image.open(f)))
                    results_list.append({"filename": f.name, "result": res})
                bar.empty()
                st.success(f"✅ {len(results_list)} images processed!")
                st.divider()

                preds    = [r["result"]["predicted_class"] for r in results_list]
                avg_conf = np.mean([r["result"]["confidence"] for r in results_list])
                n_high   = sum(1 for r in results_list if r["result"]["is_confident"])

                cols = st.columns(4)
                stats = [
                    ("Total Images",    str(len(results_list))),
                    ("Avg Confidence",  f"{avg_conf:.1%}"),
                    ("High Confidence", str(n_high)),
                    ("Most Common",     max(set(preds), key=preds.count)),
                ]
                for col, (label, val) in zip(cols, stats):
                    with col:
                        st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Individual Results")
                for idx, item in enumerate(results_list):
                    r = item["result"]
                    icon = "🔴" if r["predicted_class"] == "Tumor" else ("🟢" if r["predicted_class"] == "No Tumor" else "🧠")
                    with st.expander(f"{icon} {item['filename']} — {r['predicted_class'].upper()} ({r['confidence']:.1%})"):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown(f"**Predicted:** {r['predicted_class'].upper()}")
                            st.markdown(f"**Confidence:** {r['confidence']:.2%}")
                            st.markdown(f"**Status:** {'✅ High' if r['is_confident'] else '⚠️ Low'}")
                            for cls, conf in r["sorted_predictions"]:
                                st.progress(conf, text=f"{cls}: {conf:.1%}")
                        with c2:
                            if is_binary:
                                st.plotly_chart(create_binary_gauge(r["all_predictions"]["Tumor"]), use_container_width=True, key=f"batch_gauge_{idx}")
                            else:
                                st.plotly_chart(create_prediction_chart(r["all_predictions"]), use_container_width=True, key=f"batch_chart_{idx}")

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
### 🧠 4-Class Classifier
Classifies brain MRI into four categories:
- **Glioma** — malignant glial cell tumor
- **Meningioma** — meninges tumor
- **No Tumor** — healthy scan
- **Pituitary** — pituitary gland tumor

**Architecture:** EfficientNetB4 + Transfer Learning  
**Training set:** 7,200+ images  
**Augmentation:** Rotation, flip, zoom, brightness
""")
        with c2:
            st.markdown("""
### ✅ Binary Detector
Simple yes/no detection:
- **Tumor** — any tumor present
- **No Tumor** — healthy scan

**Architecture:** EfficientNetB4 + Transfer Learning  
**Training set:** 253 images  
**Training:** 2-phase (frozen → fine-tune top 50 layers)  
**Augmentation:** Heavy (flip, rotation, zoom, brightness, contrast)
""")
        st.divider()
        st.error("⚠️ **Medical Disclaimer**: This system is for research and educational assistance only. It must NOT be used as a sole diagnostic tool. Always consult qualified medical professionals.")
        st.caption("Brain Tumor Detection System v1.2 — 4-Class + Binary Detector")

if __name__ == "__main__":
    main()
