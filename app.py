import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from frequency_analysis import analyze_frequency

st.set_page_config(page_title="Deepfake Detection Lab", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1180px;padding-top:1.4rem;padding-bottom:2rem}
.hero {padding:1.3rem 1.5rem;border:1px solid rgba(128,128,128,.25);border-radius:16px;margin-bottom:1rem}
.eyebrow {font-size:.76rem;letter-spacing:.13em;text-transform:uppercase;opacity:.7;font-weight:700}
.hero h1 {margin:.25rem 0 .4rem 0;font-size:2.1rem}
.subtle {opacity:.75;font-size:.94rem}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:.8rem 1rem}
</style>
<div class="hero"><div class="eyebrow">Academic prototype · Image forensics</div>
<h1>🔎 Deepfake Detection Lab</h1><div class="subtle">Compare a model's class scores with a separate frequency-domain visualisation.</div></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Checkpoint discovery — support both FIRMNet and MobileNetV3 checkpoints.
# Priority: FIRMNet checkpoint first (the project's main model), then
# MobileNetV3 as fallback.  Never silently swap models.
# ---------------------------------------------------------------------------
FIRMNET_CHECKPOINTS = [
    "checkpoints/firmnet_best.pth",
    "checkpoints/epoch24.pth",
    "checkpoints/epoch15.pth",
]
MOBILENET_CHECKPOINTS = [
    "checkpoints/mobilenet_best.pth",
    "checkpoints/mobilenet_deepfake.pth",
]


def _find_checkpoint():
    """Return (path, model_type) or (None, None)."""
    for path in FIRMNET_CHECKPOINTS:
        if os.path.isfile(path):
            return path, "firmnet"
    for path in MOBILENET_CHECKPOINTS:
        if os.path.isfile(path):
            return path, "mobilenet"
    return None, None


CHECKPOINT_PATH, MODEL_TYPE = _find_checkpoint()

# Sidebar —  reflects whichever model is actually loaded
with st.sidebar:
    st.header("About this prototype")
    if MODEL_TYPE == "firmnet":
        st.write("**Classifier:** FIRMNet (EfficientNetV2-S + FAGate + FER)")
    elif MODEL_TYPE == "mobilenet":
        st.write("**Classifier:** fine-tuned MobileNetV3-Small")
    else:
        st.write("**Classifier:** ⚠️ No checkpoint found")
    st.write("**Frequency view:** 2D Fast Fourier Transform (FFT)")
    if CHECKPOINT_PATH is None:
        st.error(
            "No model checkpoint found in `checkpoints/`. "
            "Train a model first with `main.py` or `train_mobilenet.py`."
        )
    st.caption(
        "Softmax scores are not calibrated probabilities. "
        "This demo cannot verify authenticity."
    )


@st.cache_resource
def load_model():
    """Load the detected model checkpoint.  Raises RuntimeError on failure."""
    if CHECKPOINT_PATH is None:
        raise RuntimeError(
            "No model checkpoint found in the checkpoints/ directory.\n\n"
            "To train a model:\n"
            "  FIRMNet:    python main.py --config config/config.yaml --data_dir data/images\n"
            "  MobileNet:  python train_mobilenet.py\n"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if MODEL_TYPE == "firmnet":
        from models.backbones import FIRMNet
        from torchvision import transforms

        model = FIRMNet(num_classes=2).to(device)
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()

        # FIRMNet was trained with ToTensor() only (no ImageNet normalisation)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return model, device, preprocess

    else:  # mobilenet
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model, device, weights.transforms()


# ---------------------------------------------------------------------------
# Upload and analysis UI — preserving the existing layout
# ---------------------------------------------------------------------------
left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("1 · Upload an image")
    uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png", "webp"])
    image = None
    if uploaded_file is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            st.image(image, caption="Image ready for analysis", use_container_width=True)
            st.caption(f"Image dimensions: {image.width} × {image.height} pixels")
        except Exception:
            st.error("This file could not be opened as an image.")
with right:
    st.subheader("2 · Run analysis")
    st.write("The FFT display is a separate visual analysis and does not alter the classifier's result.")
    run_analysis = st.button("Run analysis", type="primary", use_container_width=True, disabled=image is None)
    st.caption("Supported formats: JPG, JPEG, PNG, WEBP")

if run_analysis and image is not None:
    fft_figure = None
    try:
        with st.spinner("Running classification and FFT analysis…"):
            model, device, preprocess = load_model()
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                scores = F.softmax(model(tensor), dim=1)[0].detach().cpu()
            # ImageFolder mapping from training: FAKE=0, REAL=1
            fake_score, real_score = float(scores[0]), float(scores[1])
            predicted = "FAKE" if fake_score >= real_score else "REAL"
            score_gap = abs(fake_score - real_score)
            fft_figure, frequency_ratios = analyze_frequency(image)

        st.divider()
        st.subheader("Analysis report")
        st.warning("Use this result only as a demonstration. Model predictions are not reliable evidence that an image is real or fake.")
        a, b, c = st.columns(3)
        a.metric("Top model class", predicted)
        b.metric("Difference between scores", f"{score_gap:.1%}")
        c.metric("Compute device", str(device).upper())
        st.caption("A larger score difference means the model preferred one class more strongly; it does not establish correctness.")

        fake_col, real_col = st.columns(2)
        with fake_col:
            st.markdown("**FAKE score**")
            st.progress(fake_score)
            st.write(f"{fake_score:.1%}")
        with real_col:
            st.markdown("**REAL score**")
            st.progress(real_score)
            st.write(f"{real_score:.1%}")

        st.divider()
        st.subheader("Frequency-domain view")
        fft_col, explanation_col = st.columns([1.15, 1], gap="large")
        with fft_col:
            st.pyplot(fft_figure, use_container_width=True)
        with explanation_col:
            st.metric("Low-frequency energy", f"{frequency_ratios['low']:.2%}")
            st.metric("Mid-frequency energy", f"{frequency_ratios['mid']:.2%}")
            st.metric("High-frequency energy", f"{frequency_ratios['high']:.2%}")
            st.write(
                "The FFT spectrum shows how image energy is distributed across "
                "spatial frequencies. The centre represents lower frequencies "
                "(smooth regions); areas farther from the centre represent higher "
                "frequencies (edges, textures, noise)."
            )
            st.info(
                "These frequency ratios are descriptive image statistics, not "
                "standalone proof of a deepfake. They are shown for educational "
                "purposes and do not change the model's prediction."
            )
        with st.expander("Model method and limitations"):
            model_name = (
                "FIRMNet (EfficientNetV2-S + FAGate + FER)"
                if MODEL_TYPE == "firmnet"
                else "fine-tuned MobileNetV3-Small"
            )
            st.markdown(f"""
- **Classifier:** {model_name}.
- **Checkpoint:** `{CHECKPOINT_PATH}`
- **Input:** a single image, resized and normalised with the model's standard preprocessing.
- **Class mapping:** FAKE = 0, REAL = 1.
- **Frequency analysis:** FFT spectrum with low / mid / high energy ratios, shown separately.
- **Limitation:** results depend on training data quality and quantity. This prototype cannot reliably verify authenticity.
""")
    except Exception as error:
        st.error("Analysis could not be completed. Check that the model checkpoint and dependencies are available.")
        st.exception(error)
    finally:
        if fft_figure is not None:
            plt.close(fft_figure)

st.divider()
st.caption("Robust Deepfake Detection using Frequency-Domain Analysis · Academic prototype")
