# web/server.py
# Flask backend for FIRMNet AI — authentication, image analysis,
# and per-user history.  Integrates with the existing model and
# frequency_analysis pipeline without modifying any original files.

import os
import sys
import uuid
import secrets
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_from_directory, abort,
)
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────
# Add the project root to sys.path so we can import the existing
# frequency_analysis module and models package unchanged.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frequency_analysis import analyze_frequency   # existing FFT pipeline
from web.database import (
    init_db, create_user, verify_user, get_user_by_id,
    save_analysis, get_user_analyses, get_analysis_by_id,
    get_user_stats, username_exists, email_exists,
)

# ── Flask application ───────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Secret key: prefer environment variable; fall back to random (dev only).
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

# Directories for user uploads, FFT figures, and Grad-CAM heatmaps
UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), "static", "uploads")
FFT_DIR     = os.path.join(os.path.dirname(__file__), "static", "fft")
GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "static", "gradcam")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FFT_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ── Checkpoint discovery (mirrors app.py logic exactly) ─────────
FIRMNET_CHECKPOINTS = [
    os.path.join(PROJECT_ROOT, "checkpoints", "firmnet_best.pth"),
    os.path.join(PROJECT_ROOT, "checkpoints", "epoch24.pth"),
    os.path.join(PROJECT_ROOT, "checkpoints", "epoch15.pth"),
]
MOBILENET_CHECKPOINTS = [
    os.path.join(PROJECT_ROOT, "checkpoints", "mobilenet_best.pth"),
    os.path.join(PROJECT_ROOT, "checkpoints", "mobilenet_deepfake.pth"),
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


# ── Model loading (cached in-process) ──────────────────────────
_model_cache: dict = {}


def _get_model():
    """Load and cache the detection model.  Returns (model, device, preprocess, model_type)."""
    if "model" in _model_cache:
        return (
            _model_cache["model"],
            _model_cache["device"],
            _model_cache["preprocess"],
            _model_cache["model_type"],
        )

    import torch
    from torchvision import transforms

    checkpoint_path, model_type = _find_checkpoint()
    if checkpoint_path is None:
        raise RuntimeError(
            "No model checkpoint found in checkpoints/. "
            "Train a model first with main.py or train_mobilenet.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path}: {exc}"
        )

    if model_type == "firmnet":
        from models.backbones import FIRMNet

        model = FIRMNet(num_classes=2).to(device)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with FIRMNet architecture: {exc}"
            )
        model.to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:  # mobilenet
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, 2
        )
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with MobileNetV3-Small architecture: {exc}"
            )
        model.to(device).eval()
        preprocess = weights.transforms()

    _model_cache.update(
        model=model, device=device, preprocess=preprocess,
        model_type=model_type, checkpoint_path=checkpoint_path,
    )
    return model, device, preprocess, model_type


def _run_analysis(pil_image):
    """Run classification + FFT + Grad-CAM on a PIL Image.

    Returns a dict with prediction results, frequency ratios,
    and Grad-CAM spatial feature visualization.
    """
    import torch
    import torch.nn.functional as F

    model, device, preprocess, model_type = _get_model()

    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = F.softmax(model(tensor), dim=1)[0].detach().cpu()

    fake_score = float(scores[0])
    real_score = float(scores[1])
    predicted = "FAKE" if fake_score >= real_score else "REAL"
    confidence = max(fake_score, real_score)

    # FFT analysis (reuses existing pipeline unchanged)
    fft_figure, frequency_ratios = analyze_frequency(pil_image)

    # Grad-CAM spatial feature analysis
    gradcam_figure = None
    try:
        from gradcam_utils import generate_gradcam_figure, get_mobilenet_target_layer, get_firmnet_target_layer

        if model_type == "mobilenet":
            target_layer = get_mobilenet_target_layer(model)
        else:
            target_layer = get_firmnet_target_layer(model)

        # Need a fresh tensor with gradients for Grad-CAM (not inference_mode)
        cam_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        gradcam_figure, _ = generate_gradcam_figure(
            model, cam_tensor, pil_image, target_layer,
            class_names={0: "FAKE", 1: "REAL"},
        )
    except Exception as exc:
        print(f"[Grad-CAM] Visualization failed (non-critical): {exc}")
        gradcam_figure = None

    return {
        "predicted": predicted,
        "fake_score": fake_score,
        "real_score": real_score,
        "confidence": confidence,
        "freq_low": frequency_ratios["low"],
        "freq_mid": frequency_ratios["mid"],
        "freq_high": frequency_ratios["high"],
        "fft_figure": fft_figure,
        "gradcam_figure": gradcam_figure,
        "model_type": model_type,
    }


# ── Auth helpers ────────────────────────────────────────────────
def login_required(f):
    """Decorator that redirects anonymous users to the login page."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_globals():
    """Make the current user and year available in every template."""
    user = None
    if "user_id" in session:
        user = get_user_by_id(session["user_id"])
    return dict(current_user=user, now=datetime.now().year)


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", active_page="home")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        login_id = request.form.get("login_id", "").strip()
        password = request.form.get("password", "")

        if not login_id or not password:
            flash("Please fill in all fields.", "error")
            return render_template("login.html", active_page="login")

        user = verify_user(login_id, password)
        if user is None:
            flash("Invalid username/email or password.", "error")
            return render_template("login.html", active_page="login")

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        flash(f"Welcome back, {user['full_name']}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html", active_page="login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username  = request.form.get("username", "").strip()
        email     = request.form.get("email", "").strip()
        password  = request.form.get("password", "")
        confirm   = request.form.get("confirm_password", "")

        # ── Server-side validation ──
        errors = []
        if not full_name:
            errors.append("Full name is required.")
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters.")
        elif not username.replace("_", "").isalnum():
            errors.append("Username may only contain letters, numbers, and underscores.")
        if not email or "@" not in email:
            errors.append("A valid email address is required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        if username_exists(username):
            errors.append("That username is already taken.")
        if email_exists(email):
            errors.append("That email is already registered.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template(
                "signup.html", active_page="signup",
                form=dict(full_name=full_name, username=username, email=email),
            )

        ok, msg = create_user(full_name, username, email, password)
        if ok:
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash(msg, "error")
            return render_template(
                "signup.html", active_page="signup",
                form=dict(full_name=full_name, username=username, email=email),
            )

    return render_template("signup.html", active_page="signup", form={})


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = get_user_by_id(session["user_id"])
    stats = get_user_stats(session["user_id"])
    analyses = get_user_analyses(session["user_id"])
    checkpoint_path, model_type = _find_checkpoint()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        user=user,
        stats=stats,
        analyses=analyses,
        model_available=checkpoint_path is not None,
        model_type=model_type,
    )


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Please select an image to analyze.", "error")
        return redirect(url_for("dashboard"))

    if not _allowed_file(file.filename):
        flash("Unsupported file type. Use JPG, PNG, or WebP.", "error")
        return redirect(url_for("dashboard"))

    try:
        # Save uploaded image with a unique name
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(save_path)

        # Open and prepare image
        pil_image = ImageOps.exif_transpose(Image.open(save_path)).convert("RGB")

        # Run the actual model + FFT pipeline
        result = _run_analysis(pil_image)

        # Save FFT figure to disk
        fft_name = f"{uuid.uuid4().hex}.png"
        fft_path = os.path.join(FFT_DIR, fft_name)
        result["fft_figure"].savefig(fft_path, dpi=120, bbox_inches="tight",
                                     facecolor="#0a0e1a", edgecolor="none")
        plt.close(result["fft_figure"])

        # Save Grad-CAM figure to disk
        gradcam_name = None
        if result.get("gradcam_figure") is not None:
            gradcam_name = f"{uuid.uuid4().hex}.png"
            gradcam_path = os.path.join(GRADCAM_DIR, gradcam_name)
            result["gradcam_figure"].savefig(
                gradcam_path, dpi=120, bbox_inches="tight",
                facecolor="#0a0e1a", edgecolor="none",
            )
            plt.close(result["gradcam_figure"])

        # Persist to database
        analysis_id = save_analysis(
            user_id=session["user_id"],
            image_filename=unique_name,
            original_filename=file.filename,
            predicted_class=result["predicted"],
            fake_score=result["fake_score"],
            real_score=result["real_score"],
            confidence=result["confidence"],
            freq_low=result["freq_low"],
            freq_mid=result["freq_mid"],
            freq_high=result["freq_high"],
            fft_image_path=fft_name,
            gradcam_image_path=gradcam_name,
            model_type=result["model_type"],
        )

        return redirect(url_for("result", analysis_id=analysis_id))

    except RuntimeError as exc:
        flash(str(exc), "error")
        return redirect(url_for("dashboard"))
    except Exception as exc:
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("dashboard"))


@app.route("/result/<int:analysis_id>")
@login_required
def result(analysis_id):
    analysis = get_analysis_by_id(analysis_id, session["user_id"])
    if analysis is None:
        abort(404)
    return render_template(
        "result.html",
        active_page="dashboard",
        analysis=analysis,
    )


# ── Static file routes for uploads and FFT images ──────────────
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/fft/<path:filename>")
def fft_file(filename):
    return send_from_directory(FFT_DIR, filename)


@app.route("/gradcam/<path:filename>")
def gradcam_file(filename):
    return send_from_directory(GRADCAM_DIR, filename)


# ── Error handlers ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    flash("Page not found.", "error")
    return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(e):
    flash("File too large — maximum size is 16 MB.", "error")
    return redirect(url_for("dashboard"))


# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        app.secret_key = os.environ.get("SECRET_KEY", app.secret_key)
    except ImportError:
        pass

    init_db()
    print("=" * 55)
    print("  FIRMNet AI -- Deepfake Detection Website")
    print("=" * 55)

    checkpoint_path, model_type = _find_checkpoint()
    if checkpoint_path:
        print(f"  Model:      {model_type}")
        print(f"  Checkpoint: {os.path.relpath(checkpoint_path, PROJECT_ROOT)}")
    else:
        print("  [!] No model checkpoint found -- analysis disabled.")
        print("      Train a model first (see README).")

    print(f"  Database:   {os.path.relpath(os.path.join(os.path.dirname(__file__), 'firmnet.db'), PROJECT_ROOT)}")
    print(f"  Server:     http://127.0.0.1:5000")
    print("=" * 55)

    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
