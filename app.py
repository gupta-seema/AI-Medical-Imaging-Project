import os, io, json, uuid, logging
from datetime import datetime
from typing import Tuple, Dict, List

from flask import Flask, render_template, request, send_file, url_for, jsonify
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# your existing modules
from model_chestray import ChestRayNet
from chestray_labels import LABELS
from report_builder import build_card

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

import sys, pathlib, os

def _res_path(rel: str) -> str:
    """Resolve files when frozen with PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

# ------------ config ------------
WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", _res_path("outputs/chestray_best.pt"))
THRESH_PATH  = os.environ.get("THRESHOLDS", None)  # e.g. "eval_outputs/thresholds.json"
IMG_SIZE     = int(os.environ.get("IMG_SIZE", 320))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.getcwd(), "web_uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
USE_GOOGLE_TRANSLATE = os.environ.get("USE_GOOGLE_TRANSLATE", "1") == "1"

app = Flask(__name__, template_folder=_res_path("templates"), static_folder=_res_path("static"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insightx")

# ------------ device/model ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChestRayNet(backbone="efficientnet_b0", pretrained=False)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval().to(device)

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------ helpers ------------
def _translate_texts(texts: List[str], target: str = "ne") -> List[str]:
    """Translate a list of strings using Google Cloud Translate v2, if available.
    Returns input texts on any failure (graceful fallback)."""
    if not texts:
        return texts
    if not USE_GOOGLE_TRANSLATE:
        return texts
    try:
        from google.cloud import translate_v2 as translate
        client = translate.Client()  
        out = []
        for t in texts:
            try:
                r = client.translate(t, target_language=target, format_='text')
                out.append(r.get('translatedText', t))
            except Exception:
                out.append(t)
        return out
    except Exception as e:
        logger.warning(f"Translate fallback: {e}")
        return texts


def run_inference_and_cam(pil_img: Image.Image):
    """Returns (probs, resized PIL, gradcam PIL)."""
    img_rgb = pil_img.convert("RGB")
    t = TFM(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(t))[0].cpu().numpy()

    # top-1 for CAM target
    top_idx = int(np.argmax(probs))
    target_layer = model.gradcam_target_layer()
    cam = GradCAM(model=model, target_layers=[target_layer])
    heat = cam(input_tensor=t, targets=[ClassifierOutputTarget(top_idx)])[0]

    h, w = heat.shape
    img_resized = img_rgb.resize((w, h))
    rgb = np.asarray(img_resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    overlay_pil = Image.fromarray(overlay)

    return probs, img_resized, overlay_pil


def build_pdf(uid: str, patient: dict, card: dict, orig_path: str, cam_path: str) -> str:
    """Create a simple, clean PDF (monospace fonts) and return its path."""
    pdf_path = os.path.join(UPLOAD_DIR, f"{uid}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Header (Courier-Bold)
    c.setFillColorRGB(0.06, 0.49, 0.42)  # green bar
    c.rect(0, h-2*cm, w, 2*cm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Courier-Bold", 16)
    c.drawString(2*cm, h-1.2*cm, "InsightX – Screening Report")

    # Patient block (Courier)
    y = h-3.2*cm
    c.setFillColorRGB(0,0,0)
    c.setFont("Courier", 11)
    c.drawString(2*cm, y, f"Patient: {patient.get('name','—')}  |  Age: {patient.get('age','—')}  |  Sex: {patient.get('sex','—')}")
    y -= 0.6*cm
    c.drawString(2*cm, y, f"ID: {patient.get('id','—')}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Summary (Courier-Bold for heading, Courier for text)
    y -= 1.0*cm
    c.setFont("Courier-Bold", 12)
    c.drawString(2*cm, y, "Summary")
    y -= 0.5*cm
    c.setFont("Courier", 11)
    c.drawString(2*cm, y, f"Decision: {card.get('decision','—')}  |  Risk tier: {card.get('risk_tier','—')}")
    y -= 0.5*cm
    headline = (card.get('headline','') or '')[:90]
    c.drawString(2*cm, y, f"Headline: {headline}")

    # Top findings
    y -= 0.8*cm
    c.setFont("Courier-Bold", 12)
    c.drawString(2*cm, y, "Top Findings")
    y -= 0.5*cm
    c.setFont("Courier", 11)
    for tf in card.get('top_findings', [])[:3]:
        c.drawString(2.2*cm, y, f"* {tf['label']}: {tf['score']:.2f}")
        y -= 0.45*cm

    # Images (best-effort)
    y -= 0.2*cm
    try:
        c.drawImage(orig_path, 2*cm, y-7.0*cm, width=7.5*cm, height=7.0*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    try:
        c.drawImage(cam_path, 10.5*cm, y-7.0*cm, width=7.5*cm, height=7.0*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    # Footer note (Courier-Oblique)
    c.setFont("Courier-Oblique", 9)
    c.drawString(2*cm, 1.5*cm, "This is decision support for health professionals, not a diagnosis.")

    c.showPage()
    c.save()
    return pdf_path


# ------------ Flask ------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # inputs
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose an image (JPG/PNG).")

    lang = request.form.get("lang", "en")  # 'en' or 'np'
    patient = {
        "name": request.form.get("patient_name", "Unknown"),
        "age": request.form.get("patient_age", "—"),
        "sex": request.form.get("patient_sex", "—"),
        "id": request.form.get("patient_id", uuid.uuid4().hex[:6].upper())
    }

    # image -> inference
    try:
        pil = Image.open(file.stream)
    except Exception:
        return render_template("index.html", error="Could not read image. Use JPG/PNG.")

    probs, img_resized, cam_pil = run_inference_and_cam(pil)

    # build screening card (bilingual-ready; uses safe defaults for thresholds)
    card = build_card(
        probs_or_scores=probs,
        thresholds_path=THRESH_PATH,
        lang=("np" if lang == "np" else "en"),
        patient_id=patient["id"],
        frontal_view=True
    )

    impression_en = card.get("summary_en", {}).get("impression", "")
    consider_en = card.get("summary_en", {}).get("consider", [])
    if lang == "np":
        translated = _translate_texts([impression_en] + consider_en, target="ne")
        if translated:
            card["summary_np"] = {
                "impression": translated[0],
                "consider": translated[1:] if len(translated) > 1 else []
            }

    # persist artifacts for viewing & PDF
    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(UPLOAD_DIR, f"{uid}_orig.jpg")
    cam_path  = os.path.join(UPLOAD_DIR, f"{uid}_cam.jpg")
    card_path = os.path.join(UPLOAD_DIR, f"{uid}_card.json")
    patient_path = os.path.join(UPLOAD_DIR, f"{uid}_patient.json")

    img_resized.save(orig_path, quality=92)
    cam_pil.save(cam_path, quality=92)
    with open(card_path, 'w') as f: json.dump(card, f, indent=2)
    with open(patient_path, 'w') as f: json.dump(patient, f, indent=2)

    # Generate PDF now so the button opens instantly
    pdf_path = build_pdf(uid, patient, card, orig_path, cam_path)

    # concise top-3 for chips
    top3 = sorted(card["scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    suspected = f"{top3[0][0]} ({top3[0][1]:.2f})" if top3 else "—"

    return render_template(
        "index.html",
        # media
        orig_url=url_for("serve_file", path=os.path.basename(orig_path)),
        cam_url=url_for("serve_file", path=os.path.basename(cam_path)),
        pdf_url=url_for("serve_file", path=os.path.basename(pdf_path)),
        # kpis
        decision=card.get("decision","—"),
        risk_tier=card.get("risk_tier","—"),
        suspected=suspected,
        # texts
        headline=card.get("headline",""),
        summary_en=card.get("summary_en", {}),
        summary_np=card.get("summary_np", None),
        # patient
        patient=patient,
        # raw card
        card_json=json.dumps(card, indent=2)
    )


@app.route("/files/<path:path>")
def serve_file(path):
    return send_file(os.path.join(UPLOAD_DIR, path))


@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    target = data.get("target", "ne")
    return jsonify({"texts": _translate_texts(texts, target)})


@app.route("/feedback", methods=["POST"])
def feedback():
    msg = request.get_json(force=True).get("message", "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "Empty message"}), 400
    with open(os.path.join(UPLOAD_DIR, "feedback.log"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    return jsonify({"ok": True})


if __name__ == "__main__":
    import webbrowser, threading, time

    def _open_browser():
        # wait a second for Flask to start
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:5000/")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
