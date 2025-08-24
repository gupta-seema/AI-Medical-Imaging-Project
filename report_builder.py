from __future__ import annotations
import os, json, time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from chestray_labels import LABELS
from post_process import load_thresholds, scores_from_probs, triage

VERSION = "1.2.0"

KB = [
  {
    "key": "Lung Lesion",
    "en": "Small round opacity may represent a nodule. Compare with prior films and consider CT if persistent.",
    "np": "साना गोलो छायाँ नोड्युल हुनसक्छ। अघिल्ला एक्सरेसँग तुलना गर्नुहोस् र आवश्यक परे CT विचार गर्नुहोस्।"
  },
  {
    "key": "Lung Opacity",
    "en": "Focal or diffuse opacity may be infection or fluid. Correlate with symptoms and temperature.",
    "np": "फोकल वा फैलिएको छायाँ संक्रमण वा तरल हुन सक्छ। लक्षण र ज्वरोसँग मिलाएर हेर्नुहोस्।"
  },
  {
    "key": "No Finding",
    "en": "No specific abnormality detected on this view.",
    "np": "यस दृश्यमा खास असामान्यता भेटिएन।"
  }
]

# ====== English summary templates for all 14 CheXpert labels ======
LABEL_IMPRESSIONS = {
    "No Finding":                 "No specific abnormality detected on this view.",
    "Enlarged Cardiomediastinum": "Prominent cardiomediastinal contour, which may reflect cardiomegaly or technical factors.",
    "Cardiomegaly":               "Cardiac silhouette appears enlarged.",
    "Lung Opacity":               "Pulmonary opacity is present.",
    "Lung Lesion":                "Focal round opacity may represent a pulmonary nodule.",
    "Edema":                      "Interstitial/alveolar markings raise concern for pulmonary edema.",
    "Consolidation":              "Focal air-space consolidation is present.",
    "Pneumonia":                  "Pattern may be compatible with pneumonia in the appropriate clinical context.",
    "Atelectasis":                "Linear/bandlike opacity may represent atelectasis.",
    "Pneumothorax":               "Lucency without lung markings suggests possible pneumothorax.",
    "Pleural Effusion":           "Blunting/meniscus suggests pleural effusion.",
    "Pleural Other":              "Pleural abnormality is suspected.",
    "Fracture":                   "Osseous contour abnormality may represent fracture.",
    "Support Devices":            "Support devices are visualized."
}

LABEL_CONSIDER = {
    "Lung Lesion":      ["Compare with prior films.", "Consider CT if persistent or suspicious."],
    "Lung Opacity":     ["Correlate with symptoms and temperature.", "Consider follow-up imaging if warranted."],
    "Consolidation":    ["Consider antibiotics if clinically indicated.", "Short-interval follow-up if unresolved."],
    "Pneumonia":        ["Treat per clinical judgment.", "Repeat film if symptoms persist."],
    "Atelectasis":      ["Encourage pulmonary hygiene.", "Reassess on follow-up imaging."],
    "Pleural Effusion": ["Consider ultrasound for characterization.", "Evaluate for heart/renal/hepatic causes."],
    "Pneumothorax":     ["Urgent clinical correlation recommended.", "Consider chest CT/expedited care if symptomatic."],
    "Cardiomegaly":     ["Correlate with echocardiography if indicated."],
    "Edema":            ["Assess volume status; consider diuresis per clinician."],
    "Fracture":         ["Correlate with point tenderness.", "Dedicated skeletal imaging if needed."],
    "Enlarged Cardiomediastinum": ["Assess technique vs pathology.", "Consider echocardiography if persistent."],
    "Pleural Other":    ["Consider CT if clinically warranted."],
    "Support Devices":  ["Confirm positioning as per protocol."],
    "No Finding":       []
}

# ====================== helpers ======================

def _lang_pick(en_text: str, np_text: str, lang: str) -> str:
    return en_text if lang == "en" else np_text

def _round_sig(x: float, decimals: int = 2) -> float:
    try:
        return float(np.clip(round(float(x), decimals), 0.0, 1.0))
    except Exception:
        return 0.0

def _topk(scores: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return items[:k]

def _rag_snippets(positives: List[str], lang="en", max_items=2) -> List[str]:
    notes, keys = [], set(positives) & {k["key"] for k in KB}
    if not keys:
        keys = {"No Finding"}
    for item in KB:
        if item["key"] in keys:
            notes.append(item[lang])
    return notes[:max_items]

def _risk_tier(nodule_proxy: float) -> str:
    if nodule_proxy >= 0.75: return "High"
    if nodule_proxy >= 0.50: return "Moderate"
    if nodule_proxy >= 0.30: return "Slight"
    return "Low"

def _severity(score: float) -> str:
    if score >= 0.8:  return "marked"
    if score >= 0.6:  return "moderate"
    if score >= 0.4:  return "mild"
    return "possible"

def _compose_impression(positives: List[str], scores: Dict[str, float], max_len=2) -> str:
    """Turn top positives into one or two readable clauses."""
    ranked = sorted(positives, key=lambda k: scores.get(k, 0.0), reverse=True)
    lines = []
    for lab in ranked[:max_len]:
        base = LABEL_IMPRESSIONS.get(lab, lab)
        sev = _severity(scores.get(lab, 0.0))
        if lab == "No Finding":
            text = base
        else:
            text = f"{base} ({sev})."
        lines.append(text)
    if not lines:
        lines = [LABEL_IMPRESSIONS["No Finding"]]
    return " ".join(lines)

def _compose_consider(positives: List[str], scores: Dict[str, float], max_items=3) -> List[str]:
    ranked = sorted(positives, key=lambda k: scores.get(k, 0.0), reverse=True)
    out: List[str] = []
    for lab in ranked:
        for tip in LABEL_CONSIDER.get(lab, []):
            if tip not in out:
                out.append(tip)
        if len(out) >= max_items:
            break
    return out[:max_items]

def _fuse_scores_with_report(
    image_scores: Dict[str, float],
    report_labels: Optional[Dict[str, Union[int, str]]] = None,
    pos_floor: float = 0.6,
    neg_floor: float = 0.0
) -> Dict[str, float]:
    """
    Conservative fusion of image scores with report NLP labels:
    - If report is positive (1), raise score to at least pos_floor.
    - If report is negative (0), clamp to at least neg_floor (does not suppress strong positives).
    - 'U' or missing -> unchanged.
    """
    if not report_labels:
        return image_scores
    fused = dict(image_scores)
    for k, v in report_labels.items():
        if k not in fused:
            continue
        if v == 1:
            fused[k] = max(fused[k], pos_floor)
        elif v == 0:
            fused[k] = max(fused[k], neg_floor)
    return fused

# =================== public API ===================

def build_english_summary(scores: Dict[str, float], positives: List[str], nodule_proxy: float) -> Dict[str, Union[str, List[str]]]:
    """English-only concise summary (Impression/Consider + nodule risk tag)."""
    impression = _compose_impression(positives, scores, max_len=2)
    consider = _compose_consider(positives, scores, max_items=3)
    if nodule_proxy >= 0.75:      nodule_risk = "high"
    elif nodule_proxy >= 0.50:    nodule_risk = "moderate"
    elif nodule_proxy >= 0.30:    nodule_risk = "slight"
    else:                         nodule_risk = "low"
    return {"impression": impression, "consider": consider, "nodule_risk": nodule_risk}

def build_card(
    probs_or_scores: Union[np.ndarray, Dict[str, float]],
    thresholds_path: Optional[str],
    lang: str = "en",                               # UI can translate to 'np' client-side if preferred
    report_labels: Optional[Dict[str, Union[int, str]]] = None,
    patient_id: Optional[str] = None,
    study_uid: Optional[str] = None,
    cam_image_path: Optional[str] = None,
    frontal_view: Optional[bool] = None
) -> Dict:
    """
    Robust screening card builder.
    - Accepts np.ndarray (len==len(LABELS)) or dict {label: prob in 0..1}.
    - Loads thresholds (with safe defaults).
    - Optionally fuses report-derived labels.
    - Returns a stable JSON payload for frontend rendering.
    """
    # ---- normalize input into scores dict ----
    if isinstance(probs_or_scores, dict):
        scores = {k: _round_sig(v) for k, v in probs_or_scores.items() if k in LABELS}
    else:
        scores = scores_from_probs(np.asarray(probs_or_scores, dtype=float))
        scores = {k: _round_sig(v) for k, v in scores.items()}

    # ---- thresholds & triage ----
    thr = load_thresholds(thresholds_path)  # safe defaults if file missing
    fused_scores = _fuse_scores_with_report(scores, report_labels)
    decision, positives, nodule_proxy, suspect_nodule = triage(fused_scores, thr)

    # ---- headline (bilingual-ready; EN default) ----
    headline = _lang_pick(
        en_text = "Screening result: low suspicion on this view." if decision == "Likely healthy"
                 else "Screening result: findings that may need follow-up.",
        np_text = "स्क्रिनिङ नतिजा: यो दृश्यमा शंका कम देखिन्छ।" if decision == "Likely healthy"
                 else "स्क्रिनिङ नतिजा: थप मूल्याङ्कन आवश्यक हुनसक्ने संकेत।",
        lang = lang
    )

    # ---- risk tier on nodule proxy ----
    tier = _risk_tier(float(nodule_proxy))

    # ---- top findings (limit to top-3 by score) ----
    top_findings = [
        {"label": lab, "score": _round_sig(fused_scores[lab])}
        for lab, _ in _topk(fused_scores, k=3)
    ]

    # ---- QC warnings ----
    warnings: List[str] = []
    if thresholds_path is None or not os.path.isfile(thresholds_path):
        warnings.append("Threshold file not found; using safe defaults.")
    if len(positives) == 0 and max(fused_scores.values(), default=0.0) < 0.05:
        warnings.append("Very low confidences across all labels.")
    if frontal_view is False:
        warnings.append("Non-frontal view; model tuned primarily for frontal CXRs.")
    if "Lung Lesion" not in fused_scores or "Lung Opacity" not in fused_scores:
        warnings.append("Nodule proxy labels missing; check label set.")

    # ---- bilingual guidance snippets (RAG-lite) ----
    explanations = _rag_snippets(positives, lang=lang)

    # ---- concise English summary (for UI to translate if needed) ----
    summary_en = build_english_summary(fused_scores, positives, float(nodule_proxy))

    # ---- assemble payload ----
    payload = {
        "version": VERSION,
        "generated_at": int(time.time()),
        "language": lang,
        "headline": headline,
        "decision": decision,                       # "Likely healthy" | "Suspicious findings"
        "risk_tier": tier,                          # "Low" | "Slight" | "Moderate" | "High"
        "suspect_nodule": bool(suspect_nodule),
        "nodule_proxy_score": _round_sig(float(nodule_proxy)),
        "positives": positives,                     # labels over threshold
        "top_findings": top_findings,               # top-3 by score
        "scores": fused_scores,                     # full dict 0..1 (rounded)
        "explanations": explanations,               # short bilingual hints
        "summary_en": summary_en,                   # impression/consider/nodule_risk (English)
        "safety_notice_en": "This is decision support for health professionals, not a diagnosis.",
        "safety_notice_np": "यो स्वास्थ्यकर्मीको सहायक हो, अन्तिम निदान होइन।",
        "cam_image_path": cam_image_path,
        "patient_id": patient_id,
        "study_uid": study_uid,
        "warnings": warnings,
    }
    return payload


# # Optional: tiny CLI for quick manual test
# if __name__ == "__main__":
#     rng = np.random.RandomState(7)
#     probs = rng.rand(len(LABELS))
#     card = build_card(
#         probs_or_scores=probs,
#         thresholds_path="eval_outputs/thresholds.json" if os.path.isfile("eval_outputs/thresholds.json") else None,
#         lang="en",
#         report_labels=None,
#         cam_image_path="cam_outputs/cam_Lung_Lesion.png",
#         frontal_view=True
#     )
#     print(json.dumps(card, indent=2))
