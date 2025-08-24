import os, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from chestray_labels import LABELS, NUM_CLASSES

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P=[]; Y=[]; M=[]
    for x, y, m, _ in loader:
        x = x.to(device, non_blocking=True)
        p = torch.sigmoid(model(x)).cpu().numpy()
        P.append(p); Y.append(y.numpy()); M.append(m.numpy())
    P = np.vstack(P); Y = np.vstack(Y); M = np.vstack(M)

    metrics = {}
    aucs, aps = [], []
    for i, lab in enumerate(LABELS):
        mask = M[:, i] > 0.5
        y = Y[mask, i]; p = P[mask, i]
        if len(y) == 0 or len(np.unique(y)) < 2:
            metrics[lab] = {'auroc': None, 'auprc': None}
            aucs.append(np.nan); aps.append(np.nan)
            continue
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        metrics[lab] = {'auroc': float(auroc), 'auprc': float(auprc)}
        aucs.append(auroc); aps.append(auprc)
    metrics['macro_auroc'] = float(np.nanmean(aucs))
    metrics['macro_auprc'] = float(np.nanmean(aps))
    return metrics

def f1_opt_threshold(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    if len(thr) == 0: return 0.5
    i = int(np.nanargmax(f1))
    return float(thr[i]) if i < len(thr) else 0.5

@torch.no_grad()
def compute_thresholds(model, loader, device):
    model.eval()
    P=[]; Y=[]; M=[]
    for x, y, m, _ in loader:
        x = x.to(device, non_blocking=True)
        P.append(torch.sigmoid(model(x)).cpu().numpy())
        Y.append(y.numpy()); M.append(m.numpy())
    P = np.vstack(P); Y = np.vstack(Y); M = np.vstack(M)

    thr = {}
    for i, lab in enumerate(LABELS):
        mask = M[:, i] > 0.5
        y = Y[mask, i]; p = P[mask, i]
        thr[lab] = f1_opt_threshold(y, p) if len(np.unique(y)) == 2 else 0.5
    return thr

def load_thresholds(path):
    if path and os.path.isfile(path):
        with open(path, 'r') as f: return json.load(f)
    d = {lab: 0.5 for lab in LABELS}
    d.update({"No Finding": 0.9, "Lung Lesion": 0.4})
    return d

def scores_from_probs(probs: np.ndarray):
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

def triage(scores: dict, thr: dict):
    positives = [lab for lab, s in scores.items() if s >= thr.get(lab, 0.5)]
    healthy = ("No Finding" in positives) and all(lab=="No Finding" or scores[lab] < thr.get(lab,0.5) for lab in LABELS)
    decision = "Likely healthy" if healthy else "Suspicious findings"
    nodule_proxy = max(scores.get("Lung Lesion",0.0), scores.get("Lung Opacity",0.0))
    suspect_nodule = nodule_proxy >= thr.get("Lung Lesion", 0.4)
    return decision, positives, nodule_proxy, suspect_nodule
