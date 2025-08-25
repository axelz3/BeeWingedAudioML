# evaluate_in_idle.py — run directly in IDLE (no CMD needed)

import re, pickle
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
import librosa
import pandas as pd
from datetime import datetime
import warnings
from scipy.io.wavfile import WavFileWarning
warnings.simplefilter("ignore", WavFileWarning)


# --- Simple GUI pickers for IDLE ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# -----------------------------
# CONFIG you can tweak (optional)
# -----------------------------
THRESHOLD = 0.62          # file-level decision on mean probability
WIN_S = None             # None = use window size from model config; else e.g. 5.0
HOP_S = None             # None = use hop from model config; else e.g. 2.5
MAX_WINS_PER_FILE = 50  # cap windows per file to save time/memory
GLOB_SUFFIXES = {".wav", ".WAV"}  # which file suffixes to include

# -----------------------------
# Audio + features (match training)
# -----------------------------
def to_float_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind == "u":
        max_val = np.iinfo(x.dtype).max
        x = (x.astype(np.float32) - max_val/2) / (max_val/2)
    elif x.dtype.kind == "i":
        max_val = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x

def extract_features(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=cfg["n_mels"])
    log_mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=cfg["n_mfcc"])
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zc = librosa.feature.zero_crossing_rate(y)

    feats = np.hstack([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_d1.mean(axis=1), mfcc_d1.std(axis=1),
        mfcc_d2.mean(axis=1), mfcc_d2.std(axis=1),
        sc.mean(axis=1), sc.std(axis=1),
        bw.mean(axis=1), bw.std(axis=1),
        ro.mean(axis=1), ro.std(axis=1),
        zc.mean(axis=1), zc.std(axis=1),
    ]).astype(np.float32)
    return feats

def extract_windows_memmap(raw_pcm, sr: int, cfg: dict, win_s: float, hop_s: float, keep_short: bool=True):
    n = int(raw_pcm.shape[0])
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))

    if raw_pcm.ndim == 2:
        raw_pcm = raw_pcm.mean(axis=1)

    # short file → single window
    if n < win:
        if not keep_short:
            return np.empty((0,)), []
        y = to_float_mono(raw_pcm)
        sr_eff = sr
        if sr > cfg["target_sr"]:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg["target_sr"], res_type=cfg["resample_type"])
            sr_eff = cfg["target_sr"]
        feats = extract_features(y, sr_eff, cfg)
        return feats.reshape(1, -1), [(0.0, n / sr)]

    Xw, intervals = [], []
    for start in range(0, n - win + 1, hop):
        seg = raw_pcm[start:start + win]
        y = to_float_mono(seg)
        sr_eff = sr
        if sr > cfg["target_sr"]:
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg["target_sr"], res_type=cfg["resample_type"])
            sr_eff = cfg["target_sr"]
        feats = extract_features(y, sr_eff, cfg)
        Xw.append(feats)
        intervals.append((start / sr, (start + win) / sr))
    return np.vstack(Xw), intervals

# -----------------------------
# Label inference from filename
# -----------------------------
NEG_PATTERNS = [r'no[_\s-]?queenbee', r'no[_\s-]?queen', r'queenless']
POS_PATTERNS = [r'queen[_\s-]?bee']
NEG_RE = re.compile("|".join(NEG_PATTERNS), re.IGNORECASE)
POS_RE = re.compile("|".join(POS_PATTERNS), re.IGNORECASE)

def infer_label_from_name(name: str) -> Optional[int]:
    if NEG_RE.search(name):
        return 0
    if POS_RE.search(name) and not NEG_RE.search(name):
        return 1
    return None

# -----------------------------
# Model I/O
# -----------------------------
def load_model(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    model = blob["model"]
    cfg = blob["config"]
    cfg.setdefault("win_s", 5.0)
    cfg.setdefault("hop_s", 2.5)
    return model, cfg

# -----------------------------
# Evaluation
# -----------------------------
def find_labeled_wavs(root: Path) -> List[Tuple[Path, int]]:
    pairs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in GLOB_SUFFIXES:
            lab = infer_label_from_name(p.name)
            if lab is not None:
                pairs.append((p, lab))
    return pairs

def evaluate_on_folder(model, cfg: dict, audio_root: Path,
                       threshold: float = THRESHOLD,
                       win_s: Optional[float] = WIN_S,
                       hop_s: Optional[float] = HOP_S,
                       max_wins_per_file: int = MAX_WINS_PER_FILE):
    win_s = cfg.get("win_s") if win_s is None else win_s
    hop_s = cfg.get("hop_s") if hop_s is None else hop_s

    rows = find_labeled_wavs(audio_root)
    if not rows:
        raise RuntimeError("No labeled WAVs found (by filename). Check patterns or folder.")

    n_pos = sum(l == 1 for _, l in rows)
    n_neg = sum(l == 0 for _, l in rows)
    print(f"[INFO] Found {len(rows)} labeled files ({n_pos} pos / {n_neg} neg)")
    print(f"[INFO] Using win_s={win_s} hop_s={hop_s} threshold={threshold} max_wins_per_file={max_wins_per_file}")

    all_win_y, all_win_pred, all_win_proba = [], [], []
    file_records = []

    for i, (p, label) in enumerate(rows, 1):
        try:
            sr, raw = wavfile.read(str(p), mmap=True)
        except Exception as e:
            print(f"[SKIP] {p.name} - {e}")
            continue

        Xw, _ = extract_windows_memmap(raw, sr, cfg, win_s, hop_s, keep_short=True)
        if Xw.size == 0:
            print(f"[SKIP] {p.name} - no windows")
            continue

        if max_wins_per_file and Xw.shape[0] > max_wins_per_file:
            Xw = Xw[:max_wins_per_file]

        win_pred = model.predict(Xw)
        if hasattr(model, "predict_proba"):
            win_proba = model.predict_proba(Xw)[:, 1]
        else:
            win_proba = win_pred.astype(float)

        all_win_y.extend([label] * len(win_pred))
        all_win_pred.extend(win_pred.tolist())
        all_win_proba.extend(win_proba.tolist())

        mean_prob = float(np.mean(win_proba))
        maj_vote = int(np.mean(win_pred) >= 0.5)
        thr_pred = int(mean_prob >= threshold)

        file_records.append({
            "file_path": str(p),
            "true_label": int(label),
            "n_windows": int(len(win_pred)),
            "mean_prob": mean_prob,
            "pos_win_frac": float(np.mean(win_pred)),
            "maj_vote": int(maj_vote),
            "thr_pred": int(thr_pred),
        })

        if i % 25 == 0 or i == len(rows):
            print(f"[PROGRESS] {i}/{len(rows)} files...")

    # --- Window-level metrics ---
    wy = np.array(all_win_y, dtype=int)
    wp = np.array(all_win_pred, dtype=int)
    wpr = np.array(all_win_proba, dtype=float)

    print("\n=== Window-level metrics ===")
    if len(np.unique(wy)) == 2:
        try:
            print(f"ROC-AUC: {roc_auc_score(wy, wpr):.3f}")
        except Exception:
            pass
    print(f"Accuracy: {accuracy_score(wy, wp):.3f}")
    print(classification_report(wy, wp, labels=[0,1], digits=3, zero_division=0))
    print("Confusion (rows=true 0/1, cols=pred 0/1):\n", confusion_matrix(wy, wp, labels=[0,1]))

    # --- File-level metrics ---
    fdf = pd.DataFrame(file_records)
    fy = fdf["true_label"].to_numpy(int)
    fhat_thr = fdf["thr_pred"].to_numpy(int)
    fhat_maj = fdf["maj_vote"].to_numpy(int)

    def summarize(name, y_true, y_pred):
        print(f"\n=== File-level ({name}) ===")
        print("Accuracy:", f"{accuracy_score(y_true, y_pred):.3f}")
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        print(f"Precision: {pr:.3f}  Recall: {rc:.3f}  F1: {f1:.3f}")
        print("Confusion (rows=true 0/1, cols=pred 0/1):\n", confusion_matrix(y_true, y_pred, labels=[0,1]))

    summarize(f"mean_prob ≥ {threshold:.2f}", fy, fhat_thr)
    summarize("majority_vote", fy, fhat_maj)

    return fdf

# -----------------------------
# Tiny GUI runner for IDLE
# -----------------------------
def pick_model_and_folder():
    if not TK_AVAILABLE:
        raise RuntimeError("tkinter not available. Install it or set paths manually.")
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Select bee_audio_rf.pkl",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    if not model_path:
        raise RuntimeError("No model selected.")
    audio_dir = filedialog.askdirectory(title="Select root folder with WAV files")
    if not audio_dir:
        raise RuntimeError("No audio folder selected.")
    return Path(model_path), Path(audio_dir)

if __name__ == "__main__":
    try:
        model_pkl, audio_root = pick_model_and_folder()
        print(f"[INFO] Model: {model_pkl}")
        print(f"[INFO] Audio folder: {audio_root}")

        model, cfg = load_model(model_pkl)
        print(f"[INFO] Loaded model. Feature config: {cfg}")

        results_df = evaluate_on_folder(
            model, cfg, audio_root,
            threshold=THRESHOLD, win_s=WIN_S, hop_s=HOP_S,
            max_wins_per_file=MAX_WINS_PER_FILE
        )

        # Save per-file results CSV next to the audio folder
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = audio_root / f"eval_results_per_file_{ts}.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"\n[INFO] Per-file results written to: {out_csv}")

        if TK_AVAILABLE:
            messagebox.showinfo("Done", f"Evaluation finished.\nResults CSV:\n{out_csv}")

    except Exception as e:
        print(f"[ERROR] {e}")
        if TK_AVAILABLE:
            try:
                messagebox.showerror("Error", str(e))
            except Exception:
                pass
