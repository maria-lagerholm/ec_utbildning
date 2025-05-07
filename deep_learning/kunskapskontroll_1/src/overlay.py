# overlay.py
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd
import numpy as np

###############################################################################
# CONSTANTS ###################################################################
###############################################################################

EMOTION_COLS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

EMOTION_COLOURS = {
    "angry":    (0,   0, 255),   # red
    "disgust":  (0, 255,   0),   # green
    "fear":     (255, 0, 255),   # magenta
    "happy":    (0, 255, 255),   # yellow
    "neutral":  (180,180,180),   # grey
    "sad":      (255, 0,   0),   # blue
    "surprise": (255,255,   0),  # cyan
    "low":      (120,120,120),   # colour for low‑confidence label
}

LOW_CONF_LABEL = "LOW‑CONF"

###############################################################################
# CSV helpers #################################################################
###############################################################################

def _detect_units(first_col: pd.Series, total_frames: int) -> str:
    """Return 'frame' or 'ms' depending on the magnitude of the first column."""
    numeric = pd.to_numeric(first_col, errors="coerce")
    if numeric.max(skipna=True) > total_frames:
        return "ms"
    return "frame"

def _read_csv(csv_path: Path, csv_units: str, total_frames: int) -> Tuple[pd.DataFrame,str]:
    df = pd.read_csv(csv_path)
    if csv_units == "auto":
        csv_units = _detect_units(df.iloc[:,0], total_frames)

    # rename columns
    if csv_units == "frame":
        df = df.rename(columns={df.columns[0]: "frame_idx"})
    else:
        df = df.rename(columns={df.columns[0]: "time_ms"})
        df["frame_idx"] = (df["time_ms"] / 1000).mul(df["fps"], fill_value=0).round().astype(int)

    # force numeric
    for col in EMOTION_COLS + ["x","y","width","height","frame_idx"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # dominant class & prob
    df["dominant"] = df[EMOTION_COLS].idxmax(axis=1)
    # map emotion name  →  column index (0‑based)
    col_index = {e: i for i, e in enumerate(EMOTION_COLS)}
    # take the value from the row’s dominant‑class column
    df["dom_prob"] = df[EMOTION_COLS].to_numpy()[
        np.arange(len(df)),
        df["dominant"].map(col_index).to_numpy()
    ]

    return df, csv_units

###############################################################################
# annotation ##################################################################
###############################################################################

def _scale_factors(det_size: Tuple[int,int]|None, vid_w:int, vid_h:int,
                   df: pd.DataFrame) -> Tuple[float,float]:
    if det_size:
        return vid_w/det_size[0], vid_h/det_size[1]

    # auto: if coords already fit video, use 1.0
    if df["x"].max() < vid_w and df["y"].max() < vid_h:
        return 1.0, 1.0

    # otherwise assume square input resized to min(vid_w, vid_h)
    short = min(vid_w, vid_h)
    sx = sy = short / df[["width","height"]].max().max()
    return sx, sy

def annotate_video(
        video_path: Path,
        csv_path: Path,
        output: Path,
        codec: str = "mp4v",
        det_size: Tuple[int,int]|None = None,
        rotate:int = 0,
        csv_units:str = "auto",
        hold:int = 0,
        min_prob:float = 0.10,
        show_prob:bool = False):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    v_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df, csv_units = _read_csv(csv_path, csv_units, total)
    df["fps"] = fps                                # for ms→frame conversion
    buckets: Dict[int, List[dict]] = {}
    for row in df.to_dict("records"):
        buckets.setdefault(row["frame_idx"], []).append(row)

    sx, sy = _scale_factors(det_size, v_w, v_h, df)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, (v_w, v_h))

    last: List[dict] = []
    ttl = 0

    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        current = buckets.get(idx, [])
        if current:
            last = current
            ttl  = hold
        elif ttl>0:
            ttl -= 1
        else:
            last = []

        for det in last:
            x = int(det["x"] * sx);  y = int(det["y"] * sy)
            w = int(det["width"] * sx); h = int(det["height"] * sy)

            label  = det["dominant"] if det["dom_prob"] >= min_prob else LOW_CONF_LABEL
            colour = EMOTION_COLOURS.get(label, (0,255,0))

            cv2.rectangle(frame, (x,y), (x+w, y+h), colour, 2)

            txt = label.upper()
            if show_prob and label != LOW_CONF_LABEL:
                txt += f" ({int(det['dom_prob']*100):02d}%)"

            (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-th-6), (x+tw+4, y), colour, -1)
            cv2.putText(frame, txt, (x+2, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        writer.write(frame)

    cap.release(); writer.release()
    print(f"✅ Annotated saved →  {output}")

###############################################################################
# CLI #########################################################################
###############################################################################

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Overlay emotion CSV onto video")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--csv",   type=Path, required=True)
    p.add_argument("--output",type=Path, default=Path("annotated.mp4"))
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--det-size", help="W×H used for detection, e.g. 640x480")
    p.add_argument("--rotate", type=int, default=0, choices=[0,90,-90,180])
    p.add_argument("--units", choices=["auto","frame","ms"], default="auto")
    p.add_argument("--hold", type=int, default=0)
    p.add_argument("--min-prob", type=float, default=0.10,
                   help="Min prob to draw an emotion label")
    p.add_argument("--show-prob", action="store_true",
                   help="Show probability next to label")
    args = p.parse_args()

    ds = None
    if args.det_size:
        if "x" not in args.det_size:
            raise ValueError("det-size must look like 640x480")
        ds = tuple(map(int, args.det_size.split("x")))

    annotate_video(video_path=args.video,
                   csv_path=args.csv,
                   output=args.output,
                   codec=args.codec,
                   det_size=ds,
                   rotate=args.rotate,
                   csv_units=args.units,
                   hold=args.hold,
                   min_prob=args.min_prob,
                   show_prob=args.show_prob)