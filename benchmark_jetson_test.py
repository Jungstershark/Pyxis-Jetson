import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
import cv2

from ultralytics import YOLO

# =========================
# CONFIG (EDIT ONLY THIS)
# =========================

DATASET_ROOT = Path("data/dataset_split_v2_gpu_corrected_t0")

TEST_IMAGES_DIR = DATASET_ROOT / "images" / "test"
TEST_LABELS_DIR = DATASET_ROOT / "labels" / "test"

# Your finetuned YOLOv11 checkpoints (closed-set: class 0 ladder, class 1 pilot)
YOLOV11_MODELS = {
    "y11s": "weights/y11s_t0_960/best.pt",
    "y11m": "weights/y11m_t0_960/best.pt",
}

# Your YOLOE checkpoints (open-vocab)
YOLOE_MODELS = {
    # "yoloe_m": "weights/yoloe_m.pt",
    "yoloe_l": "weights/yoloe-11l-seg.pt",
    # example if you use the official checkpoint:
    # "yoloe_11l_seg": "yoloe-11l-seg.pt",
}

# Prompts (UPDATED: no ship hull)
# index 0 -> ladder, index 1 -> person/pilot
NAMES_PROMPT = [
    "pilot ladder, rope ladder used for pilot transfer, wooden-step ladder with side ropes, maritime boarding ladder hanging off ship hull",
    "person, human crew member standing or climbing",
]

# Mapping from YOLOE prompt index -> your dataset class id
# ladder prompt -> class 0, person prompt -> class 1
YOLOE_CLASS_MAP = {0: 0, 1: 1}

# Class names for reporting (must match dataset class ids)
CLASS_NAMES = {0: "ladder", 1: "pilot"}

# Inference params
IMG_SIZE = 960
CONF = 0.25
IOU_NMS = 0.70
DEVICE = 0          # 0 for Jetson GPU, or "cpu"
WARMUP_ITERS = 10
MAX_IMAGES = None   # set e.g. 500 for quick run

# Evaluation params
IOU_MATCH = 0.50    # for TP matching
OUT_CSV = "jetson_benchmark_results.csv"

# =========================
# UTIL
# =========================

def list_test_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    if not imgs:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return sorted(imgs)

def yolo_txt_to_boxes(label_path: Path, img_w: int, img_h: int):
    """
    Reads YOLO txt: class x_center y_center w h (normalized)
    Returns list of dict: {cls, xyxy}
    """
    if not label_path.exists():
        return []

    out = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])

            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h

            out.append({"cls": cls, "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32)})
    return out

def iou_xyxy(a, b):
    """
    a,b: [x1,y1,x2,y2]
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def percentile(arr, p):
    if not arr:
        return float("nan")
    return float(np.percentile(np.array(arr, dtype=np.float32), p))

def warmup(model: YOLO, sample_frame):
    for _ in range(WARMUP_ITERS):
        _ = model.predict(sample_frame, imgsz=IMG_SIZE, conf=CONF, iou=IOU_NMS, device=DEVICE, verbose=False)

def match_detections_to_gt(gt, preds, iou_thr=0.5):
    """
    Greedy matching per class.
    gt: list of {cls, xyxy}
    preds: list of {cls, xyxy}
    returns TP/FP/FN per class
    """
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES.keys()}

    for cls in CLASS_NAMES.keys():
        gt_cls = [g for g in gt if g["cls"] == cls]
        pr_cls = [p for p in preds if p["cls"] == cls]

        used_gt = set()

        # Sort preds by (optional) conf - if available; here no conf stored, keep order
        for p in pr_cls:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gt_cls):
                if j in used_gt:
                    continue
                iou = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thr and best_j >= 0:
                stats[cls]["tp"] += 1
                used_gt.add(best_j)
            else:
                stats[cls]["fp"] += 1

        # FNs = unmatched GTs
        stats[cls]["fn"] += (len(gt_cls) - len(used_gt))

    return stats

def stats_to_metrics(stats):
    """
    stats: dict cls -> {tp,fp,fn}
    returns precision/recall/f1 per class + overall
    """
    out = {}
    total_tp = total_fp = total_fn = 0

    for cls, d in stats.items():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        out[f"{CLASS_NAMES[cls]}_tp"] = tp
        out[f"{CLASS_NAMES[cls]}_fp"] = fp
        out[f"{CLASS_NAMES[cls]}_fn"] = fn
        out[f"{CLASS_NAMES[cls]}_precision"] = prec
        out[f"{CLASS_NAMES[cls]}_recall"] = rec
        out[f"{CLASS_NAMES[cls]}_f1"] = f1

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1   = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    out["overall_tp"] = total_tp
    out["overall_fp"] = total_fp
    out["overall_fn"] = total_fn
    out["overall_precision"] = overall_prec
    out["overall_recall"] = overall_rec
    out["overall_f1"] = overall_f1

    return out

def preds_from_ultralytics(results0, yoloe_mode=False):
    """
    Convert ultralytics result to list of {cls, xyxy}
    For YOLOE, we map prompt-index class ids to dataset ids using YOLOE_CLASS_MAP.
    """
    preds = []

    boxes = results0.boxes
    if boxes is None or len(boxes) == 0:
        return preds

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        cls = int(cls_ids[i])
        if yoloe_mode:
            if cls not in YOLOE_CLASS_MAP:
                continue
            cls = YOLOE_CLASS_MAP[cls]

        preds.append({"cls": cls, "xyxy": xyxy[i].astype(np.float32)})

    return preds

def benchmark_model(model_name: str, model: YOLO, image_paths):
    # Warmup
    sample = cv2.imread(str(image_paths[0]))
    if sample is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")
    warmup(model, sample)

    lat_ms = []
    agg_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES.keys()}
    n_imgs = min(len(image_paths), MAX_IMAGES) if MAX_IMAGES else len(image_paths)

    is_yoloe = model_name.startswith("yoloe")

    for i, img_path in enumerate(image_paths[:n_imgs]):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
        gt = yolo_txt_to_boxes(label_path, w, h)

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, iou=IOU_NMS, device=DEVICE, verbose=False)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)

        preds = preds_from_ultralytics(results[0], yoloe_mode=is_yoloe)

        stats = match_detections_to_gt(gt, preds, iou_thr=IOU_MATCH)
        for cls in agg_stats.keys():
            agg_stats[cls]["tp"] += stats[cls]["tp"]
            agg_stats[cls]["fp"] += stats[cls]["fp"]
            agg_stats[cls]["fn"] += stats[cls]["fn"]

        if (i + 1) % 200 == 0:
            print(f"[{model_name}] {i+1}/{n_imgs} processed...")

    row = {
        "model": model_name,
        "n_images": len(lat_ms),
        "lat_ms_mean": mean(lat_ms) if lat_ms else float("nan"),
        "lat_ms_median": median(lat_ms) if lat_ms else float("nan"),
        "lat_ms_p90": percentile(lat_ms, 90),
        "lat_ms_p95": percentile(lat_ms, 95),
        "lat_ms_min": float(np.min(lat_ms)) if lat_ms else float("nan"),
        "lat_ms_max": float(np.max(lat_ms)) if lat_ms else float("nan"),
        "fps_mean": 1000.0 / (mean(lat_ms) if lat_ms else float("nan")),
    }

    row.update(stats_to_metrics(agg_stats))
    return row

def main():
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing {TEST_IMAGES_DIR}")
    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(f"Missing {TEST_LABELS_DIR}")

    image_paths = list_test_images(TEST_IMAGES_DIR)
    print(f"Found {len(image_paths)} test images.")

    rows = []

    # ---- YOLOv11 models (normal closed-set)
    for name, path in YOLOV11_MODELS.items():
        print(f"\n=== Benchmark YOLOv11: {name} @ {path} ===")
        model = YOLO(path)
        rows.append(benchmark_model(name, model, image_paths))

    # ---- YOLOE models (prompted)
    for name, path in YOLOE_MODELS.items():
        print(f"\n=== Benchmark YOLOE: {name} @ {path} ===")
        model = YOLO(path)

        # Set prompt classes (2 prompts only: ladder + person)
        model.set_classes(NAMES_PROMPT, model.get_text_pe(NAMES_PROMPT))

        rows.append(benchmark_model(name, model, image_paths))

    df = pd.DataFrame(rows)

    # Make it more slide-friendly: show percentages
    pct_cols = [c for c in df.columns if c.endswith("_precision") or c.endswith("_recall") or c.endswith("_f1")]
    for c in pct_cols:
        df[c] = df[c] * 100.0

    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved: {OUT_CSV}")
    print(df[[
        "model",
        "lat_ms_mean", "lat_ms_p95", "fps_mean",
        "overall_precision", "overall_recall", "overall_f1",
        "ladder_precision", "ladder_recall", "pilot_precision", "pilot_recall"
    ]])

if __name__ == "__main__":
    main()