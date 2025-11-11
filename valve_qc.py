# valve_qc.py
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class QCConfig:
    # HSV thresholds for blue (tune if needed)
    hsv_lower: tuple = (95, 40, 40)
    hsv_upper: tuple = (135, 255, 255)

    # Morphology + area
    min_blob_area: int = 700       # ignore tiny specks
    open_ksize: int = 3            # gentle noise removal
    dilate_ksize: int = 5          # mild gap closing

    # Geometry tolerances (loose for now)
    max_extra_blobs: float = 4
    max_area_cv: float = 0.50          # area similarity of kept blobs
    max_edge_mismatch: float = 0.45    # rectangle edge symmetry
    max_off_rect_rmse: float = 140.0    # fit error to ideal rectangle
    allowed_angle_deg: float = 40.0
    expected_angle_deg: float = 28.0  # learned mean orientation for a correct part
    angle_pass_deg: float = 6.0  # must be within ±6° to gain PASS
    angle_fail_deg: float = 9.0  # loses PASS only if |err| > 9°
    require_angle_pass: bool = True  # orientation must pass or overall FAIL
    angle_tolerance_deg: float = 35.0  # allowed deviation from expected
    # rectangle orientation

    draw: bool = True


def _blue_mask(bgr: np.ndarray, cfg: QCConfig) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(cfg.hsv_lower, np.uint8),
        np.array(cfg.hsv_upper, np.uint8),
    )

    # Close then open: bridge tiny gaps, then drop specks
    if cfg.dilate_ksize > 1:
        kernel_close = np.ones((cfg.dilate_ksize, cfg.dilate_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    if cfg.open_ksize > 1:
        kernel_open = np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def _find_blue_blobs(mask: np.ndarray, cfg: QCConfig):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < cfg.min_blob_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blobs.append({
            "area": float(a),
            "bbox": (x, y, w, h),
            "center": (x + w / 2.0, y + h / 2.0),
            "contour": c
        })
    blobs.sort(key=lambda d: d["area"], reverse=True)
    return blobs

def _wrap_deg(a):
    # Wrap angle to (-180, 180]
    a = (a + 180.0) % 360.0 - 180.0
    return a

def _order_corners_xy(centers):
    """
    Returns TL, TR, BR, BL deterministically using PCA axes.
    """
    pts = np.array(centers, dtype=np.float32)  # (4,2)
    c = pts.mean(axis=0)
    X = pts - c
    cov = (X.T @ X) / 4.0
    vals, vecs = np.linalg.eig(cov)
    i0 = int(np.argmax(vals))           # principal axis
    v0 = vecs[:, i0]                    # left–right axis
    v1 = np.array([-v0[1], v0[0]])      # top–bottom axis (perpendicular)

    # project points onto axes
    u = X @ v0   # left–right coordinate
    v = X @ v1   # top–bottom coordinate

    # left is smaller u, right is larger u; top is smaller v
    left_idxs  = np.argsort(u)[:2]
    right_idxs = np.argsort(u)[-2:]
    left_pts   = pts[left_idxs]
    right_pts  = pts[right_idxs]

    # on each side, sort by v (top first)
    left_v  = (left_pts - c) @ v1
    right_v = (right_pts - c) @ v1

    TL = left_pts[np.argmin(left_v)]
    BL = left_pts[np.argmax(left_v)]
    TR = right_pts[np.argmin(right_v)]
    BR = right_pts[np.argmax(right_v)]
    return TL, TR, BR, BL


def _wrap_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a

def _order_corners_xy(centers):
    """
    centers: list/array of 4 (x, y) points
    returns TL, TR, BR, BL in that order
    """
    pts = np.array(centers, dtype=np.float32)
    idx_y = np.argsort(pts[:, 1])
    top2 = pts[idx_y[:2]]
    bot2 = pts[idx_y[2:]]
    top2 = top2[np.argsort(top2[:, 0])]
    bot2 = bot2[np.argsort(bot2[:, 0])]
    TL, TR = top2[0], top2[1]
    BL, BR = bot2[0], bot2[1]
    return TL, TR, BR, BL

def _rectangle_features(centers):
    """
    centers: list/array of 4 (x, y) points
    returns: dict(angle, edge_cv, rmse, short_mean, long_mean, diag_mean)
    """
    pts = np.array(centers, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Need 4 points, got {pts.shape}")

    # --- Deterministic angle from the top edge ---
    TL, TR, BR, BL = _order_corners_xy(pts)
    vx, vy = (TR[0] - TL[0], TR[1] - TL[1])
    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    angle_deg = _wrap_deg(angle_deg)

    # Pairwise distances (6)
    d = []
    for i in range(4):
        for j in range(i + 1, 4):
            d.append(np.linalg.norm(pts[i] - pts[j]))
    d = np.sort(np.array(d, dtype=np.float32))

    short = d[0:2]       # two short edges
    long_ = d[2:4]       # two long edges
    diag = d[4:6]        # two diagonals

    # Edge symmetry
    short_cv = float(np.std(short) / (np.mean(short) + 1e-6))
    long_cv  = float(np.std(long_) / (np.mean(long_) + 1e-6))

    # --- Procrustes-ish fit to an ideal rectangle (needs mean) ---
    w = np.mean(short)
    h = np.mean(long_)
    target = np.array(
        [[-w/2, -h/2],
         [ w/2, -h/2],
         [ w/2,  h/2],
         [-w/2,  h/2]], dtype=np.float32
    )

    mean = pts.mean(axis=0)                # ← define mean here
    src = pts - mean
    tgt = target - target.mean(axis=0)

    den = (tgt.T @ tgt)
    if np.linalg.det(den) == 0 or np.any(np.isnan(den)):
        rmse = 9999.0
    else:
        A = src.T @ tgt @ np.linalg.pinv(den)
        fitted = (tgt @ A.T) + mean
        rmse = float(np.sqrt(np.mean(np.sum((pts - fitted) ** 2, axis=1))))

    return dict(
        angle=angle_deg,
        edge_cv=max(short_cv, long_cv),
        rmse=rmse,
        short_mean=float(np.mean(short)),
        long_mean=float(np.mean(long_)),
        diag_mean=float(np.mean(diag)),
    )

def _label_valves_by_position(centers):
    """
    centers: list of 4 (x,y). Uses _order_corners_xy to get TL,TR,BR,BL points,
    then maps each label to the nearest center index.
    Returns: label2idx dict like {"TL": i, "TR": j, "BR": k, "BL": m}
    """
    TL, TR, BR, BL = _order_corners_xy(centers)  # these are points
    pts = np.array(centers, dtype=np.float32)
    labels = {"TL": TL, "TR": TR, "BR": BR, "BL": BL}
    label2idx = {}
    for name, p in labels.items():
        d = np.linalg.norm(pts - p, axis=1)
        label2idx[name] = int(np.argmin(d))
    return label2idx


def score_bgra_frame(bgra: np.ndarray, cfg: QCConfig = QCConfig()):
    """
    Returns (overlay_bgra, info)
    info = {
        'decision': 'PASS' | 'FAIL',
        'reason': str,
        'angle': float,
        'rmse': float,
        'edge_cv': float,
        'num_blobs': int,
        'angle_error': float,
        'valves': {TL/TR/BR/BL: {angle, area, bbox, center}},
        'valve_reasons': {TL/TR/BR/BL: ["OK" or reasons...]},
    }
    """
    # 1) Color → mask → blobs
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    mask = _blue_mask(bgr, cfg)

    # ---- HSV/Mask diagnostics (helps when blobs==0) ----
    blue_px = int(np.count_nonzero(mask))
    pct = 100.0 * blue_px / (mask.size + 1e-6)
    if pct < 0.1:  # practically nothing detected
        hsv_sample = cv2.cvtColor(cv2.resize(bgr, (64, 36)), cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean = map(float, hsv_sample.reshape(-1, 3).mean(axis=0))
        print(f"[HSV DEBUG] blue%={pct:.3f}  H≈{h_mean:.1f}  S≈{s_mean:.1f}  V≈{v_mean:.1f}")

    blobs = _find_blue_blobs(mask, cfg)
    print(f"[MASK DEBUG] blue_px={blue_px} ({pct:.2f}%), blobs={len(blobs)}")

    overlay = bgr.copy()
    decision = "FAIL"
    reason = ""

    # always-init locals that may be filled later
    feats = {}
    angle_err = 0.0
    main4 = []

    if len(blobs) < 2:
        # not enough blue to proceed
        reason = f"Found only {len(blobs)} blue blobs (<2)"
    else:
        # Keep up to the 4 largest (expected valves)
        main4 = blobs[:4] if len(blobs) >= 4 else blobs
        centers = [b["center"] for b in main4]
        areas = np.array([b["area"] for b in main4], np.float32)
        area_cv = float(np.std(areas) / (np.mean(areas) + 1e-6))

        # --- Normalize global rotation so valves are upright ---
        if len(centers) >= 3:
            # Fit a minimum bounding rectangle to all valve centers
            rect = cv2.minAreaRect(np.array(centers, dtype=np.float32))
            board_angle = rect[-1]  # degrees from horizontal, in range [-90, 0)

            # Normalize OpenCV's angle convention (make it 0° when horizontal)
            if rect[1][0] < rect[1][1]:
                board_angle += 90.0

            # Rotate the entire frame to deskew the part
            center = (bgr.shape[1] // 2, bgr.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, board_angle, 1.0)
            norm = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]))

            # Re-run color mask and blob detection on normalized image
            mask = _blue_mask(norm, cfg)
            blobs = _find_blue_blobs(mask, cfg)
            main4 = blobs[:4] if len(blobs) >= 4 else blobs
            centers = [b["center"] for b in main4]
            centers = [b["center"] for b in blobs[:4]] if len(blobs) >= 4 else [b["center"] for b in blobs]

            print(f"[ROT NORM] board_angle={board_angle:.1f}°, blobs after norm={len(blobs)}")
        else:
            norm = bgr.copy()

        n_centers = len(centers)

        # Not enough geometry? Return an explicit status for the caller to skip.
        if n_centers < 3:
            info = {
                "status": "INSUFFICIENT_BLOBS",
                "num_blobs": int(n_centers),
                "angle": float('nan'),
                "angle_error": float('nan'),
                "edge_cv": float('nan'),
                "rmse": float('inf'),
                "decision": "REJECT",
                "reason": f"Need >=3 blobs for geometry, got {n_centers}",
            }
            print(f"[QC] num_blobs={n_centers}, decision=REJECT ({info['reason']})")
            return None, info

        # ---- Per-valve orientation consistency (now that main4 exists) ----
        per_ok = True
        if len(main4) >= 2:
            per_angles = []
            for b in main4:
                rect = cv2.minAreaRect(b["contour"])  # ((cx,cy),(w,h),theta in [-90,0))
                w, h = rect[1]
                theta = rect[2]
                if w < h:
                    theta += 90.0
                theta = float(_wrap_deg(theta))
                per_angles.append(theta)

            per_angles = np.array(per_angles, np.float32)
            # normalize around median to kill 90° flips and outliers
            med = float(np.median(per_angles))
            adj = np.array([_wrap_deg(a - med) for a in per_angles], dtype=np.float32)
            # allow a bit more spread; 10° was too strict for these blobs
            per_ok = (np.ptp(adj) <= 20.0)

        # ---- Branches by how many blobs we have ----
        if n_centers >= 4:
            feats = _rectangle_features(np.array(centers, np.float32))
            angle_err = _wrap_deg(feats["angle"] - cfg.expected_angle_deg)
            angle_ok = abs(angle_err) <= cfg.angle_tolerance_deg

            # area_cv should be recomputed for the post-norm 4
            areas = np.array([b["area"] for b in blobs[:4]], np.float32)
            area_cv = float(np.std(areas) / (np.mean(areas) + 1e-6))

            edges_ok = feats["edge_cv"] <= cfg.max_edge_mismatch
            rmse_ok = feats["rmse"] <= cfg.max_off_rect_rmse
            area_ok = area_cv <= cfg.max_area_cv

            if not per_ok:
                decision, reason = "FAIL", "per-valve twist"
            elif cfg.require_angle_pass and not angle_ok:
                decision, reason = "FAIL", "orientation"
            else:
                pass_conditions = sum([angle_ok, edges_ok, rmse_ok, area_ok])
                if pass_conditions >= 2:
                    decision, reason = "PASS", "OK"
                else:
                    decision = "FAIL"
                    reason = (f"angle_ok={angle_ok}, edges_ok={edges_ok}, "
                              f"rmse_ok={rmse_ok}, area_ok={area_ok}")

            # ... (labeling + drawing stays the same, but be sure to use blobs[:4])


        elif len(main4) == 3:

            pts = np.array(centers, np.float32)

            # (keep your dist_cv, area_ok, spread_ok calculations)

            # do NOT gate on per_ok here:

            if area_ok and spread_ok:

                decision, reason = "PASS", "3-blobs heuristic"

            else:

                reason = f"Only {len(main4)} blobs (area_ok={area_ok}, spread_ok={spread_ok})"

    # PASS/FAIL text
    color = (0, 200, 0) if decision == "PASS" else (0, 0, 255)
    cv2.putText(overlay, f"{decision} - {reason}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    bgra_out = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # Safe info dict
    info = {
        "decision": decision,
        "reason": reason,
        "num_blobs": len(blobs),
        "angle": float(feats.get("angle", 0.0)) if "feats" in locals() else 0.0,
        "rmse": float(feats.get("rmse", 0.0)) if "feats" in locals() else 0.0,
        "edge_cv": float(feats.get("edge_cv", 0.0)) if "feats" in locals() else 0.0,
        "angle_error": float(angle_err),
    }

    # Attach per-valve data if computed
    if 'label2idx' in locals():
        info["valves"] = {n: per_valve[n] for n in ["TL", "TR", "BR", "BL"]}
        info["valve_reasons"] = {n: valve_reasons[n] for n in ["TL", "TR", "BR", "BL"]}

    print(
        f"[QC] angle={info['angle']:.1f}deg, "
        f"angle_err={info['angle_error']:.1f}deg, "
        f"edge_cv={info['edge_cv']:.3f}, "
        f"rmse={info['rmse']:.1f}, "
        f"num_blobs={info['num_blobs']}, "
        f"decision={info['decision']} ({info['reason']})"
    )
    n_centers = len(blobs)

    info = {
        "decision": decision,
        "reason": reason,
        "num_blobs": int(n_centers),
        "angle": float(feats.get("angle", 0.0)) if 'feats' in locals() else 0.0,
        "rmse": float(feats.get("rmse", 0.0)) if 'feats' in locals() else 0.0,
        "edge_cv": float(feats.get("edge_cv", 0.0)) if 'feats' in locals() else 0.0,
        "angle_error": float(angle_err),
        "status": "OK",  # ← add this
    }

    return bgra_out, info




# ---------------------------------------------------------------------
# Helper to learn averages from PASS images (use for tuning thresholds)
# ---------------------------------------------------------------------
def learn_geometry_from_passes(folder: str):
    """
    Reads all PNGs in a folder, runs the QC, and prints averages for tuning.
    Usage:
        learn_geometry_from_passes(r"C:/path/to/Pass")
    """
    import glob

    vals = []
    for path in glob.glob(folder.rstrip("/\\") + "/*.png"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        _, info = score_bgra_frame(bgra)
        if info.get("decision") == "PASS":
            vals.append(info)

    if not vals:
        print("No PASS images found or all classified as FAIL.")
        return

    mean_angle = float(np.mean([v.get("angle", 0.0) for v in vals]))
    mean_rmse  = float(np.mean([v.get("rmse", 0.0) for v in vals]))
    mean_edge  = float(np.mean([v.get("edge_cv", 0.0) for v in vals]))

    print(f"--- Learned from {len(vals)} PASS images ---")
    print(f"Mean angle: {mean_angle:.2f}°")
    print(f"Mean rmse:  {mean_rmse:.2f}")
    print(f"Mean edge CV: {mean_edge:.3f}")
