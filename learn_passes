from valve_qc import score_bgra_frame
import cv2
import glob
import os

folder = r"C:\Users\roman\PycharmProjects\PythonProject2\.venv\Pass"

vals = []

for path in glob.glob(os.path.join(folder, "*.png")):
    img = cv2.imread(path)
    if img is None:
        print(f"[SKIP] {path}: could not read image")
        continue

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # --- Run the QC function ---
    score, info = score_bgra_frame(bgra)

    # --- 🔽 Add this immediately after calling score_bgra_frame ---
    img_name = os.path.basename(path)
    if info.get("status") != "OK":
        print(f"[SKIP] {img_name}: {info.get('status')} - {info.get('reason','')}")
        continue
    # --- 🔼 This skips frames that don't have enough blobs or otherwise failed ---

    if info.get("decision") == "PASS":
        vals.append(info)

# --- After the loop, summarize results ---
if not vals:
    print("No PASS images found or all classified as FAIL.")
else:
    mean_angle = float(np.mean([v.get("angle", 0.0) for v in vals]))
    mean_rmse  = float(np.mean([v.get("rmse", 0.0) for v in vals]))
    mean_edge  = float(np.mean([v.get("edge_cv", 0.0) for v in vals]))

    print(f"--- Learned from {len(vals)} PASS images ---")
    print(f"Mean angle: {mean_angle:.2f}°")
    print(f"Mean rmse:  {mean_rmse:.2f}")
    print(f"Mean edge CV: {mean_edge:.3f}")
