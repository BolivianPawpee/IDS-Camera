# pattern_detection.py
import cv2
import numpy as np
import os
from dataclasses import dataclass
from glob import glob

@dataclass
class PDConfig:
    pass_dir: str = r"C:\Users\roman\PycharmProjects\PythonProject2\.venv\Pass"
    fail_dir: str = r"C:\Users\roman\PycharmProjects\PythonProject2\.venv\Fail"
    min_match_count: int = 4
    angle_tolerance: float = 15.0   # degrees
    score_threshold: float = 0.2   # higher = stricter
    debug: bool = False

class PatternDetector:
    def __init__(self, config: PDConfig | None = None):
        self.cfg = config or PDConfig()
        self.orb = cv2.ORB_create(1000)
        self.pass_features = self._load_features(self.cfg.pass_dir)
        self.fail_features = self._load_features(self.cfg.fail_dir)

    def _load_features(self, folder):
        images = []
        for path in glob(os.path.join(folder, "*.png")) + glob(os.path.join(folder, "*.jpg")):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                kp, desc = self.orb.detectAndCompute(img, None)
                if desc is not None:
                    images.append({"kp": kp, "desc": desc, "path": path})
        if self.cfg.debug:
            print(f"Loaded {len(images)} samples from {folder}")
        return images

    def _match_score(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        if len(matches) == 0:
            return 0
        distances = [m.distance for m in matches]
        score = 1 - (np.mean(distances) / 100)  # smaller distance = better
        return max(0, min(score, 1))

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        kp_live, desc_live = self.orb.detectAndCompute(frame_bgr, None)
        if desc_live is None:
            return {"decision": "FAIL", "reason": "no features"}

        pass_scores = [self._match_score(p["desc"], desc_live) for p in self.pass_features]
        fail_scores = [self._match_score(f["desc"], desc_live) for f in self.fail_features]

        mean_pass = np.mean(pass_scores) if pass_scores else 0
        mean_fail = np.mean(fail_scores) if fail_scores else 0

        decision = "PASS" if mean_pass - mean_fail > 0.05 else "FAIL"
        confidence = abs(mean_pass - mean_fail)

        if self.cfg.debug:
            print(f"PASS avg={mean_pass:.2f}, FAIL avg={mean_fail:.2f}, → {decision}")

        return {
            "decision": decision,
            "confidence": confidence,
            "mean_pass_score": mean_pass,
            "mean_fail_score": mean_fail,
        }
