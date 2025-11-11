# color_detection.py
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CDConfig:
    color: str = "blue"      # "green", "red", "blue", "yellow"...
    min_area: int = 150       # ignore tiny blobs
    open_ksize: int = 3       # morphology open kernel
    dilate_ksize: int = 5     # morphology dilate kernel
    draw_boxes: bool = True   # draw bounding boxes
    overlay_alpha: float = 0  # 0 = no overlay, 0.3 = semi-transparent tint

class ColorDetector:
    def __init__(self, config: CDConfig | None = None):
        self.cfg = config or CDConfig()
        # HSV ranges for a few common colors; adjust as needed for your lighting
        self.ranges = {
            "green": [(np.array([35, 60, 60], np.uint8), np.array([85, 255, 255], np.uint8))],
            "blue":  [(np.array([95, 60, 60], np.uint8), np.array([130, 255, 255], np.uint8))],
            "yellow":[(np.array([20, 70, 70], np.uint8), np.array([35, 255, 255], np.uint8))],
            # red is split across 0° and 180° in HSV, so it needs two ranges:
            "red":   [
                (np.array([0, 80, 80], np.uint8),   np.array([10, 255, 255], np.uint8)),
                (np.array([170, 80, 80], np.uint8), np.array([180, 255, 255], np.uint8)),
            ],
        }

    def set_color(self, color: str):
        if color not in self.ranges:
            raise ValueError(f"Unsupported color '{color}'. Valid: {list(self.ranges)}")
        self.cfg.color = color

    def process_bgra_to_bgra(self, bgra: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Input: BGRA image from your IDS pipeline (h, w, 4) uint8
        Output: (BGRA image with drawings, info dict)
        """
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image with shape (H, W, 4).")

        # BGRA -> BGR -> HSV
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Build mask (handles multi-range colors like red)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.ranges[self.cfg.color]:
            mask |= cv2.inRange(hsv, lower, upper)

        # Clean up mask
        if self.cfg.open_ksize > 1:
            k1 = np.ones((self.cfg.open_ksize, self.cfg.open_ksize), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        if self.cfg.dilate_ksize > 1:
            k2 = np.ones((self.cfg.dilate_ksize, self.cfg.dilate_ksize), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k2)

        # Draw detections
        result = bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            kept.append({"bbox": (x, y, w, h), "area": float(area), "center": (cx, cy)})
            if self.cfg.draw_boxes:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 3, (0, 255, 0), -1)

        # Optional colored overlay
        if self.cfg.overlay_alpha and 0 < self.cfg.overlay_alpha < 1:
            overlay = result.copy()
            overlay[mask > 0] = (0, 255, 0)
            result = cv2.addWeighted(result, 1 - self.cfg.overlay_alpha, overlay, self.cfg.overlay_alpha, 0)

        # Back to BGRA for your Qt path
        bgra_out = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        info = {"num_detections": len(kept), "detections": kept}
        return bgra_out, info
