import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CDConfig:
    color: str = "blue"
    min_area: int = 150
    open_ksize: int = 3
    dilate_ksize: int = 5
    draw_boxes: bool = True
    overlay_alpha: float = 0.0  # set 0.3 for a see-through highlight

class ColorDetector:
    def __init__(self, config: CDConfig | None = None):
        self.cfg = config or CDConfig()

        # Define HSV ranges — tuned for typical indoor lighting
        self.ranges = {
            "blue": [
                (np.array([95, 60, 60], np.uint8), np.array([130, 255, 255], np.uint8)),
            ],
        }

    def process_bgra_to_bgra(self, bgra: np.ndarray) -> tuple[np.ndarray, dict]:
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image with shape (H, W, 4).")

        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.ranges[self.cfg.color]:
            mask |= cv2.inRange(hsv, lower, upper)

        k1 = np.ones((self.cfg.open_ksize, self.cfg.open_ksize), np.uint8)
        k2 = np.ones((self.cfg.dilate_ksize, self.cfg.dilate_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k2)

        result = bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            detections.append({"center": (cx, cy), "area": area})
            if self.cfg.draw_boxes:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(result, (cx, cy), 3, (255, 0, 0), -1)

        if 0 < self.cfg.overlay_alpha < 1:
            overlay = result.copy()
            overlay[mask > 0] = (255, 0, 0)
            result = cv2.addWeighted(result, 1 - self.cfg.overlay_alpha, overlay, self.cfg.overlay_alpha, 0)

        bgra_out = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        return bgra_out, {"num_detections": len(detections), "detections": detections}
