import numpy as np


class WeightEstimator:
    def __init__(self, base_area: float = 5000.0, base_weight: float = 1200.0):
        self.base_area   = base_area
        self.base_weight = base_weight

    def estimate(self, bbox) -> float:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = max((x2 - x1) * (y2 - y1), 1.0)
        return round(float(area / self.base_area * self.base_weight), 2)

    def aggregate(self, weights: list):
        if not weights:
            return None
        return {
            "average_grams": round(float(np.mean(weights)), 2),
            "min_grams":     round(float(np.min(weights)),  2),
            "max_grams":     round(float(np.max(weights)),  2),
        }
