import pandas as pd
from typing import List
from river.drift import ADWIN  # Import ADWIN from river library

class ADWINDriftDetector:
    """Detects concept drift points using the ADWIN algorithm.
    Only counts drifts with sufficient data and merges the last fold if insufficient data"""

    def __init__(self, delta: float = 0.002, min_fold_len: int = 60):
        self.detector = ADWIN(delta=delta)
        self.min_fold_len = min_fold_len
        self.drift_points_: List[int] = []

    def detect(self, data: pd.DataFrame, target_column: str, seq_len: int = 30, test_ratio: float = 0.2) -> List[int]:
        self.drift_points_ = []
        series_to_monitor = data[target_column]
        drift_points_temp = []
        last_point = 0

        # Step 1: Detect drift points
        for i, val in enumerate(series_to_monitor):
            self.detector.update(val)
            if self.detector.drift_detected:
                fold_len = i - last_point
                split_idx = int((1-test_ratio)*fold_len)
                train_len = split_idx
                test_len = fold_len - split_idx
                if fold_len >= self.min_fold_len and train_len >= seq_len and test_len >= seq_len:
                    drift_points_temp.append(i)
                    last_point = i
                # If insufficient data, skip this drift

        # Step 2: Consider the last fold (period from last drift -> len(data))
        all_points = [0] + drift_points_temp + [len(data)]
        last_drift_idx = len(drift_points_temp) - 1
        prev_drift = drift_points_temp[last_drift_idx] if last_drift_idx >= 0 else 0
        final_fold_len = len(data) - prev_drift
        split_idx = int((1-test_ratio)*final_fold_len)
        train_len = split_idx
        test_len = final_fold_len - split_idx

        # If the last fold has "insufficient data", merge with previous drift
        if final_fold_len < self.min_fold_len or train_len < seq_len or test_len < seq_len:
            # Remove the last drift (if there are more than 1 drift)
            if len(drift_points_temp) > 0:
                drift_points_temp = drift_points_temp[:-1]

        self.drift_points_ = drift_points_temp
        return self.drift_points_