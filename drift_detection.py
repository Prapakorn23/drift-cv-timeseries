import pandas as pd
from typing import List
from river.drift import ADWIN  # นำเข้า ADWIN จากไลบรารี river

class ADWINDriftDetector:
    """Detects concept drift points using the ADWIN algorithm.
    จะนับเฉพาะ drift ที่มีข้อมูลพอ และ merge fold สุดท้ายหากข้อมูลไม่พอ"""

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
                # ถ้าไม่พอข้อมูล ให้ข้าม drift นี้ไปเลย

        # Step 2: พิจารณา fold สุดท้าย (ช่วง drift สุดท้าย -> len(data))
        all_points = [0] + drift_points_temp + [len(data)]
        last_drift_idx = len(drift_points_temp) - 1
        prev_drift = drift_points_temp[last_drift_idx] if last_drift_idx >= 0 else 0
        final_fold_len = len(data) - prev_drift
        split_idx = int((1-test_ratio)*final_fold_len)
        train_len = split_idx
        test_len = final_fold_len - split_idx

        # ถ้า fold สุดท้าย "ไม่พอข้อมูล" ให้ merge กับ drift ก่อนหน้า
        if final_fold_len < self.min_fold_len or train_len < seq_len or test_len < seq_len:
            # ลบ drift สุดท้ายออก (ถ้ามีมากกว่า 1 drift)
            if len(drift_points_temp) > 0:
                drift_points_temp = drift_points_temp[:-1]

        self.drift_points_ = drift_points_temp
        return self.drift_points_