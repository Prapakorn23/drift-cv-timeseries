import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import RNNRegressor, LinearRegressionModel

class DriftAdaptiveTimeSeriesCV:
    """Performs cross-validation using a rolling window approach based on detected drift points.
    แก้ไขให้ข้ามจุด drift ที่แบ่ง train/test ไม่ได้"""
    def __init__(self, model_type: str = 'LSTM', model_params: dict = None):
        self.model_type = model_type.upper()
        self.model_params = model_params or {}

    def run(self, X: pd.DataFrame, y: pd.Series, drift_points: List[int]) -> Tuple[List[float], List[float]]:
        metrics_rmse, metrics_mae = [], []
        seq_len = self.model_params.get('sequence_length', 30)
        min_fold_len = max(seq_len * 2, 40)
        test_ratio = 0.2

        all_points = sorted(list(set([0] + drift_points + [len(X)])))
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i+1]
            fold_length = end - start
            split_point = int(fold_length * (1 - test_ratio))
            train_start = start
            train_end = start + split_point
            test_start = train_end
            test_end = end

            train_len = train_end - train_start
            test_len = test_end - test_start

            # เงื่อนไขสำหรับ Linear ไม่ต้องใช้ seq_len
            if self.model_type == 'LINEAR':
                if train_len <= 0 or test_len <= 0:
                    print(f"[Adaptive Fold {i+1}] Skipping (train/test < 1): train({train_len}), test({test_len})")
                    continue
            else: # เงื่อนไขสำหรับ RNN/LSTM/GRU
                if train_len <= seq_len or test_len <= seq_len:
                    print(f"[Adaptive Fold {i+1}] Skipping (train/test < seq_len): train({train_len}), test({test_len}), seq_len({seq_len})")
                    continue

            split_point = int(fold_length * (1 - test_ratio))
            train_start = start
            train_end = start + split_point
            test_start = train_end
            test_end = end

            train_len = train_end - train_start
            test_len = test_end - test_start

            # ตรวจสอบ train/test ต้องมีขนาดมากกว่า sequence_length
            if train_len <= seq_len or test_len <= seq_len:
                print(f"[Adaptive Fold {i+1}] Skipping (train/test < seq_len): train({train_len}), test({test_len}), seq_len({seq_len})")
                continue

            X_train, X_test = X.iloc[train_start:train_end], X.iloc[test_start:test_end]
            y_train, y_test = y.iloc[train_start:train_end], y.iloc[test_start:test_end]

            rmse, mae = np.nan, np.nan
            if self.model_type in ['RNN', 'LSTM', 'GRU']:
                model = RNNRegressor(model_type=self.model_type, **self.model_params)
            elif self.model_type == 'LINEAR':
                model = LinearRegressionModel(**self.model_params)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if self.model_type in ['RNN', 'LSTM', 'GRU']:
                    y_test_aligned = y_test.iloc[model.seq_generator.sequence_length:]
                    y_pred = y_pred[:len(y_test_aligned)]
                else:
                    y_test_aligned = y_test
                if len(y_pred) > 0 and len(y_test_aligned) > 0:
                    rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
                    mae = mean_absolute_error(y_test_aligned, y_pred)
                    metrics_rmse.append(rmse)
                    metrics_mae.append(mae)
                    print(f"[Adaptive Fold {i+1}] RMSE={rmse:.3f}, MAE={mae:.3f}")
                else:
                    print(f"[Adaptive Fold {i+1}] Not enough data to calculate metrics.")
            except Exception as e:
                print(f"[Adaptive Fold {i+1}] Error: {e}")

        return metrics_rmse, metrics_mae

class BaselineTimeSeriesCV:
    """
    Performs cross-validation for time series data using a rolling window.
    The data is split into n_splits + 1 parts.
    """
    def __init__(self, model_type: str = 'LSTM', model_params: dict = None,
                 n_splits: int = 4):
        self.model_type = model_type.upper()
        self.model_params = model_params or {}
        self.n_splits = n_splits
        if self.n_splits < 1:
            raise ValueError("n_splits must be at least 1.")

    def run(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[float], List[float]]:
        metrics_rmse, metrics_mae = [], []
        total_size = len(X)
        
        # Calculate the size of each part. The data is split into n_splits + 1 parts.
        part_size = total_size // (self.n_splits + 1)
        test_ratio = 0.2
        
        if part_size <= 0:
            raise ValueError("Not enough data for the specified number of splits.")

        for i in range(self.n_splits):
            # กำหนดช่วงข้อมูลสำหรับ Fold ปัจจุบัน (ข้อมูลส่วนที่ i+1)
            fold_start = i * part_size
            fold_end = (i + 1) * part_size
            if i == self.n_splits - 1: # จัดการส่วนสุดท้ายที่อาจมีขนาดไม่เท่ากัน
                fold_end = total_size

            fold_data_X = X.iloc[fold_start:fold_end]
            fold_data_y = y.iloc[fold_start:fold_end]

            # แบ่งชุดข้อมูลภายใน Fold เป็น Train และ Test
            split_point = int(len(fold_data_X) * (1 - test_ratio))
            
            X_train = fold_data_X.iloc[:split_point]
            y_train = fold_data_y.iloc[:split_point]
            
            X_test = fold_data_X.iloc[split_point:]
            y_test = fold_data_y.iloc[split_point:]

            # Check for sufficient data size for RNN-based models
            seq_len = self.model_params.get('sequence_length', 30) if self.model_type != 'LINEAR' else 0
            train_len = len(X_train)
            test_len = len(X_test)
            
            if train_len <= seq_len or test_len <= seq_len:
                print(f"[Baseline Fold {i+1}] Skipping (train/test < seq_len): train({train_len}), test({test_len}), seq_len({seq_len})")
                continue

            rmse, mae = np.nan, np.nan
            try:
                # Model instantiation and fitting
                if self.model_type in ['RNN', 'LSTM', 'GRU']:
                    model = RNNRegressor(model_type=self.model_type, **self.model_params)
                elif self.model_type == 'LINEAR':
                    model = LinearRegressionModel(**self.model_params)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Align prediction and test data length for RNN models
                if self.model_type in ['RNN', 'LSTM', 'GRU']:
                    y_test_aligned = y_test.iloc[seq_len:]
                else:
                    y_test_aligned = y_test
                
                min_len = min(len(y_pred), len(y_test_aligned))
                if min_len > 0:
                    y_pred_trimmed = y_pred[:min_len]
                    y_test_trimmed = y_test_aligned[:min_len]
                    rmse = np.sqrt(mean_squared_error(y_test_trimmed, y_pred_trimmed))
                    mae = mean_absolute_error(y_test_trimmed, y_pred_trimmed)
                    metrics_rmse.append(rmse)
                    metrics_mae.append(mae)
                    print(f"[Baseline Fold {i+1}] RMSE={rmse:.3f}, MAE={mae:.3f}")
                else:
                    print(f"[Baseline Fold {i+1}] Not enough data to calculate metrics.")
            except Exception as e:
                print(f"[Baseline Fold {i+1}] Error during model training/prediction: {e}")
                continue
                
        return metrics_rmse, metrics_mae