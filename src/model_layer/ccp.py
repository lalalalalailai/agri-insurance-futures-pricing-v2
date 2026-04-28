import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


class CCP:

    def __init__(self, alpha: float = 0.1, gamma: float = 0.08,
                 n_windows: int = 8, xgb_params: dict = None):
        self.alpha = alpha
        self.gamma = gamma
        self.n_windows = n_windows
        self.xgb_params = xgb_params or {
            "n_estimators": 150, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "random_state": 42,
        }
        self.point_model = None
        self.sigma_model = None
        self.calibration_scores = None
        self.alpha_history = [alpha]
        self.coverage_history = []
        self.is_fitted = False
        self.feature_names = None

    def _compute_causal_residuals(self, Y, g_pred, tau_pred, D):
        return Y - tau_pred * D - g_pred

    def _conformal_quantile(self, scores, alpha):
        n = len(scores)
        if n == 0:
            return 1.96
        q_idx = int(np.ceil((1 - alpha) * (n + 1)))
        q_idx = min(q_idx, n)
        sorted_scores = np.sort(scores)
        return sorted_scores[q_idx - 1]

    def _adaptive_update(self, alpha_t, coverage_rate, target_coverage):
        error = target_coverage - coverage_rate
        alpha_t = alpha_t - self.gamma * error
        alpha_t = np.clip(alpha_t, 0.01, 0.3)
        return alpha_t

    def fit(self, data: pd.DataFrame, feature_cols: list = None,
            acml_model=None) -> dict:
        if feature_cols is None:
            feature_cols = [c for c in data.columns if c not in
                            ["date", "variety_code", "variety_name"] and
                            data[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        self.feature_names = feature_cols
        X = data[feature_cols].copy()
        Y = X["close"].copy() if "close" in X.columns else pd.Series(0.0, index=X.index)

        if "extreme_precip_index" in X.columns and "extreme_temp_index" in X.columns:
            risk = X["extreme_precip_index"] + X["extreme_temp_index"]
            D = (risk > risk.median()).astype(float)
        else:
            D = pd.Series(0.0, index=X.index)

        valid_mask = Y.notna()
        for col in feature_cols:
            if col in X.columns:
                valid_mask &= X[col].notna()
        X, Y, D = X[valid_mask], Y[valid_mask], D[valid_mask]

        if len(X) < 50:
            return {"status": "insufficient_data", "n_samples": len(X)}

        n = len(X)
        target_coverage = 1 - self.alpha

        cumulative_cal_scores = []
        all_covered = []
        alpha_t = self.alpha

        n_splits = min(self.n_windows, max(3, n // 60))
        window_size = n // (n_splits + 1)

        for w in range(n_splits):
            train_end = (w + 1) * window_size
            cal_start = train_end
            cal_end = min(cal_start + window_size // 2, n - window_size // 2)
            test_start = cal_end
            test_end = min(test_start + window_size // 2, n)

            if test_end <= test_start or cal_end <= cal_start or train_end < 20:
                continue

            X_train = X.iloc[:train_end]
            Y_train = Y.iloc[:train_end]
            X_cal = X.iloc[cal_start:cal_end]
            Y_cal = Y.iloc[cal_start:cal_end]
            X_test = X.iloc[test_start:test_end]
            Y_test = Y.iloc[test_start:test_end]

            point_model = XGBRegressor(**self.xgb_params)
            point_model.fit(X_train, Y_train)

            g_pred_cal = point_model.predict(X_cal)
            g_pred_test = point_model.predict(X_test)

            if acml_model is not None and acml_model.is_fitted:
                tau_pred_cal = acml_model.predict_cate(X_cal)
                tau_pred_test = acml_model.predict_cate(X_test)
            else:
                tau_pred_cal = np.zeros(len(X_cal))
                tau_pred_test = np.zeros(len(X_test))

            D_cal = D.iloc[cal_start:cal_end].values
            D_test = D.iloc[test_start:test_end].values

            train_pred = point_model.predict(X_train)
            train_res = np.abs(Y_train.values - train_pred)
            sigma_model = XGBRegressor(**{**self.xgb_params, "n_estimators": 100})
            sigma_model.fit(X_train, train_res)
            sigma_cal = sigma_model.predict(X_cal)
            sigma_test = sigma_model.predict(X_test)

            causal_res_cal = self._compute_causal_residuals(
                Y_cal.values, g_pred_cal, tau_pred_cal, D_cal
            )
            cal_scores = np.abs(causal_res_cal) / (sigma_cal + 1e-10)
            cumulative_cal_scores.extend(cal_scores.tolist())

            q = self._conformal_quantile(np.array(cumulative_cal_scores), alpha_t)

            cold_start_factor = 1.0 + max(0, 2 - w) * 0.3
            lower = g_pred_test + tau_pred_test * D_test - q * sigma_test * cold_start_factor
            upper = g_pred_test + tau_pred_test * D_test + q * sigma_test * cold_start_factor
            covered = (Y_test.values >= lower) & (Y_test.values <= upper)
            coverage_rate = covered.mean()
            all_covered.append(coverage_rate)

            alpha_t = self._adaptive_update(alpha_t, coverage_rate, target_coverage)
            self.alpha_history.append(alpha_t)
            self.coverage_history.append(coverage_rate)

        self.point_model = XGBRegressor(**self.xgb_params)
        self.point_model.fit(X, Y)
        train_pred = self.point_model.predict(X)
        train_res = np.abs(Y.values - train_pred)
        self.sigma_model = XGBRegressor(**{**self.xgb_params, "n_estimators": 100})
        self.sigma_model.fit(X, train_res)

        g_pred_all = self.point_model.predict(X)
        if acml_model is not None and acml_model.is_fitted:
            tau_pred_all = acml_model.predict_cate(X)
        else:
            tau_pred_all = np.zeros(len(X))
        D_all = D.values
        causal_res_all = self._compute_causal_residuals(Y.values, g_pred_all, tau_pred_all, D_all)
        sigma_all = self.sigma_model.predict(X)
        self.calibration_scores = np.abs(causal_res_all) / (sigma_all + 1e-10)

        self.is_fitted = True

        avg_coverage = np.mean(all_covered) if all_covered else 0
        return {
            "status": "success",
            "avg_coverage": round(avg_coverage, 4),
            "final_alpha": round(self.alpha_history[-1], 4),
            "n_windows": len(all_covered),
            "n_samples": len(X),
        }

    def predict_interval(self, X: pd.DataFrame, alpha: float = None) -> dict:
        if not self.is_fitted:
            return {"lower": np.array([]), "upper": np.array([]), "point": np.array([])}

        a = alpha or self.alpha
        point_pred = self.point_model.predict(X)
        sigma_pred = self.sigma_model.predict(X)

        if self.calibration_scores is not None and len(self.calibration_scores) > 0:
            q = self._conformal_quantile(self.calibration_scores, a)
        else:
            q = 1.645

        lower = point_pred - q * sigma_pred
        upper = point_pred + q * sigma_pred

        return {
            "point": point_pred,
            "lower": lower,
            "upper": upper,
            "sigma": sigma_pred,
        }

    def get_coverage_stats(self) -> dict:
        return {
            "coverage_history": self.coverage_history,
            "alpha_history": self.alpha_history,
            "avg_coverage": round(np.mean(self.coverage_history), 4) if self.coverage_history else 0,
            "final_alpha": round(self.alpha_history[-1], 4) if self.alpha_history else self.alpha,
        }
