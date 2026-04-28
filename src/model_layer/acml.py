import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from scipy import stats


class ACML:

    def __init__(self, n_splits: int = 5, risk_lambda: float = 0.1,
                 xgb_params: dict = None):
        self.n_splits = n_splits
        self.risk_lambda = risk_lambda
        self.xgb_params = xgb_params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
        }
        self.tau_model = None
        self.g_model = None
        self.m_model = None
        self.feature_names = None
        self.is_fitted = False

    def _create_treatment(self, data: pd.DataFrame, feature_cols: list) -> tuple:
        X = data[feature_cols].copy()

        if "extreme_precip_index" in X.columns and "extreme_temp_index" in X.columns:
            risk = X["extreme_precip_index"] + X["extreme_temp_index"]
            threshold = risk.median()
            D = (risk > threshold).astype(float)
        elif "drought_index" in X.columns:
            threshold = X["drought_index"].median()
            D = (X["drought_index"] > threshold).astype(float)
        else:
            D = pd.Series(0.0, index=X.index)

        Y = X["close"].copy() if "close" in X.columns else pd.Series(0.0, index=X.index)
        return X, Y, D

    def _risk_weights(self, X: pd.DataFrame) -> np.ndarray:
        risk_cols = [c for c in ["extreme_precip_index", "extreme_temp_index",
                                  "drought_index"] if c in X.columns]
        if not risk_cols:
            return np.ones(len(X))
        risk = X[risk_cols].sum(axis=1).values
        risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-10)
        return 1 + risk

    def fit(self, data: pd.DataFrame, feature_cols: list = None) -> dict:
        if feature_cols is None:
            feature_cols = [c for c in data.columns if c not in
                            ["date", "variety_code", "variety_name"] and
                            data[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        self.feature_names = feature_cols
        X, Y, D = self._create_treatment(data, feature_cols)

        valid_mask = Y.notna() & D.notna()
        for col in feature_cols:
            if col in X.columns:
                valid_mask &= X[col].notna()
        X, Y, D = X[valid_mask], Y[valid_mask], D[valid_mask]

        if len(X) < 50:
            return {"status": "insufficient_data", "n_samples": len(X)}

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        tau_preds = np.zeros(len(X))
        g_residuals = np.zeros(len(X))
        m_residuals = np.zeros(len(X))

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            D_train, D_test = D.iloc[train_idx], D.iloc[test_idx]

            g_model = XGBRegressor(**self.xgb_params)
            g_model.fit(X_train, Y_train)
            g_pred = g_model.predict(X_test)
            g_res = Y_test.values - g_pred

            m_model = XGBRegressor(**self.xgb_params)
            m_model.fit(X_train, D_train)
            m_pred = m_model.predict(X_test)
            m_res = D_test.values - m_pred

            g_residuals[test_idx] = g_res
            m_residuals[test_idx] = m_res

            if np.var(m_res) > 1e-10:
                tau_local = np.dot(g_res, m_res) / np.dot(m_res, m_res)
                tau_preds[test_idx] = tau_local
            else:
                tau_preds[test_idx] = 0.0

        self.g_model = XGBRegressor(**self.xgb_params)
        self.g_model.fit(X, Y)
        self.m_model = XGBRegressor(**self.xgb_params)
        self.m_model.fit(X, D)

        risk_weights = self._risk_weights(X)
        tau_mean = np.average(tau_preds, weights=risk_weights)
        risk_penalty = self.risk_lambda * np.mean(
            risk_weights * (tau_preds - tau_mean) ** 2
        )

        orth_residual = np.dot(g_residuals, m_residuals) / (
            np.dot(m_residuals, m_residuals) + 1e-10
        )

        self.tau_model = {
            "tau_mean": tau_mean,
            "tau_preds": tau_preds,
            "risk_penalty": risk_penalty,
            "orth_residual": orth_residual,
            "feature_names": feature_cols,
        }
        self.is_fitted = True

        return {
            "status": "success",
            "tau_mean": round(tau_mean, 6),
            "risk_penalty": round(risk_penalty, 6),
            "orth_residual": round(orth_residual, 6),
            "n_samples": len(X),
            "n_features": len(feature_cols),
        }

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        return np.full(len(X), self.tau_model["tau_mean"])

    def predict_price(self, data: pd.DataFrame, feature_cols: list = None) -> dict:
        if not self.is_fitted:
            return {"base_price": 0, "risk_premium": 0, "total": 0}

        if feature_cols is None:
            feature_cols = self.feature_names

        X = data[feature_cols].copy()
        for col in feature_cols:
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

        g_pred = self.g_model.predict(X)
        cate = self.predict_cate(X)
        D = np.ones(len(X))

        base_price = float(np.mean(g_pred))
        risk_premium = float(np.mean(np.abs(cate * D)))
        total = base_price + risk_premium

        return {
            "base_price": round(base_price, 2),
            "risk_premium": round(risk_premium, 2),
            "total": round(total, 2),
        }

    def get_feature_importance(self) -> dict:
        if not self.is_fitted or self.g_model is None:
            return {}
        imp = self.g_model.feature_importances_
        model_features = self.g_model.get_booster().feature_names
        if model_features is None:
            model_features = self.feature_names[:len(imp)]
        result = {}
        for i, name in enumerate(model_features):
            if i < len(imp) and imp[i] > 0:
                result[name] = round(float(imp[i]), 4)
        return result

    def neyman_orthogonality_test(self, data: pd.DataFrame, n_perturb: int = 100,
                                   eps: float = 0.01) -> dict:
        if not self.is_fitted:
            return {"status": "not_fitted"}

        X, Y, D = self._create_treatment(data, self.feature_names)
        valid_mask = Y.notna()
        X, Y, D = X[valid_mask], Y[valid_mask], D[valid_mask]

        g_base = self.g_model.predict(X)
        m_base = self.m_model.predict(X)

        original_tau = self.tau_model["tau_mean"]
        perturbed_taus = []

        for _ in range(n_perturb):
            g_perturbed = g_base + np.random.normal(0, eps * np.std(g_base), len(g_base))
            m_perturbed = m_base + np.random.normal(0, eps * np.std(m_base), len(m_base))

            g_res = Y.values - g_perturbed
            m_res = D.values - m_perturbed

            if np.var(m_res) > 1e-10:
                tau_p = np.dot(g_res, m_res) / np.dot(m_res, m_res)
                perturbed_taus.append(tau_p)

        if not perturbed_taus:
            return {"status": "failed", "sensitivity": float("inf")}

        perturbed_taus = np.array(perturbed_taus)
        sensitivity = np.std(perturbed_taus) / (abs(original_tau) + 1e-10)

        return {
            "status": "passed" if sensitivity < 1.0 else "warning",
            "original_tau": round(original_tau, 6),
            "perturbed_mean": round(np.mean(perturbed_taus), 6),
            "perturbed_std": round(np.std(perturbed_taus), 6),
            "sensitivity": round(sensitivity, 4),
        }
