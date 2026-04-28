import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor
from scipy import stats


class SLearner:

    def __init__(self, xgb_params: dict = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "random_state": 42,
        }
        self.model = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, Y: np.ndarray, D: np.ndarray):
        X_aug = X.copy()
        X_aug["treatment"] = D
        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(X_aug, Y)
        self.is_fitted = True

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        X1 = X.copy()
        X1["treatment"] = 1
        X0 = X.copy()
        X0["treatment"] = 0
        return self.model.predict(X1) - self.model.predict(X0)


class TLearner:

    def __init__(self, xgb_params: dict = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "random_state": 42,
        }
        self.model1 = None
        self.model0 = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, Y: np.ndarray, D: np.ndarray):
        mask1 = D == 1
        mask0 = D == 0
        self.model1 = XGBRegressor(**self.xgb_params)
        self.model0 = XGBRegressor(**self.xgb_params)
        if mask1.sum() > 10:
            self.model1.fit(X[mask1], Y[mask1])
        if mask0.sum() > 10:
            self.model0.fit(X[mask0], Y[mask0])
        self.is_fitted = True

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.model1.predict(X) - self.model0.predict(X)


class DMLBaseline:

    def __init__(self):
        self.ate = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, Y: np.ndarray, D: np.ndarray):
        tscv = TimeSeriesSplit(n_splits=3)
        tau_estimates = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]

            g_model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
            g_model.fit(X_train, Y_train)
            g_res = Y_test - g_model.predict(X_test)

            m_model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
            m_model.fit(X_train, D_train)
            m_res = D_test - m_model.predict(X_test)

            if np.var(m_res) > 1e-10:
                tau = np.dot(g_res, m_res) / np.dot(m_res, m_res)
                tau_estimates.append(tau)

        self.ate = np.mean(tau_estimates) if tau_estimates else 0
        self.is_fitted = True

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.ate) if self.is_fitted else np.zeros(len(X))


class IVBaseline:

    def __init__(self):
        self.ate = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, Y: np.ndarray, D: np.ndarray):
        if "drought_index" in X.columns:
            Z = X["drought_index"].values.reshape(-1, 1)
        elif "precipitation" in X.columns:
            Z = X["precipitation"].values.reshape(-1, 1)
        else:
            self.ate = 0
            self.is_fitted = True
            return

        stage1 = LinearRegression()
        stage1.fit(Z, D)
        D_hat = stage1.predict(Z)

        stage2 = LinearRegression()
        stage2.fit(D_hat.reshape(-1, 1), Y)
        self.ate = float(stage2.coef_[0])
        self.is_fitted = True

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.ate) if self.is_fitted else np.zeros(len(X))


class PSMBaseline:

    def __init__(self):
        self.ate = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, Y: np.ndarray, D: np.ndarray):
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, D)
        ps = ps_model.predict_proba(X)[:, 1]

        matched_effects = []
        treated_idx = np.where(D == 1)[0]
        control_idx = np.where(D == 0)[0]

        for t in treated_idx:
            distances = np.abs(ps[control_idx] - ps[t])
            nearest = control_idx[np.argmin(distances)]
            matched_effects.append(Y[t] - Y[nearest])

        self.ate = np.mean(matched_effects) if matched_effects else 0
        self.is_fitted = True

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.ate) if self.is_fitted else np.zeros(len(X))


class XGBoostBaseline:

    def __init__(self, xgb_params: dict = None):
        self.xgb_params = xgb_params or {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "random_state": 42,
        }
        self.model = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, Y: np.ndarray):
        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(X, Y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.model.predict(X)


class TraditionalActuary:

    def __init__(self, base_loss_rate: float = 0.045, risk_coeff: float = 1.5):
        self.base_loss_rate = base_loss_rate
        self.risk_coeff = risk_coeff
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        if "close" in data.columns:
            prices = data["close"].dropna()
            if len(prices) > 0:
                returns = prices.pct_change().dropna()
                vol = returns.std() if len(returns) > 0 else 0.2
                self.base_loss_rate = 0.03 + vol * self.risk_coeff
        self.is_fitted = True

    def predict_premium(self, price: float, area: float = 1.0) -> dict:
        rate = self.base_loss_rate
        premium = price * area * rate
        return {
            "premium_rate": round(rate, 4),
            "premium": round(premium, 2),
        }


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
