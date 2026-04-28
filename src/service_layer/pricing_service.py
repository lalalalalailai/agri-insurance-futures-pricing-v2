import numpy as np
import pandas as pd
import streamlit as st
from config import CACHE_TTL_RESULT, FUTURES_VARIETIES
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.acml import ACML
from src.model_layer.ccp import CCP
from src.model_layer.baselines import TraditionalActuary, compute_mape
from src.data_layer.cache_manager import CacheManager


class PricingService:

    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()
        self.cache = CacheManager()

    @st.cache_resource
    def _get_acml_model(_self, variety_code: str) -> ACML:
        acml = ACML()
        panel = _self.loader.load_variety_panel(variety_code)
        panel = _self.preprocessor.preprocess_panel(panel)
        panel = _self.fe.build_features(panel)
        feature_cols = _self.fe.get_feature_columns()
        available = [c for c in feature_cols if c in panel.columns]
        acml.fit(panel, available)
        return acml

    def single_pricing(self, variety_code: str, date: str = None,
                       area: float = 1.0, risk_level: str = "中") -> dict:
        try:
            acml = self._get_acml_model(variety_code)
            panel = self.loader.load_variety_panel(variety_code)
            panel = self.preprocessor.preprocess_panel(panel)
            panel = self.fe.build_features(panel)

            if date:
                panel = panel[panel["date"] <= date].tail(30)
            else:
                panel = panel.tail(30)

            feature_cols = self.fe.get_feature_columns()
            available = [c for c in feature_cols if c in panel.columns]

            result = acml.predict_price(panel, available)

            risk_mult = {"低": 0.8, "中": 1.0, "高": 1.3}
            mult = risk_mult.get(risk_level, 1.0)
            result["risk_premium"] = round(result["risk_premium"] * mult, 2)
            result["total"] = round(result["base_price"] + result["risk_premium"], 2)
            result["total_premium"] = round(result["total"] * area, 2)

            trad = TraditionalActuary()
            trad.fit(panel)
            trad_result = trad.predict_premium(result["base_price"], area)
            result["traditional_rate"] = trad_result["premium_rate"]
            result["model_rate"] = round(
                result["risk_premium"] / (result["base_price"] + 1e-10), 4
            )
            if trad_result["premium_rate"] > 0:
                result["reduction_pct"] = round(
                    (1 - result["model_rate"] / trad_result["premium_rate"]) * 100, 2
                )
            else:
                result["reduction_pct"] = 0

            importance = acml.get_feature_importance()
            result["key_factors"] = dict(list(importance.items())[:5])

            return result
        except Exception as e:
            return {"error": str(e), "base_price": 0, "risk_premium": 0, "total": 0}

    def batch_pricing(self, variety_codes: list = None) -> pd.DataFrame:
        if variety_codes is None:
            variety_codes = list(FUTURES_VARIETIES.keys())

        results = []
        for code in variety_codes:
            try:
                r = self.single_pricing(code)
                results.append({
                    "品种代码": code,
                    "品种名称": FUTURES_VARIETIES.get(code, code),
                    "基准价格": r.get("base_price", 0),
                    "风险溢价": r.get("risk_premium", 0),
                    "总保费": r.get("total_premium", 0),
                    "模型费率": r.get("model_rate", 0),
                    "传统费率": r.get("traditional_rate", 0),
                    "降幅%": r.get("reduction_pct", 0),
                })
            except Exception:
                continue
        return pd.DataFrame(results)
