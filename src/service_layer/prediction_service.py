import numpy as np
import pandas as pd
import streamlit as st
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.ccp import CCP
from src.model_layer.acml import ACML


class PredictionService:

    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()

    @st.cache_resource
    def _get_ccp_model(_self, variety_code: str) -> CCP:
        panel = _self.loader.load_variety_panel(variety_code)
        panel = _self.preprocessor.preprocess_panel(panel)
        panel = _self.fe.build_features(panel)
        feature_cols = _self.fe.get_feature_columns()
        available = [c for c in feature_cols if c in panel.columns]

        acml = ACML()
        acml.fit(panel, available)

        ccp = CCP()
        ccp.fit(panel, available, acml_model=acml)
        return ccp

    def predict_price(self, variety_code: str, horizon: int = 30) -> dict:
        try:
            ccp = self._get_ccp_model(variety_code)
            panel = self.loader.load_variety_panel(variety_code)
            panel = self.preprocessor.preprocess_panel(panel)
            panel = self.fe.build_features(panel)

            feature_cols = self.fe.get_feature_columns()
            available = [c for c in feature_cols if c in panel.columns]
            X = panel[available].tail(horizon).fillna(0)

            interval = ccp.predict_interval(X)
            coverage = ccp.get_coverage_stats()

            return {
                "point_predictions": interval["point"].tolist() if len(interval["point"]) > 0 else [],
                "lower_bound": interval["lower"].tolist() if len(interval["lower"]) > 0 else [],
                "upper_bound": interval["upper"].tolist() if len(interval["upper"]) > 0 else [],
                "avg_coverage": coverage["avg_coverage"],
                "final_alpha": coverage["final_alpha"],
                "coverage_history": coverage["coverage_history"],
                "alpha_history": coverage["alpha_history"],
            }
        except Exception as e:
            return {"error": str(e)}

    def get_conformal_interval(self, variety_code: str,
                                alpha: float = 0.1) -> dict:
        try:
            ccp = self._get_ccp_model(variety_code)
            panel = self.loader.load_variety_panel(variety_code)
            panel = self.preprocessor.preprocess_panel(panel)
            panel = self.fe.build_features(panel)

            feature_cols = self.fe.get_feature_columns()
            available = [c for c in feature_cols if c in panel.columns]
            X = panel[available].tail(30).fillna(0)

            interval = ccp.predict_interval(X, alpha=alpha)
            return {
                "point": interval["point"].tolist(),
                "lower": interval["lower"].tolist(),
                "upper": interval["upper"].tolist(),
            }
        except Exception as e:
            return {"error": str(e)}
