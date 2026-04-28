import streamlit as st
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.validation_engine import ValidationEngine


class ValidationService:

    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()
        self.engine = ValidationEngine()

    def _prepare_data(self, variety_code: str):
        panel = self.loader.load_variety_panel(variety_code)
        panel = self.preprocessor.preprocess_panel(panel)
        panel = self.fe.build_features(panel)
        feature_cols = self.fe.get_feature_columns()
        available = [c for c in feature_cols if c in panel.columns]
        return panel, available

    def five_fold_causal_validation(self, variety_code: str) -> dict:
        panel, available = self._prepare_data(variety_code)
        return self.engine.five_fold_causal_validation(panel, available)

    def ablation_study(self, variety_code: str) -> dict:
        panel, available = self._prepare_data(variety_code)
        return self.engine.ablation_study(panel, available)

    def pure_prediction_test(self, variety_code: str) -> dict:
        panel, available = self._prepare_data(variety_code)
        return self.engine.pure_prediction_test(panel, available)

    def sichuan_validation(self) -> dict:
        data_dict = {}
        for code in ["LH0", "OI0", "AP0"]:
            try:
                panel, available = self._prepare_data(code)
                data_dict[code] = panel
            except Exception:
                continue
        return self.engine.sichuan_validation(data_dict)

    def extreme_disaster_test(self, variety_code: str = "M0") -> dict:
        panel, available = self._prepare_data(variety_code)
        return self.engine.extreme_disaster_test(panel, available)

    def social_value_calculation(self) -> dict:
        return self.engine.social_value_calculation()
