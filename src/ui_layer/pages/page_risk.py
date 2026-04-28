import streamlit as st
import plotly.graph_objects as go
import numpy as np
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.acml import ACML
from config import FUTURES_VARIETIES, VARIETY_REGION_MAP


def render():
    st.markdown('<div class="main-title">⚠️ 风险评估</div>', unsafe_allow_html=True)

    loader = DataLoader()
    variety_list = loader.futures.get_variety_list()
    selected = st.selectbox("选择品种", variety_list, format_func=lambda x: x["label"])

    if st.button("🔍 评估风险", use_container_width=True):
        with st.spinner("风险评估中..."):
            try:
                panel = loader.load_variety_panel(selected["code"])
                preprocessor = Preprocessor()
                fe = FeatureEngineer()
                panel = preprocessor.preprocess_panel(panel)
                panel = fe.build_features(panel)

                feature_cols = fe.get_feature_columns()
                available = [c for c in feature_cols if c in panel.columns]

                acml = ACML()
                acml.fit(panel, available)

                importance = acml.get_feature_importance()
                risk_factor_names = [
                    "extreme_temp_index", "extreme_precip_index",
                    "drought_index", "precipitation", "temperature",
                    "humidity", "wind_speed", "solar_radiation",
                ]
                risk_factors = {k: abs(v) for k, v in importance.items()
                               if k in risk_factor_names and abs(v) > 1e-8}

                if risk_factors:
                    total_risk = sum(risk_factors.values())
                    risk_level = "高" if total_risk > 0.3 else "中" if total_risk > 0.1 else "低"
                    st.metric("综合风险等级", risk_level)

                    names = list(risk_factors.keys())
                    values = list(risk_factors.values())
                    colors = ["#DC2626" if v > 0.1 else "#C2410C" if v > 0.05 else "#0D9488"
                              for v in values]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=names,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in values],
                        textposition="outside",
                        textfont=dict(color="#1E293B", size=11),
                    ))
                    fig.update_layout(
                        title="风险因子贡献",
                        height=400,
                        template="agri_green_light",
                        xaxis_title="贡献度",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("未检测到显著风险因子，该品种风险较低")
            except Exception as e:
                st.error(f"评估失败: {e}")
