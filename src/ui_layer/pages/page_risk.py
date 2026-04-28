import streamlit as st
import plotly.graph_objects as go
import numpy as np
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.acml import ACML


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
                fit_result = acml.fit(panel, available)

                if fit_result.get("status") != "success":
                    st.error("模型训练失败")
                    return

                importance = acml.get_feature_importance()

                risk_factor_names = [
                    "extreme_temp_index", "extreme_precip_index",
                    "drought_index", "precipitation", "temperature",
                    "humidity", "wind_speed", "solar_radiation",
                    "ndvi", "yield_proxy", "surface_pressure",
                ]

                risk_factors = {}
                for k in risk_factor_names:
                    if k in importance and importance[k] > 0:
                        risk_factors[k] = importance[k]

                if not risk_factors:
                    all_factors = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])
                    risk_factors = all_factors

                if risk_factors:
                    sorted_factors = dict(sorted(risk_factors.items(), key=lambda x: x[1], reverse=True))
                    total_risk = sum(sorted_factors.values())
                    risk_level = "高" if total_risk > 0.3 else "中" if total_risk > 0.1 else "低"
                    st.metric("综合风险等级", risk_level)

                    names = list(sorted_factors.keys())
                    values = list(sorted_factors.values())
                    colors = ["#DC2626" if v > 0.1 else "#C2410C" if v > 0.05 else "#0D9488"
                              for v in values]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=names,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:.4f}" for v in values],
                        textposition="outside",
                        textfont=dict(color="#1E293B", size=11),
                    ))
                    fig.update_layout(
                        title="风险因子贡献",
                        height=max(350, len(names) * 30),
                        template="agri_green_light",
                        xaxis_title="贡献度",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 风险因子解读")
                    top_factor = names[0]
                    top_value = values[0]
                    if top_value > 0.1:
                        st.warning(f"⚠️ 主要风险因子: **{top_factor}** (贡献度: {top_value:.4f})，建议重点关注")
                    else:
                        st.success("✅ 各风险因子贡献度较低，该品种整体风险可控")
                else:
                    st.info("未检测到显著风险因子，该品种风险较低")
            except Exception as e:
                st.error(f"评估失败: {e}")
