import streamlit as st
import pandas as pd
from src.service_layer.validation_service import ValidationService


def render():
    st.markdown('<div class="main-title">🧪 消融实验</div>', unsafe_allow_html=True)

    service = ValidationService()
    variety_list = service.loader.futures.get_variety_list()
    selected = st.selectbox("选择品种", variety_list, format_func=lambda x: x["label"])

    if st.button("🔍 运行消融实验", use_container_width=True):
        with st.spinner("消融实验中..."):
            result = service.ablation_study(selected["code"])

            if result.get("status") == "full_model_failed":
                st.error("完整模型训练失败")
                return

            st.metric("完整模型τ", f"{result.get('full_model_tau', 0):.6f}")

            ablations = result.get("ablations", {})
            if ablations:
                data = []
                for key, val in ablations.items():
                    data.append({
                        "创新点": val.get("name", key),
                        "消融后τ": val.get("tau", 0),
                        "贡献度%": val.get("contribution_pct", 0),
                    })
                st.dataframe(pd.DataFrame(data), use_container_width=True)

                import plotly.graph_objects as go
                names = [v["name"] for v in ablations.values()]
                contributions = [v["contribution_pct"] for v in ablations.values()]
                fig = go.Figure(go.Bar(
                    x=contributions, y=names, orientation="h",
                    marker_color="#15803D",
                    text=[f"{c:.1f}%" for c in contributions],
                    textposition="outside",
                    textfont=dict(color="#1E293B", size=11),
                ))
                fig.update_layout(title="创新点贡献度", height=400, template="agri_green_light")
                st.plotly_chart(fig, use_container_width=True)
