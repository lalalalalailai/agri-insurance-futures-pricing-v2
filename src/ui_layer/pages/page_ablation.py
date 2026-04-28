import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

            c1, c2, c3 = st.columns(3)
            c1.metric("Agri-PC F1", f"{result.get('full_f1_score', 0):.4f}")
            c2.metric("ACML τ", f"{result.get('full_model_tau', 0):.6f}")
            c3.metric("CCP覆盖率", f"{result.get('full_ccp_coverage', 0)*100:.1f}%")

            ablations = result.get("ablations", {})
            if ablations:
                groups = {
                    "Agri-PC (因果发现)": ([], []),
                    "ACML (因果定价)": ([], []),
                    "CCP (保形预测)": ([], []),
                }
                group_colors = {
                    "Agri-PC (因果发现)": "#15803D",
                    "ACML (因果定价)": "#165DFF",
                    "CCP (保形预测)": "#0D9488",
                }
                for key, val in ablations.items():
                    name = val.get("name", key)
                    contrib = val.get("contribution_pct", 0)
                    if key.startswith("A"):
                        groups["Agri-PC (因果发现)"][0].append(name)
                        groups["Agri-PC (因果发现)"][1].append(contrib)
                    elif key.startswith("B"):
                        groups["ACML (因果定价)"][0].append(name)
                        groups["ACML (因果定价)"][1].append(contrib)
                    elif key.startswith("C"):
                        groups["CCP (保形预测)"][0].append(name)
                        groups["CCP (保形预测)"][1].append(contrib)

                fig = go.Figure()
                for group_name, (names, contribs) in groups.items():
                    if names:
                        fig.add_trace(go.Bar(
                            x=contribs, y=names, orientation="h",
                            name=group_name,
                            marker_color=group_colors[group_name],
                            text=[f"{c:.1f}%" for c in contribs],
                            textposition="outside",
                            textfont=dict(color="#1E293B", size=11),
                        ))
                fig.update_layout(
                    title="9大创新点贡献度（按算法分组）",
                    height=450, template="agri_green_light",
                    barmode="group",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 详细消融结果")
                data = []
                for key, val in ablations.items():
                    row = {"创新点": val.get("name", key), "贡献度%": val.get("contribution_pct", 0)}
                    if "metric" in val:
                        row["消融后F1"] = val["metric"]
                    if "tau" in val:
                        row["消融后τ"] = val["tau"]
                    if "coverage" in val:
                        row["消融后覆盖率"] = val["coverage"]
                    data.append(row)
                st.dataframe(pd.DataFrame(data), use_container_width=True)
