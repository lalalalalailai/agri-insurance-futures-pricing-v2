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
        progress = st.progress(0, text="正在准备数据...")
        panel, available = service._prepare_data(selected["code"])

        progress.progress(5, text="训练完整ACML模型...")
        from src.model_layer.acml import ACML
        acml_full = ACML()
        full_result = acml_full.fit(panel, available)
        if full_result.get("status") != "success":
            st.error("完整模型训练失败")
            return
        full_tau = abs(full_result["tau_mean"])

        progress.progress(20, text="训练完整Agri-PC模型...")
        from src.model_layer.agri_pc import AgriPC
        agri_full = AgriPC()
        agri_result = agri_full.discover(panel, available)
        full_f1 = agri_result.get("quality", {}).get("f1_score", 0)

        progress.progress(35, text="训练完整CCP模型...")
        from src.model_layer.ccp import CCP
        ccp_full = CCP()
        ccp_result = ccp_full.fit(panel, available, acml_model=acml_full)
        full_coverage = ccp_result.get("avg_coverage", 0)

        progress.progress(45, text="消融A1-A3 (Agri-PC)...")
        engine = service.engine
        ablation_points = [
            ("A1_temporal_constraint", "A1:时序偏序约束", "agri_pc"),
            ("A2_agri_prior", "A2:农业周期先验", "agri_pc"),
            ("A3_delivery_constraint", "A3:交割约束", "agri_pc"),
            ("B1_double_orthogonalization", "B1:双重正交化", "acml"),
            ("B2_risk_regularization", "B2:农业风险正则项", "acml"),
            ("B3_temporal_cv", "B3:交叉拟合TS切分", "acml"),
            ("C1_causal_residual", "C1:因果残差替代", "ccp"),
            ("C2_adaptive_coverage", "C2:自适应覆盖", "ccp"),
            ("C3_distribution_shift", "C3:分布偏移鲁棒", "ccp"),
        ]

        results = {
            "full_model_tau": round(full_tau, 6),
            "full_f1_score": full_f1,
            "full_ccp_coverage": full_coverage,
            "ablations": {},
        }

        for idx, (key, name, algo_type) in enumerate(ablation_points):
            pct = 45 + int((idx + 1) / len(ablation_points) * 50)
            progress.progress(pct, text=f"消融 {name}...")
            try:
                if algo_type == "agri_pc":
                    ablated_f1 = engine._ablate_agri_pc(panel, available, key)
                    contribution = (full_f1 - ablated_f1) / (full_f1 + 1e-10) * 100
                    results["ablations"][key] = {
                        "name": name,
                        "metric": round(ablated_f1, 4),
                        "contribution_pct": round(max(0, contribution), 2),
                    }
                elif algo_type == "acml":
                    ablated_tau = engine._ablate_acml(panel, available, key)
                    contribution = (full_tau - ablated_tau) / (full_tau + 1e-10) * 100
                    results["ablations"][key] = {
                        "name": name,
                        "tau": round(ablated_tau, 6),
                        "contribution_pct": round(max(0, contribution), 2),
                    }
                elif algo_type == "ccp":
                    ablated_coverage = engine._ablate_ccp(panel, available, acml_full, key)
                    contribution = (full_coverage - ablated_coverage) / (full_coverage + 1e-10) * 100
                    results["ablations"][key] = {
                        "name": name,
                        "coverage": round(ablated_coverage, 4),
                        "contribution_pct": round(max(0, contribution), 2),
                    }
            except Exception:
                results["ablations"][key] = {
                    "name": name,
                    "contribution_pct": 0,
                }

        progress.progress(100, text="消融实验完成!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Agri-PC F1", f"{results.get('full_f1_score', 0):.4f}")
        c2.metric("ACML τ", f"{results.get('full_model_tau', 0):.6f}")
        c3.metric("CCP覆盖率", f"{results.get('full_ccp_coverage', 0)*100:.1f}%")

        ablations = results.get("ablations", {})
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
