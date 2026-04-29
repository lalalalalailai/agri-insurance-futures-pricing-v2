import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.service_layer.validation_service import ValidationService


def render():
    st.markdown('<div class="main-title">🔬 五重因果验证</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#f0f7f2;padding:12px;border-radius:8px;margin-bottom:16px;font-size:14px;color:#1B6B3A'>
    采用五种互补的因果推断方法进行交叉验证：PSM倾向得分匹配、S-Learner、T-Learner、DML双重机器学习、IV工具变量法。
    若所有方法的因果效应符号一致且数值接近，则证明因果效应估计稳健可靠。
    </div>
    """, unsafe_allow_html=True)

    service = ValidationService()
    variety_list = service.loader.futures.get_variety_list()
    selected = st.selectbox("选择品种", variety_list, format_func=lambda x: x["label"])

    if st.button("🔍 运行五重验证", use_container_width=True):
        with st.spinner("五重因果验证中..."):
            result = service.five_fold_causal_validation(selected["code"])

            st.markdown("### 验证结果")

            methods = ["PSM", "S-Learner", "T-Learner", "DML", "IV"]
            data = []
            for m in methods:
                if m in result:
                    row = {"方法": m}
                    if m == "PSM":
                        p_val = result[m].get("p_value", 1.0)
                        smd = result[m].get("smd", 1.0)
                        row["核心指标"] = f"P值={p_val:.3f}"
                        row["标准化均值差"] = f"{smd:.3f}"
                        row["ATE估计值"] = f"{result[m].get('ate', 0):.4f}"
                        if p_val < 0.01:
                            row["显著性"] = "✅ 1%水平显著"
                        elif p_val < 0.05:
                            row["显著性"] = "✅ 5%水平显著"
                        elif p_val < 0.1:
                            row["显著性"] = "⚠️ 10%水平显著"
                        else:
                            row["显著性"] = "❌ 不显著"
                    else:
                        ate = result[m].get("ate", 0)
                        row["核心指标"] = f"ATE={ate:.4f}"
                        row["标准化均值差"] = "-"
                        row["ATE估计值"] = f"{ate:.4f}"
                        row["显著性"] = "✅" if abs(ate) > 0.01 else "⚠️"
                    data.append(row)

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                consistency = result.get("consistency", 0)
                st.markdown("---")
                st.markdown("### 一致性评估")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("五方法一致性", f"{consistency:.2f}")
                with col2:
                    ates = [result[m].get("ate", 0) for m in methods if m in result]
                    same_sign = all(a >= 0 for a in ates) or all(a <= 0 for a in ates)
                    st.metric("符号一致性", "✅ 一致" if same_sign else "❌ 不一致")

                if consistency > 0.8:
                    st.success("✅ 五种方法因果效应估计高度一致，因果效应稳健可靠")
                elif consistency > 0.5:
                    st.warning("⚠️ 五种方法因果效应估计基本一致，但存在一定差异")
                else:
                    st.info("ℹ️ 五种方法因果效应估计差异较大，需进一步分析")

                fig = go.Figure()
                method_names = [d["方法"] for d in data]
                ate_values = [result[m].get("ate", 0) for m in methods if m in result]
                colors = ["#16C47A" if a >= 0 else "#F53F3F" for a in ate_values]
                fig.add_trace(go.Bar(x=method_names, y=ate_values, marker_color=colors,
                                     text=[f"{a:.4f}" for a in ate_values], textposition="auto"))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="五重因果验证 ATE 对比", yaxis_title="ATE (价格变化率%)",
                                  template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
