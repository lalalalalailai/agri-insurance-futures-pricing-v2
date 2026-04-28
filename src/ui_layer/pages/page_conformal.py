import streamlit as st
import numpy as np
from src.service_layer.prediction_service import PredictionService
from src.ui_layer.plotly_templates import prediction_interval_chart, coverage_chart
from config import FUTURES_VARIETIES


def render():
    st.markdown('<div class="main-title">📐 保形预测 (CCP)</div>', unsafe_allow_html=True)

    service = PredictionService()

    col1, col2 = st.columns([1, 4])
    with col1:
        variety_list = service.loader.futures.get_variety_list()
        selected = st.selectbox("选择品种", variety_list,
                                format_func=lambda x: x["label"])
        target_cov = st.slider("目标覆盖率", 0.80, 0.99, 0.90, 0.01)
        run_btn = st.button("📈 运行CCP预测", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("CCP保形预测中..."):
                result = service.predict_price(selected["code"])

                if "error" in result:
                    st.error(f"预测失败: {result['error']}")
                else:
                    st.markdown("### 预测区间")
                    st.metric("平均覆盖率", f"{result.get('avg_coverage', 0)*100:.1f}%")
                    st.metric("最终α值", f"{result.get('final_alpha', 0.1):.4f}")

                    try:
                        panel = service.loader.load_variety_panel(selected["code"])
                        dates = panel["date"].tail(30).values
                        actual = panel["close"].tail(30).values

                        point = np.array(result.get("point_predictions", []))
                        lower = np.array(result.get("lower_bound", []))
                        upper = np.array(result.get("upper_bound", []))

                        n_show = min(len(dates), len(point), 30)
                        if n_show > 0:
                            fig = prediction_interval_chart(
                                dates[:n_show],
                                actual[:n_show] if len(actual) >= n_show else None,
                                point[:n_show],
                                lower[:n_show],
                                upper[:n_show],
                                title=f"{selected['label']} 保形预测区间",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"图表渲染: {e}")

                    cov_hist = result.get("coverage_history", [])
                    alpha_hist = result.get("alpha_history", [])
                    if cov_hist:
                        fig = coverage_chart(cov_hist, alpha_hist, target=target_cov)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请选择品种后点击「运行CCP预测」")
