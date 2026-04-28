import streamlit as st
import time
from src.service_layer.report_service import ReportService
from config import FUTURES_VARIETIES


def render():
    st.markdown('<div class="main-title">🔄 一键复现</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-card"><h4>一键复现论文全部实验</h4><p>运行Agri-PC因果发现、ACML因果定价、CCP保形预测、五重因果验证、消融实验等全部实验。</p><p>使用真实数据，结果与论文结论印证。</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        variety_options = [{"code": k, "name": v, "label": f"{v}({k})"}
                          for k, v in FUTURES_VARIETIES.items()]
        selected = st.selectbox("选择品种", variety_options,
                                format_func=lambda x: x["label"])
        st.markdown("---")
        run_btn = st.button("🚀 一键复现", use_container_width=True, type="primary")

    with col2:
        if run_btn:
            service = ReportService()
            progress = st.progress(0)
            status = st.empty()

            def update_progress(pct, msg):
                progress.progress(pct)
                status.text(f"正在执行: {msg}")

            with st.spinner("实验运行中..."):
                results = service.run_full_experiment(
                    selected["code"], progress_callback=update_progress
                )

            progress.progress(1.0)
            status.text("✅ 全部实验完成!")

            st.markdown("### 实验结果摘要")

            di = results.get("data_info", {})
            st.markdown(f"**品种**: {di.get('variety', 'N/A')} | **样本数**: {di.get('n_samples', 0)} | **特征数**: {di.get('n_features', 0)}")

            c1, c2, c3 = st.columns(3)
            with c1:
                pc = results.get("agri_pc", {})
                st.metric("Agri-PC F1", f"{pc.get('f1_score', 0):.4f}")
            with c2:
                ccp = results.get("ccp", {})
                st.metric("CCP覆盖率", f"{ccp.get('avg_coverage', 0)*100:.1f}%")
            with c3:
                acml = results.get("acml", {})
                st.metric("Neyman正交性", acml.get('neyman_status', 'N/A'))

            text_report = service.generate_text_report(results)
            st.markdown("### 详细报告")
            st.text(text_report)

            try:
                json_report = service.generate_json_report(results)
                st.download_button(
                    "📥 下载JSON报告",
                    data=json_report,
                    file_name=f"experiment_{selected['code']}_report.json",
                    mime="application/json",
                )
            except Exception as e:
                st.warning(f"JSON报告生成异常: {e}")

            st.download_button(
                "📥 下载文本报告",
                data=text_report,
                file_name=f"experiment_{selected['code']}_report.txt",
                mime="text/plain",
            )
        else:
            st.info("请选择品种后点击「一键复现」开始实验")
