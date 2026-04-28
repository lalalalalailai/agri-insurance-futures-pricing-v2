import streamlit as st
from src.service_layer.validation_service import ValidationService
from src.ui_layer.theme import metric_card


def render():
    st.markdown('<div class="main-title">🌍 社会价值测算</div>', unsafe_allow_html=True)

    service = ValidationService()
    result = service.social_value_calculation()

    st.markdown("### 全国效益测算")
    national = result.get("national", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("保费优化空间", f"{national.get('premium_optimization_billion', 0)}亿元/年")
    with c2:
        st.metric("惠及农户", f"{national.get('farmers_benefited_hundred_million', 0)}亿户")
    with c3:
        st.metric("保费优化率", f"{national.get('optimization_rate', 0):.1f}%")

    st.markdown("---")
    st.markdown("### 四川省效益测算")
    sichuan = result.get("sichuan", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("财政补贴节省", f"{sichuan.get('fiscal_saving_billion', 0)}亿元/年")
    with c2:
        st.metric("惠及农户", f"{sichuan.get('farmers_benefited_million', 0)}万户")
    with c3:
        st.metric("保费优化率", f"{sichuan.get('optimization_rate', 0):.1f}%")

    st.markdown("---")
    st.markdown("""<div class="info-card"><h4>社会价值说明</h4><p>本模型通过因果推断精准定价，降低保费率15-25%，全国每年可优化保费180-300亿元，惠及2.3亿农户。四川省作为农业大省，可节省财政补贴18-22亿元。</p></div>""", unsafe_allow_html=True)
