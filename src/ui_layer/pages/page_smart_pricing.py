import streamlit as st
from src.service_layer.pricing_service import PricingService
from src.ui_layer.plotly_templates import feature_importance_chart
from src.ui_layer.theme import info_card


def render():
    st.markdown('<div class="main-title">🧠 智能定价</div>', unsafe_allow_html=True)
    service = PricingService()

    col1, col2 = st.columns([1, 3])
    with col1:
        variety_list = service.loader.futures.get_variety_list()
        selected = st.selectbox("选择品种", variety_list, format_func=lambda x: x["label"])
        area = st.number_input("投保面积(亩)", min_value=1, value=500)
        risk_level = st.selectbox("风险等级", ["低", "中", "高"])
        run_btn = st.button("📋 生成报告", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("生成综合定价报告..."):
                result = service.single_pricing(selected["code"], area=area, risk_level=risk_level)
                if "error" not in result:
                    st.markdown("### 综合定价报告")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""<div class="info-card"><h4>基本信息</h4><p>品种: <b>{selected['label']}</b></p><p>投保面积: <b>{area}亩</b></p><p>风险等级: <b>{risk_level}</b></p></div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="info-card"><h4>定价结果</h4><p>基准价格: <b class="highlight">{result['base_price']:,.0f}元/吨</b></p><p>风险溢价: <b class="highlight">{result['risk_premium']:,.0f}元/吨</b></p><p>总保费: <b class="highlight-green">{result.get('total_premium',0):,.0f}元</b></p></div>""", unsafe_allow_html=True)

                    if result.get("key_factors"):
                        fig = feature_importance_chart(result["key_factors"], title="因果因子重要性")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"定价失败: {result.get('error', '')}")
