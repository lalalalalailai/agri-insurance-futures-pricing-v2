import streamlit as st
from src.service_layer.pricing_service import PricingService
from src.ui_layer.plotly_templates import bar_comparison, feature_importance_chart
from src.ui_layer.theme import info_card
from config import FUTURES_VARIETIES


def render():
    st.markdown('<div class="main-title">💰 因果定价 (ACML)</div>', unsafe_allow_html=True)

    service = PricingService()

    col1, col2 = st.columns([1, 2])
    with col1:
        variety_list = service.loader.futures.get_variety_list()
        selected = st.selectbox("选择品种", variety_list,
                                format_func=lambda x: x["label"])
        date = st.date_input("定价日期")
        area = st.number_input("投保面积(亩)", min_value=1, value=500)
        risk_level = st.selectbox("风险等级", ["低", "中", "高"])
        price_btn = st.button("💰 立即定价", use_container_width=True)

    with col2:
        if price_btn:
            with st.spinner("ACML因果定价中..."):
                result = service.single_pricing(
                    selected["code"], str(date), area, risk_level
                )

                if "error" in result:
                    st.error(f"定价失败: {result['error']}")
                else:
                    st.markdown("### 定价结果")
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.metric("基准价格", f"{result['base_price']:,.0f}元/吨")
                    with r2:
                        st.metric("风险溢价", f"{result['risk_premium']:,.0f}元/吨")
                    with r3:
                        st.metric("总保费", f"{result.get('total_premium', 0):,.0f}元")
                    with r4:
                        reduction = result.get("reduction_pct", 0)
                        st.metric("保费降幅", f"{reduction:.1f}%")

                    st.markdown("### 定价对比")
                    model_rate = result.get("model_rate", 0) * 100
                    trad_rate = result.get("traditional_rate", 0) * 100
                    fig = bar_comparison(
                        ["费率"],
                        [model_rate],
                        [trad_rate],
                        "本模型",
                        "传统精算",
                        title="保费率对比(%)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if result.get("key_factors"):
                        st.markdown("### 关键影响因子")
                        fig = feature_importance_chart(result["key_factors"])
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请选择品种和参数后点击「立即定价」")
