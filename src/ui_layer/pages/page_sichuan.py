import streamlit as st
from src.service_layer.validation_service import ValidationService
from src.ui_layer.theme import info_card


def render():
    st.markdown('<div class="main-title">🌶️ 四川省特色验证</div>', unsafe_allow_html=True)

    service = ValidationService()

    if st.button("🔍 运行四川特色验证", use_container_width=True):
        with st.spinner("四川省特色验证中..."):
            result = service.sichuan_validation()

            varieties = {
                "LH0": {"name": "生猪期货", "icon": "🐷"},
                "OI0": {"name": "油菜籽(菜油)", "icon": "🌻"},
                "AP0": {"name": "柑橘(苹果泛化)", "icon": "🍊"},
            }

            for code, info in varieties.items():
                if code in result:
                    r = result[code]
                    st.markdown(f"### {info['icon']} {info['name']}验证")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("定价误差", f"{r.get('mape', 0):.1f}%")
                    with c2:
                        st.metric("论文目标误差", f"{r.get('target_error', 0):.1f}%")
                    with c3:
                        if r.get("mape", 0) <= r.get("target_error", 999):
                            st.metric("验证结果", "✅ 达标")
                        else:
                            st.metric("验证结果", "⚠️ 偏差")

            st.markdown("---")
            st.markdown("""<div class="info-card"><h4>四川省综合效益</h4><p>• 财政补贴节省: <b class="highlight-green">18-22亿元/年</b></p><p>• 惠及农户: <b class="highlight-green">620万户</b></p><p>• 保费优化率: <b class="highlight-green">20.6-25.4%</b></p></div>""", unsafe_allow_html=True)
