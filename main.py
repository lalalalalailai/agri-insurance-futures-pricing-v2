import streamlit as st
from config import PAGES, FUTURES_VARIETIES
from src.ui_layer.theme import apply_theme

st.set_page_config(
    page_title="农险期货智能定价系统",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🌾 农险期货智能定价")
    st.markdown("---")
    page_names = [p[1] for p in PAGES]
    selected_page = st.radio("功能导航", page_names, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:13px; color:#CBD5E1;">
    基于因果推断的农险期货智能定价模型<br>
    Agri-PC / ACML / CCP 三大原创算法<br>
    77,297条真实多源数据(2020.01-2025.12)
    </div>
    """, unsafe_allow_html=True)

page_map = {p[1]: p[0] for p in PAGES}
page_key = page_map.get(selected_page, "01_home")

if page_key == "01_home":
    from src.ui_layer.pages.page_home import render
    render()
elif page_key == "02_data_explorer":
    from src.ui_layer.pages.page_data_explorer import render
    render()
elif page_key == "03_causal_analysis":
    from src.ui_layer.pages.page_causal_analysis import render
    render()
elif page_key == "04_causal_pricing":
    from src.ui_layer.pages.page_causal_pricing import render
    render()
elif page_key == "05_conformal_prediction":
    from src.ui_layer.pages.page_conformal import render
    render()
elif page_key == "06_smart_pricing":
    from src.ui_layer.pages.page_smart_pricing import render
    render()
elif page_key == "07_risk_assessment":
    from src.ui_layer.pages.page_risk import render
    render()
elif page_key == "08_social_value":
    from src.ui_layer.pages.page_social import render
    render()
elif page_key == "09_five_validation":
    from src.ui_layer.pages.page_five_val import render
    render()
elif page_key == "10_benchmark":
    from src.ui_layer.pages.page_benchmark import render
    render()
elif page_key == "11_ablation":
    from src.ui_layer.pages.page_ablation import render
    render()
elif page_key == "12_pure_prediction":
    from src.ui_layer.pages.page_pure_pred import render
    render()
elif page_key == "13_sichuan":
    from src.ui_layer.pages.page_sichuan import render
    render()
elif page_key == "14_extreme_disaster":
    from src.ui_layer.pages.page_disaster import render
    render()
elif page_key == "15_one_click":
    from src.ui_layer.pages.page_one_click import render
    render()
