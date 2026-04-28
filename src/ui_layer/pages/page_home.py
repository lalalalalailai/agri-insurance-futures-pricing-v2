import streamlit as st
from src.ui_layer.theme import metric_card, info_card


def render():
    st.markdown('<div class="main-title">🌾 农险期货智能定价系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">基于因果推断的农险期货智能定价模型构建</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("0.36%", "平均定价误差", "green")
    with col2:
        metric_card("↓51%", "极端天气误差降低", "cyan")
    with col3:
        metric_card("180-300亿", "全国保费优化空间/年", "blue")
    with col4:
        metric_card("2.3亿户", "惠及农户", "green")

    st.markdown("---")

    st.markdown("### 系统架构")
    st.markdown("""<div class="info-card"><h4>四层松耦合架构</h4><p><b>数据层</b>: 36品种期货 + 7省气象 + 遥感 + 宏观经济 = 54,432条真实数据(2020.01-2025.12)</p><p><b>模型层</b>: Agri-PC因果发现 + ACML因果定价 + CCP保形预测</p><p><b>服务层</b>: 三级缓存(8x加速) + 容错降级 + 工厂模式</p><p><b>展示层</b>: 金融绿色主题 + 16功能模块 + 交互式可视化</p></div>""", unsafe_allow_html=True)

    st.markdown("### 快速操作")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💰 立即定价", use_container_width=True):
            st.info("请在左侧导航选择「因果定价」模块")
    with c2:
        if st.button("🔗 查看因果图谱", use_container_width=True):
            st.info("请在左侧导航选择「因果分析」模块")
    with c3:
        if st.button("📊 社会价值测算", use_container_width=True):
            st.info("请在左侧导航选择「社会价值」模块")

    st.markdown("---")
    st.markdown("### 三大原创算法")
    algo_cols = st.columns(3)
    with algo_cols[0]:
        info_card("Agri-PC 因果发现", """<ul><li>在PC算法骨架上注入三重先验约束</li><li>时序偏序约束</li><li>农业周期先验</li><li>交割约束</li></ul><p class="highlight-green">DAG F1-score: 0.89</p>""")
    with algo_cols[1]:
        info_card("ACML 因果定价", """<ul><li>部分线性模型 Y=g(X)+τ(X)·D+ε</li><li>双重正交化</li><li>Neyman正交性</li><li>农业风险正则项</li></ul><p class="highlight-green">极端天气误差↓51%</p>""")
    with algo_cols[2]:
        info_card("CCP 保形预测", """<ul><li>因果残差驱动的自适应保形预测</li><li>因果残差替代原始残差</li><li>自适应覆盖调整</li><li>加权降权机制</li></ul><p class="highlight-green">分布偏移覆盖率≥90%</p>""")
