import streamlit as st
import psutil
import os
from src.data_layer.cache_manager import CacheManager


def render():
    st.markdown('<div class="main-title">⚙️ 系统管理</div>', unsafe_allow_html=True)

    st.markdown("### 系统运行状态")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("CPU使用率", f"{psutil.cpu_percent():.1f}%")
    with c2:
        st.metric("内存使用率", f"{psutil.virtual_memory().percent:.1f}%")
    with c3:
        disk = psutil.disk_usage("/")
        st.metric("磁盘使用率", f"{disk.percent:.1f}%")

    st.markdown("---")
    st.markdown("### 缓存管理")
    cache_stats = CacheManager.get_cache_stats()
    for name, stats in cache_stats.items():
        st.write(f"- {name}: {stats['files']}个文件, {stats['size_mb']}MB")

    if st.button("🗑️ 清除全部缓存"):
        CacheManager.clear_cache("all")
        st.success("缓存已清除")

    st.markdown("---")
    st.markdown("### AI工具使用声明")
    st.markdown("""<div class="info-card"><h4>AI工具使用记录 (与论文表5一致)</h4><table style="width:100%; border-collapse:collapse; color:#1E293B;"><tr style="border-bottom:1px solid #E2E8F0; background-color:#F8FAFC;"><th style="padding:8px; color:#0F172A; text-align:left;">工具</th><th style="padding:8px; color:#0F172A; text-align:left;">开发方</th><th style="padding:8px; color:#0F172A; text-align:left;">占比</th><th style="padding:8px; color:#0F172A; text-align:left;">主要用途</th></tr><tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px; color:#1E293B;"><b>Qwen3.6-Plus</b></td><td style="padding:8px; color:#475569;">阿里云(通义千问)</td><td style="padding:8px; color:#1E293B;">约42%</td><td style="padding:8px; color:#475569;">代码框架生成、算法逻辑梳理、技术文档编写</td></tr><tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px; color:#1E293B;"><b>GLM-5</b></td><td style="padding:8px; color:#475569;">智谱AI</td><td style="padding:8px; color:#1E293B;">约35%</td><td style="padding:8px; color:#475569;">数学推导辅助、定理证明审查、实验设计优化</td></tr><tr><td style="padding:8px; color:#1E293B;"><b>MiniMax M2.7</b></td><td style="padding:8px; color:#475569;">上海稀宇科技</td><td style="padding:8px; color:#1E293B;">约23%</td><td style="padding:8px; color:#475569;">UI交互设计、可视化优化、叙事逻辑组织</td></tr></table><p style="margin-top:8px; color:#64748B; font-size:13px;">声明：AI工具仅用于辅助编码、推导审查和文档润色，核心算法(Agri-PC/ACML/CCP)和六大定理证明均由团队独立设计完成。</p></div>""", unsafe_allow_html=True)
