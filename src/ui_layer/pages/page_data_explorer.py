import streamlit as st
import pandas as pd
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.ui_layer.plotly_templates import time_series_chart, correlation_heatmap
from config import FUTURES_VARIETIES


def render():
    st.markdown('<div class="main-title">📊 数据探索</div>', unsafe_allow_html=True)

    loader = DataLoader()
    summary = loader.get_data_summary()

    st.markdown("### 数据总览")
    cols = st.columns(4)
    futures_total = sum(v["records"] for v in summary["futures"].values())
    weather_total = sum(v["records"] for v in summary["weather"].values())
    rs_total = sum(v["records"] for v in summary["remote_sensing"].values())

    with cols[0]:
        st.metric("期货数据", f"{len(summary['futures'])}品种", f"{futures_total:,}条")
    with cols[1]:
        st.metric("气象数据", f"{len(summary['weather'])}区域", f"{weather_total:,}条")
    with cols[2]:
        st.metric("遥感数据", f"{len(summary['remote_sensing'])}指标", f"{rs_total:,}条")
    with cols[3]:
        st.metric("总数据量", f"{futures_total + weather_total + rs_total:,}条")

    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    with col1:
        variety_list = loader.futures.get_variety_list()
        selected = st.selectbox("选择品种", variety_list,
                                format_func=lambda x: x["label"])
        variety_code = selected["code"]

    with col2:
        st.markdown("### 价格走势")
        try:
            df = loader.futures.load_variety(variety_code)
            fig = time_series_chart(
                df, "date", ["close", "open"],
                title=f"{selected['label']} 价格走势",
                labels={"close": "收盘价", "open": "开盘价"},
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"数据加载失败: {e}")

    st.markdown("### 特征相关性热力图")
    try:
        panel = loader.load_variety_panel(variety_code)
        preprocessor = Preprocessor()
        fe = FeatureEngineer()
        panel = preprocessor.preprocess_panel(panel)
        panel = fe.build_features(panel)

        feature_cols = fe.get_feature_columns()
        available = [c for c in feature_cols if c in panel.columns]
        if len(available) > 2:
            corr = panel[available].corr()
            fig = correlation_heatmap(corr, title=f"{selected['label']} 特征相关性")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"相关性计算失败: {e}")

    st.markdown("### 数据统计")
    try:
        stats_data = []
        for code, info in summary["futures"].items():
            stats_data.append({
                "品种": f"{info['name']}({code})",
                "记录数": info["records"],
                "字段数": len(info["columns"]),
            })
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    except Exception:
        pass
