import streamlit as st
import pandas as pd
from src.service_layer.validation_service import ValidationService


def render():
    st.markdown('<div class="main-title">🔬 五重因果验证</div>', unsafe_allow_html=True)

    service = ValidationService()
    variety_list = service.loader.futures.get_variety_list()
    selected = st.selectbox("选择品种", variety_list, format_func=lambda x: x["label"])

    if st.button("🔍 运行五重验证", use_container_width=True):
        with st.spinner("五重因果验证中..."):
            result = service.five_fold_causal_validation(selected["code"])

            st.markdown("### 验证结果")
            methods = ["PSM", "S-Learner", "T-Learner", "DML", "IV"]
            data = []
            for m in methods:
                if m in result:
                    data.append({"方法": m, "ATE估计值": result[m].get("ate", 0)})

            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)

                consistency = result.get("consistency", 0)
                st.metric("五方法一致性", f"{consistency:.4f}")

                if consistency > 0.8:
                    st.success("✅ 五种方法ATE估计高度一致，因果效应稳健")
                elif consistency > 0.5:
                    st.warning("⚠️ 五种方法ATE估计基本一致，但存在一定差异")
                else:
                    st.info("ℹ️ 五种方法ATE估计差异较大，需进一步分析")
