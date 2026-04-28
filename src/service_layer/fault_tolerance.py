import os
import time
import traceback
import streamlit as st
from config import CACHE_DATA_DIR


class FaultTolerance:

    @staticmethod
    def safe_load(fn, fallback_fn=None, message=""):
        try:
            return fn()
        except Exception as e:
            st.warning(f"⚠️ {message or '数据加载异常'}: {str(e)}")
            if fallback_fn:
                try:
                    return fallback_fn()
                except Exception:
                    st.error("降级方案也失败，请检查数据文件")
                    return None
            return None

    @staticmethod
    def safe_model_train(fn, timeout: float = 30.0, fallback_fn=None):
        try:
            start = time.time()
            result = fn()
            elapsed = time.time() - start
            if elapsed > timeout:
                st.warning(f"⚠️ 模型训练超时({elapsed:.1f}s)，切换轻量模型")
                if fallback_fn:
                    return fallback_fn()
            return result
        except Exception as e:
            st.warning(f"⚠️ 模型训练异常: {str(e)}")
            if fallback_fn:
                try:
                    return fallback_fn()
                except Exception:
                    return None
            return None

    @staticmethod
    def safe_visualize(fn, data=None):
        try:
            return fn()
        except Exception as e:
            st.warning(f"⚠️ 图表渲染失败: {str(e)}")
            if data is not None:
                st.json(data if isinstance(data, dict) else {"data": str(data)[:500]})
            return None

    @staticmethod
    def safe_operation(fn, error_msg: str = "操作失败"):
        try:
            return fn()
        except Exception as e:
            st.error(f"❌ {error_msg}: {str(e)}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())
            return None
