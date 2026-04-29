import streamlit as st
import pandas as pd
import numpy as np
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.acml import ACML
from src.model_layer.validation_engine import ValidationEngine
from src.model_layer.baselines import (
    TLearner, XGBoostBaseline, TraditionalActuary, compute_mape, compute_rmse
)


def render():
    st.markdown('<div class="main-title">📊 基准对比</div>', unsafe_allow_html=True)

    loader = DataLoader()
    variety_list = loader.futures.get_variety_list()
    selected = st.selectbox("选择品种", variety_list, format_func=lambda x: x["label"])

    if st.button("🔍 运行基准对比", use_container_width=True):
        with st.spinner("基准对比中..."):
            try:
                panel = loader.load_variety_panel(selected["code"])
                preprocessor = Preprocessor()
                fe = FeatureEngineer()
                panel = preprocessor.preprocess_panel(panel)
                panel = fe.build_features(panel)

                feature_cols = fe.get_feature_columns()
                available = [c for c in feature_cols if c in panel.columns]

                n = len(panel)
                split = int(n * 0.8)
                train, test = panel.iloc[:split], panel.iloc[split:]

                Y_test = test["close"].values
                X_test = test[available].fillna(0)
                X_train = train[available].fillna(0)
                Y_train = train["close"].values

                acml = ACML()
                acml.fit(train, available)
                acml_pred = acml.predict_price(test, available)
                acml_y = np.full(len(Y_test), acml_pred["base_price"])

                xgb = XGBoostBaseline()
                xgb.fit(X_train, Y_train)
                xgb_y = xgb.predict(X_test)

                trad = TraditionalActuary()
                trad.fit(train)
                mean_price = float(train["close"].mean())
                trad_result = trad.predict_premium(mean_price)
                trad_rate = trad_result["premium_rate"]
                trad_y = Y_test * (1 + trad_rate)

                results = []
                results.append({"模型": "ACML(本模型)", "MAPE": round(compute_mape(Y_test, acml_y), 4),
                                "RMSE": round(compute_rmse(Y_test, acml_y), 2)})
                results.append({"模型": "XGBoost", "MAPE": round(compute_mape(Y_test, xgb_y), 4),
                                "RMSE": round(compute_rmse(Y_test, xgb_y), 2)})
                results.append({"模型": "传统精算", "MAPE": round(compute_mape(Y_test, trad_y), 4),
                                "RMSE": round(compute_rmse(Y_test, trad_y), 2)})

                st.markdown("### 性能对比")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

                st.markdown("### Diebold-Mariano检验")
                ve = ValidationEngine()
                errors_acml = Y_test - acml_y
                errors_xgb = Y_test - xgb_y
                dm_result = ve.diebold_mariano_test(errors_acml, errors_xgb)
                dm_data = {
                    "检验项": ["DM统计量", "p值", "显著性(α=0.05)"],
                    "结果": [
                        f"{dm_result['statistic']:.4f}",
                        f"{dm_result['p_value']:.6f}",
                        "✅ 显著" if dm_result["significant"] else "❌ 不显著",
                    ],
                }
                st.dataframe(pd.DataFrame(dm_data), use_container_width=True)
                if dm_result["significant"]:
                    st.success(f"ACML定价误差显著优于XGBoost基准 (DM={dm_result['statistic']:.4f}, p={dm_result['p_value']:.6f})")
                else:
                    st.info("ACML与XGBoost定价误差差异不显著")

            except Exception as e:
                st.error(f"对比失败: {e}")
