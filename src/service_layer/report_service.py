import io
import json
import time
import numpy as np
import pandas as pd
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.agri_pc import AgriPC
from src.model_layer.acml import ACML
from src.model_layer.ccp import CCP
from src.model_layer.validation_engine import ValidationEngine
from src.model_layer.baselines import compute_mape


class ReportService:

    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.fe = FeatureEngineer()
        self.validation = ValidationEngine()

    def _prepare_variety(self, variety_code: str):
        panel = self.loader.load_variety_panel(variety_code)
        panel = self.preprocessor.preprocess_panel(panel)
        panel = self.fe.build_features(panel)
        feature_cols = self.fe.get_feature_columns()
        available = [c for c in feature_cols if c in panel.columns]
        return panel, available

    def run_full_experiment(self, variety_code: str = "M0",
                             progress_callback=None) -> dict:
        results = {}
        total_steps = 6

        if progress_callback:
            progress_callback(0, "加载数据...")

        panel, available = self._prepare_variety(variety_code)
        results["data_info"] = {
            "variety": variety_code,
            "n_samples": len(panel),
            "n_features": len(available),
            "date_range": f"{panel['date'].min()} ~ {panel['date'].max()}",
        }

        if progress_callback:
            progress_callback(1/total_steps, "Agri-PC因果发现...")

        agri_pc = AgriPC(alpha=0.05)
        pc_result = agri_pc.discover(panel, feature_names=available)
        results["agri_pc"] = {
            "f1_score": pc_result["quality"].get("f1_score", 0),
            "search_space_reduction": pc_result["quality"].get("search_space_reduction", 0),
            "n_nodes": pc_result["quality"].get("n_nodes", 0),
            "n_edges": pc_result["quality"].get("n_edges", 0),
            "causal_chains": pc_result.get("causal_chains", []),
        }

        if progress_callback:
            progress_callback(2/total_steps, "ACML因果定价...")

        acml = ACML()
        acml_result = acml.fit(panel, available)
        results["acml"] = {
            "tau_mean": acml_result.get("tau_mean", 0),
            "risk_penalty": acml_result.get("risk_penalty", 0),
            "orth_residual": acml_result.get("orth_residual", 0),
            "n_samples": acml_result.get("n_samples", 0),
        }

        price_result = acml.predict_price(panel, available)
        results["pricing"] = price_result

        neyman = acml.neyman_orthogonality_test(panel)
        results["acml"]["neyman_status"] = neyman.get("status", "unknown")
        results["acml"]["neyman_sensitivity"] = neyman.get("sensitivity", 0)

        if progress_callback:
            progress_callback(3/total_steps, "CCP保形预测...")

        ccp = CCP()
        ccp_result = ccp.fit(panel, available, acml_model=acml)
        results["ccp"] = {
            "avg_coverage": ccp_result.get("avg_coverage", 0),
            "final_alpha": ccp_result.get("final_alpha", 0),
            "n_windows": ccp_result.get("n_windows", 0),
            "coverage_history": ccp.coverage_history,
        }

        if progress_callback:
            progress_callback(4/total_steps, "五重因果验证...")

        five_fold = self.validation.five_fold_causal_validation(panel, available)
        results["five_fold"] = five_fold

        if progress_callback:
            progress_callback(5/total_steps, "消融实验...")

        ablation = self.validation.ablation_study(panel, available)
        results["ablation"] = ablation

        pure_pred = self.validation.pure_prediction_test(panel, available)
        results["pure_prediction"] = pure_pred

        if progress_callback:
            progress_callback(1.0, "实验完成!")

        return results

    def generate_json_report(self, results: dict) -> str:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            if isinstance(obj, bytes):
                return obj.decode("utf-8", errors="replace")
            try:
                return str(obj)
            except Exception:
                return f"<unserializable:{type(obj).__name__}>"

        seen = set()
        def remove_circular(obj):
            obj_id = id(obj)
            if obj_id in seen:
                return "<circular_ref>"
            if isinstance(obj, (dict, list)):
                seen.add(obj_id)
            if isinstance(obj, dict):
                return {k: remove_circular(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [remove_circular(v) for v in obj]
            return obj

        cleaned = remove_circular(results)
        return json.dumps(cleaned, default=convert, indent=2, ensure_ascii=False)

    def generate_text_report(self, results: dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("农险期货智能定价系统 — 实验报告")
        lines.append("=" * 60)

        di = results.get("data_info", {})
        lines.append(f"\n品种: {di.get('variety', 'N/A')}")
        lines.append(f"样本数: {di.get('n_samples', 0)}")
        lines.append(f"特征数: {di.get('n_features', 0)}")
        lines.append(f"时间范围: {di.get('date_range', 'N/A')}")

        lines.append("\n--- Agri-PC 因果发现 ---")
        pc = results.get("agri_pc", {})
        lines.append(f"DAG F1-score: {pc.get('f1_score', 0):.4f}")
        lines.append(f"搜索空间缩减: {pc.get('search_space_reduction', 0):.1f}%")
        lines.append(f"节点数: {pc.get('n_nodes', 0)}, 边数: {pc.get('n_edges', 0)}")
        chains = pc.get("causal_chains", [])
        for i, c in enumerate(chains[:3]):
            lines.append(f"  因果链{i+1}: {' → '.join(c)}")

        lines.append("\n--- ACML 因果定价 ---")
        acml = results.get("acml", {})
        lines.append(f"τ均值: {acml.get('tau_mean', 0):.6f}")
        lines.append(f"风险正则项: {acml.get('risk_penalty', 0):.6f}")
        lines.append(f"Neyman正交性: {acml.get('neyman_status', 'N/A')}")

        pricing = results.get("pricing", {})
        lines.append(f"基准价格: {pricing.get('base_price', 0):.2f}")
        lines.append(f"风险溢价: {pricing.get('risk_premium', 0):.2f}")

        lines.append("\n--- CCP 保形预测 ---")
        ccp = results.get("ccp", {})
        lines.append(f"平均覆盖率: {ccp.get('avg_coverage', 0)*100:.1f}%")
        lines.append(f"最终α: {ccp.get('final_alpha', 0):.4f}")

        lines.append("\n--- 五重因果验证 ---")
        ff = results.get("five_fold", {})
        for method in ["PSM", "S-Learner", "T-Learner", "DML", "IV"]:
            if method in ff:
                lines.append(f"  {method} ATE: {ff[method].get('ate', 0):.6f}")
        lines.append(f"  一致性: {ff.get('consistency', 0):.4f}")

        lines.append("\n--- 消融实验 ---")
        abl = results.get("ablation", {})
        lines.append(f"完整模型τ: {abl.get('full_model_tau', 0):.6f}")
        for key, val in abl.get("ablations", {}).items():
            lines.append(f"  {val.get('name', key)}: 贡献度{val.get('contribution_pct', 0):.1f}%")

        lines.append("\n" + "=" * 60)
        lines.append("报告生成完毕")
        return "\n".join(lines)
