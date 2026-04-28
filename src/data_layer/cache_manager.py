import os
import pickle
import pandas as pd
import streamlit as st
from config import CACHE_DATA_DIR, CACHE_MODELS_DIR, CACHE_RESULTS_DIR, CACHE_TTL_DATA, CACHE_TTL_MODEL, CACHE_TTL_RESULT


class CacheManager:

    @staticmethod
    def save_data(key: str, data, subdir: str = ""):
        cache_dir = os.path.join(CACHE_DATA_DIR, subdir)
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, f"{key}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(key: str, subdir: str = ""):
        filepath = os.path.join(CACHE_DATA_DIR, subdir, f"{key}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None

    @staticmethod
    def save_model(key: str, model):
        os.makedirs(CACHE_MODELS_DIR, exist_ok=True)
        filepath = os.path.join(CACHE_MODELS_DIR, f"{key}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(key: str):
        filepath = os.path.join(CACHE_MODELS_DIR, f"{key}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None

    @staticmethod
    def save_result(key: str, result):
        os.makedirs(CACHE_RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(CACHE_RESULTS_DIR, f"{key}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_result(key: str):
        filepath = os.path.join(CACHE_RESULTS_DIR, f"{key}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None

    @staticmethod
    def clear_cache(cache_type: str = "all"):
        import shutil
        dirs = []
        if cache_type in ("all", "data"):
            dirs.append(CACHE_DATA_DIR)
        if cache_type in ("all", "models"):
            dirs.append(CACHE_MODELS_DIR)
        if cache_type in ("all", "results"):
            dirs.append(CACHE_RESULTS_DIR)
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)

    @staticmethod
    def get_cache_stats() -> dict:
        stats = {}
        for name, path in [("data", CACHE_DATA_DIR), ("models", CACHE_MODELS_DIR),
                           ("results", CACHE_RESULTS_DIR)]:
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith(".pkl")]
                total_size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
                stats[name] = {"files": len(files), "size_mb": round(total_size / 1024 / 1024, 2)}
            else:
                stats[name] = {"files": 0, "size_mb": 0}
        return stats
