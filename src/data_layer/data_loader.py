import os
import pandas as pd
import numpy as np
import streamlit as st
from config import (
    FUTURES_DIR, WEATHER_DIR, RS_DIR, DATA_DIR,
    FUTURES_VARIETIES, VARIETY_REGION_MAP, WEATHER_REGIONS,
    RS_PROVINCES, CACHE_DATA_DIR, CACHE_TTL_DATA,
    DATA_DATE_START, DATA_DATE_END,
)


class FuturesDataLoader:

    def load_variety(self, code: str) -> pd.DataFrame:
        name = FUTURES_VARIETIES.get(code, code)
        filepath = os.path.join(FUTURES_DIR, f"{code}_{name}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"期货数据文件不存在: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[(df["date"] >= DATA_DATE_START) & (df["date"] <= DATA_DATE_END)].copy()
        df["variety_code"] = code
        df["variety_name"] = name
        return df

    def load_all(self) -> dict:
        result = {}
        for code in FUTURES_VARIETIES:
            try:
                result[code] = self.load_variety(code)
            except FileNotFoundError:
                continue
        return result

    def get_variety_list(self) -> list:
        existing = []
        for code, name in FUTURES_VARIETIES.items():
            filepath = os.path.join(FUTURES_DIR, f"{code}_{name}.csv")
            if os.path.exists(filepath):
                existing.append({"code": code, "name": name, "label": f"{name}({code})"})
        return existing


class WeatherDataLoader:

    def load_region(self, region: str) -> pd.DataFrame:
        filepath = os.path.join(WEATHER_DIR, f"{region}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"气象数据文件不存在: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[(df["date"] >= DATA_DATE_START) & (df["date"] <= DATA_DATE_END)].copy()
        return df

    def load_all(self) -> dict:
        result = {}
        for region in WEATHER_REGIONS:
            try:
                result[region] = self.load_region(region)
            except FileNotFoundError:
                continue
        return result

    def load_for_variety(self, variety_code: str) -> pd.DataFrame:
        region = VARIETY_REGION_MAP.get(variety_code)
        if not region:
            return pd.DataFrame()
        return self.load_region(region)


class RemoteSensingLoader:

    def load_ndvi(self) -> pd.DataFrame:
        filepath = os.path.join(RS_DIR, "ndvi_monthly.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NDVI数据文件不存在: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df[(df["date"] >= DATA_DATE_START) & (df["date"] <= DATA_DATE_END)].copy()
        return df

    def load_evi(self) -> pd.DataFrame:
        filepath = os.path.join(RS_DIR, "evi_monthly.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"EVI数据文件不存在: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df[(df["date"] >= DATA_DATE_START) & (df["date"] <= DATA_DATE_END)].copy()
        return df

    def load_lst(self) -> pd.DataFrame:
        filepath = os.path.join(RS_DIR, "lst_monthly.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LST数据文件不存在: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df[(df["date"] >= DATA_DATE_START) & (df["date"] <= DATA_DATE_END)].copy()
        return df

    def load_drought(self) -> pd.DataFrame:
        filepath = os.path.join(RS_DIR, "drought_index_monthly.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"干旱指数数据文件不存在: {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df[(df["date"] >= DATA_DATE_START) & (df["date"] <= DATA_DATE_END)].copy()
        return df

    def load_all(self) -> dict:
        result = {}
        for name, fn in [("ndvi", self.load_ndvi), ("evi", self.load_evi),
                         ("lst", self.load_lst), ("drought", self.load_drought)]:
            try:
                result[name] = fn()
            except FileNotFoundError:
                continue
        return result

    def load_for_province(self, province: str) -> pd.DataFrame:
        dfs = {}
        for name, fn in [("ndvi", self.load_ndvi), ("evi", self.load_evi),
                         ("lst", self.load_lst), ("drought", self.load_drought)]:
            try:
                df = fn()
                df_prov = df[df["province"] == province].copy()
                if len(df_prov) > 0:
                    dfs[name] = df_prov
            except FileNotFoundError:
                continue
        if not dfs:
            return pd.DataFrame()
        merged = None
        for name, df in dfs.items():
            cols_to_keep = ["date"]
            value_cols = [c for c in df.columns if c not in
                          ["date", "province", "latitude", "year", "month"]]
            cols_to_keep.extend(value_cols)
            df_sub = df[cols_to_keep].copy()
            df_sub = df_sub.rename(columns={c: f"{name}_{c}" if c != "date" else c
                                            for c in df_sub.columns})
            if merged is None:
                merged = df_sub
            else:
                merged = merged.merge(df_sub, on="date", how="outer")
        return merged if merged is not None else pd.DataFrame()


class DataLoader:

    def __init__(self):
        self.futures = FuturesDataLoader()
        self.weather = WeatherDataLoader()
        self.rs = RemoteSensingLoader()

    @st.cache_data(ttl=CACHE_TTL_DATA, show_spinner="加载真实数据中...")
    def load_variety_panel(_self, variety_code: str) -> pd.DataFrame:
        futures_df = _self.futures.load_variety(variety_code)
        weather_df = _self.weather.load_for_variety(variety_code)
        region = VARIETY_REGION_MAP.get(variety_code, "")
        from config import REGION_PROVINCE_MAP
        province = REGION_PROVINCE_MAP.get(region, "")
        rs_df = _self.rs.load_for_province(province) if province else pd.DataFrame()

        panel = futures_df.copy()

        if len(weather_df) > 0:
            weather_daily = weather_df.copy()
            weather_daily = weather_daily.rename(columns={
                c: f"weather_{c}" for c in weather_daily.columns if c != "date"
            })
            panel = panel.merge(weather_daily, on="date", how="left")

        if len(rs_df) > 0:
            rs_df_copy = rs_df.copy()
            rs_df_copy["date"] = pd.to_datetime(rs_df_copy["date"])
            rs_df_copy["year"] = rs_df_copy["date"].dt.year
            rs_df_copy["month"] = rs_df_copy["date"].dt.month
            rs_monthly = rs_df_copy.groupby(["year", "month"]).first().reset_index()
            rs_monthly = rs_monthly.drop(columns=["date"], errors="ignore")

            panel["year"] = panel["date"].dt.year
            panel["month"] = panel["date"].dt.month
            panel = panel.merge(rs_monthly, on=["year", "month"], how="left")
            panel = panel.drop(columns=["year", "month"])

        panel = panel.sort_values("date").reset_index(drop=True)
        panel = panel[(panel["date"] >= DATA_DATE_START) & (panel["date"] <= DATA_DATE_END)].copy()
        return panel

    def get_data_summary(self) -> dict:
        summary = {"futures": {}, "weather": {}, "remote_sensing": {}}
        for code, name in FUTURES_VARIETIES.items():
            filepath = os.path.join(FUTURES_DIR, f"{code}_{name}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                summary["futures"][code] = {
                    "name": name, "records": len(df), "columns": list(df.columns),
                }
        for region in WEATHER_REGIONS:
            filepath = os.path.join(WEATHER_DIR, f"{region}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                summary["weather"][region] = {"records": len(df), "columns": list(df.columns)}
        for name in ["ndvi_monthly", "evi_monthly", "lst_monthly", "drought_index_monthly"]:
            filepath = os.path.join(RS_DIR, f"{name}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                summary["remote_sensing"][name] = {
                    "records": len(df), "columns": list(df.columns),
                }
        return summary
