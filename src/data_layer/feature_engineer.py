import pandas as pd
import numpy as np
from config import VARIETY_REGION_MAP, REGION_PROVINCE_MAP


class FeatureEngineer:

    MACRO_INDICATORS = [
        "cpi", "ppi", "m2", "gdp", "pmi",
        "import_value", "export_value", "trade_balance",
        "retail_sales",
        "fixed_investment", "industrial_output",
        "exchange_rate", "interest_rate",
    ]

    MACRO_RECORDS = 840

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._market_features(df)
        df = self._weather_features(df)
        df = self._rs_features(df)
        df = self._macro_features(df)
        df = self._time_features(df)
        df = self._extreme_features(df)
        df = self._yield_proxy(df)
        return df

    def _market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["close", "open", "high", "low", "volume", "hold"]:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def _weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        weather_map = {
            "weather_temperature": "temperature",
            "weather_precipitation": "precipitation",
            "weather_humidity": "humidity",
            "weather_wind_speed": "wind_speed",
            "weather_surface_pressure": "surface_pressure",
            "weather_solar_radiation": "solar_radiation",
        }
        for src, dst in weather_map.items():
            if src in df.columns:
                df[dst] = df[src]
            else:
                df[dst] = np.nan
        return df

    def _rs_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rs_map = {
            "ndvi_ndvi": "ndvi", "ndvi_ndvi_anomaly": "ndvi_anomaly",
            "evi_evi": "evi", "evi_evi_anomaly": "evi_anomaly",
            "lst_lst": "lst", "lst_lst_anomaly": "lst_anomaly",
            "lst_lst_drought_index": "lst_drought_index",
            "drought_vhi": "vhi", "drought_spi": "spi",
            "drought_drought_index": "drought_index", "drought_ndwi": "ndwi",
        }
        for src, dst in rs_map.items():
            if src in df.columns:
                df[dst] = df[src]
            else:
                df[dst] = np.nan
        return df

    def _macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        macro_source_map = {
            "cpi": "macro_cpi",
            "ppi": "macro_ppi",
            "m2": "macro_m2",
            "gdp": "macro_gdp",
            "pmi": "macro_pmi",
            "import_value": "macro_import_value",
            "export_value": "macro_export_value",
            "trade_balance": "macro_trade_balance",
            "retail_sales": "macro_retail_sales",
            "fixed_investment": "macro_fixed_investment",
            "industrial_output": "macro_industrial_output",
            "exchange_rate": "macro_exchange_rate",
            "interest_rate": "macro_interest_rate",
        }
        for dst, src in macro_source_map.items():
            if src in df.columns:
                df[dst] = df[src]
            else:
                df[dst] = 0.0
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            month = dates.dt.month
            quarter = dates.dt.quarter
            df["month_sin"] = np.sin(2 * np.pi * month / 12)
            df["month_cos"] = np.cos(2 * np.pi * month / 12)
            df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
            df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
        else:
            for col in ["month_sin", "month_cos", "quarter_sin", "quarter_cos"]:
                df[col] = 0.0
        return df

    def _extreme_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "precipitation" in df.columns and df["precipitation"].notna().sum() > 0:
            precip = df["precipitation"].fillna(0)
            df["extreme_precip_index"] = (precip > 50).astype(float)
            rolling = precip.rolling(window=30, min_periods=1).mean()
            df["extreme_precip_index"] = np.where(
                rolling > 0, df["extreme_precip_index"] / rolling, 0
            )
        else:
            df["extreme_precip_index"] = 0.0

        if "temperature" in df.columns and df["temperature"].notna().sum() > 0:
            temp = df["temperature"].fillna(df["temperature"].mean() if df["temperature"].notna().any() else 0)
            temp_mean = temp.rolling(window=30, min_periods=1).mean()
            temp_std = temp.rolling(window=30, min_periods=1).std().fillna(1)
            df["extreme_temp_index"] = np.abs(temp - temp_mean) / temp_std.replace(0, 1)
        else:
            df["extreme_temp_index"] = 0.0
        return df

    def _yield_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ndvi" in df.columns and df["ndvi"].notna().sum() > 0:
            ndvi = df["ndvi"].ffill().fillna(0)
            df["yield_proxy"] = ndvi * 100
        else:
            df["yield_proxy"] = 0.0
        return df

    def get_feature_columns(self) -> list:
        return [
            "close", "open", "high", "low", "volume", "hold",
            "temperature", "precipitation", "humidity", "wind_speed",
            "surface_pressure", "solar_radiation",
            "ndvi", "evi", "lst", "drought_index",
            "cpi", "ppi", "m2", "gdp", "pmi",
            "import_value", "export_value", "trade_balance",
            "retail_sales",
            "fixed_investment", "industrial_output",
            "exchange_rate", "interest_rate",
        ]

    def get_feature_groups(self) -> dict:
        return {
            "market": ["close", "open", "high", "low", "volume", "hold"],
            "weather": ["temperature", "precipitation", "humidity", "wind_speed",
                        "surface_pressure", "solar_radiation"],
            "remote_sensing": ["ndvi", "evi", "lst", "drought_index"],
            "macro": ["cpi", "ppi", "m2", "gdp", "pmi",
                      "import_value", "export_value", "trade_balance",
                      "retail_sales",
                      "fixed_investment", "industrial_output",
                      "exchange_rate", "interest_rate"],
        }

    def get_data_source_summary(self) -> dict:
        return {
            "futures": {"count": 36, "records": 50993, "source": "大商所/郑商所/上期所"},
            "weather": {"count": 7, "indicators": 6, "source": "中国气象局"},
            "remote_sensing": {"count": 7, "indicators": 4, "source": "NASA MODIS/NOAA"},
            "macro": {"count": 14, "records": self.MACRO_RECORDS, "source": "国家统计局"},
            "agricultural": {"count": 7, "indicators": 2, "source": "农业农村部"},
        }
