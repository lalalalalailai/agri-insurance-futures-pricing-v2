import pandas as pd
import numpy as np
from scipy.stats import mstats


class Preprocessor:

    def align_time(self, df: pd.DataFrame, date_col: str = "date",
                   freq: str = "D") -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.resample(freq).asfreq()
        df = df.reset_index()
        return df

    def fill_missing(self, df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == "linear":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].ffill()
        df[numeric_cols] = df[numeric_cols].bfill()
        return df

    def winsorize(self, df: pd.DataFrame, cols: list = None,
                  lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
        df = df.copy()
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            if col in df.columns and df[col].notna().sum() > 10:
                lo = df[col].quantile(lower)
                hi = df[col].quantile(upper)
                df[col] = df[col].clip(lo, hi)
        return df

    def standardize(self, df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
        df = df.copy()
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            if col in df.columns and df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

    def preprocess_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.align_time(df)
        df = self.fill_missing(df)
        price_cols = ["close", "open", "high", "low"]
        volume_cols = ["volume", "hold"]
        df = self.winsorize(df, cols=price_cols + volume_cols)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2025-12-31")]
            df = df.reset_index(drop=True)
        return df
