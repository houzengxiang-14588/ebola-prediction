"""数据采集模块 — 从公开数据源拉取历史疫情时序数据。

支持的数据源：
- OWID (Our World in Data): COVID-19 全球数据
- 自定义 CSV: 用户本地数据文件

数据格式约定（统一输出）：
    date, confirmed, deaths, recovered, [region]
    2020-01-22, 555, 17, 28, Hubei
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def list_sources() -> list[dict]:
    """列出可用数据源及其描述。"""
    return [
        {
            "name": "owid-covid",
            "description": "OWID COVID-19 全球时序数据",
            "url": "https://github.com/owid/covid-19-data",
        },
        {
            "name": "csv",
            "description": "用户本地 CSV 文件",
        },
    ]


def fetch_owid_covid(cache_dir: str = "data/raw") -> pd.DataFrame:
    """从 OWID GitHub 仓库拉取 COVID-19 全球数据。"""
    url = (
        "https://raw.githubusercontent.com/owid/covid-19-data/master/"
        "public/data/owid-covid-data.csv"
    )
    cache_path = Path(cache_dir) / "owid_covid.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        df = pd.read_csv(url, parse_dates=["date"])
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def fetch_data(
    source: str = "owid-covid",
    region: str | None = None,
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """统一入口：按数据源名称拉取数据，返回标准化 DataFrame。

    返回列: date, confirmed, deaths, recovered, region
    """
    if source == "owid-covid":
        df = fetch_owid_covid(cache_dir)
        if region:
            df = df[df["location"] == region]
            df = df.rename(columns={"location": "region"})
        df = df.rename(columns={
            "total_cases": "confirmed",
            "total_deaths": "deaths",
        })
        if "recovered" not in df.columns:
            df["recovered"] = 0
        return df[["date", "confirmed", "deaths", "recovered"]]

    elif source == "csv":
        raise NotImplementedError("请提供 CSV 文件路径并在此处加载")

    raise ValueError(f"未知数据源: {source}")
