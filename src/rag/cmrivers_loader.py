"""加载 cmrivers/ebola 数据到 RAG 知识库"""

import os, json
from glob import glob
import pandas as pd

CMRIVERS_DIR = "ebola_raw_data/ebola-master"


def load_country_timeseries(base_dir=CMRIVERS_DIR):
    path = os.path.join(base_dir, "country_timeseries.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    chunks = []
    for country, case_col, death_col in [
        ("Guinea", "Cases_Guinea", "Deaths_Guinea"),
        ("Liberia", "Cases_Liberia", "Deaths_Liberia"),
        ("Sierra Leone", "Cases_SierraLeone", "Deaths_SierraLeone"),
        ("Nigeria", "Cases_Nigeria", "Deaths_Nigeria"),
        ("Senegal", "Cases_Senegal", "Deaths_Senegal"),
        ("Mali", "Cases_Mali", "Deaths_Mali"),
    ]:
        sub = df[["Date", case_col, death_col]].dropna(subset=[case_col], how="all")
        if sub.empty:
            continue
        sub = sub.sort_values("Date")
        total = int(sub[case_col].sum())
        peak_row = sub.loc[sub[case_col].idxmax()]
        max_single = int(peak_row[case_col])
        deaths = int(sub[death_col].max()) if not sub[death_col].isna().all() else 0
        first = sub.iloc[0]["Date"].strftime("%Y-%m-%d")
        last = sub.iloc[-1]["Date"].strftime("%Y-%m-%d")
        peak_date = peak_row["Date"].strftime("%Y-%m-%d")
        chunks.append({
            "id": f"cmrivers_ts_{country.replace(' ', '_').lower()}",
            "text": (
                f"2014年西非埃博拉暴发期间，{country}逐日病例数据："
                f"报告期{first}至{last}，累计{total}例，"
                f"单日峰值{peak_date}达{max_single}例，累计死亡约{deaths}例。"
                f"数据来源：WHO每日情况报告(cmrivers/ebola)。"
            ),
            "metadata": {
                "type": "cmrivers_timeseries", "country": country,
                "first_date": first, "last_date": last,
                "peak_date": peak_date, "peak_cases": max_single,
                "total_cases": total, "total_deaths": deaths,
            },
        })
    return chunks


def load_line_list(base_dir=CMRIVERS_DIR):
    path = os.path.join(base_dir, "line_list.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    total = len(df)
    countries = df["country"].value_counts()
    desc = "、".join(f"{c}{n}例" for c, n in countries.head(8).items())
    return [{
        "id": "cmrivers_line_list",
        "text": (
            f"2014年西非埃博拉暴发行列表：共{total}例个案，"
            f"覆盖{len(countries)}个国家。主要分布：{desc}。"
        ),
        "metadata": {"type": "cmrivers_line_list", "total_cases": total, "num_countries": len(countries)},
    }]


def load_daily_reports(base_dir=CMRIVERS_DIR):
    """处理各国卫生部每日报告CSV（每个文件是一天的快照），按国家汇总。"""
    patterns = {
        "Guinea": "guinea_data/*.csv",
        "Liberia": "liberia_data/*.csv",
        "Sierra Leone": "sl_data/*.csv",
        "Mali": "mali_data/*.csv",
    }
    all_summaries = {}
    for country, pattern in patterns.items():
        files = sorted(glob(os.path.join(base_dir, pattern)))
        if not files:
            continue
        dates = []
        total_confirmed = []
        for fpath in files:
            try:
                df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
            except Exception:
                continue
            if "Date" not in df.columns or "Description" not in df.columns:
                continue
            # 找 confirmed 行
            confirmed_row = df[df["Description"].str.contains("confirmed", case=False, na=False)]
            if confirmed_row.empty:
                continue
            row = confirmed_row.iloc[0]
            date_val = str(row["Date"]).strip()
            if "Totals" in df.columns:
                try:
                    total_val = int(float(row["Totals"]))
                except (ValueError, TypeError):
                    continue
            else:
                continue
            dates.append(date_val)
            total_confirmed.append(total_val)

        if not dates:
            continue

        peak_idx = total_confirmed.index(max(total_confirmed))
        all_summaries[country] = {
            "first_date": dates[0], "last_date": dates[-1],
            "num_reports": len(dates),
            "peak_date": dates[peak_idx],
            "peak_confirmed": total_confirmed[peak_idx],
            "max_cumulative": max(total_confirmed),
        }

    chunks = []
    for country, s in all_summaries.items():
        chunks.append({
            "id": f"cmrivers_daily_{country.replace(' ', '_').lower()}",
            "text": (
                f"2014年埃博拉暴发期间，{country}卫生部每日报告："
                f"共{s['num_reports']}份日报，首份{s['first_date']}，末份{s['last_date']}。"
                f"累计确诊病例峰值{s['peak_date']}达{s['peak_confirmed']}例，"
                f"报告期内最高累计确诊{s['max_cumulative']}例。"
            ),
            "metadata": {
                "type": "cmrivers_daily_report", "country": country,
                "num_reports": s["num_reports"],
                "first_date": s["first_date"], "last_date": s["last_date"],
                "peak_date": s["peak_date"], "peak_confirmed": s["peak_confirmed"],
            },
        })
    return chunks


def build_cmrivers_chunks(base_dir=CMRIVERS_DIR):
    all_chunks = []
    ts = load_country_timeseries(base_dir); all_chunks.extend(ts); print(f"  时序: {len(ts)} 块")
    ll = load_line_list(base_dir); all_chunks.extend(ll); print(f"  行列表: {len(ll)} 块")
    dr = load_daily_reports(base_dir); all_chunks.extend(dr); print(f"  每日报告汇总: {len(dr)} 块")
    print(f"cmrivers 分块完成: {len(all_chunks)} 块")
    return all_chunks


if __name__ == "__main__":
    os.chdir("D:/Ebola")
    chunks = build_cmrivers_chunks()
    os.makedirs("data/knowledge", exist_ok=True)
    with open("data/knowledge/cmrivers_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"已保存 data/knowledge/cmrivers_chunks.json")
