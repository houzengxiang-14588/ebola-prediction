"""知识加载与分块 — 处理WHO埃博拉疫情报告文本、结构化数据、专家规则"""

import os, json, re


def load_pdf_texts(knowledge_dir="data/knowledge/extracted"):
    """加载所有已提取的PDF文本，按报告年份分块。"""
    docs = []
    if not os.path.exists(knowledge_dir):
        return docs
    for fname in sorted(os.listdir(knowledge_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(knowledge_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        year_match = re.search(r"(20\d{2})", fname)
        year = int(year_match.group(1)) if year_match else 0

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        docs.append({"source": fname.replace(".txt", ""), "year": year, "text": text})

    return sorted(docs, key=lambda d: d["year"], reverse=True)


def chunk_text(text, chunk_size=500, overlap=50):
    """将长文本按段落边界分块，支持中文。"""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
            while len(current) > chunk_size:
                split_at = current[:chunk_size].rfind(".")
                if split_at == -1:
                    split_at = chunk_size
                chunks.append(current[:split_at + 1].strip())
                current = current[split_at + 1:]

    if current.strip():
        chunks.append(current.strip())

    overlapped = []
    for i, chunk in enumerate(chunks):
        prev_tail = chunks[i - 1][-overlap:] if i > 0 else ""
        next_head = chunks[i + 1][:overlap] if i < len(chunks) - 1 else ""
        overlapped.append(prev_tail + chunk + next_head)

    return overlapped


def load_country_profiles(csv_path="data/ebola_africa.csv"):
    """加载非洲国家埃博拉概况，转为自然语言描述。"""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    profiles = []

    for _, row in df.iterrows():
        name = str(row["country"])
        if name.startswith("EU_"):
            continue

        cases = {y: int(row[f"cases_{y}"]) for y in range(2019, 2024)}
        rates = {y: float(row[f"rate_{y}"]) for y in range(2019, 2024)}
        total_5y = sum(cases.values())

        if total_5y > 1000:
            level = "历史重灾区"
        elif total_5y > 10:
            level = "近期疫区"
        elif total_5y > 0:
            level = "散发疫区"
        else:
            level = "风险监测国"

        text = (
            f"{name}是埃博拉病毒病{level}。"
            f"2019-2023年病例数分别为: "
            + ", ".join(f"{y}年{cases[y]}例" for y in range(2019, 2024))
            + f"。五年累计{total_5y}例。"
        )

        profiles.append({
            "country": name,
            "level": level,
            "total_cases": total_5y,
            "text": text,
            "cases": cases,
            "rates": rates,
        })

    return profiles


def load_expert_rules():
    """埃博拉病毒病流行病学专家规则。"""
    rules = [
        {
            "category": "传播途径",
            "content": (
                "埃博拉病毒通过直接接触感染者的血液、分泌物、器官或其他体液传播。"
                "动物-人传播(溢出事件)：接触感染动物(果蝠、黑猩猩、大猩猩)体液。"
                "人-人传播：直接接触患者体液(血液、唾液、汗液、呕吐物、粪便、尿液、精液)，"
                "通过破损皮肤或黏膜进入体内。葬礼习俗中的尸体接触是重要传播环节。"
                "康复后男性精液可携带病毒长达7周以上。不通过空气传播。"
                "医护人员和密切接触者是高风险人群。"
            ),
        },
        {
            "category": "暴发模式",
            "content": (
                "埃博拉呈间歇性暴发模式，始于单一溢出事件(index case)后在社区人传人扩散。"
                "暴发规模从几例到数万例不等，取决于响应速度和卫生基础设施。"
                "2014-2016年西非大流行是史上最大规模(28646例/11323死亡)。"
                "2018-2020年刚果(金)北基伍省暴发为第二大规模(3481例/2299死亡)。"
                "2022-2023年乌干达暴发由苏丹型引起(164例/77死亡)。"
                "小规模暴发(<20例)在刚果(金)频繁发生，通常能快速控制。"
                "暴发的关键特征：病例呈聚集性增长，而非全年持续散发。"
            ),
        },
        {
            "category": "病毒学特征",
            "content": (
                "埃博拉病毒属(Ebolavirus)属丝状病毒科。6种型别中扎伊尔型致病性最强。"
                "病死率(CFR)：扎伊尔型可达90%(平均约50%)，苏丹型约50%。"
                "潜伏期2-21天(最常见5-9天)，潜伏期后突发症状。"
                "早期症状：发热、剧烈头痛、极度乏力、肌肉疼痛、咽痛。"
                "后期症状：呕吐、腹泻、皮疹、肝肾功能损伤、内/外出血。"
                "rVSV-ZEBOV疫苗对扎伊尔型有效，已获WHO预认证。"
            ),
        },
        {
            "category": "地理分布",
            "content": (
                "埃博拉主要流行于中非和西非热带雨林地区。"
                "刚果民主共和国(DRC)是全球暴发次数最多的国家，自1976年以来超过14次。"
                "乌干达暴发7次以上，加蓬、刚果共和国、苏丹、几内亚、利比里亚、塞拉利昂均有暴发史。"
                "2014-2016年西非大流行扩散至几内亚、利比里亚、塞拉利昂首都城市。"
                "尼日利亚、马里、塞内加尔出现少量输入病例。"
            ),
        },
        {
            "category": "风险评估与预警",
            "content": (
                "单个疑似病例即触发紧急响应。关键预警信号：聚集性不明原因发热+出血症状、"
                "近期动物接触史或葬礼参加史的患者、边境地区不明原因死亡。"
                "暴发规模的决定因素：首例发现延迟、接触者追踪覆盖率、安全埋葬实施率、"
                "社区信任度与配合度、国际援助响应速度。"
                "病例倍增时间是衡量传播强度的核心指标。"
                "有效再生数Rt持续>1时疫情扩张，Rt<1时趋于结束。"
                "疫情宣告结束标准：末例后经两个最长潜伏期(42天)无新病例。"
                "埃博拉可在康复者免疫豁免部位持续存在，极少数可导致疫情复燃(flare-up)。"
            ),
        },
    ]
    return rules


def build_all_chunks(knowledge_dir="data/knowledge/extracted"):
    """构建全部知识分块，用于向量库存储。"""
    all_chunks = []

    # 1. PDF报告分块
    docs = load_pdf_texts(knowledge_dir)
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            if len(chunk) < 50:
                continue
            all_chunks.append({
                "id": f"pdf_{doc['year']}_{i:03d}",
                "text": chunk,
                "metadata": {
                    "type": "pdf_report",
                    "source": doc["source"],
                    "year": doc["year"],
                    "chunk_index": i,
                },
            })

    # 2. 国家概况分块
    profiles = load_country_profiles()
    for p in profiles:
        all_chunks.append({
            "id": f"country_{p['country']}",
            "text": p["text"],
            "metadata": {
                "type": "country_profile",
                "country": p["country"],
                "level": p["level"],
                "total_cases": p["total_cases"],
            },
        })

    # 3. 专家规则分块
    rules = load_expert_rules()
    for r in rules:
        all_chunks.append({
            "id": f"rule_{r['category']}",
            "text": r["content"],
            "metadata": {
                "type": "expert_rule",
                "category": r["category"],
            },
        })

    print(f"知识分块完成: {len(all_chunks)} 块")
    print(f"  PDF报告: {len([c for c in all_chunks if c['metadata']['type'] == 'pdf_report'])} 块")
    print(f"  国家概况: {len([c for c in all_chunks if c['metadata']['type'] == 'country_profile'])} 块")
    print(f"  专家规则: {len([c for c in all_chunks if c['metadata']['type'] == 'expert_rule'])} 块")

    return all_chunks


if __name__ == "__main__":
    os.chdir("D:/Ebola")
    chunks = build_all_chunks()
    with open("data/knowledge/all_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("已保存到 data/knowledge/all_chunks.json")
