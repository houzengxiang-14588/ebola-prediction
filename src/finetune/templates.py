"""埃博拉病毒预测 — 训练数据模板

3类模板覆盖不同预测任务：
1. trend_forecast: 给历史年份数据，预测后续年份趋势
2. country_analogy: 给相似国家数据，预测目标国家
3. seasonal_fill: 给部分月份数据，预测缺失月份
"""

# 模板1: 趋势外推 — 最核心的预测任务
TREND_FORECAST = """你是一位埃博拉病毒（Ebola）流行病学预测专家。根据以下国家/地区的历史疫情数据，预测未来的病例变化趋势。

{context_data}

请完成以下预测任务：
1. 预测 {target_period} 每年的病例数（整数）
2. 判断整体趋势方向（上升/下降/平稳）
3. 给出置信度（高/中/低）

请以JSON格式输出：
{{"predictions": [{{"year": YYYY, "cases": N}}], "trend": "上升/下降/平稳", "confidence": "高/中/低"}}"""

# 模板2: 国家类比 — 利用相似国家模式
COUNTRY_ANALOGY = """你是一位埃博拉病毒流行病学分析专家。已知以下参考国家/地区的疫情数据，请据此推断目标国家的可能情况。

参考国家数据：
{reference_data}

目标国家：{target_country}
目标国家已知数据：{target_context}

请分析目标国家与参考国家的相似性，并预测 {target_period} 的病例趋势。
注意：考虑发病率量级、季节性、多年周期等因素。

JSON输出：{{"predictions": [], "similar_to": "国家名", "reasoning": "简要理由"}}"""

# 模板3: 季节性填空 — 月份级别预测
SEASONAL_FILL = """你是一位埃博拉病毒暴发模式分析专家。埃博拉呈间歇性暴发模式，而非固定季节性。

已知以下国家在 {context_period} 的月度病例分布：
{monthly_context}

请预测 {target_period} 各月的病例数，注意埃博拉的暴发动力学特征。
注意：整年总病例数约 {yearly_estimate} 例。

JSON输出：{{"monthly_cases": [{{"month": M, "cases": N}}]}}"""

# 模板4: 多国交叉验证 — 利用地理邻近性
CROSS_VALIDATION = """你是一位传染病空间流行病学专家。根据非洲多国埃博拉监测数据的空间分布特征，进行交叉推断。

已知以下国家 {context_years} 年的发病率数据：
{multi_country_data}

请基于空间自相关性和地理邻近性，预测 {target_country} 在 {target_years} 年的病例情况。
埃博拉在邻近疫区常呈现相似的暴发传播模式。

JSON输出：{{"predictions": []}}"""

TEMPLATES = {
    "trend_forecast": TREND_FORECAST,
    "country_analogy": COUNTRY_ANALOGY,
    "seasonal_fill": SEASONAL_FILL,
    "cross_validation": CROSS_VALIDATION,
}
