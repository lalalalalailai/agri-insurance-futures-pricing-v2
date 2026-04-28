# 农险期货智能定价系统 — 系统规范文档 (spec.md)

> **项目名称**: 乡村振兴背景下基于因果推断的农险期货智能定价模型构建  
> **版本**: V1.0 | **日期**: 2026-04-27  
> **目标赛事**: 2026年全国大学生计算机设计大赛  
> **QL大脑引擎**: V211 (ASI Candidate 98.14 + V67 Professional Depth)

---

## 1. 项目概述

### 1.1 核心目标
基于论文《乡村振兴背景下基于因果推断的农险期货智能定价模型构建》，开发一套部署于Streamlit的完整系统，严格复现三大原创算法（Agri-PC、ACML、CCP），使用真实免费数据，界面采用金融绿色风格，通过自动化测试持续迭代直至与论文结论完全印证。

### 1.2 系统定位
- **学术复现系统**: 严格遵循论文算法原理与六大定理证明
- **交互式定价平台**: 提供一站式农险期货智能定价服务
- **实验验证工具**: 支持一键复现论文全部实验并生成报告

### 1.3 关键约束
1. 严格遵循论文算法原理与六大定理证明，不得简化或篡改核心逻辑
2. 所有数据须来自免费公开源（AKShare、mootdx、国家统计局、NASA POWER API）
3. 单品种定价响应时间<1秒，Agri-PC运行≤37.2秒，ACML训练≤12.5秒，CCP预测≤0.8秒
4. 缓存策略须提升数据加载速度8倍以上（st.cache_data/st.cache_resource）
5. 容错降级机制覆盖数据加载失败、模型超时、可视化渲染失败、并发过载四类场景
6. 纯预测验证模块须诚实报告lag特征伪精度与真实预测能力（MAPE 3-8%）
7. 四川省特色验证与河南暴雨极端场景须独立成模块
8. AI工具使用记录须与论文表5完全一致（Qwen3.6-Plus 42%/GLM-5 35%/MiniMax M2.7 23%）

---

## 2. 系统架构设计

### 2.1 四层松耦合架构

```
┌─────────────────────────────────────────────────────────────────┐
│  展示层 (Streamlit 1.30+)                                       │
│  16个功能模块 + 金融绿色主题CSS(600+行) + Plotly交互可视化       │
├─────────────────────────────────────────────────────────────────┤
│  服务层 (Python 3.11+)                                          │
│  缓存管理(st.cache_data/st.cache_resource) + 容错降级            │
│  工厂模式动态加载 + RESTful API接口 + 日志记录                    │
├─────────────────────────────────────────────────────────────────┤
│  模型层 (核心算法引擎)                                           │
│  Agri-PC因果发现 + ACML因果定价 + CCP保形预测                    │
│  基准模型(XGBoost/传统精算) + 实验验证引擎                        │
├─────────────────────────────────────────────────────────────────┤
│  数据层                                                         │
│  多源数据采集 + 预处理 + 特征工程(28维) + 缓存存储(Pickle)        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 项目目录结构

```
代码/
├── main.py                          # Streamlit入口
├── requirements.txt                 # 依赖清单
├── config.py                        # 全局配置
├── data/                            # 真实数据目录(已存在)
│   ├── futures/                     # 36品种期货日度数据
│   ├── weather/                     # 7省气象日度数据
│   ├── remote_sensing/              # 遥感月度数据
│   ├── macro/                       # 宏观经济数据
│   ├── insurance_pricing.csv        # 保险定价数据
│   ├── disaster_panel.csv           # 灾害面板数据
│   └── policy_panel.csv             # 政策面板数据
├── cache/                           # 缓存目录
│   ├── data/                        # 数据缓存(24h)
│   ├── models/                      # 模型缓存(7d)
│   └── results/                     # 结果缓存(1h)
├── src/
│   ├── __init__.py
│   ├── data_layer/                  # 数据层
│   │   ├── __init__.py
│   │   ├── data_loader.py           # 多源数据加载器
│   │   ├── preprocessor.py          # 数据预处理(对齐/缺失/异常)
│   │   ├── feature_engineer.py      # 28维特征工程
│   │   └── cache_manager.py         # 三级缓存管理
│   ├── model_layer/                 # 模型层
│   │   ├── __init__.py
│   │   ├── agri_pc.py               # Agri-PC因果发现算法
│   │   ├── acml.py                  # ACML因果元学习器
│   │   ├── ccp.py                   # CCP因果保形预测
│   │   ├── baselines.py             # 基准模型(XGBoost/ARIMA/传统精算)
│   │   └── validation_engine.py     # 实验验证引擎
│   ├── service_layer/               # 服务层
│   │   ├── __init__.py
│   │   ├── pricing_service.py       # 定价服务
│   │   ├── prediction_service.py    # 预测服务
│   │   ├── validation_service.py    # 验证服务
│   │   ├── report_service.py        # 报告生成服务
│   │   └── fault_tolerance.py       # 容错降级机制
│   └── ui_layer/                    # 展示层
│       ├── __init__.py
│       ├── theme.py                 # 金融绿色主题CSS
│       ├── components.py            # 通用UI组件
│       ├── plotly_templates.py      # Plotly图表模板
│       └── pages/                   # 16个功能模块页面
│           ├── 01_home.py           # 首页概览
│           ├── 02_data_explorer.py  # 数据探索
│           ├── 03_causal_analysis.py # 因果分析
│           ├── 04_causal_pricing.py # 因果定价
│           ├── 05_conformal_prediction.py # 保形预测
│           ├── 06_smart_pricing.py  # 智能定价
│           ├── 07_risk_assessment.py # 风险评估
│           ├── 08_social_value.py   # 社会价值
│           ├── 09_five_validation.py # 五重验证
│           ├── 10_benchmark.py      # 基准对比
│           ├── 11_ablation.py       # 消融实验
│           ├── 12_pure_prediction.py # 纯预测验证
│           ├── 13_sichuan.py        # 四川特色
│           ├── 14_extreme_disaster.py # 极端灾害
│           ├── 15_one_click.py      # 一键复现
│           └── 16_system_admin.py   # 系统管理
└── tests/                           # 测试目录
    ├── test_algorithms.py           # 算法一致性测试
    ├── test_performance.py          # 性能测试
    └── test_ui.py                   # UI功能测试
```

---

## 3. 数据层规范

### 3.1 数据源清单

| 数据类型 | 来源 | 文件数 | 记录数 | 字段数 | 时间范围 |
|---------|------|--------|--------|--------|---------|
| 期货数据 | AKShare(新浪财经) | 36 | 53,058 | 8 | 2020-2026 |
| 气象数据 | NASA POWER API | 7 | 16,016 | 12 | 2020-2026 |
| 遥感数据 | MODIS/Landsat | 4 | 2,016 | 4-7 | 2020-2025 |
| 宏观经济 | 国家统计局/央行 | 14 | 3,145 | 4-13 | 2020-2026 |
| 保险定价 | AKShare | 1 | 138 | 9 | — |
| 灾害面板 | 统计局 | 1 | 18 | 10 | — |
| 政策面板 | 统计局 | 1 | 18 | 10 | — |
| 极端天气 | NASA POWER | 3 | 6,864 | 12 | 2020-2026 |

### 3.2 期货数据字段

```csv
date,open,high,low,close,volume,hold,settle
```

### 3.3 气象数据字段

```csv
date,temperature,precipitation,humidity,wind_speed,surface_pressure,solar_radiation,region,latitude,longitude,data_source
```

### 3.4 遥感数据字段

- **NDVI**: ndvi, ndvi_anomaly, ndvi_lag5, ndvi_ma20
- **EVI**: evi, evi_anomaly
- **LST**: lst, lst_anomaly, lst_drought_index
- **干旱指数**: vhi, spi, drought_index, ndwi

### 3.5 特征工程规范 (28维)

| 维度类别 | 特征名 | 维度数 | 计算方式 |
|---------|--------|--------|---------|
| 市场风险 | 收盘价/开盘价/最高价/最低价/成交量/持仓量 | 6 | 直接提取 |
| 天气风险 | 温度/降水/湿度/风速/气压/辐射 | 6 | 按品种主产区匹配气象区域 |
| 遥感指数 | NDVI/EVI/LST/干旱指数 | 4 | 按品种主产区匹配省份数据 |
| 宏观风险 | CPI/PPI/M2/GDP/PMI | 5 | 月度数据向前填充至日度 |
| 时间特征 | 月度/季度/年度周期/趋势 | 4 | sin/cos编码+线性趋势 |
| 极端指标 | 极端温度指数/极端降水指数 | 2 | 日降水量>50mm天数标准化 |
| 产量代理 | 播种面积/产量估计 | 1 | 基于遥感指数推算 |

### 3.6 数据预处理流程

```
原始数据 → 时间对齐(日度统一) → 缺失插值(线性+前向填充) → 异常缩尾(1%/99%分位数) → 特征工程(28维) → 标准化(Z-score) → 缓存序列化(Pickle)
```

### 3.7 缓存策略

| 缓存类型 | 缓存内容 | 过期时间 | 装饰器 | 预期加速 |
|---------|---------|---------|--------|---------|
| 数据缓存 | 预处理后全量数据 | 24小时 | st.cache_data(ttl=86400) | 8x+ |
| 模型缓存 | 预训练36品种模型 | 7天 | st.cache_resource | 10x+ |
| 结果缓存 | 常用定价/实验结果 | 1小时 | st.cache_data(ttl=3600) | 5x+ |

---

## 4. 模型层规范

### 4.1 Agri-PC因果发现算法

**论文依据**: 5.1节 Agri-PC算法、6.1节因果发现实验

**核心原理**: 在PC算法骨架上注入三重先验约束

**三重约束机制**:
1. **时序偏序约束**: T(i)<T(j)时移除违例边（原因必须先于结果）
2. **农业周期先验**: 播种→产量→价格因果链（强制保留农业逻辑链）
3. **交割约束**: 现货→期货交割期边（期货到期收敛于现货）

**算法流程**:
```
1. 构建完全无向图(28节点)
2. 注入三重先验约束移除违例边
3. PC骨架: 基于条件独立性检验逐步删边(Fisher-Z检验, α=0.05)
4. 方向规则: V结构识别 + Meek规则传播方向
5. 输出Markov等价类DAG(CPDAG)
```

**目标指标**:
- DAG F1-score: 0.89
- 搜索空间缩减率: 21.9%
- 运行时间: ≤37.2秒

**输出格式**:
- NetworkX DiGraph对象
- DAG质量评估字典(F1-score, 搜索空间缩减率, 边数, 节点数)
- 因果链高亮路径列表

### 4.2 ACML因果元学习器

**论文依据**: 5.2节 ACML算法、6.2节因果定价实验

**核心原理**: 部分线性模型 Y = g(X) + τ(X)·D + ε

**四大创新组件**:
1. **双重正交化**: 结果模型残差与处理模型残差正交，消除混淆偏差
2. **Neyman正交性验证**: 确保估计量对 nuisance 参数的微小扰动不敏感
3. **农业风险正则项**: λ×E[Risk×(τ(X)-τ̄)²]，防止极端天气下过拟合
4. **时间序列K折交叉拟合**: 避免数据泄露，保持时序结构

**算法流程**:
```
1. 数据分割: 时间序列K折(K=5)
2. 第一阶段(处理模型): XGBoost拟合 D=f(X)+η, 得残差 D̃=D-f̂(X)
3. 第一阶段(结果模型): XGBoost拟合 Y=g(X)+ε, 得残差 Ỹ=Y-ĝ(X)
4. 第二阶段(CATE估计): Ỹ = τ(X)·D̃ + ν
5. 农业风险正则化: 添加λ×E[Risk×(τ(X)-τ̄)²]
6. Neyman正交性验证: 扰动nuisance参数检验稳定性
7. 输出CATE函数τ̂(X)
```

**目标指标**:
- 极端天气下误差率较T-Learner降低51%
- 训练时间: ≤12.5秒
- MAPE: 0.36%(全品种平均)

### 4.3 CCP因果保形预测

**论文依据**: 5.3节 CCP算法、7.2节Walk-Forward验证

**核心原理**: 以因果残差替代原始残差，设计自适应覆盖调整

**三大创新组件**:
1. **因果残差**: r_i = Y_i - τ̂(X_i)·D_i - ĝ(X_i)，替代传统保形预测的简单残差
2. **自适应覆盖调整**: α_{t+1} = α_t + γ(1 - 1{Y_t ∈ Ĉ_t})，动态追踪分布偏移
3. **加权降低历史分布偏移样本权重**: 对历史分布偏移样本降权，增强近期数据影响

**算法流程**:
```
1. 计算因果残差: r_i = Y_i - τ̂(X_i)·D_i - ĝ(X_i)
2. 计算非一致性分数: s_i = |r_i| / σ̂(X_i)
3. 排序非一致性分数
4. 计算分位数: q = ceil((1-α)(n+1))/n
5. 构建预测区间: [ŷ - q·σ̂(X), ŷ + q·σ̂(X)]
6. 自适应调整: α_{t+1} = α_t + γ(1 - 1{Y_t ∈ Ĉ_t})
7. 加权更新: 降低历史分布偏移样本权重
```

**目标指标**:
- 分布偏移下覆盖率≥90%
- 预测时间: ≤0.8秒
- 自适应收敛速度: 5步内达到目标覆盖率

### 4.4 基准模型

| 模型 | 用途 | 实现方式 |
|------|------|---------|
| XGBoost | 基准对比 | xgboost.XGBRegressor |
| ARIMA | 时间序列基准 | statsmodels.tsa.arima.ARIMA |
| 传统精算 | 保险定价基准 | 历史损失率×风险系数 |
| S-Learner | 因果推断基准 | 单模型CATE估计 |
| T-Learner | 因果推断基准 | 双模型CATE估计 |
| DML | 因果推断基准 | EconML LinearDML |
| IV | 因果推断基准 | 两阶段最小二乘 |

---

## 5. 服务层规范

### 5.1 定价服务 (PricingService)

```python
class PricingService:
    def single_pricing(variety, date, area, risk_level) -> PricingResult
    def batch_pricing(csv_file) -> BatchPricingResult
    def get_risk_premium(variety, risk_level) -> float
    def get_confidence_interval(variety, date, alpha) -> tuple
```

**PricingResult字段**:
- base_price: 基准价格(元/吨)
- risk_premium: 风险溢价(元/吨)
- total_premium: 总保费(元)
- confidence_interval: 90%置信区间
- key_factors: 关键影响因子排序

### 5.2 预测服务 (PredictionService)

```python
class PredictionService:
    def predict_price(variety, date, horizon) -> PredictionResult
    def get_conformal_interval(variety, date, alpha) -> ConformalResult
    def walk_forward_validation(variety, windows) -> WFResult
```

### 5.3 验证服务 (ValidationService)

```python
class ValidationService:
    def five_fold_causal_validation(variety) -> FiveFoldResult
    def ablation_study(variety) -> AblationResult
    def pure_prediction_test(variety) -> PurePredResult
    def sichuan_validation() -> SichuanResult
    def extreme_disaster_test() -> DisasterResult
```

### 5.4 报告服务 (ReportService)

```python
class ReportService:
    def generate_pdf_report(experiment_results) -> bytes
    def one_click_reproduction() -> ReproductionResult
    def export_charts(figure_list) -> zip_bytes
```

### 5.5 容错降级机制

| 故障场景 | 降级策略 | 用户提示 |
|---------|---------|---------|
| 数据加载失败 | 加载预存2020-2022样例数据 | "当前使用样例数据，部分功能受限" |
| 模型训练超时(>30s) | 切换预训练轻量XGBoost | "已切换至轻量模型，精度略有降低" |
| 可视化渲染失败 | 回退JSON文本输出+下载链接 | "图表渲染失败，已提供数据下载" |
| 并发过载 | 启用队列机制 | "当前请求较多，请稍候" |

---

## 6. 展示层规范

### 6.1 金融绿色主题配色

| 颜色类型 | 色值 | 使用场景 |
|---------|------|---------|
| 主色调 | #165DFF | 标题、按钮、关键数据高亮 |
| 辅助色1 | #36CFC9 | 成功状态、正向指标 |
| 辅助色2 | #FF7D00 | 警告状态、风险指标 |
| 辅助色3 | #F53F3F | 错误状态、高风险预警 |
| 农业绿底色 | #16C47A | 农业相关指标、背景色 |
| 页面背景 | #FFFFFF | 页面背景 |
| 分隔线 | #F2F3F5 | 分隔线、输入框背景 |
| 正文文字 | #4E5969 | 正文文字 |
| 标题文字 | #1D2129 | 标题文字 |

### 6.2 Plotly图表模板 (agri_green_light)

- 背景: #F7FBF9 (浅奶白)
- 网格线: #E2E8E5 (浅灰)
- 主色: #57B894 (清新农绿)
- 涨色: #57B894 (绿涨)
- 跌色: #E76F51 (红跌)
- 字体: Microsoft YaHei, Inter, Arial

### 6.3 16个功能模块规范

| 编号 | 模块名 | 核心功能 | 论文依据 | 关键指标 |
|------|--------|---------|---------|---------|
| 01 | 首页概览 | 核心指标卡片+架构图+快速操作 | 9.1/7.3 | 4指标卡片 |
| 02 | 数据探索 | 统计表格+时序图+热力图 | 2.1-2.3 | 5类数据统计 |
| 03 | 因果分析 | 交互式DAG+质量评估+因果链 | 5.1/6.1 | F1=0.89, 缩减21.9% |
| 04 | 因果定价 | 单品种/批量定价+对比图 | 5.2/7.3 | 响应<1s |
| 05 | 保形预测 | 预测区间+覆盖率+自适应α | 5.3/7.2 | 覆盖率≥90% |
| 06 | 智能定价 | 综合报告+风险等级+PDF导出 | 4.3/5.4 | 一站式 |
| 07 | 风险评估 | 3D热力图+饼图+高风险列表 | 3.1/6.2 | 5级风险 |
| 08 | 社会价值 | 全国/四川效益测算+对比图 | 7.3/7.4 | 180-300亿/年 |
| 09 | 五重验证 | PSM/S-L/T-L/DML/IV | 6.5 | 5方法一致性 |
| 10 | 基准对比 | 性能表格+误差对比+时序对比 | 6.2/表3 | 降低51% |
| 11 | 消融实验 | 9创新点消融+贡献度+协同效应 | 6.4/表4 | 9点消融 |
| 12 | 纯预测验证 | MAPE箱线图+Clark-West+特征重要性 | 6.3 | MAPE 3-8% |
| 13 | 四川特色 | 生猪/油菜籽/柑橘验证+效益 | 7.4 | 3品种验证 |
| 14 | 极端灾害 | 河南暴雨场景+性能对比+理赔响应 | 7.4 | 5天vs15天 |
| 15 | 一键复现 | 全实验异步执行+进度条+PDF报告 | 全部 | 5分钟 |
| 16 | 系统管理 | 运行监控+日志+AI工具声明 | 第八章 | AI声明一致 |

---

## 7. 性能指标规范

| 指标 | 目标值 | 测量方式 |
|------|--------|---------|
| 单品种定价响应 | <1秒 | 从请求到结果返回 |
| Agri-PC运行时间 | ≤37.2秒 | 28节点DAG构建完成 |
| ACML训练时间 | ≤12.5秒 | 单品种5折交叉拟合 |
| CCP预测时间 | ≤0.8秒 | 单品种预测区间生成 |
| 全量数据加载 | ≤30秒 | 54,432条数据加载+预处理 |
| 缓存命中加载 | ≤3秒 | 缓存命中后数据加载 |
| 一键复现总时间 | ≤5分钟 | 全部实验执行 |

---

## 8. 论文印证目标

### 8.1 六大定理验证

| 定理 | 内容 | 系统验证方式 |
|------|------|-------------|
| 定理1 | Agri-PC因果可识别性 | DAG F1-score ≥ 0.89 |
| 定理2 | ACML一致性 | CATE估计收敛性验证 |
| 定理3 | Neyman正交性 | 扰动nuisance参数稳定性 |
| 定理4 | CCP覆盖保证 | 分布偏移下覆盖率≥90% |
| 定理5 | 自适应收敛性 | α_t收敛速度验证 |
| 定理6 | 风险正则化界 | 极端天气误差降低≥51% |

### 8.2 五重因果验证目标

| 方法 | ATE估计目标 | 置信区间 |
|------|-----------|---------|
| PSM | 与论文一致 | 与论文一致 |
| S-Learner | 与论文一致 | 与论文一致 |
| T-Learner | 与论文一致 | 与论文一致 |
| DML | 与论文一致 | 与论文一致 |
| IV | 与论文一致 | 与论文一致 |

### 8.3 消融实验目标 (9创新点)

1. 时序偏序约束移除
2. 农业周期先验移除
3. 交割约束移除
4. 双重正交化移除
5. Neyman正交性移除
6. 农业风险正则项移除
7. 时间序列交叉拟合移除
8. 因果残差替代原始残差移除
9. 自适应覆盖调整移除

### 8.4 数值偏差容忍度

- 算法一致性测试: 与论文表2-4数值偏差 ≤ ±2%
- 超限自动回溯算法实现，参数调优/逻辑重构

---

## 9. 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 主语言 |
| Streamlit | 1.30+ | 前端框架 |
| Pandas | 2.0+ | 数据处理 |
| NumPy | 1.26+ | 矩阵运算 |
| Plotly | 5.18+ | 交互可视化 |
| NetworkX | 3.1+ | 图算法(DAG) |
| XGBoost | 2.0+ | 机器学习 |
| EconML | 0.14+ | 因果推断 |
| DoWhy | 0.11+ | 因果推断 |
| SciPy | 1.13+ | 统计分析 |
| Statsmodels | 0.14+ | 时间序列 |
| Matplotlib | 3.8+ | 静态图表 |
| AKShare | 1.10+ | 数据采集 |
| mootdx | 0.9+ | 期货数据 |
| reportlab | 4.0+ | PDF生成 |
