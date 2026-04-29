import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

FUTURES_DIR = os.path.join(DATA_DIR, "futures")
WEATHER_DIR = os.path.join(DATA_DIR, "weather")
RS_DIR = os.path.join(DATA_DIR, "remote_sensing")

CACHE_DATA_DIR = os.path.join(CACHE_DIR, "data")
CACHE_MODELS_DIR = os.path.join(CACHE_DIR, "models")
CACHE_RESULTS_DIR = os.path.join(CACHE_DIR, "results")

DATA_DATE_START = "2020-01-01"
DATA_DATE_END = "2025-12-31"
DATA_DATE_RANGE_DESC = "2020.01-2025.12"

COLORS = {
    "primary": "#165DFF",
    "secondary": "#36CFC9",
    "agri_green": "#16C47A",
    "warning": "#FF7D00",
    "danger": "#F53F3F",
    "bg": "#FFFFFF",
    "divider": "#F2F3F5",
    "text": "#4E5969",
    "title": "#1D2129",
    "plotly_bg": "#F7FBF9",
    "plotly_grid": "#E2E8E5",
    "plotly_green": "#57B894",
    "plotly_red": "#E76F51",
}

FUTURES_VARIETIES = {
    "A0": "豆一", "AP0": "苹果", "C0": "玉米", "CF0": "棉花",
    "CJ0": "红枣", "CS0": "玉米淀粉", "EB0": "苯乙烯", "EG0": "乙二醇",
    "FG0": "玻璃", "HC0": "热卷", "I0": "铁矿石", "J0": "焦炭",
    "JD0": "鸡蛋", "JM0": "焦煤", "L0": "塑料", "LH0": "生猪",
    "M0": "豆粕", "MA0": "甲醇", "OI0": "菜油", "P0": "棕榈油",
    "PG0": "LPG", "PK0": "花生", "PP0": "PP", "RM0": "菜粕",
    "RU0": "橡胶", "SA0": "纯碱", "SF0": "硅铁", "SM0": "锰硅",
    "SP0": "纸浆", "SR0": "白糖", "SS0": "不锈钢", "TA0": "PTA",
    "UR0": "尿素", "V0": "PVC", "Y0": "豆油", "ZC0": "动力煤",
}

AGRI_VARIETIES = [
    "A0", "AP0", "C0", "CF0", "CJ0", "CS0", "JD0", "LH0",
    "M0", "OI0", "P0", "PK0", "RM0", "RU0", "SR0", "SP0", "Y0",
]

IMPORT_DEPENDENT_VARIETIES = ["A0", "C0", "CF0", "OI0", "P0", "SR0"]

INDUSTRIAL_VARIETIES = [
    "EB0", "EG0", "FG0", "HC0", "I0", "J0", "JM0", "L0",
    "MA0", "PG0", "PP0", "SA0", "SF0", "SM0", "SS0", "TA0",
    "UR0", "V0", "ZC0",
]

VARIETY_REGION_MAP = {
    "A0": "东北_黑龙江", "C0": "东北_吉林", "CS0": "东北_吉林",
    "M0": "华北_山东", "Y0": "华北_山东", "P0": "华南_广西",
    "JD0": "华北_山东", "LH0": "华中_湖北", "CF0": "西北_新疆",
    "SR0": "华南_广西", "AP0": "西北_新疆", "CJ0": "西北_新疆",
    "OI0": "华中_湖北", "RM0": "华中_湖北", "PK0": "华北_河南",
    "RU0": "华南_广西", "SP0": "东北_黑龙江",
    "FG0": "华北_山东", "SA0": "华北_山东",
    "EB0": "华北_山东", "EG0": "华北_山东",
    "HC0": "华北_山东", "I0": "华北_山东",
    "J0": "华北_山东", "JM0": "华北_山东",
    "L0": "华北_山东", "MA0": "华北_山东",
    "PG0": "华北_山东", "PP0": "华北_山东",
    "SF0": "西北_新疆", "SM0": "西北_新疆",
    "SS0": "华北_山东", "TA0": "华北_山东",
    "UR0": "华北_山东", "V0": "华北_山东",
    "ZC0": "华北_河南",
}

REGION_PROVINCE_MAP = {
    "东北_黑龙江": "黑龙江", "东北_吉林": "吉林",
    "华北_河南": "河南", "华北_山东": "山东",
    "华中_湖北": "湖北", "华南_广西": "广西",
    "西北_新疆": "新疆",
}

WEATHER_REGIONS = [
    "东北_黑龙江", "东北_吉林", "华北_河南", "华北_山东",
    "华中_湖北", "华南_广西", "西北_新疆",
]

RS_PROVINCES = ["黑龙江", "吉林", "山东", "河南", "湖北", "广西", "新疆"]

FEATURE_NAMES = [
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

PAGES = [
    ("01_home", "首页概览"),
    ("02_data_explorer", "数据探索"),
    ("03_causal_analysis", "因果分析"),
    ("04_causal_pricing", "因果定价"),
    ("05_conformal_prediction", "保形预测"),
    ("06_smart_pricing", "智能定价"),
    ("07_risk_assessment", "风险评估"),
    ("08_social_value", "社会价值"),
    ("09_five_validation", "五重验证"),
    ("10_benchmark", "基准对比"),
    ("11_ablation", "消融实验"),
    ("12_pure_prediction", "纯预测验证"),
    ("13_sichuan", "四川特色"),
    ("14_extreme_disaster", "极端灾害"),
    ("15_one_click", "一键复现"),
]

CACHE_TTL_DATA = 86400
CACHE_TTL_MODEL = 604800
CACHE_TTL_RESULT = 3600
