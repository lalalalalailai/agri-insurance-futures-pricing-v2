import streamlit as st

THEME_CSS = """
<style>
    :root {
        --primary: #165DFF;
        --primary-dark: #0E42D2;
        --primary-light: #E8F0FF;
        --secondary: #0D9488;
        --agri-green: #15803D;
        --agri-green-light: #DCFCE7;
        --warning: #C2410C;
        --danger: #DC2626;
        --bg: #FFFFFF;
        --bg-alt: #F8FAFC;
        --bg-card: #FFFFFF;
        --divider: #E2E8F0;
        --text: #1E293B;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --title: #0F172A;
        --card-shadow: 0 2px 8px rgba(0,0,0,0.10);
        --sidebar-bg: #0A1F14;
        --sidebar-text: #F1F5F9;
        --sidebar-hover: #163D2B;
        --sidebar-muted: #94A3B8;
    }

    .stApp {
        font-family: 'Microsoft YaHei', 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: var(--title);
        margin-bottom: 8px;
    }

    .sub-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 16px;
    }

    .metric-card {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 20px;
        box-shadow: var(--card-shadow);
        text-align: center;
        border-left: 4px solid var(--primary);
    }

    .metric-card .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--primary);
        line-height: 1.2;
    }

    .metric-card .metric-label {
        font-size: 14px;
        color: var(--text);
        margin-top: 4px;
        font-weight: 500;
    }

    .metric-card.green { border-left-color: var(--agri-green); }
    .metric-card.green .metric-value { color: var(--agri-green); }
    .metric-card.cyan { border-left-color: var(--secondary); }
    .metric-card.cyan .metric-value { color: var(--secondary); }
    .metric-card.orange { border-left-color: var(--warning); }
    .metric-card.orange .metric-value { color: var(--warning); }

    .info-card {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 16px;
        box-shadow: var(--card-shadow);
        margin-bottom: 12px;
        border: 1px solid var(--divider);
    }

    .info-card h4 {
        color: var(--title) !important;
        margin-bottom: 8px;
        font-size: 16px;
        font-weight: 600;
    }

    .info-card p {
        color: var(--text) !important;
        font-size: 14px;
        margin: 4px 0;
        line-height: 1.6;
    }

    .highlight { color: var(--primary); font-weight: 600; }
    .highlight-green { color: var(--agri-green); font-weight: 600; }
    .highlight-red { color: var(--danger); font-weight: 600; }

    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--sidebar-text) !important;
    }

    section[data-testid="stSidebar"] .stRadio > div > label {
        color: var(--sidebar-text) !important;
        font-size: 16px;
        padding: 8px 14px;
        border-radius: 6px;
        transition: all 0.2s;
    }

    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: var(--sidebar-hover);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div {
        color: #CBD5E1 !important;
        font-size: 13px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        padding: 8px 16px;
        border-radius: 4px;
        border: none;
        background: transparent;
        color: var(--sidebar-text) !important;
        font-size: 14px;
        transition: all 0.2s;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--sidebar-hover);
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: var(--sidebar-text) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    .stButton > button {
        border-radius: 4px;
        background: var(--primary);
        color: #FFFFFF !important;
        border: none;
        padding: 8px 24px;
        font-weight: 500;
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        color: #FFFFFF !important;
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    .success-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background: var(--agri-green-light);
        color: var(--agri-green);
        font-size: 12px;
        font-weight: 600;
    }

    .warning-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background: #FFF7ED;
        color: var(--warning);
        font-size: 12px;
        font-weight: 600;
    }

    .danger-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background: #FEF2F2;
        color: var(--danger);
        font-size: 12px;
        font-weight: 600;
    }

    .stMetric label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: var(--title) !important;
        font-weight: 700;
    }

    .stMetric [data-testid="stMetricDelta"] {
        font-weight: 600;
    }

    h1, h2, h3 {
        color: var(--title) !important;
    }

    h4, h5, h6 {
        color: var(--text) !important;
    }

    p, li, td, th {
        color: var(--text) !important;
    }

    label {
        color: var(--text) !important;
    }

    .stMarkdown {
        color: var(--text) !important;
    }

    .stAlert {
        color: var(--text) !important;
    }

    .stInfo > div {
        color: var(--text) !important;
    }

    .stSuccess > div {
        color: var(--text) !important;
    }

    .stWarning > div {
        color: var(--text) !important;
    }

    .stError > div {
        color: var(--text) !important;
    }

    table {
        color: var(--text) !important;
    }

    table th {
        color: var(--title) !important;
        background-color: var(--bg-alt) !important;
    }

    table td {
        color: var(--text) !important;
    }

    .stSidebar .stMarkdown {
        color: var(--sidebar-text) !important;
    }

    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        color: var(--sidebar-text) !important;
    }

    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
        color: var(--sidebar-text) !important;
    }

    [data-testid="stSidebar"] .stSlider [data-testid="stSliderLabel"] {
        color: var(--sidebar-text) !important;
    }

    [data-testid="stSidebar"] .stNumberInput label {
        color: var(--sidebar-text) !important;
    }

    [data-testid="stSidebar"] .stDateInput label {
        color: var(--sidebar-text) !important;
    }
</style>
"""


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def metric_card(value, label, color="blue"):
    color_class = {"blue": "", "green": "green", "cyan": "cyan", "orange": "orange"}.get(color, "")
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def info_card(title, content_html):
    st.markdown(f"""
    <div class="info-card">
        <h4>{title}</h4>
        {content_html}
    </div>
    """, unsafe_allow_html=True)
