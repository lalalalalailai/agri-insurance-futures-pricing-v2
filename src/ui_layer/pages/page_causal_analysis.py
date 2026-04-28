import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from src.data_layer.data_loader import DataLoader
from src.data_layer.preprocessor import Preprocessor
from src.data_layer.feature_engineer import FeatureEngineer
from src.model_layer.agri_pc import AgriPC
from src.service_layer.fault_tolerance import FaultTolerance
from config import FUTURES_VARIETIES


def render():
    st.markdown('<div class="main-title">🔗 因果分析 (Agri-PC)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        variety_list = DataLoader().futures.get_variety_list()
        selected = st.selectbox("选择品种", variety_list,
                                format_func=lambda x: x["label"])
        alpha = st.slider("显著性水平", 0.01, 0.10, 0.05, 0.01)
        run_btn = st.button("🔍 运行Agri-PC", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Agri-PC因果发现中..."):
                result = FaultTolerance.safe_operation(lambda: _run_agri_pc(
                    selected["code"], alpha
                ), "Agri-PC运行失败")

                if result:
                    quality = result["quality"]
                    dag = result["dag"]

                    q1, q2, q3 = st.columns(3)
                    with q1:
                        st.metric("DAG F1-score", f"{quality.get('f1_score', 0):.4f}")
                    with q2:
                        st.metric("搜索空间缩减率", f"{quality.get('search_space_reduction', 0):.1f}%")
                    with q3:
                        st.metric("边数/节点数", f"{quality.get('n_edges', 0)}/{quality.get('n_nodes', 0)}")

                    st.markdown("### 因果DAG图")
                    fig = _plot_dag(dag)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    chains = result.get("causal_chains", [])
                    if chains:
                        st.markdown("### 核心因果链")
                        for i, chain in enumerate(chains[:5]):
                            st.markdown(f"**链{i+1}**: {' → '.join(chain)}")
                else:
                    st.warning("请点击「运行Agri-PC」开始因果发现")


def _run_agri_pc(variety_code, alpha):
    loader = DataLoader()
    preprocessor = Preprocessor()
    fe = FeatureEngineer()

    panel = loader.load_variety_panel(variety_code)
    panel = preprocessor.preprocess_panel(panel)
    panel = fe.build_features(panel)

    feature_cols = fe.get_feature_columns()
    available = [c for c in feature_cols if c in panel.columns]

    agri_pc = AgriPC(alpha=alpha)
    return agri_pc.discover(panel, feature_names=available)


def _plot_dag(dag):
    if dag.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(dag, seed=42, k=2)

    node_colors = []
    for node in dag.nodes():
        if node in ["temperature", "precipitation", "humidity", "wind_speed",
                     "surface_pressure", "solar_radiation",
                     "extreme_temp_index", "extreme_precip_index"]:
            node_colors.append("#57B894")
        elif node in ["ndvi", "evi", "lst", "drought_index", "yield_proxy"]:
            node_colors.append("#E9C46A")
        elif node in ["close", "open", "high", "low", "volume", "hold"]:
            node_colors.append("#165DFF")
        elif node in ["cpi", "ppi", "m2", "gdp", "pmi"]:
            node_colors.append("#9B59B6")
        else:
            node_colors.append("#95A5A6")

    edge_x, edge_y = [], []
    for edge in dag.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[n][0] for n in dag.nodes()]
    node_y = [pos[n][1] for n in dag.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#CBD5D1"), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=20, color=node_colors, line=dict(width=2, color="white")),
        text=list(dag.nodes()), textposition="top center",
        textfont=dict(size=10, color="#0F172A"),
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Agri-PC 因果DAG", height=600, template="agri_green_light",
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig
