import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import numpy as np

pio.templates["agri_green_light"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Microsoft YaHei, Inter, Arial", color="#1E293B", size=12),
        title=dict(font=dict(size=17, color="#0F172A"), x=0.02),
        xaxis=dict(
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5D1",
            linecolor="#94A3B8",
            tickcolor="#475569",
            title=dict(font=dict(color="#1E293B", size=13)),
            tickfont=dict(color="#475569"),
        ),
        yaxis=dict(
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5D1",
            linecolor="#94A3B8",
            tickcolor="#475569",
            title=dict(font=dict(color="#1E293B", size=13)),
            tickfont=dict(color="#475569"),
        ),
        colorway=["#165DFF", "#0D9488", "#E76F51", "#15803D", "#E9C46A", "#7C3AED"],
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(color="#1E293B", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(color="#0F172A", size=13),
            bordercolor="#165DFF",
        ),
    ),
    data=dict(
        candlestick=[go.Candlestick(increasing_line_color="#15803D", decreasing_line_color="#E76F51")]
    )
)
pio.templates.default = "agri_green_light"


def time_series_chart(df, x_col, y_cols, title="", labels=None):
    fig = go.Figure()
    colors = ["#165DFF", "#0D9488", "#15803D", "#E76F51", "#7C3AED", "#E9C46A"]
    for i, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[col], mode="lines",
                name=labels.get(col, col) if labels else col,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
    fig.update_layout(title=title, height=480, template="agri_green_light")
    return fig


def correlation_heatmap(corr_matrix, title="特征相关性热力图"):
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, "#165DFF"], [0.5, "#F8FAFC"], [1, "#E76F51"]],
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#1E293B"),
    ))
    fig.update_layout(title=title, height=600, template="agri_green_light")
    return fig


def bar_comparison(categories, values1, values2, name1, name2, title=""):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values1, name=name1, marker_color="#165DFF"))
    fig.add_trace(go.Bar(x=categories, y=values2, name=name2, marker_color="#0D9488"))
    fig.update_layout(
        title=title, barmode="group", height=480, template="agri_green_light",
        xaxis=dict(tickfont=dict(color="#1E293B", size=13)),
        yaxis=dict(tickfont=dict(color="#475569")),
    )
    return fig


def prediction_interval_chart(dates, actual, predicted, lower, upper, title=""):
    fig = go.Figure()
    if upper is not None and lower is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(22,93,255,0.12)",
            line=dict(color="rgba(22,93,255,0)"), name="90%置信区间",
        ))
    if predicted is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=predicted, mode="lines",
            name="模型预测", line=dict(color="#165DFF", width=2),
        ))
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=actual, mode="markers",
            name="真实价格", marker=dict(color="#E76F51", size=4),
        ))
    fig.update_layout(title=title, height=480, template="agri_green_light")
    return fig


def feature_importance_chart(importance_dict, title="特征重要性排序"):
    if not importance_dict:
        return go.Figure()
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color="#15803D",
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#1E293B", size=11),
    ))
    fig.update_layout(title=title, height=max(400, len(names) * 25), template="agri_green_light")
    return fig


def coverage_chart(coverage_history, alpha_history, target=0.9):
    fig = go.Figure()
    if coverage_history:
        fig.add_trace(go.Scatter(
            y=coverage_history, mode="lines+markers",
            name="实际覆盖率", line=dict(color="#165DFF", width=2),
        ))
        fig.add_hline(y=target, line_dash="dash", line_color="#DC2626",
                      annotation_text=f"目标{target*100}%",
                      annotation_font_color="#DC2626")
    if alpha_history:
        fig.add_trace(go.Scatter(
            y=alpha_history, mode="lines+markers",
            name="自适应α", line=dict(color="#0D9488", width=2),
            yaxis="y2",
        ))
    fig.update_layout(
        title="覆盖率与自适应置信水平变化",
        height=400, template="agri_green_light",
        yaxis2=dict(overlaying="y", side="right", title="α值",
                    titlefont=dict(color="#475569"), tickfont=dict(color="#475569")),
    )
    return fig
