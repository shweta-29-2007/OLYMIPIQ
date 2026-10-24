"""
OLYMPIQ - Olympic Data Analytics and Predictive Visualization
=============================================================

Main Streamlit dashboard.

Run with:
    streamlit run app.py

The app uses only authentic historical Olympic data. If the dataset is
missing, it explains how to obtain it instead of substituting synthetic data.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import OlympicAnalyzer
from data_loader import OlympicDataLoader, load_data
from prediction import (
    ForecastResult,
    OlympicDataError,
    chronological_backtest,
    describe_schema,
    explain_metrics,
    forecast_medals,
)

# ---------------------------------------------------------------------------
# Page configuration and theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OLYMPIQ | Olympic Analytics",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0a0e27"
PANEL = "#111827"
GOLD = "#fbbf24"
BLUE = "#60a5fa"
SILVER = "#d1d5db"
BRONZE = "#b45309"

st.markdown(
    f"""
    <style>
        .stApp {{ background: linear-gradient(180deg, {NAVY} 0%, #070a1a 100%); }}
        [data-testid="stSidebar"] {{
            background: #0d1226;
            border-right: 1px solid #1e3a8a;
        }}
        h1, h2, h3 {{ color: {GOLD}; letter-spacing: 0.3px; }}
        h4, h5, h6 {{ color: {BLUE}; }}
        p, li, label, span {{ color: #e5e7eb; }}
        .hero {{
            padding: 1.4rem 1.8rem;
            border: 1px solid #1e3a8a;
            border-radius: 16px;
            background: radial-gradient(circle at top left, #15204a 0%, #0b1024 70%);
            margin-bottom: 1rem;
        }}
        .hero h1 {{ margin: 0; font-size: 2.7rem; }}
        .hero p {{ margin: 0.35rem 0 0 0; color: #c7d2fe; }}
        .card {{
            background: {PANEL};
            border: 1px solid #1f2a4a;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }}
        div[data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid #1f2a4a;
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }}
        div[data-testid="stMetricLabel"] p {{ color: #93a4c3 !important; }}
        div[data-testid="stMetricValue"] {{ color: {GOLD} !important; }}
        .stDataFrame {{ border: 1px solid #1f2a4a; border-radius: 10px; }}
        .note {{
            border-left: 4px solid {GOLD};
            background: #171b31;
            padding: 0.7rem 1rem;
            border-radius: 6px;
            color: #e5e7eb;
        }}
        .warn {{
            border-left: 4px solid #f97316;
            background: #2a1a12;
            padding: 0.7rem 1rem;
            border-radius: 6px;
            color: #fed7aa;
        }}
        footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_figure(fig: go.Figure, height: int = 430) -> go.Figure:
    """Apply the shared dark plot theme."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color="#e5e7eb"),
        margin=dict(l=30, r=20, t=55, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937"),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading authentic Olympic history...")
def get_loader() -> OlympicDataLoader:
    return OlympicDataLoader()


@st.cache_resource(show_spinner="Preparing analysis tables...")
def get_analyzer() -> OlympicAnalyzer:
    return OlympicAnalyzer(get_loader())


@st.cache_data(show_spinner="Running chronological backtest and forecast...")
def run_forecast(season: str, target_year: int, use_host: bool = True) -> ForecastResult:
    """Fit and validate the transparent hurdle model.

    ``forecast_medals`` internally uses expanding-window chronological
    validation (train on earlier editions only), so no future information
    leaks into a past prediction.
    """
    history = load_data()
    return forecast_medals(
        history,
        season=season,
        target_year=target_year,
        n_test_editions=3,
        min_train_editions=4,
        min_history_year=1960,
        active_window=2,
        use_host=use_host,
    )


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)


def show_error_box(exc: Exception) -> None:
    st.error(str(exc))
    st.markdown(
        """
        <div class="warn">
        The app will not substitute synthetic data. Place the authentic dataset at
        <code>data/olympics.csv</code> (or set <code>OLYMPIQ_DATA_PATH</code>) and
        reload. See the <b>Dataset &amp; Methodology</b> page for the download link.
        </div>
        """,
        unsafe_allow_html=True,
    )


def season_select(key: str) -> str:
    return st.selectbox(
        "Season",
        ["Summer", "Winter"],
        key=key,
        help="Summer and Winter Games have different sports and medal scales, so they are analysed separately.",
    )


def medal_breakdown_chart(frame: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    """Grouped gold/silver/bronze bar chart."""
    melted = frame.melt(
        id_vars=[label_col],
        value_vars=[c for c in ["gold", "silver", "bronze"] if c in frame.columns],
        var_name="medal",
        value_name="count",
    )
    colours = {"gold": GOLD, "silver": SILVER, "bronze": BRONZE}
    fig = px.bar(
        melted,
        x=label_col,
        y="count",
        color="medal",
        barmode="group",
        title=title,
        color_discrete_map=colours,
        labels={"count": "Medals", label_col: ""},
    )
    fig.update_layout(xaxis_tickangle=-35)
    return style_figure(fig)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame, analyzer: OlympicAnalyzer) -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🏅 OLYMPIQ</h1>
          <p>Olympic history, country trajectories and transparent medal forecasting — built on authentic public data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### What this dashboard does

        OLYMPIQ explores **120 years of Olympic history** (Athens 1896 – Rio 2016) and turns it into:

        - **Interactive history** — medal trends by edition, country, sport and medal colour.
        - **Country trajectories** — how a nation's performance changes across Games.
        - **Breakthrough Watch** — countries with improving historical trends, *not* guaranteed future medallists.
        - **Sport Opportunity Explorer** — sports where a country has relative historical strength.
        - **Chronologically validated forecasts** — the model is trained on earlier editions and tested on later ones.
        - **Honest uncertainty** — every forecast is an estimate with documented limitations.

        > **Important:** predictions are statistical estimates, not promised medal counts.
        > The dataset ends in 2016, so 2020+ forecasts extrapolate historical patterns.
        """
    )

    info = analyzer.get_dataset_explanation()["stats"]
    st.markdown("### Dataset at a glance")
    metric_row(
        [
            ("Athlete-event rows", f"{info['total_rows']:,}"),
            ("Unique athletes", f"{info['unique_athletes']:,}"),
            ("NOCs", f"{info['unique_countries']}"),
            ("Editions", f"{info['years'][0]}–{info['years'][1]}"),
        ]
    )
    metric_row(
        [
            ("Sports", f"{info['unique_sports']}"),
            ("Events", f"{info['unique_events']}"),
            ("Seasons", ", ".join(info["seasons"])),
            ("Source", "TidyTuesday / Kaggle"),
        ]
    )

    st.markdown("### Explore the pages")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="card">
            <h4>📈 Historical analysis</h4>
            Historical Trends · Country Analysis · Sport Analysis ·
            Performance Trajectory · Breakthrough Watch · Sport Opportunity Explorer
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
            <h4>🔮 Predictive analysis</h4>
            Predictive Analytics · Prediction vs Actual backtest ·
            Model card · Data quality · Metric explanations
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Quick preview: all-time top nations")
    top = analyzer.get_top_countries(top_n=10)
    fig = px.bar(
        top,
        x="country",
        y=["gold", "silver", "bronze"],
        barmode="stack",
        title="All-time medals (top 10 NOCs)",
        color_discrete_map={"gold": GOLD, "silver": SILVER, "bronze": BRONZE},
    )
    st.plotly_chart(style_figure(fig), use_container_width=True)


def page_historical_trends(df: pd.DataFrame, analyzer: OlympicAnalyzer) -> None:
    st.title("📈 Historical Olympic Trends")
    st.caption("Authentic medal counts by edition. Team medals are counted once per NOC/event.")

    col1, col2, col3 = st.columns(3)
    with col1:
        season = season_select("trend_season")
    with col2:
        years = sorted(int(y) for y in df.loc[df["season"] == season, "year"].unique())
        year_range = st.slider("Year range", min_value=years[0], max_value=years[-1], value=(years[0], years[-1]))
    with col3:
        metric = st.selectbox(
            "Chart metric",
            ["Total medals", "Gold medals", "Countries participating"],
        )

    trends = analyzer.get_historical_trends(season=season, year_range=year_range)
    if trends.empty:
        st.warning("No editions match these filters.")
        return

    if metric == "Total medals":
        y_col, title = "total", "Total medals awarded"
    elif metric == "Gold medals":
        y_col, title = "gold", "Gold medals awarded"
    else:
        y_col, title = "countries_participating", "Countries participating"

    fig = px.line(trends, x="year", y=y_col, markers=True, title=title, labels={"year": "Edition", y_col: metric})
    fig.update_traces(line=dict(color=GOLD, width=3), marker=dict(size=7))
    st.plotly_chart(style_figure(fig), use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Gold", x=trends["year"], y=trends["gold"], marker_color=GOLD))
    fig2.add_trace(go.Bar(name="Silver", x=trends["year"], y=trends["silver"], marker_color=SILVER))
    fig2.add_trace(go.Bar(name="Bronze", x=trends["year"], y=trends["bronze"], marker_color=BRONZE))
    fig2.update_layout(barmode="stack", title="Medal mix by edition")
    st.plotly_chart(style_figure(fig2, height=390), use_container_width=True)

    st.markdown("### Edition-level table")
    st.dataframe(trends, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note">
        <b>Limitation:</b> the Olympic programme changed repeatedly (new sports, dropped events,
        boycotts, different participation rules). A rise in total medals can reflect a larger
        programme rather than better performance. Compare <i>shares</i> and per-country trends
        when making cross-era claims.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_country_analysis(df: pd.DataFrame, analyzer: OlympicAnalyzer) -> None:
    st.title("🌍 Country Analysis")
    st.caption("Gold, silver, bronze and total medals by country.")

    col1, col2, col3 = st.columns(3)
    with col1:
        season = season_select("country_season")
    with col2:
        years = sorted(int(y) for y in df.loc[df["season"] == season, "year"].unique())
        year_option = st.selectbox("Edition", ["All-time"] + [str(y) for y in years])
    with col3:
        top_n = st.slider("Number of countries", 5, 30, 15)

    year = None if year_option == "All-time" else int(year_option)
    top = analyzer.get_top_countries(season=season, year=year, top_n=top_n)

    if top.empty:
        st.warning("No data for this selection.")
        return

    fig = px.bar(
        top,
        x="country",
        y=["gold", "silver", "bronze"],
        barmode="group",
        title=f"Top {len(top)} countries — {season} {'all-time' if year is None else year}",
        color_discrete_map={"gold": GOLD, "silver": SILVER, "bronze": BRONZE},
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(style_figure(fig, height=460), use_container_width=True)

    st.markdown("### Rankings")
    st.dataframe(
        top[["country", "noc", "gold", "silver", "bronze", "total"]],
        use_container_width=True,
        hide_index=True,
    )

    # Country vs edition heatmap for the selected top countries
    st.markdown("### Top nations across editions")
    codes = top["noc"].tolist()
    trajectory = analyzer.country_medals[
        (analyzer.country_medals["season"] == season)
        & (analyzer.country_medals["noc"].isin(codes))
    ].copy()
    if not trajectory.empty:
        pivot = trajectory.pivot_table(index="noc", columns="year", values="total", aggfunc="sum", fill_value=0)
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Total medals by edition",
            labels=dict(x="Edition", y="NOC", color="Medals"),
        )
        st.plotly_chart(style_figure(fig_heat, height=420), use_container_width=True)


def page_sport_analysis(df: pd.DataFrame, analyzer: OlympicAnalyzer) -> None:
    st.title("🏃 Sport Analysis")
    st.caption("Which sports produce the most medals, and how does a country perform within them?")

    col1, col2 = st.columns(2)
    with col1:
        season = season_select("sport_season")
    with col2:
        top_n = st.slider("Sports to show", 5, 30, 15, key="sport_top_n")

    sports = analyzer.loader.get_sport_summary(season=season)
    if sports.empty:
        st.warning("No sport data available.")
        return

    fig = px.treemap(
        sports.head(25),
        path=["sport"],
        values="total",
        color="gold",
        color_continuous_scale="YlOrRd",
        title="Medal share by sport (top 25)",
    )
    st.plotly_chart(style_figure(fig, height=460), use_container_width=True)

    fig2 = px.bar(
        sports.head(top_n),
        x="sport",
        y=["gold", "silver", "bronze"],
        barmode="group",
        title=f"Top {top_n} sports by total medals — {season}",
        color_discrete_map={"gold": GOLD, "silver": SILVER, "bronze": BRONZE},
    )
    fig2.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(style_figure(fig2, height=460), use_container_width=True)

    with st.expander("Browse sport table"):
        st.dataframe(sports, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note">
        Sport names are standardised by the source dataset. Some sports were renamed,
        split or merged across eras; comparisons should therefore be read as
        <i>historical</i> rather than perfectly like-for-like.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_trajectory(df: pd.DataFrame, analyzer: OlympicAnalyzer) -> None:
    st.title("🎯 Olympic Performance Trajectory")
    st.caption("How a country's medal performance changes across editions.")

    col1, col2 = st.columns([2, 1])
    with col1:
        countries = sorted(df["noc"].dropna().unique().tolist())
        default_index = countries.index("USA") if "USA" in countries else 0
        noc = st.selectbox("Country (NOC)", countries, index=default_index, key="traj_noc")
    with col2:
        season = season_select("traj_season")

    trajectory = analyzer.get_performance_trajectory(noc, season=season)
    if trajectory.empty:
        st.warning("No trajectory data for this country and season.")
        return

    name = trajectory["country_name"].iloc[0]
    metric_row(
        [
            ("Country", name),
            ("Editions", str(int(trajectory["year"].nunique()))),
            ("Best total", str(int(trajectory["total"].max()))),
            ("Latest total", str(int(trajectory["total"].iloc[-1]))),
        ]
    )

    fig = go.Figure()
    for col, colour, width in [
        ("gold", GOLD, 2),
        ("silver", SILVER, 2),
        ("bronze", BRONZE, 2),
        ("total", BLUE, 3),
    ]:
        fig.add_trace(
            go.Scatter(
                x=trajectory["year"],
                y=trajectory[col],
                mode="lines+markers",
                name=col.title(),
                line=dict(color=colour, width=width),
            )
        )
    fig.update_layout(title=f"{name} — medals by edition", xaxis_title="Edition", yaxis_title="Medals")
    st.plotly_chart(style_figure(fig, height=470), use_container_width=True)

    st.markdown("### Recent form (3-edition rolling average)")
    fig2 = px.line(
        trajectory,
        x="year",
        y=["total", "total_rolling"],
        markers=True,
        title="Total medals and rolling average",
        labels={"value": "Medals", "year": "Edition", "variable": "Series"},
    )
    fig2.update_traces(selector=dict(name="total"), line=dict(color=BLUE))
    fig2.update_traces(selector=dict(name="total_rolling"), line=dict(color=GOLD, dash="dash"))
    st.plotly_chart(style_figure(fig2, height=380), use_container_width=True)

    with st.expander("Trajectory data"):
        st.dataframe(trajectory, use_container_width=True, hide_index=True)


def page_breakthrough(analyzer: OlympicAnalyzer) -> None:
    st.title("🔍 Breakthrough Watch")
    st.caption("Countries with improving historical trends. This is not a guarantee of future medals.")

    threshold = st.slider(
        "Minimum improvement ratio",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Compares the average of the earlier half of a country's recent editions with the later half.",
    )
    result = analyzer.get_breakthrough_watch(improvement_threshold=threshold)

    if result.empty:
        st.info("No countries meet this improvement threshold in the available data.")
        return

    metric_row(
        [
            ("Countries flagged", str(result["noc"].nunique())),
            ("Summer entries", str(int((result["season"] == "Summer").sum()))),
            ("Winter entries", str(int((result["season"] == "Winter").sum()))),
            ("Threshold", f"{threshold:.0%}"),
        ]
    )

    fig = px.bar(
        result.head(20),
        x="country",
        y="improvement_ratio",
        color="season",
        title="Largest historical improvements (top 20)",
        labels={"improvement_ratio": "Improvement ratio", "country": ""},
        color_discrete_sequence=[GOLD, BLUE],
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(style_figure(fig, height=460), use_container_width=True)

    st.dataframe(
        result[
            [
                "country",
                "noc",
                "season",
                "editions",
                "first_medals",
                "last_medals",
                "recent_avg",
                "improvement_ratio",
                "trend_slope",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="warn">
        <b>Read this carefully:</b> an improving past trend is an <i>association in history</i>,
        not proof that a country will keep improving. Small medal counts make trends noisy:
        going from 0 to 1 medal is a huge percentage change but may be random. Use this page to
        start questions, not to promise outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_sport_opportunity(df: pd.DataFrame, analyzer: OlympicAnalyzer) -> None:
    st.title("💡 Sport Opportunity Explorer")
    st.caption("Sports where a country has relative historical strength — or little history to build on.")

    countries = sorted(df["noc"].dropna().unique().tolist())
    default_index = countries.index("USA") if "USA" in countries else 0
    noc = st.selectbox("Country (NOC)", countries, index=default_index, key="opp_noc")
    opportunities = analyzer.get_sport_opportunity_explorer(noc, top_n=15)

    if opportunities.empty:
        st.warning("No participation data for this country.")
        return

    country_name = (
        df.loc[df["noc"] == noc, "team"].mode().iloc[0]
        if not df.loc[df["noc"] == noc, "team"].mode().empty
        else noc
    )
    st.markdown(f"### {country_name} ({noc})")

    if "category" in opportunities.columns:
        label = "category"
    elif "medals" in opportunities.columns:
        label = "medals"
    else:
        label = opportunities.columns[0]

    fig = px.bar(
        opportunities,
        x="sport",
        y="opportunity_score",
        color=label,
        title="Relative sport opportunity score",
        labels={"opportunity_score": "Score", "sport": ""},
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(style_figure(fig, height=440), use_container_width=True)

    st.dataframe(opportunities, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note">
        <b>How the score works:</b> it combines historical medals, gold medals and the share of
        entered events that produced a medal. It is a descriptive ranking of <i>past</i> strength,
        not a causal forecast. A low score can also mean the country has simply entered few events.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_predictions(season: str, target_year: int, use_host: bool) -> None:
    st.title("🔮 Predictive Analytics")
    st.caption("Transparent hurdle model, validated chronologically. All outputs are estimates.")

    st.markdown(
        f"""
        <div class="warn">
        <b>Dataset cutoff:</b> the historical data ends in 2016. A forecast for {target_year} is an
        extrapolation of historical patterns, not observed data. Medal tables are highly sensitive to
        injuries, host decisions, programme changes and political events that this model cannot see.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Fitting occurrence and count stages; running chronological backtests..."):
            result = run_forecast(season, target_year, use_host)
    except (OlympicDataError, ValueError, FileNotFoundError) as exc:
        show_error_box(exc)
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unexpected prediction error: {exc}")
        return

    pred = result.predictions
    metric_row(
        [
            ("Target", f"{season} {target_year}"),
            ("Candidates", f"{len(pred):,}"),
            ("Backtest editions", str(len(result.backtest_folds))),
            ("Data through", str(int(result.backtest_folds["trained_through_year"].max()))),
        ]
    )

    st.markdown("### Forecast medal table")
    display_cols = [
        c
        for c in [
            "rank",
            "noc",
            "country",
            "predicted_total",
            "predicted_total_rounded",
            "medal_probability",
            "count_if_medal",
            "predicted_gold",
            "lower_80",
            "upper_80",
            "last_total",
            "host",
        ]
        if c in pred.columns
    ]
    st.dataframe(pred[display_cols].head(40), use_container_width=True, hide_index=True)

    top = pred.head(20)
    fig = px.bar(
        top,
        x="country",
        y="predicted_total",
        error_y=(top["upper_80"] - top["predicted_total"]).clip(lower=0) if {"upper_80", "predicted_total"} <= set(top) else None,
        title=f"Top 20 forecast medal totals — {season} {target_year}",
        labels={"predicted_total": "Expected medals", "country": ""},
        color="medal_probability" if "medal_probability" in top.columns else None,
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(style_figure(fig, height=470), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Medal probability vs count")
        show = pred.head(40).copy()
        fig_p = px.scatter(
            show,
            x="medal_probability",
            y="count_if_medal",
            size="predicted_total",
            color="predicted_total",
            hover_name="country",
            title="P(any medal) and expected count if medalling",
            labels={"medal_probability": "P(at least one medal)", "count_if_medal": "E[medals | medal]"},
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(style_figure(fig_p, height=420), use_container_width=True)
    with col2:
        st.markdown("### What the columns mean")
        st.markdown(
            """
            - **predicted_total** — expected medals: `P(any medal) × E[medals | medal]`.
            - **medal_probability** — probability the NOC wins at least one medal.
            - **count_if_medal** — expected medals *given* it wins at least one.
            - **predicted_gold** — recent historical gold share applied to the total (heuristic, not a separate model).
            - **lower_80 / upper_80** — approximate 80% interval from backtest residuals.
            """
        )
        st.info(
            "A zero-medal country is not deleted. The hurdle model estimates a low probability "
            "of winning anything instead of forcing a negative or fabricated count."
        )

    st.markdown("### Chronological backtest metrics")
    st.caption(
        "Each test edition is predicted using earlier editions only. POOLED averages over all folds."
    )
    metrics = result.metrics.copy()
    show_metrics = [
        c
        for c in [
            "season",
            "test_year",
            "n_countries",
            "n_zero_medal",
            "mae_model",
            "mae_baseline_last",
            "mae_baseline_mean3",
            "mae_improvement_pct",
            "rmse_model",
            "mae_model_medal_winners",
            "mae_model_zero_medal",
            "brier_medal_occurrence_model",
            "brier_medal_occurrence_baseline",
            "spearman_model",
            "top10_overlap_model",
        ]
        if c in metrics.columns
    ]
    st.dataframe(metrics[show_metrics].round(3), use_container_width=True, hide_index=True)

    st.markdown("### Model card (coefficients)")
    st.caption(
        "Occurrence stage: logistic regression for P(medal > 0). Count stage: Poisson GLM with a "
        "log(previous total + 1) offset, fitted on medal-winning rows."
    )
    st.dataframe(result.model_card.round(4), use_container_width=True, hide_index=True)

    st.markdown("### Data quality")
    st.dataframe(result.data_quality, use_container_width=True, hide_index=True)

    st.markdown("### Limitations and uncertainty")
    for note in result.notes:
        st.markdown(f"- {note}")

    with st.expander("Full forecast table (all candidates)"):
        st.dataframe(pred, use_container_width=True, hide_index=True)


def page_backtest(season: str, target_year: int, use_host: bool) -> None:
    st.title("📊 Prediction vs Actual")
    st.caption("Historical backtesting view. Train on earlier editions, test on a later edition.")

    try:
        result = run_forecast(season, target_year, use_host)
    except (OlympicDataError, ValueError, FileNotFoundError) as exc:
        show_error_box(exc)
        return
    except Exception as exc:  # pragma: no cover
        st.error(f"Unexpected backtest error: {exc}")
        return

    backtest = result.backtest
    folds = result.backtest_folds
    if backtest.empty:
        st.warning("The backtest returned no rows.")
        return

    test_years = sorted(int(y) for y in folds["test_year"].unique())
    selected = st.selectbox("Test edition", test_years, index=len(test_years) - 1)
    group = backtest.loc[backtest["test_year"] == selected].copy()

    fold = folds.loc[folds["test_year"] == selected].iloc[0]
    metric_row(
        [
            ("Training editions", str(int(fold["n_train_editions"]))),
            ("Trained through", str(int(fold["trained_through_year"]))),
            ("Scored countries", str(len(group))),
            ("Debutants", str(int(fold["n_debutants"]))),
        ]
    )

    st.markdown(
        f"""
        <div class="note">
        For <b>{season} {selected}</b> the model was fitted only on editions up to
        <b>{int(fold['trained_through_year'])}</b> and scored on every participating NOC in {selected}.
        Zero-medal countries are included, so the metrics are not flattered by dropping the long tail.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = px.scatter(
        group,
        x="actual_total",
        y="predicted_total",
        hover_name="country",
        color="actual_total",
        color_continuous_scale="YlOrRd",
        title=f"Predicted vs actual medals — {selected}",
        labels={"actual_total": "Actual medals", "predicted_total": "Predicted medals"},
    )
    maximum = float(max(group["actual_total"].max(), group["predicted_total"].max(), 1))
    fig.add_trace(
        go.Scatter(
            x=[0, maximum],
            y=[0, maximum],
            mode="lines",
            name="Perfect forecast",
            line=dict(color=BLUE, dash="dash"),
        )
    )
    st.plotly_chart(style_figure(fig, height=470), use_container_width=True)

    st.markdown("### Largest misses")
    misses = group.reindex(group["abs_error"].sort_values(ascending=False).index)
    st.dataframe(
        misses[
            [
                "country",
                "noc",
                "actual_total",
                "predicted_total",
                "baseline_last",
                "baseline_mean3",
                "error",
                "abs_error",
                "predicted_medal_probability",
                "seen_before",
            ]
        ].head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Metrics for every backtested edition")
    st.dataframe(result.metrics.round(3), use_container_width=True, hide_index=True)

    st.markdown("### Metric definitions")
    definitions = explain_metrics()
    st.dataframe(definitions, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="warn">
        <b>Interpretation:</b> MAE is the average absolute gap between predicted and actual medals
        per country. It is in medal units and is dominated by the many countries that win zero.
        A model that predicts "same as last time" is a strong baseline in this domain. If our model
        does not beat it on an edition, that is reported honestly rather than hidden.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dataset(analyzer: OlympicAnalyzer) -> None:
    st.title("📚 Dataset & Methodology")

    info = analyzer.get_dataset_explanation()
    st.markdown("### Source")
    st.markdown(
        f"""
        - **Dataset:** {info['source']}
        - **Public URL:** [{info['url']}]({info['url']})
        - **Coverage:** {info['coverage']} ({', '.join(info['seasons'])})
        - **Description:** {info['description']}
        """
    )

    st.markdown("### Required input columns")
    st.markdown(
        """
        The predictor accepts either raw athlete-event rows or an aggregated panel.
        Column names are matched case-insensitively.
        """
    )
    st.dataframe(describe_schema(), use_container_width=True, hide_index=True)

    st.markdown("### Data quality and cleaning")
    st.dataframe(analyzer.loader.data_quality(), use_container_width=True, hide_index=True)

    st.markdown("### Historical country-code changes")
    st.caption(
        "Codes are intentionally NOT merged across political eras. Read the table before "
        "interpreting long-run country trajectories."
    )
    st.dataframe(analyzer.loader.country_code_change_note(), use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Cleaning steps in `data_loader.py`**

        1. Exact duplicate athlete rows are removed.
        2. NOC codes are upper-cased and trimmed; team, sport, event and city names are trimmed.
        3. Medal values are standardised to Gold / Silver / Bronze; blanks mean "no medal".
        4. Team medals are counted once per NOC/event/medal, not once per athlete.
        5. Every NOC that participated is kept, including countries with zero medals.
        6. 1906 Intercalated Games are excluded from the predictor (not an official IOC edition).
        7. Summer and Winter are modelled separately because their sports and medal scales differ.

        **Important limitations**

        - The dataset ends at 2016; 2020+ values are forecasts, not observations.
        - NOC codes are not merged across political eras (URS, GDR, EUN, TCH, YUG keep separate histories).
        - Sport programmes and event counts change between editions.
        - Historical comparisons are descriptive, not causal.
        - Missing athlete attributes (age, height, weight) are not used by the model, so their gaps do not bias it.
        """
    )

    st.markdown("### Model methodology")
    st.markdown(
        """
        **Two-part hurdle model (`prediction.py`)**

        1. **Occurrence stage** — ridge logistic regression estimates `P(medals > 0)` for every NOC.
        2. **Count stage** — ridge Poisson regression estimates `E[medals | medals > 0]`, anchored on
           `log(previous total + 1)`, fitted only on medal-winning rows with a previous total.
        3. **Forecast** — `P(medal) × E[medals | medal]`.

        All features are computed from **earlier editions only**:

        - previous medal total and previous three-edition mean,
        - recent trend and momentum,
        - recent depth (best recent edition vs last edition),
        - recent gold share,
        - number of prior appearances,
        - host flag (estimated from history, not hard-coded as a medal bonus).

        **Validation:** expanding-window chronological backtest. For each test edition, the model is
        fitted only on earlier editions. There is **no random train/test split**, because that would
        leak future information into the past.

        **Baselines:** persistence (last edition's total) and mean of the previous three editions.

        **Metrics:** MAE, RMSE, median absolute error, bias, MAE split by zero-medal and medalling
        countries, zero-medal false-positive rate, Brier score for the occurrence stage, Spearman
        rank correlation and top-10 overlap. See `prediction.explain_metrics()`.
        """
    )

    st.markdown("### All metric definitions")
    st.dataframe(explain_metrics(), use_container_width=True, hide_index=True)

    st.markdown("### Data source and honesty statement")
    st.markdown(
        """
        - All historical records come from the public TidyTuesday / Kaggle Olympic dataset.
        - No synthetic medal counts, countries or accuracy scores are generated.
        - If the data are missing or insufficient, the predictor raises an error instead of inventing results.
        - Forecasts are estimates with uncertainty; they are not guarantees and do not imply causation.
        """
    )


# ---------------------------------------------------------------------------
# Sidebar + routing
# ---------------------------------------------------------------------------

def main() -> None:
    st.sidebar.markdown(
        f"<h1 style='color:{GOLD}; margin-bottom:0;'>🏅 OLYMPIQ</h1>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Olympic analytics & transparent forecasting")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Historical Trends",
            "Country Analysis",
            "Sport Analysis",
            "Performance Trajectory",
            "Breakthrough Watch",
            "Sport Opportunity",
            "Predictive Analytics",
            "Prediction vs Actual",
            "Dataset & Methodology",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Forecast settings")
    season = st.sidebar.selectbox("Season", ["Summer", "Winter"], key="global_season")
    if season == "Summer":
        # 2020/2024 are deliberately excluded: those Games are already in the
        # past and are not in this dataset, so showing them as "forecasts" would
        # confuse a prediction with a known result.
        years = [2028, 2032, 2036]
        default_year = 2028
    else:
        years = [2026, 2030, 2034]
        default_year = 2026
    target_year = st.sidebar.selectbox(
        "Target edition",
        years,
        index=years.index(default_year),
        help="The dataset ends in 2016. Choose a later edition to forecast.",
    )
    use_host = st.sidebar.checkbox("Estimate host effect", value=True)
    st.sidebar.caption(
        "The model is refitted chronologically for each target. "
        "Changing these settings invalidates cached forecasts."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <small>Authentic data: TidyTuesday / Kaggle Olympic history (1896–2016).<br>
        Estimates only — not guaranteed medal counts.</small>
        """,
        unsafe_allow_html=True,
    )

    # Load data / analyzer with friendly errors
    try:
        analyzer = get_analyzer()
        df = analyzer.df
    except Exception as exc:
        st.title("🏅 OLYMPIQ")
        st.error("Could not load the Olympic dataset.")
        show_error_box(exc)
        st.stop()

    if page == "Overview":
        page_overview(df, analyzer)
    elif page == "Historical Trends":
        page_historical_trends(df, analyzer)
    elif page == "Country Analysis":
        page_country_analysis(df, analyzer)
    elif page == "Sport Analysis":
        page_sport_analysis(df, analyzer)
    elif page == "Performance Trajectory":
        page_trajectory(df, analyzer)
    elif page == "Breakthrough Watch":
        page_breakthrough(analyzer)
    elif page == "Sport Opportunity":
        page_sport_opportunity(df, analyzer)
    elif page == "Predictive Analytics":
        page_predictions(season, target_year, use_host)
    elif page == "Prediction vs Actual":
        page_backtest(season, target_year, use_host)
    elif page == "Dataset & Methodology":
        page_dataset(analyzer)


if __name__ == "__main__":
    main()
