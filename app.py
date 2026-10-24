"""OLYMPIQ — historical Olympic data exploration and cautious forecasting."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import MEDAL_ORDER, breakthrough_watch, medal_events, medal_table, sport_opportunities, sport_table, trajectory, trend_by_edition
from data_loader import ATHLETE_SOURCE, DataUnavailableError, load_olympic_data
from prediction import build_forecast


st.set_page_config(page_title="OLYMPIQ | Olympic Data Analytics", page_icon="🏅", layout="wide")

COLORS = {"Gold": "#F4BF54", "Silver": "#B7C5D8", "Bronze": "#CF815F", "Total": "#45D6C3"}
CHART_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#DCE7F3", size=12),
    margin=dict(l=18, r=18, t=42, b=16), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --navy:#07111F; --panel:#101D2D; --line:#24354A; --muted:#92A5BA; --aqua:#45D6C3; }
.stApp { background: radial-gradient(ellipse at 90% 0%, #142842 0%, #07111F 48%, #050C16 100%); color:#E8F0F8; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0B1726,#08121F); border-right:1px solid #21344A; }
[data-testid="stSidebar"] * { font-family:'Manrope',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em; }
p, label, div { font-family:'Manrope',sans-serif; }
.brand {font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;letter-spacing:.16em;color:#F4F8FC}
.brand span {color:#45D6C3}
.eyebrow {font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.14em;text-transform:uppercase;color:#8FA6BE}
.hero {background:linear-gradient(125deg,rgba(23,44,67,.95),rgba(12,25,40,.82));border:1px solid #29415A;border-radius:18px;padding:26px 30px;margin:8px 0 22px;position:relative;overflow:hidden}
.hero:after {content:'';position:absolute;right:-32px;top:-110px;width:270px;height:270px;border:1px solid rgba(69,214,195,.22);border-radius:50%;box-shadow:0 0 0 22px rgba(69,214,195,.035),0 0 0 48px rgba(69,214,195,.025)}
.hero h1 {font-size:2.35rem;margin:5px 0 5px;color:#F5F8FC}.hero p {color:#B3C4D5;max-width:720px;margin:0}
.section-kicker {color:#45D6C3;font-family:'DM Mono';font-size:.72rem;letter-spacing:.1em;text-transform:uppercase}
[data-testid="stMetric"] {background:linear-gradient(145deg,#122338,#0E1A29);border:1px solid #263A51;padding:15px 17px;border-radius:14px}
[data-testid="stMetricLabel"] {color:#9CB0C4!important}.stCaption {color:#91A5B9!important}
[data-testid="stDataFrame"] {border:1px solid #263A51;border-radius:12px;overflow:hidden}
.note {background:#102539;border-left:3px solid #45D6C3;padding:12px 15px;border-radius:5px;color:#C6D7E6;font-size:.9rem}
.warning {background:#31261C;border-left:3px solid #F4BF54;padding:12px 15px;border-radius:5px;color:#E7D7BF;font-size:.9rem}
.stButton>button {border:1px solid #30475F;border-radius:9px}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Reading the historical Olympic records…")
def get_data() -> pd.DataFrame:
    return load_olympic_data()


@st.cache_data(show_spinner=False)
def cached_forecast(season: str, year: int):
    # The source table is fixed for this app version, so key this expensive
    # rolling backtest by the two controls that change the output.
    return build_forecast(olympics, season, year)


try:
    olympics = get_data()
except (DataUnavailableError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.markdown(f"**Source:** [{ATHLETE_SOURCE}]({ATHLETE_SOURCE})")
    st.stop()


def metric_card(label: str, value: str | int, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def chart(fig: go.Figure, height: int = 390) -> None:
    fig.update_layout(**CHART_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, linecolor="#27384B", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(132,157,182,.13)", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


with st.sidebar:
    st.markdown('<div class="brand">OLYMPI<span>Q</span></div>', unsafe_allow_html=True)
    st.caption("OLYMPIC DATA · SIGNALS · STORIES")
    page = st.radio("Explore", ["Overview", "Medal trends", "Country explorer", "Breakthrough watch", "Predictions & backtesting", "Data & methodology"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="eyebrow">Dashboard filters</div>', unsafe_allow_html=True)
    seasons = st.multiselect("Games", ["Summer", "Winter"], default=["Summer", "Winter"])
    min_year, max_year = int(olympics.Year.min()), int(olympics.Year.max())
    year_range = st.slider("Edition year", min_year, max_year, (min_year, max_year), step=1)
    available_countries = sorted(olympics.Country.dropna().unique().tolist())
    countries = st.multiselect("Countries / NOCs", available_countries, default=[] , help="Leave empty to include all countries. Historic NOCs remain distinct.")
    available_sports = sorted(olympics.Sport.dropna().unique().tolist())
    sports = st.multiselect("Sports", available_sports, default=[])
    medals = st.multiselect("Medal categories", MEDAL_ORDER, default=MEDAL_ORDER)

filtered = olympics[
    olympics["Season"].isin(seasons)
    & olympics["Year"].between(year_range[0], year_range[1])
    & olympics["Sport"].isin(sports if sports else available_sports)
]
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if medals:
    filtered = filtered[filtered["Medal"].isin(medals) | filtered["Medal"].isna()]
else:
    filtered = filtered[filtered["Medal"].isna()]

st.markdown('<div class="eyebrow">A century of competition, made explorable</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>OLYMPIQ</h1><p>Explore Olympic history through event-aware medal analysis, country trajectories and transparent forecasts. Historical records end at Rio 2016; every estimate carries that context.</p></div>', unsafe_allow_html=True)


if page == "Overview":
    awards = medal_events(filtered)
    st.markdown('<div class="section-kicker">The field at a glance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("MEDALS IN VIEW", f"{len(awards):,}", "Event-level medal awards. Team roster members are counted once per event medal.")
    with c2: metric_card("COUNTRIES / NOCs", f"{filtered.NOC.nunique():,}", "Historic NOC codes stay separate across political changes.")
    with c3: metric_card("SPORTS", f"{filtered.Sport.nunique():,}")
    with c4: metric_card("EDITIONS", f"{filtered[['Games']].drop_duplicates().shape[0]:,}")
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Medal momentum")
        trends = trend_by_edition(filtered)
        if not trends.empty:
            fig = px.line(trends, x="Year", y="Total", color="Season", markers=True,
                          color_discrete_map={"Summer":"#45D6C3","Winter":"#8BA8FF"}, hover_data=["Gold","Silver","Bronze"])
            chart(fig)
        else: st.info("No medal records match these filters.")
    with right:
        st.subheader("Medal mix")
        mix = awards.groupby("Medal").size().reindex(MEDAL_ORDER, fill_value=0).rename_axis("Medal").reset_index(name="Awards")
        fig = px.pie(mix, names="Medal", values="Awards", hole=.67, color="Medal", color_discrete_map=COLORS)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        chart(fig, 390)
    st.subheader("Countries leading this view")
    leaders = medal_table(filtered).sort_values(["Gold", "Silver", "Bronze"], ascending=False).head(12)
    if not leaders.empty:
        leaders = leaders.rename(columns={"Country":"Country / NOC"})
        st.dataframe(leaders, width="stretch", hide_index=True)
    st.markdown('<div class="note">Medal totals count one award per NOC, sport, event, edition and medal type. Team-event rosters are de-duplicated; separate events and legitimate ties remain separate awards.</div>', unsafe_allow_html=True)


elif page == "Medal trends":
    st.markdown('<div class="section-kicker">Compare across time and disciplines</div>', unsafe_allow_html=True)
    trends = trend_by_edition(filtered)
    if not trends.empty:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Awards by Olympic edition")
            long = trends.melt(id_vars=["Year", "Season"], value_vars=MEDAL_ORDER, var_name="Medal", value_name="Awards")
            fig = px.bar(long, x="Year", y="Awards", color="Medal", facet_row="Season", barmode="stack", color_discrete_map=COLORS)
            chart(fig, 510)
        with col2:
            st.subheader("Country comparison")
            table = medal_table(filtered).sort_values(["Gold", "Silver", "Bronze"], ascending=False).head(18)
            long = table.melt(id_vars=["Country", "NOC"], value_vars=MEDAL_ORDER, var_name="Medal", value_name="Awards")
            fig = px.bar(long, x="Awards", y="Country", color="Medal", orientation="h", barmode="stack", color_discrete_map=COLORS)
            fig.update_layout(yaxis={"categoryorder":"total ascending"})
            chart(fig, 510)
        st.subheader("Sport medal analysis")
        sports_table = sport_table(filtered).sort_values("Total", ascending=False)
        sport_chart, sport_data = st.columns([1.1, 1])
        with sport_chart:
            top_sports = sports_table.head(14).sort_values("Total")
            fig = px.bar(top_sports, x="Total", y="Sport", orientation="h", color="Total",
                         color_continuous_scale=["#1C5261", "#45D6C3"], hover_data=MEDAL_ORDER)
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Event-level medal awards")
            chart(fig, 430)
        with sport_data:
            st.dataframe(sports_table, width="stretch", hide_index=True, height=430)
    else:
        st.info("No medal records match this filter combination.")


elif page == "Country explorer":
    st.markdown('<div class="section-kicker">Country lens · trajectories · sport opportunities</div>', unsafe_allow_html=True)
    country_options = sorted(olympics.Country.unique())
    default_country = countries[0] if len(countries) == 1 else ("United States" if "United States" in country_options else country_options[0])
    selected_country = st.selectbox("Choose a country label (underlying historic NOC codes are preserved)", country_options, index=country_options.index(default_country))
    season = st.selectbox("Season for trajectory", ["All", "Summer", "Winter"])
    trajectory_data = trajectory(olympics, selected_country, season)
    if not trajectory_data.empty:
        trajectory_data["Label"] = trajectory_data["Year"].astype(str) + " · " + trajectory_data["Season"]
        fig = px.area(trajectory_data, x="Year", y="Total", color_discrete_sequence=["#45D6C3"], markers=True,
                      hover_data=["Games", "Gold", "Silver", "Bronze"])
        fig.update_traces(line=dict(width=3), fillcolor="rgba(69,214,195,.13)")
        st.subheader("Olympic Performance Trajectory")
        chart(fig, 340)
        st.caption("Each point is one source edition in which this NOC participated. A zero means participants were recorded but no medal award was recorded.")
        st.subheader("Sport Opportunity Explorer")
        opportunities = sport_opportunities(olympics[olympics["Season"].isin(seasons)], selected_country)
        st.dataframe(opportunities.head(15), width="stretch", hide_index=True)
        st.markdown('<div class="note">Sports are ranked by historical medal awards and editions with medals. A high share is descriptive, not a causal advantage; it does not adjust for event availability, delegation size, or changing programs. “Opportunity” means an area to investigate, not a guarantee of future medals.</div>', unsafe_allow_html=True)
    else:
        st.info("No records found for the selected country and season.")


elif page == "Breakthrough watch":
    st.markdown('<div class="section-kicker">Positive historical slopes deserve a closer look</div>', unsafe_allow_html=True)
    season = st.selectbox("Season", ["All", "Summer", "Winter"])
    watch = breakthrough_watch(olympics, season)
    if not watch.empty:
        fig = px.bar(watch.sort_values("Trend per edition"), x="Trend per edition", y="Country", orientation="h",
                     color="Recent medals", color_continuous_scale=["#274C65", "#45D6C3"], hover_data=["NOC", "Editions"])
        st.subheader("Breakthrough Watch · improving medal trajectories")
        chart(fig, max(380, min(650, len(watch) * 32)))
        st.dataframe(watch, width="stretch", hide_index=True)
    else:
        st.info("No countries meet the minimum history and positive-trend rule for this season.")
    st.markdown('<div class="note">Calculation: count event-aware awards by NOC and edition, use each participating edition (including genuine zero-medal editions), fit a straight-line slope to up to the latest five editions, and show only positive slopes with at least four editions. This is retrospective trend screening, not a forecast or medal guarantee.</div>', unsafe_allow_html=True)


elif page == "Predictions & backtesting":
    st.markdown('<div class="section-kicker">Transparent estimates · time-based validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning"><b>Data-vintage warning.</b> The included dataset ends in 2016. The estimates below are long-range scenarios that do not incorporate Tokyo, Beijing, Paris, or current qualification and athlete information.</div>', unsafe_allow_html=True)
    season = st.selectbox("Forecast season", ["Summer", "Winter"])
    max_year = int(olympics.loc[olympics.Season.eq(season), "Year"].max())
    cadence = 4
    current_year = pd.Timestamp.now().year
    future_steps = max(1, (current_year - max_year) // cadence + 1)
    first_future = max(max_year + cadence * future_steps, max_year + cadence)
    future_year = st.number_input("Scenario year", min_value=max_year + cadence, max_value=2040, value=min(first_future, 2040), step=cadence)
    with st.spinner("Building chronological backtest and scenario estimates…"):
        bundle = cached_forecast(season, int(future_year))
    st.caption(bundle.note)
    if bundle.metrics.get("status") == "ok":
        metrics = bundle.metrics
        if bundle.selected_model.startswith("Last-edition"):
            st.info(f"Scenario method selected: **{bundle.selected_model}**. It had lower historical MAE than the Random Forest in rolling validation, so the app uses the simpler estimate for the scenario.")
        else:
            st.success("The Random Forest improved on the last-edition baseline in this rolling validation.")
        x1, x2, x3, x4 = st.columns(4)
        with x1: metric_card("MODEL MAE", f"{metrics['mae']:.2f}", "Mean absolute error: average medal-count distance from actual, per tested country-edition.")
        with x2: metric_card("LAST-EDITION BASELINE", f"{metrics['baseline_mae']:.2f}", "Predict each country repeats its previous edition's total.")
        with x3: metric_card("MODEL RMSE", f"{metrics['rmse']:.2f}", "Root mean squared error penalizes large misses more heavily than MAE.")
        with x4: metric_card("TEST EDITIONS", metrics["test_editions"], "Rolling chronological holdouts, never a random split.")
        st.subheader("Scenario medal estimates")
        estimates = bundle.forecast.head(20).sort_values("Estimated medals")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=estimates["Estimated medals"], y=estimates["Country"], orientation="h", name="Estimate", marker_color="#45D6C3",
                             error_x=dict(type="data", symmetric=False, array=estimates["Upper estimate"]-estimates["Estimated medals"],
                                          arrayminus=estimates["Estimated medals"]-estimates["Lower estimate"], color="#A4B7CA")))
        fig.update_layout(xaxis_title="Estimated event-level medals", yaxis_title="", showlegend=False)
        chart(fig, 510)
        st.caption("Bars show model estimates. Whiskers show a rough residual-based uncertainty band, not a calibrated confidence interval or a promise of outcomes.")
        st.subheader("Prediction vs actual · rolling historical holdouts")
        backtest = bundle.backtest
        edition_options = sorted(backtest.Year.unique())
        selected_edition = st.selectbox("Holdout edition", edition_options, index=len(edition_options)-1)
        comparison = backtest[backtest.Year.eq(selected_edition)].nlargest(15, "Actual")
        fig = px.scatter(comparison, x="Actual", y="Prediction", color="Country", hover_data=["NOC", "Baseline"],
                         labels={"Actual":"Actual event-level medals", "Prediction":"Predicted medals"})
        lim = max(float(comparison[["Actual", "Prediction"]].max().max()), 1)
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color="#F4BF54", dash="dash"))
        chart(fig, 400)
        st.dataframe(comparison.sort_values("Actual", ascending=False), width="stretch", hide_index=True)
        st.markdown(f'<div class="note">Candidate model: scikit-learn Random Forest using prior edition medal counts (three lags, three-edition mean and linear trend). Scenario method selected: {bundle.selected_model}. Validation trains on earlier editions only and rolls forward; the baseline repeats the last edition. A non-negative cap is applied. New NOCs without prior data are not guessed. Host effect is omitted because this source has host city but no dependable host-NOC field. Sport-specific strength is shown separately in the Country Explorer.</div>', unsafe_allow_html=True)
    else:
        st.warning(bundle.metrics.get("status", "Not enough data for a responsible backtest."))


else:
    st.markdown('<div class="section-kicker">Know what the numbers can and cannot say</div>', unsafe_allow_html=True)
    st.subheader("Dataset at a glance")
    overview = pd.DataFrame({"Item":["Source dataset","Rows","Coverage","File","Fields used","Primary unit"],
                             "Details":["Kaggle · 120 years of Olympic history: athletes and results",
                                        f"{len(olympics):,} cleaned athlete-event rows",
                                        f"{olympics.Year.min()}–{olympics.Year.max()} · Summer and Winter",
                                        "data/athlete_events.csv or olympics_raw.csv + noc_regions.csv",
                                        "Year, Season, NOC, region, Sport, Event, Medal, Games",
                                        "One athlete participating in one event"]})
    st.dataframe(overview, width="stretch", hide_index=True)
    st.markdown(f"Source page: [{ATHLETE_SOURCE}]({ATHLETE_SOURCE}) · The raw files are excluded from Git by `.gitignore`; this project uses the publicly reachable Kaggle data mirror and preserves the original CSVs locally.")
    st.subheader("Cleaning and comparison decisions")
    st.markdown("""
- Exact duplicate source rows are dropped; valid athlete appearances in separate events remain.
- Missing ages and body measurements are kept missing. Medal blanks remain non-medallists; unrecognized medal values never become medals.
- Sport labels are trimmed and normalized for spacing/case. The source's event names are retained.
- The NOC code is the identity used for grouping. Display labels come from `noc_regions.csv`; historical NOCs such as USSR, East Germany and modern successor NOCs remain distinct rather than being retrospectively merged.
- Team-event rosters are deduplicated at NOC + sport + event + edition + medal category, so a team medal counts once, while separate events and tied medal categories remain.
- Historical comparisons are affected by changing borders, qualification rules, event programs, delegation sizes, boycotts, wartime cancellations, medal reassignment, and record quality. Early editions and different sports are not directly comparable in all respects.
""")
    st.subheader("What the prediction does")
    st.markdown("""
The exploratory model predicts a country's next-edition total from its own previous three edition totals and their short-run trend. Backtesting walks forward through editions: every test prediction is trained only on earlier Games. We compare the model against a persistence baseline that repeats the latest observed total. MAE reports the typical absolute error in medals per country-edition; RMSE puts extra weight on large misses.

The residual band is a rough uncertainty aid, not a calibrated probability statement. The source ends at 2016; current athletes, qualifying fields, current results, host-NOC identity, budgets, sanctions, and program changes are unavailable. The app therefore labels its 2028/2030 output as a long-range scenario and should not be used as a current betting, funding, or selection forecast. A country with no recorded medal can legitimately remain at zero; a country with no prior history is not assigned invented results.
""")
    st.caption("OLYMPIQ is an independent educational project and is not affiliated with or endorsed by the International Olympic Committee. Olympic terminology is used descriptively.")

st.divider()
st.caption("OLYMPIQ · Historical records through 2016 · Estimates are not results")
