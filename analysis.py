"""Reusable medal aggregation and hackathon insight calculations."""

from __future__ import annotations

import pandas as pd

MEDAL_ORDER = ["Gold", "Silver", "Bronze"]


def medal_events(data: pd.DataFrame) -> pd.DataFrame:
    """One medal per NOC/event/category/edition, avoiding team-roster inflation.

    Olympic team events award a single medal to a team, while the raw source has
    one row for each team member. Deduplicating on this event key avoids counting
    every athlete as a separate medal. Distinct events and medal categories stay.
    """
    winners = data[data["Medal"].isin(MEDAL_ORDER)].copy()
    key = ["Games", "Year", "Season", "NOC", "Country", "Sport", "Event", "Medal"]
    return winners.drop_duplicates(subset=key)


def medal_table(data: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Pivot event-level medal awards into gold/silver/bronze/total columns."""
    group_by = group_by or ["Country", "NOC"]
    awards = medal_events(data)
    if awards.empty:
        return pd.DataFrame(columns=group_by + MEDAL_ORDER + ["Total"])
    counts = awards.groupby(group_by + ["Medal"], dropna=False).size().unstack("Medal", fill_value=0)
    counts = counts.reindex(columns=MEDAL_ORDER, fill_value=0)
    counts["Total"] = counts.sum(axis=1)
    return counts.reset_index()


def trend_by_edition(data: pd.DataFrame) -> pd.DataFrame:
    awards = medal_events(data)
    if awards.empty:
        return pd.DataFrame(columns=["Year", "Season", "Gold", "Silver", "Bronze", "Total"])
    result = awards.groupby(["Year", "Season", "Medal"]).size().unstack("Medal", fill_value=0)
    result = result.reindex(columns=MEDAL_ORDER, fill_value=0)
    result["Total"] = result.sum(axis=1)
    return result.reset_index().sort_values(["Season", "Year"])


def sport_table(data: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    subset = data if not country or country == "All countries" else data[data["Country"].eq(country)]
    return medal_table(subset, ["Sport"])


def trajectory(data: pd.DataFrame, country: str, season: str = "All") -> pd.DataFrame:
    """Country medal history with true zeros after its first recorded appearance."""
    rows = data[data["Country"].eq(country)]
    if season != "All":
        rows = rows[rows["Season"].eq(season)]
    if rows.empty:
        return pd.DataFrame(columns=["Games", "Year", "Season", "Gold", "Silver", "Bronze", "Total"])
    editions = rows[["Games", "Year", "Season"]].drop_duplicates().sort_values(["Year", "Season"])
    awards = medal_events(rows)
    per_game = awards.groupby(["Games", "Year", "Season", "Medal"]).size().unstack("Medal", fill_value=0)
    per_game = per_game.reindex(columns=MEDAL_ORDER, fill_value=0)
    per_game["Total"] = per_game.sum(axis=1)
    return editions.merge(per_game.reset_index(), how="left", on=["Games", "Year", "Season"]).fillna(0)


def breakthrough_watch(data: pd.DataFrame, season: str = "All", min_recent_medals: int = 1) -> pd.DataFrame:
    """Rank countries with positive three-edition least-squares medal slopes.

    A positive fitted slope is a retrospective signal, not a probability of a
    future medal. Countries need at least four recorded editions to qualify.
    """
    source = data if season == "All" else data[data["Season"].eq(season)]
    rows = source
    awards = medal_events(source)
    if awards.empty:
        return pd.DataFrame(columns=["Country", "NOC", "Recent medals", "Trend per edition", "Editions"])
    totals = awards.groupby(["NOC", "Country", "Games", "Year", "Season"]).size().reset_index(name="Total")
    editions = source[["Games", "Year", "Season"]].drop_duplicates().sort_values(["Year", "Season"])
    countries = source[["NOC", "Country"]].drop_duplicates()
    output = []
    for _, country_row in countries.iterrows():
        history = totals[totals["NOC"].eq(country_row["NOC"])].set_index("Games")["Total"]
        participated = rows[rows["NOC"].eq(country_row["NOC"])]["Games"].unique().tolist()
        country_editions = [game for game in editions["Games"].tolist() if game in participated]
        if len(country_editions) < 4:
            continue
        values = [int(history.get(game, 0)) for game in country_editions]
        recent = values[-5:]
        if len(recent) < 4 or sum(recent[-2:]) < min_recent_medals:
            continue
        # Polyfit uses edition position, not calendar year: gaps do not imply a
        # uniform Olympic-year distance and all entries represent one edition.
        import numpy as np
        slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
        if slope > 0:
            output.append({"Country": country_row["Country"], "NOC": country_row["NOC"],
                           "Recent medals": recent[-1], "Trend per edition": slope,
                           "Editions": len(history)})
    return pd.DataFrame(output).sort_values("Trend per edition", ascending=False).head(15) if output else pd.DataFrame(
        columns=["Country", "NOC", "Recent medals", "Trend per edition", "Editions"]
    )


def sport_opportunities(data: pd.DataFrame, country: str) -> pd.DataFrame:
    """Show sports where the NOC has repeated results and a strong medal share."""
    rows = data[data["Country"].eq(country)]
    if rows.empty:
        return pd.DataFrame(columns=["Sport", "Medals", "Editions with medals", "Share of this sport's medals", "Readout"])
    awards = medal_events(rows)
    all_awards = medal_events(data)
    country_counts = awards.groupby("Sport").size().rename("Medals")
    appearances = awards.groupby("Sport")["Games"].nunique().rename("Editions with medals")
    global_counts = all_awards.groupby("Sport").size().rename("All countries medals")
    result = pd.concat([country_counts, appearances, global_counts], axis=1).fillna(0)
    result["Share of this sport's medals"] = result["Medals"] / result["All countries medals"].replace(0, 1)
    result["Readout"] = result.apply(
        lambda r: "Repeated strength" if r["Editions with medals"] >= 3 else "Early signal; small sample",
        axis=1,
    )
    return result.reset_index().drop(columns="All countries medals").sort_values(
        ["Editions with medals", "Medals"], ascending=False
    )
