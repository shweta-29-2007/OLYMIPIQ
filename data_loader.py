"""Load, validate, and clean the public Olympic athlete-event dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ATHLETE_FILE = DATA_DIR / "athlete_events.csv"
ALTERNATE_ATHLETE_FILE = DATA_DIR / "olympics_raw.csv"
NOC_FILE = DATA_DIR / "noc_regions.csv"

ATHLETE_SOURCE = "https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results"
ATHLETE_MIRROR = "https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/athlete_events.csv"
NOC_MIRROR = "https://raw.githubusercontent.com/rgriff23/Olympic_history/master/data/noc_regions.csv"

REQUIRED_COLUMNS = {
    "ID", "Name", "Sex", "Age", "Height", "Weight", "Team", "NOC",
    "Games", "Year", "Season", "City", "Sport", "Event", "Medal",
}

# These are alternate spellings found in other public Olympic tables. The
# original source labels are mostly already canonical; this map is deliberately
# small so it does not erase meaningful historical sport/event distinctions.
SPORT_ALIASES = {
    "synchronized swimming": "Synchronized Swimming",
    "taekwondo": "Taekwondo",
}
MEDAL_ALIASES = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze"}

# The source's region lookup intentionally maps some historical NOCs to a
# modern region. Keep their historical identity visible in the dashboard.
HISTORIC_NOC_LABELS = {
    "URS": "Soviet Union", "GDR": "East Germany", "FRG": "West Germany",
    "YUG": "Yugoslavia", "SCG": "Serbia and Montenegro", "EUN": "Unified Team",
    "TCH": "Czechoslovakia", "BOH": "Bohemia", "ANZ": "Australasia",
    "AHO": "Netherlands Antilles", "ROC": "Russian Olympic Committee",
    "USA": "United States", "GBR": "Great Britain",
}


class DataUnavailableError(FileNotFoundError):
    """Raised when the required public CSV is not present in the project."""


def _normalize_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def _load_region_map(path: Path) -> dict[str, str]:
    """Read the optional Kaggle NOC lookup. Unknown codes stay visible as codes."""
    if not path.exists():
        return {}
    lookup = pd.read_csv(path, dtype="string", keep_default_na=False)
    lookup.columns = [str(column).strip().lower() for column in lookup.columns]
    if not {"noc", "region"}.issubset(lookup.columns):
        return {}
    lookup["noc"] = _normalize_text(lookup["noc"]).str.upper()
    lookup["region"] = _normalize_text(lookup["region"])
    lookup = lookup[(lookup["noc"] != "") & (lookup["region"] != "")]
    return lookup.drop_duplicates("noc", keep="first").set_index("noc")["region"].to_dict()


def load_olympic_data(
    athlete_path: str | Path = ATHLETE_FILE,
    noc_path: str | Path = NOC_FILE,
) -> pd.DataFrame:
    """Return a cleaned athlete-level DataFrame with a stable ``Country`` label.

    We preserve original NOC codes as the analysis identity. A region/name lookup
    only improves display labels; it never merges historic NOCs or successor
    states. Exact duplicate source rows are removed, but valid repeated athlete
    appearances across distinct events remain.
    """
    athlete_path = Path(athlete_path)
    if not athlete_path.exists() and athlete_path == ATHLETE_FILE and ALTERNATE_ATHLETE_FILE.exists():
        athlete_path = ALTERNATE_ATHLETE_FILE
    if not athlete_path.exists():
        raise DataUnavailableError(
            f"Historical data not found at {athlete_path}. Download the real "
            f"athlete_events.csv from {ATHLETE_SOURCE} (or the public mirror "
            f"{ATHLETE_MIRROR}) and place it in the project's data folder. "
            "Do not use generated or sample records."
        )

    frame = pd.read_csv(athlete_path, low_memory=False)
    column_names = {column.lower(): column for column in REQUIRED_COLUMNS}
    frame.columns = [
        column_names.get(str(column).strip().replace('"', "").lower(), str(column).strip().replace('"', ""))
        for column in frame.columns
    ]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Olympic CSV is missing required columns: {', '.join(sorted(missing))}")

    frame = frame.drop_duplicates().copy()
    for column in ("Team", "NOC", "Games", "Season", "City", "Sport", "Event", "Medal"):
        frame[column] = _normalize_text(frame[column])

    frame["NOC"] = frame["NOC"].str.upper()
    frame["Season"] = frame["Season"].str.title()
    frame["Medal"] = frame["Medal"].str.lower().map(MEDAL_ALIASES)
    frame["Sport"] = frame["Sport"].str.lower().map(SPORT_ALIASES).fillna(frame["Sport"])
    frame["Sport"] = frame["Sport"].str.strip().str.title()
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame = frame.dropna(subset=["Year", "NOC", "Season", "Sport", "Event"])
    frame["Year"] = frame["Year"].astype(int)
    frame = frame[frame["Season"].isin(["Summer", "Winter"]) & frame["NOC"].ne("")]

    # Empty source cells are missing data, not a medal category.
    frame["Medal"] = frame["Medal"].where(frame["Medal"].isin(["Gold", "Silver", "Bronze"]))
    region_map = _load_region_map(Path(noc_path))
    region_map.update(HISTORIC_NOC_LABELS)
    frame["Country"] = frame["NOC"].map(region_map).fillna("NOC "+frame["NOC"])
    # If two codes still share a display label, append the code so selections
    # remain unambiguous (the NOC remains the true grouping identity either way).
    duplicate_labels = frame.groupby("Country")["NOC"].nunique()
    shared = set(duplicate_labels[duplicate_labels > 1].index)
    frame.loc[frame["Country"].isin(shared), "Country"] = (
        frame.loc[frame["Country"].isin(shared), "Country"]
        + " (" + frame.loc[frame["Country"].isin(shared), "NOC"] + ")"
    )

    # Games is retained from source where valid. This fallback supports CSVs
    # with a missing Games label without inventing editions.
    frame["Games"] = frame["Games"].fillna("").astype("string").str.strip()
    blank_games = frame["Games"].eq("")
    frame.loc[blank_games, "Games"] = (
        frame.loc[blank_games, "Year"].astype(str) + " " + frame.loc[blank_games, "Season"]
    )
    return frame.reset_index(drop=True)


def available_years(data: pd.DataFrame, season: str | None = None) -> list[int]:
    """Sorted Olympic years available after optional season filtering."""
    subset = data if season in (None, "All") else data[data["Season"].eq(season)]
    return sorted(subset["Year"].dropna().astype(int).unique().tolist())


def filter_rows(
    data: pd.DataFrame,
    years: Iterable[int] | None = None,
    countries: Iterable[str] | None = None,
    sports: Iterable[str] | None = None,
    seasons: Iterable[str] | None = None,
    medals: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply optional dashboard filters while keeping non-medallists if asked."""
    result = data
    for column, values in (("Year", years), ("Country", countries), ("Sport", sports), ("Season", seasons)):
        if values:
            result = result[result[column].isin(values)]
    if medals:
        result = result[result["Medal"].isin(medals)]
    return result
