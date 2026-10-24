"""Chronological medal forecasting with an explicit persistence baseline.

Input contract
--------------
Pass the cleaned athlete-event DataFrame returned by ``data_loader.load_olympic_data``.
It must contain ``NOC``, ``Country``, ``Year``, ``Games``, ``Season`` and
``Medal``. Each source row is an athlete/event appearance; medal awards are
aggregated through ``analysis.medal_events`` before modeling.

``build_forecast`` returns a :class:`ForecastBundle`:

* ``forecast``: one row per NOC already observed in the selected season, with
  point estimate, a rough historical-error band, last actual, and horizon.
* ``backtest``: one row per eligible country/edition, with actual, model
  estimate, and last-edition baseline. New NOCs are omitted until they have
  history, so no identity from the future leaks into a past prediction.
* ``metrics``: status, MAE, baseline MAE, RMSE, and holdout counts. MAE is the
  average absolute medal-count error per country/edition row; RMSE penalizes
  larger errors more strongly. Neither is an exact-count guarantee.

All forecasts are estimates. Inputs do not include reliable host-NOC,
qualification, roster, event-program, or current-cycle information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from analysis import medal_events


@dataclass
class ForecastBundle:
    """Named result object returned by :func:`build_forecast`."""

    forecast: pd.DataFrame
    backtest: pd.DataFrame
    metrics: dict[str, float | int | str]
    note: str
    selected_model: str


def _model_class():
    try:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    except ImportError:
        return None


def _country_counts(data: pd.DataFrame, season: str) -> tuple[pd.DataFrame, dict[str, str], list[int], dict[str, int]]:
    rows = data[data["Season"].eq(season)]
    awards = medal_events(rows)
    editions = sorted(rows[["Year", "Games"]].drop_duplicates().to_records(index=False), key=lambda x: x[0])
    years = [int(y) for y, _ in editions]
    games = [str(g) for _, g in editions]
    if not games:
        return pd.DataFrame(), {}, [], {}
    known = rows[["NOC", "Country"]].drop_duplicates().drop_duplicates("NOC", keep="last")
    names = dict(zip(known["NOC"], known["Country"]))
    game_positions = {game: index for index, game in enumerate(games)}
    first_seen = rows.groupby("NOC")["Games"].agg(
        lambda values: min(game_positions[str(game)] for game in values)
    ).to_dict()
    matrix = awards.groupby(["Games", "NOC"]).size().unstack("NOC", fill_value=0)
    matrix = matrix.reindex(index=games, columns=sorted(names), fill_value=0)
    matrix.index.name = "Games"
    return matrix.astype(float), names, years, first_seen


REQUIRED_COLUMNS = {"NOC", "Country", "Year", "Games", "Season", "Medal"}


def _validate_input(data: pd.DataFrame, season: str) -> None:
    """Fail clearly for an incompatible table or unsupported season value."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame from load_olympic_data().")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Forecast input is missing required columns: " + ", ".join(sorted(missing))
        )
    if season not in {"Summer", "Winter"}:
        raise ValueError("season must be exactly 'Summer' or 'Winter'.")


def _features(history: list[float]) -> list[float]:
    last = history[-1] if history else 0.0
    prior = history[-2] if len(history) > 1 else 0.0
    third = history[-3] if len(history) > 2 else 0.0
    recent = history[-3:]
    trend = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
    return [last, prior, third, float(np.mean(recent)), trend]


def _training_rows(matrix: pd.DataFrame, stop: int, first_seen: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for target_index in range(3, stop):
        countries = [noc for noc in matrix.columns if first_seen.get(noc, stop) <= target_index]
        for noc in countries:
            history = matrix.iloc[:target_index][noc].tolist()
            xs.append(_features(history))
            ys.append(float(matrix.iloc[target_index][noc]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _fit_predict(matrix: pd.DataFrame, target_index: int, first_seen: dict[str, int]) -> dict[str, float]:
    """Train only on editions before target_index; return one-edition estimates."""
    model_class = _model_class()
    history_matrix = matrix.iloc[:target_index]
    known_nocs = [noc for noc in matrix.columns if first_seen.get(noc, target_index) < target_index]
    train_x, train_y = _training_rows(matrix, target_index, first_seen)
    if model_class is None or len(train_y) < 10:
        return {noc: max(0.0, float(history_matrix[noc].iloc[-1])) for noc in known_nocs}
    # Single-process trees avoid excessive joblib startup overhead in a small
    # Streamlit deployment while remaining reproducible and easy to explain.
    model = model_class(n_estimators=60, min_samples_leaf=2, max_features=1.0,
                        random_state=17, n_jobs=1)
    model.fit(train_x, train_y)
    return {noc: max(0.0, float(model.predict([_features(history_matrix[noc].tolist())])[0]))
            for noc in known_nocs}


def chronological_backtest(data: pd.DataFrame, season: str, min_train_editions: int = 8) -> tuple[pd.DataFrame, dict]:
    """Evaluate rolling-origin predictions against later editions, never shuffled.

    Returns ``(backtest, metrics)``. The backtest columns are ``Year``,
    ``Games``, ``NOC``, ``Country``, ``Actual``, ``Prediction``, and
    ``Baseline``. It scores countries seen before each target edition, including
    zero-medal outcomes for previously participating countries. Debut countries
    are excluded from that edition's score because no past-only estimate exists.

    ``MAE`` and ``RMSE`` are calculated across country-edition rows, not as the
    error in the total medal table. With too few editions, metrics contain a
    ``status`` message and no fabricated scores.
    """
    _validate_input(data, season)
    if min_train_editions < 3:
        raise ValueError("min_train_editions must be at least 3 to build lag features.")
    matrix, names, years, first_seen = _country_counts(data, season)
    if matrix.empty or len(matrix) <= min_train_editions:
        return pd.DataFrame(columns=["Year", "Games", "NOC", "Country", "Actual", "Prediction", "Baseline"]), {
            "status": "Insufficient editions for time-based validation.",
            "available_editions": int(len(matrix)),
            "required_editions": int(min_train_editions + 1),
        }
    observations = []
    # The test edition's NOC set uses country identities observed before the
    # test, so debut nations do not leak backward into model input.
    for target_index in range(min_train_editions, len(matrix)):
        preds = _fit_predict(matrix, target_index, first_seen)
        actual = matrix.iloc[target_index]
        for noc, predicted in preds.items():
            baseline = float(matrix[noc].iloc[target_index - 1])
            observations.append({"Year": years[target_index], "Games": matrix.index[target_index],
                                 "NOC": noc, "Country": names.get(noc, noc),
                                 "Actual": float(actual[noc]), "Prediction": predicted,
                                 "Baseline": baseline})
    result = pd.DataFrame(observations)
    if result.empty:
        return result, {"status": "No eligible backtest rows."}
    mae = float(np.mean(np.abs(result["Actual"] - result["Prediction"])))
    baseline_mae = float(np.mean(np.abs(result["Actual"] - result["Baseline"])))
    rmse = float(np.sqrt(np.mean((result["Actual"] - result["Prediction"]) ** 2)))
    return result, {"status": "ok", "mae": mae, "baseline_mae": baseline_mae,
                    "rmse": rmse, "test_editions": int(result["Games"].nunique()),
                    "test_rows": int(len(result)),
                    "estimator_available": _model_class() is not None}


def build_forecast(data: pd.DataFrame, season: str, target_year: int | None = None) -> ForecastBundle:
    """Forecast iteratively to a future year; long horizons are labeled clearly.

    An 80th-percentile historical one-step absolute residual supplies a simple
    uncertainty band which expands with the square root of the forecast horizon.
    It is a rough heuristic, not a calibrated probability interval.

    Parameters
    ----------
    data:
        Cleaned athlete-event table from ``load_olympic_data``.
    season:
        ``"Summer"`` or ``"Winter"``.
    target_year:
        Optional requested future year. It is rounded forward to the selected
        season's four-year cadence after the latest source edition.

    Returns
    -------
    ForecastBundle
        ``forecast`` columns: ``NOC``, ``Country``, ``Estimated medals``,
        ``Lower estimate``, ``Upper estimate``, ``Most recent actual``, and
        ``Editions since source's latest actual``. A rough band is omitted as a
        calibrated-confidence claim; it is only a historical residual heuristic.
    """
    _validate_input(data, season)
    matrix, names, years, first_seen = _country_counts(data, season)
    backtest, metrics = chronological_backtest(data, season)
    if matrix.empty or not years:
        return ForecastBundle(pd.DataFrame(), backtest, metrics, "No data exists for this season.", "Unavailable")
    if _model_class() is None:
        selected_model = "Last-edition baseline (scikit-learn is unavailable)"
    elif metrics.get("status") != "ok":
        selected_model = "Last-edition baseline (chronological validation is insufficient)"
    elif metrics.get("status") == "ok" and metrics["mae"] > metrics["baseline_mae"]:
        selected_model = "Last-edition baseline (Random Forest did not beat it)"
    else:
        selected_model = "Random Forest"
    cadence = 4
    # Do not present an already elapsed Olympic edition as a future forecast.
    # Summer and Winter editions each follow a four-year cadence, but their
    # calendar-year offsets differ; advancing from the source's own last year
    # preserves that season-specific alignment.
    next_edition_year = years[-1] + cadence
    while next_edition_year <= date.today().year:
        next_edition_year += cadence
    target_year = max(int(target_year or next_edition_year), next_edition_year)
    steps = max(1, int(np.ceil((target_year - years[-1]) / cadence)))
    # Keep the output on this season's four-year edition cadence.
    target_year = years[-1] + steps * cadence
    extended = matrix.copy()
    for step in range(steps):
        if selected_model.startswith("Last-edition"):
            estimates = {noc: float(extended[noc].iloc[-1]) for noc in extended.columns}
        else:
            estimates = _fit_predict(extended, len(extended.index), first_seen)
        next_year = years[-1] + cadence * (step + 1)
        next_games = f"{next_year} {season} (estimate)"
        extended.loc[next_games] = pd.Series(estimates)
    projected = extended.iloc[-1]

    if not backtest.empty:
        chosen_column = "Baseline" if selected_model.startswith("Last-edition") else "Prediction"
        backtest = backtest.copy()
        backtest["Absolute error"] = (backtest["Actual"] - backtest[chosen_column]).abs()
        residual_band = float(backtest["Absolute error"].quantile(0.80))
        country_bands = backtest.groupby("NOC").agg(
            residual_q80=("Absolute error", lambda values: float(values.quantile(0.80))),
            observations=("Absolute error", "size"),
        ).to_dict("index")
    else:
        residual_band = float(np.std(matrix.to_numpy())) if matrix.size else 0.0
    rows = []
    for noc, estimate in projected.items():
        history = matrix[noc].to_numpy(dtype=float)
        # unseen/zero-medal countries remain in the panel and are not assigned a
        # made-up medal; the model can return a legitimate zero.
        local = country_bands.get(noc, {})
        one_step_band = local.get("residual_q80", residual_band) if local.get("observations", 0) >= 3 else residual_band
        horizon_band = max(float(one_step_band), residual_band) * np.sqrt(steps)
        rows.append({"NOC": noc, "Country": names.get(noc, noc), "Estimated medals": estimate,
                     "Lower estimate": max(0.0, estimate - horizon_band),
                     "Upper estimate": estimate + horizon_band,
                     "Most recent actual": float(history[-1]),
                     "Editions since source's latest actual": steps})
    forecast = pd.DataFrame(rows).sort_values("Estimated medals", ascending=False)
    note = (
        f"Training history ends in {years[-1]}. This is a long-range scenario for {target_year} "
        f"({steps} edition step(s) beyond the source), not a current-cycle prediction. "
        "Host effects, rosters, qualification, budgets, sanctions, sport-program changes, "
        "injuries, and current results are unavailable and are not modeled."
    )
    return ForecastBundle(forecast, backtest, metrics, note, selected_model)
