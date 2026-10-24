"""Country-level Olympic medal forecasting for OLYMPIQ.

This module is deliberately transparent and dependency-light: it uses only
NumPy and Pandas, so it works without scikit-learn or statsmodels.  It never
invents accuracy numbers.  ``forecast_medals`` computes every metric from the
data you give it, and raises a clear error when the data cannot support a
chronological validation.

WHAT IT DOES
------------
1. Converts either raw athlete/event rows (the Kaggle "120 years of Olympic
   history" shape) or an already aggregated country/edition table into one row
   per (season, year, NOC).
2. Builds features using only *earlier* Olympic editions.  The forecast is
   anchored on the country's most recent medal total (a Poisson offset), and
   the model learns bounded adjustments around that anchor:

   ======================  ==============================================
   feature                 meaning
   ======================  ==============================================
   momentum_last_vs_mean   (last total - last-3 mean) / (last total + 1):
                           is the country above or below its recent level?
   trend_ratio             least-squares medals-per-edition slope over the
                           last four editions / (last-3 mean + 1)
   depth_ratio             log1p(best of last 3) - log1p(last total):
                           recent upside not visible in the last edition
   gold_share3             share of recent medals that were gold
   n_prior                 number of prior editions the country attended
   log_last                log1p(last total); occurrence stage zero signal
   was_zero_last           1 if the country won no medal last edition
   host                    1 if the country hosts the edition being predicted
   ======================  ==============================================

   The offset is ``log(prev_total + 1)``.  This matters: a plain log-link
   model on raw medal counts extrapolates exponentially for large teams
   (e.g. hundreds of medals for the USA).  Anchoring on the previous total
   keeps forecasts in a plausible range while still letting recent momentum
   and host status adjust them.  Skipped/missing editions count as zero.
   No feature for edition Y uses data from Y or later.
3. Fits a transparent two-part **hurdle model**:

   * **occurrence stage** - ridge logistic regression estimating
     ``P(medal > 0)`` on every training row, using ``log_last``,
     ``was_zero_last``, momentum, trend, depth, prior appearances and
     (optionally) gold share and host status;
   * **count stage** - ridge Poisson regression with a log link and the
     previous-total offset, fitted only on medal-winning rows, estimating
     ``E[medals | medals > 0]``.

   The forecast is ``P(medal > 0) x E[medals | medal > 0]``.  Splitting the
   model matters because most NOCs win nothing: a single count model spends its
   intercept on zero-medal countries and under-predicts the medal table.  The
   host flag is estimated from historical data in both stages (not hard-coded
   as a medal bonus).
4. Validates chronologically.  For every test edition the model is fitted only
   on earlier editions; there is no random split that leaks the future.
5. Compares against simple historical baselines: previous-edition medals
   ("persistence") and the mean of the previous three editions.
6. Reports MAE, RMSE, median absolute error, bias, MAE split by zero-medal and
   medalling countries, zero-medal false-positive rate, occurrence-stage Brier
   score, Spearman rank correlation and top-10 overlap.  Pooled Spearman and
   top-10 values are the mean of the per-edition values: ranking rows mixed from
   several editions in one table would not be meaningful.  See
   ``METRIC_DEFINITIONS`` / ``explain_metrics``.
7. Returns documented Pandas DataFrames suitable for ``app.py``.

DATA CONTRACT (coordinate this with the data_loader owner)
----------------------------------------------------------
``load_history()`` accepts either of these two shapes.  Column names are
matched case-insensitively and a few common aliases are accepted.

A. Raw athlete/event level (preferred, because it contains zero-medal NOCs)::

       NOC, Year, Season, Event, Medal, Team, City
       USA, 2016, Summer, "Basketball Men's Basketball", Gold, United States, Rio de Janeiro
       USA, 2016, Summer, "Basketball Men's Basketball", Gold, United States, Rio de Janeiro
       ...              # one row per athlete; team medals are de-duplicated

   Required: ``NOC``, ``Year``, ``Season``, ``Event``, ``Medal``.
   ``Medal`` is Gold/Silver/Bronze (NaN for non-medallists).  ``Team`` and
   ``City`` are optional.  ``Event`` is needed to count a team medal once per
   NOC rather than once per athlete.

B. Aggregated country/edition level::

       noc, year, season, total, gold, silver, bronze, country, host
       USA, 2016, Summer, 121, 46, 37, 38, United States, 0

   Required: ``noc``, ``year``, ``season`` and either ``total`` or the three
   colour columns.  ``country`` and ``host`` are optional (host defaults to a
   documented factual mapping of host NOCs).

   IMPORTANT: if shape B contains only medalling countries, MAE is optimistic
   because zero-medal countries are missing.  Include one row per participating
   NOC with ``total=0``, or pass a ``participants`` DataFrame with
   ``noc, year, season`` to ``prepare_country_edition_panel``.

   Coordination note (Person 2 / ``data_loader.py``): ``load_data()`` returns
   shape A, while ``OlympicDataLoader().get_country_medal_summary()`` returns
   shape B and already includes zero-medal participants.  Both are normalised to
   the same panel; ``test_prediction.RealDatasetIntegrationTests`` checks that
   their medal counts agree exactly, so the UI and analytics layers can use
   either source.

The output panel (``prepare_country_edition_panel``) always has:

    season, year, noc, country, gold, silver, bronze, total, host

PUBLIC API
----------
``forecast_medals(...) -> ForecastResult``
    Main entry point.  ``result.predictions``, ``result.backtest``,
    ``result.metrics``, ``result.model_card``, ``result.data_quality`` and
    ``result.notes`` are DataFrames / lists ready for display.

``prepare_country_edition_panel(data, ...) -> DataFrame``
    Normalise raw or aggregated input to the documented panel schema.

``build_feature_frame(panel, season, target_year=None) -> DataFrame``
    Leak-free feature table (also useful for tests and inspection).

``chronological_backtest(...) -> BacktestResult``
    Expanding-window validation over the last N editions.

``compute_metrics(frame) -> dict``
    Metrics for one fold or pooled folds; usable by app.py and tests.

``data_quality_report(panel) -> DataFrame``
    Sufficiency and coverage diagnostics **before** forecasting.

``load_history(data_path=None) -> DataFrame``
    Read a CSV, or call ``data_loader.load_data()`` if present.

``describe_schema() -> DataFrame`` and ``explain_metrics() -> DataFrame``
    Machine-readable documentation for the team/UI.

TYPICAL USE
-----------
>>> from prediction import forecast_medals, load_history
>>> history = load_history()                       # data_loader contract
>>> result = forecast_medals(history, season="Summer", target_year=2028)
>>> result.predictions.head(20)
>>> result.metrics

If the data are unavailable or insufficient, a ``OlympicDataError`` /
``InsufficientDataError`` is raised with an actionable message.  Nothing is
silently substituted with synthetic data.

LIMITATIONS (read before presenting results)
--------------------------------------------
* Predictions are estimates, not guaranteed medal counts.  Sport-level shocks
  (a star athlete's injury, a new event programme, doping bans, boycotts,
  political changes, COVID-style schedule changes) are not in the features.
* NOC codes are not merged across political eras.  Historical teams such as
  URS, GDR, EUN, TCH and YUG keep their own history and are naturally absent
  from future candidate lists.  This makes successor states hard to forecast.
* The model predicts total medals; ``predicted_gold`` is only a recent
  historical gold share applied to that total, not a separately validated model.
* The 80% intervals are approximate residual intervals.  They use the
  dispersion of chronological backtest errors and do not include model-
  selection or structural-break uncertainty.
* Host effects are estimated from a small number of host editions, so they are
  uncertain.  The coefficient and standard error are reported.  The host flag is
  kept on its natural 0/1 scale (not standardised) so its coefficient is a direct
  0-to-1 effect rather than an artefact of how rarely countries host.
* The backtest is the only evidence of quality.  If the model does not beat
  the persistence baseline, say so; the code reports that comparison honestly.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ForecastResult",
    "BacktestResult",
    "OlympicDataError",
    "InsufficientDataError",
    "forecast_medals",
    "prepare_country_edition_panel",
    "build_feature_frame",
    "chronological_backtest",
    "compute_metrics",
    "data_quality_report",
    "load_history",
    "describe_schema",
    "explain_metrics",
    "METRIC_DEFINITIONS",
    "HOST_NOC_BY_EDITION",
]

MODEL_VERSION = "olympiq-hurdle-poisson-ridge-v1"

SUMMER = "Summer"
WINTER = "Winter"

MEDAL_COLOURS = ("gold", "silver", "bronze")

#: Factual host NOC for every Olympic edition (both seasons).  Host status is
#: known years before the Games, so it is a legitimate prediction-time feature.
#: 1906 is intentionally not listed: the Intercalated Games are not recognised
#: by the IOC as an Olympic edition and are excluded from this analysis.
HOST_NOC_BY_EDITION: dict[tuple[str, int], str] = {
    # Summer Games
    (SUMMER, 1896): "GRE",
    (SUMMER, 1900): "FRA",
    (SUMMER, 1904): "USA",
    (SUMMER, 1908): "GBR",
    (SUMMER, 1912): "SWE",
    (SUMMER, 1920): "BEL",
    (SUMMER, 1924): "FRA",
    (SUMMER, 1928): "NED",
    (SUMMER, 1932): "USA",
    (SUMMER, 1936): "GER",
    (SUMMER, 1948): "GBR",
    (SUMMER, 1952): "FIN",
    (SUMMER, 1956): "AUS",   # Melbourne; equestrian events were in Stockholm
    (SUMMER, 1960): "ITA",
    (SUMMER, 1964): "JPN",
    (SUMMER, 1968): "MEX",
    (SUMMER, 1972): "FRG",   # West Germany in this dataset
    (SUMMER, 1976): "CAN",
    (SUMMER, 1980): "URS",
    (SUMMER, 1984): "USA",
    (SUMMER, 1988): "KOR",
    (SUMMER, 1992): "ESP",
    (SUMMER, 1996): "USA",
    (SUMMER, 2000): "AUS",
    (SUMMER, 2004): "GRE",
    (SUMMER, 2008): "CHN",
    (SUMMER, 2012): "GBR",
    (SUMMER, 2016): "BRA",
    (SUMMER, 2020): "JPN",
    (SUMMER, 2024): "FRA",
    (SUMMER, 2028): "USA",
    (SUMMER, 2032): "AUS",
    # Winter Games
    (WINTER, 1924): "FRA",
    (WINTER, 1928): "SUI",
    (WINTER, 1932): "USA",
    (WINTER, 1936): "GER",
    (WINTER, 1948): "SUI",
    (WINTER, 1952): "NOR",
    (WINTER, 1956): "ITA",
    (WINTER, 1960): "USA",
    (WINTER, 1964): "AUT",
    (WINTER, 1968): "FRA",
    (WINTER, 1972): "JPN",
    (WINTER, 1976): "AUT",
    (WINTER, 1980): "USA",
    (WINTER, 1984): "YUG",   # Sarajevo
    (WINTER, 1988): "CAN",
    (WINTER, 1992): "FRA",
    (WINTER, 1994): "NOR",
    (WINTER, 1998): "JPN",
    (WINTER, 2002): "USA",
    (WINTER, 2006): "ITA",
    (WINTER, 2010): "CAN",
    (WINTER, 2014): "RUS",
    (WINTER, 2018): "KOR",
    (WINTER, 2022): "CHN",
    (WINTER, 2026): "ITA",
}

#: Editions excluded with a reason.  1906 = Intercalated Games.
EXCLUDED_EDITIONS: dict[tuple[str, int], str] = {
    (SUMMER, 1906): "1906 Intercalated Games are not a recognised Olympic edition",
}

#: Default original feature columns.  These are bounded/relative features
#: applied on top of the ``log(prev_total + 1)`` offset.  ``gold_share3`` is
#: dropped automatically if the input has no gold-medal information.
BASE_FEATURES: tuple[str, ...] = (
    "momentum_last_vs_mean",
    "trend_ratio",
    "depth_ratio",
    "gold_share3",
    "n_prior",
)
HOST_FEATURE = "host"

#: Features for the occurrence ("does this NOC win any medal?") stage.  The
#: count stage only sees medal-winning rows, so the occurrence stage needs the
#: explicit zero signal: ``was_zero_last`` and ``log_last`` distinguish a team
#: that medalled last time from one that did not, and ``depth_ratio`` allows a
#: returning team with earlier medals to recover.  ``gold_share3`` and ``host``
#: are appended only when those columns exist in the model feature list.
OCCURRENCE_FEATURES: tuple[str, ...] = (
    "log_last",
    "was_zero_last",
    "momentum_last_vs_mean",
    "trend_ratio",
    "depth_ratio",
    "n_prior",
)

#: Offset anchor: log(prev_total + OFFSET_SHRINK).  The small constant keeps
#: zero-medal countries and debutants finite.
OFFSET_SHRINK = 1.0

#: Sparse binary features are not standardised.  The host flag is 1 for only a
#: handful of historical rows, so its standard deviation is tiny; standardising
#: it would turn a modest 0->1 change into many "standard deviations" and make
#: both the reported effect and the ridge penalty depend on how often a country
#: happened to host.  Keeping it on its natural 0/1 scale makes the coefficient
#: directly interpretable as the host log-effect (and keeps the penalty honest).
UNSCALED_FEATURES: frozenset[str] = frozenset({HOST_FEATURE})

#: Realistic minimum for a chronological split: at least this many prior
#: editions to train on, plus at least one later edition to test on.
MIN_TRAIN_EDITIONS = 4

#: Recent editions used to decide which NOCs are plausible future entrants.
#: This removes defunct teams (URS, GDR, EUN, ...) without looking ahead.
DEFAULT_ACTIVE_WINDOW = 2

#: Post-war/modern era default.  Pre-1960 editions had very different
#: participation and medal distributions; this is a training-window choice only
#: (features for a target edition still use all available earlier editions).
DEFAULT_MIN_HISTORY_YEAR = 1960

#: Approximate 80% normal quantile used for residual intervals.
Z80 = 1.2815515655446004

#: The count stage is fitted on medal-winning rows that have a prior total to
#: anchor on.  First-time/successor-state medal hauls (prev_total == 0) would
#: otherwise inflate the count intercept and over-predict established teams.
#: The fallback keeps tiny synthetic/test panels usable.
MIN_ANCHORED_COUNT_ROWS = 10


class OlympicDataError(ValueError):
    """Raised when the supplied Olympic data cannot be used."""


class InsufficientDataError(OlympicDataError):
    """Raised when there are too few editions for a chronological validation."""


# ---------------------------------------------------------------------------
# Schema documentation
# ---------------------------------------------------------------------------

SCHEMA_ROWS: list[dict[str, Any]] = [
    # raw athlete/event input
    dict(table="raw_athlete_events", column="NOC", required=True, dtype="str",
         description="Three-letter country/NOC code (e.g. USA)."),
    dict(table="raw_athlete_events", column="Year", required=True, dtype="int",
         description="Edition year (e.g. 2016)."),
    dict(table="raw_athlete_events", column="Season", required=True, dtype="str",
         description="'Summer' or 'Winter'."),
    dict(table="raw_athlete_events", column="Event", required=True, dtype="str",
         description="Event name; used to count a team medal once per NOC."),
    dict(table="raw_athlete_events", column="Medal", required=True, dtype="str",
         description="Gold/Silver/Bronze, or NaN for non-medallists."),
    dict(table="raw_athlete_events", column="Team", required=False, dtype="str",
         description="Country/team display name (optional)."),
    dict(table="raw_athlete_events", column="City", required=False, dtype="str",
         description="Host city (optional; host NOC comes from a documented map)."),
    # aggregated country/edition input
    dict(table="country_edition_panel", column="noc", required=True, dtype="str",
         description="Three-letter country/NOC code."),
    dict(table="country_edition_panel", column="year", required=True, dtype="int",
         description="Edition year."),
    dict(table="country_edition_panel", column="season", required=True, dtype="str",
         description="'Summer' or 'Winter'."),
    dict(table="country_edition_panel", column="total", required="one of",
         dtype="float", description="Total medals; or supply gold+silver+bronze."),
    dict(table="country_edition_panel", column="gold", required=False, dtype="float",
         description="Gold medals (needed only for predicted_gold)."),
    dict(table="country_edition_panel", column="silver", required=False, dtype="float",
         description="Silver medals."),
    dict(table="country_edition_panel", column="bronze", required=False, dtype="float",
         description="Bronze medals."),
    dict(table="country_edition_panel", column="country", required=False, dtype="str",
         description="Display name; defaults to NOC or a Team-derived name."),
    dict(table="country_edition_panel", column="host", required=False, dtype="bool",
         description="1 if host; defaults to a documented host-NOC map."),
    # target participants and output
    dict(table="target_participants", column="noc", required=True, dtype="str",
         description="NOCs expected to compete in the target edition."),
    dict(table="target_participants", column="country", required=False, dtype="str",
         description="Optional display name."),
    dict(table="predictions_output", column="rank", required=True, dtype="int",
         description="Rank by predicted_total among forecast candidates (1 = highest)."),
    dict(table="predictions_output", column="noc", required=True, dtype="str",
         description="Three-letter NOC code."),
    dict(table="predictions_output", column="country", required=True, dtype="str",
         description="Display name (NOC when no name is available)."),
    dict(table="predictions_output", column="season", required=True, dtype="str",
         description="'Summer' or 'Winter'."),
    dict(table="predictions_output", column="target_year", required=True, dtype="int",
         description="Edition being forecast."),
    dict(table="predictions_output", column="predicted_total", required=True, dtype="float",
         description="Expected total medals (continuous; use predicted_total_rounded for display)."),
    dict(table="predictions_output", column="predicted_total_rounded", required=True, dtype="int",
         description="predicted_total rounded to a whole medal for table display."),
    dict(table="predictions_output", column="medal_probability", required=True, dtype="float",
         description="P(at least one medal) from the logistic occurrence stage."),
    dict(table="predictions_output", column="count_if_medal", required=True, dtype="float",
         description="E[medals | at least one medal] from the Poisson count stage."),
    dict(table="predictions_output", column="predicted_gold", required=True, dtype="float",
         description="Heuristic gold share x predicted_total (NaN when gold data is absent)."),
    dict(table="predictions_output", column="lower_80", required=True, dtype="float",
         description="Lower bound of the approximate 80% interval from backtest residuals."),
    dict(table="predictions_output", column="upper_80", required=True, dtype="float",
         description="Upper bound of the approximate 80% interval from backtest residuals."),
    dict(table="predictions_output", column="last_total", required=True, dtype="float",
         description="Medal total in the country's most recent prior edition (0 if none)."),
    dict(table="predictions_output", column="last_year", required=True, dtype="Int64",
         description="Year of that most recent prior edition (NA for a debutant)."),
    dict(table="predictions_output", column="mean_last3", required=True, dtype="float",
         description="Mean medal total over the previous three editions."),
    dict(table="predictions_output", column="trend", required=True, dtype="float",
         description="Least-squares medals-per-edition slope over recent prior editions."),
    dict(table="predictions_output", column="n_prior_editions", required=True, dtype="int",
         description="Number of prior editions the country attended in this season."),
    dict(table="predictions_output", column="host", required=True, dtype="bool",
         description="True when the NOC hosts the target edition (from a factual host map)."),
    dict(table="predictions_output", column="seen_before", required=True, dtype="bool",
         description="False for a country with no prior participation in this season."),
    dict(table="predictions_output", column="note", required=True, dtype="str",
         description="Row-level caveat (host status, debutant estimate, or empty)."),
    # backtest output
    dict(table="backtest_output", column="noc", required=True, dtype="str",
         description="Three-letter NOC code scored in the test edition."),
    dict(table="backtest_output", column="country", required=True, dtype="str",
         description="Display name."),
    dict(table="backtest_output", column="season", required=True, dtype="str",
         description="'Summer' or 'Winter'."),
    dict(table="backtest_output", column="test_year", required=True, dtype="int",
         description="Edition being predicted out of sample."),
    dict(table="backtest_output", column="trained_through_year", required=True, dtype="int",
         description="Latest edition used to fit the fold; always earlier than test_year."),
    dict(table="backtest_output", column="actual_total", required=True, dtype="float",
         description="Observed total medals in the test edition."),
    dict(table="backtest_output", column="actual_gold", required=True, dtype="float",
         description="Observed gold medals in the test edition (NaN if gold data is absent)."),
    dict(table="backtest_output", column="predicted_total", required=True, dtype="float",
         description="Out-of-sample expected total medals."),
    dict(table="backtest_output", column="predicted_medal_probability", required=True,
         dtype="float", description="Occurrence-stage P(medal > 0) for the scored row."),
    dict(table="backtest_output", column="predicted_count_if_medal", required=True,
         dtype="float", description="Count-stage E[medals | medal > 0] for the scored row."),
    dict(table="backtest_output", column="predicted_gold", required=True, dtype="float",
         description="Heuristic predicted gold medals for the scored row."),
    dict(table="backtest_output", column="baseline_last", required=True, dtype="float",
         description="Persistence baseline: the previous edition's medal total."),
    dict(table="backtest_output", column="baseline_mean3", required=True, dtype="float",
         description="Baseline: mean medal total over the previous three editions."),
    dict(table="backtest_output", column="error", required=True, dtype="float",
         description="actual_total - predicted_total (positive = under-prediction)."),
    dict(table="backtest_output", column="abs_error", required=True, dtype="float",
         description="Absolute model error for the row."),
    dict(table="backtest_output", column="baseline_abs_error", required=True, dtype="float",
         description="Absolute persistence-baseline error for the row."),
    dict(table="backtest_output", column="baseline_mean3_abs_error", required=True,
         dtype="float", description="Absolute previous-three-edition-mean baseline error."),
    dict(table="backtest_output", column="seen_before", required=True, dtype="bool",
         description="Whether the country had prior history; debutants are scored at 0."),
    dict(table="backtest_output", column="n_prior_editions", required=True, dtype="int",
         description="Number of prior editions the country attended."),
    dict(table="backtest_output", column="host", required=True, dtype="bool",
         description="Host flag for the test edition."),
    dict(table="backtest_output", column="n_train_editions", required=True, dtype="int",
         description="Number of editions used to fit this fold."),
    dict(table="backtest_output", column="n_train_rows", required=True, dtype="int",
         description="Number of country-edition rows used to fit this fold."),
    # metrics output
    dict(table="metrics_output", column="test_year", required=True, dtype="object",
         description="Test edition, or 'POOLED' for the combined row."),
    dict(table="metrics_output", column="n_countries / n_debutants", required=False,
         dtype="int", description="Observations scored and debutant count."),
    dict(table="metrics_output", column="mae_model / mae_baseline_last / mae_baseline_mean3",
         required=True, dtype="float", description="Mean absolute error of the model and both baselines (medal units; lower is better)."),
    dict(table="metrics_output", column="mae_model_medal_winners / mae_model_zero_medal",
         required=True, dtype="float", description="MAE split by actual medal outcome, so the zero-heavy long tail does not hide medal-table accuracy."),
    dict(table="metrics_output", column="brier_medal_occurrence_model", required=True,
         dtype="float", description="Brier score of P(medal > 0); lower is better."),
    dict(table="metrics_output", column="spearman_model / top10_overlap_model", required=True,
         dtype="float", description="Ranking quality; in the POOLED row these are means of per-edition values."),
    dict(table="metrics_output", column="mae_improvement_pct", required=True, dtype="float",
         description="Percentage MAE reduction versus persistence; negative means the baseline won."),
    # model card and data quality
    dict(table="model_card_output", column="stage / feature / coefficient / std_error",
         required=True, dtype="mixed", description="Per-stage coefficients with approximate standard errors; effect_at_1 gives the host 0->1 effect."),
    dict(table="data_quality_output", column="metric / value / status / comment", required=True,
         dtype="mixed", description="Pre-forecast coverage and sufficiency diagnostics, including zero-medal coverage."),
]


def describe_schema() -> pd.DataFrame:
    """Return the documented input/output schema as a DataFrame."""
    return pd.DataFrame(SCHEMA_ROWS)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Output of :func:`chronological_backtest`.

    Attributes
    ----------
    rows:
        One row per country and test edition, with actual medals, model
        prediction, both baselines and signed errors.
    folds:
        One row per test edition: training window, number of training rows,
        number of candidates and debutants.  Useful for a methods slide.
    notes:
        Human-readable caveats, e.g. which editions were tested.
    """

    rows: pd.DataFrame
    folds: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"BacktestResult(folds={len(self.folds)}, rows={len(self.rows)}, "
                f"test_years={self.folds['test_year'].tolist() if len(self.folds) else []})")


@dataclass
class ForecastResult:
    """Everything app.py needs, as documented DataFrames.

    Attributes
    ----------
    predictions:
        Ranked forecast for ``target_year``.  Columns: rank, noc, country,
        season, target_year, predicted_total, predicted_total_rounded,
        medal_probability, count_if_medal, predicted_gold, lower_80, upper_80,
        last_total, last_year, mean_last3, trend, n_prior_editions, host,
        seen_before, note.  ``medal_probability`` is P(medal > 0) and
        ``count_if_medal`` is E[medals | medal > 0]; their product is
        ``predicted_total``.
    backtest:
        Country-level chronological backtest rows, including predicted_total,
        predicted_medal_probability, predicted_count_if_medal and both
        baselines.
    backtest_folds:
        Per-edition training-window summary.
    metrics:
        Per-edition and pooled ("POOLED") metrics.  ``test_year`` is object
        dtype because of the pooled label.
    model_card:
        Two-stage coefficient table (``stage`` is ``count|medal`` or
        ``occurrence``) with approximate standard errors, effect sizes and
        training scaler values for the final model.
    data_quality:
        Coverage/sufficiency diagnostics.
    notes:
        Limitations and data-driven caveats.  Display these next to results.
    """

    predictions: pd.DataFrame
    backtest: pd.DataFrame
    backtest_folds: pd.DataFrame
    metrics: pd.DataFrame
    model_card: pd.DataFrame
    data_quality: pd.DataFrame
    notes: list[str]
    target_year: int
    season: str
    model_version: str = MODEL_VERSION

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return the displayable DataFrames keyed by attribute name."""
        return {
            "predictions": self.predictions,
            "backtest": self.backtest,
            "backtest_folds": self.backtest_folds,
            "metrics": self.metrics,
            "model_card": self.model_card,
            "data_quality": self.data_quality,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ForecastResult(season={self.season!r}, target_year={self.target_year}, "
                f"predictions={len(self.predictions)}, "
                f"backtest_rows={len(self.backtest)})")

# ---------------------------------------------------------------------------
# Column/shape normalisation
# ---------------------------------------------------------------------------

#: Accepted aliases, matched after lower-casing and replacing spaces with "_".
COLUMN_ALIASES: dict[str, str] = {
    "noc": "noc",
    "team_noc": "noc",
    "country_code": "noc",
    "ioc": "noc",
    "year": "year",
    "edition_year": "year",
    "olympic_year": "year",
    "season": "season",
    "medal": "medal",
    "event": "event",
    "team": "team",
    "country": "country",
    "nation": "country",
    "city": "city",
    "games": "games",
    "total": "total",
    "total_medals": "total",
    "medals": "total",
    "gold": "gold",
    "gold_medals": "gold",
    "silver": "silver",
    "silver_medals": "silver",
    "bronze": "bronze",
    "bronze_medals": "bronze",
    "host": "host",
    "is_host": "host",
    "host_country": "host",
    "athletes": "n_athletes",
    "n_athletes": "n_athletes",
    "id": "id",
}

_SEASON_MAP = {
    "summer": SUMMER,
    "summer olympics": SUMMER,
    "summer olympic games": SUMMER,
    "s": SUMMER,
    "winter": WINTER,
    "winter olympics": WINTER,
    "winter olympic games": WINTER,
    "w": WINTER,
}


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case/alias column names and reject ambiguous duplicates."""
    rename: dict[Any, str] = {}
    for column in df.columns:
        key = re.sub(r"\s+", "_", str(column).strip().lower())
        rename[column] = COLUMN_ALIASES.get(key, key)
    out = df.rename(columns=rename)
    duplicated = out.columns[out.columns.duplicated()].unique().tolist()
    if duplicated:
        raise OlympicDataError(
            "Ambiguous duplicate columns after name normalisation: "
            f"{duplicated}. Keep one spelling, e.g. only 'total'."
        )
    return out


def _normalise_season_value(value: Any) -> str:
    """Convert one season value to 'Summer'/'Winter' or raise."""
    text = str(value).strip().lower()
    if text in _SEASON_MAP:
        return _SEASON_MAP[text]
    raise OlympicDataError(
        f"Unknown season {value!r}. Expected 'Summer' or 'Winter'."
    )


def _normalise_season_series(series: pd.Series) -> pd.Series:
    """Map a season column to 'Summer'/'Winter'; unknown values become NA."""
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(_SEASON_MAP)
    unknown = sorted(set(text[mapped.isna() & text.notna()].tolist()))
    if unknown:
        warnings.warn(
            f"Dropping rows with unknown Season values: {unknown}. "
            "Expected 'Summer' or 'Winter'.",
            UserWarning,
            stacklevel=2,
        )
    return mapped


def _normalise_noc_series(series: pd.Series) -> pd.Series:
    """Upper-case/strip NOC codes; blanks become NA."""
    out = series.astype("string").str.strip().str.upper()
    return out.mask(out == "")


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Best-effort booleans from 0/1, yes/no, true/false; NA stays NA."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({
        "1": True, "0": False, "true": True, "false": False,
        "yes": True, "no": False, "y": True, "n": False,
    })
    return mapped.astype("boolean")


def _host_noc(season: str, year: int) -> str | None:
    return HOST_NOC_BY_EDITION.get((season, int(year)))


def _derived_host_series(season: str, years: pd.Series, nocs: pd.Series) -> pd.Series:
    """1 where the NOC hosts that edition, else 0, using the factual map."""
    values = [
        1 if _host_noc(season, int(year)) == noc else 0
        for year, noc in zip(years.tolist(), nocs.tolist())
    ]
    return pd.Series(values, index=years.index, dtype="int64")


def _country_names_from_team(raw: pd.DataFrame) -> dict[str, str]:
    """Infer display names from the most common Team value per NOC."""
    if "team" not in raw.columns:
        return {}
    teams = raw.dropna(subset=["team", "noc"])
    if teams.empty:
        return {}

    def _mode(values: pd.Series) -> Any:
        counts = values.value_counts()
        return counts.index[0] if len(counts) else None

    names = teams.groupby("noc", sort=False)["team"].agg(_mode)
    names = names.astype("string").str.replace(r"-\d+$", "", regex=True).str.strip()
    return {str(noc): str(name) for noc, name in names.dropna().items()}


def _apply_country_names(
    panel: pd.DataFrame,
    country_names: Mapping[str, str] | pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply an optional NOC -> country display-name mapping."""
    panel = panel.copy()
    existing = panel["country"] if "country" in panel.columns else pd.Series(pd.NA, index=panel.index)
    existing = existing.astype("string")

    if country_names is None:
        mapped = pd.Series(pd.NA, index=panel.index, dtype="string")
    elif isinstance(country_names, Mapping):
        mapped = panel["noc"].map({str(k).upper(): str(v) for k, v in country_names.items()}).astype("string")
    elif isinstance(country_names, pd.DataFrame):
        names = _canonicalise_columns(country_names)
        if not {"noc", "country"} <= set(names.columns):
            raise OlympicDataError(
                "country_names DataFrame must contain 'noc' and 'country'."
            )
        names = names[["noc", "country"]].copy()
        names["noc"] = _normalise_noc_series(names["noc"])
        names = names.dropna(subset=["noc"]).drop_duplicates("noc")
        mapped = panel["noc"].map(names.set_index("noc")["country"]).astype("string")
    else:
        raise OlympicDataError(
            "country_names must be a dict {NOC: name} or a DataFrame with noc,country."
        )

    panel["country"] = mapped.fillna(existing).fillna(panel["noc"]).astype("string")
    return panel


def _normalise_participants(participants: pd.DataFrame | None) -> pd.DataFrame | None:
    """Validate a historical participants table (noc, year, season)."""
    if participants is None:
        return None
    if not isinstance(participants, pd.DataFrame):
        raise OlympicDataError(
            "participants must be a DataFrame with columns noc, year, season."
        )
    part = _canonicalise_columns(participants)
    missing = {"noc", "year", "season"} - set(part.columns)
    if missing:
        raise OlympicDataError(
            f"participants is missing columns: {sorted(missing)}. "
            "Required: noc, year, season."
        )
    part = part[["noc", "year", "season"]].copy()
    part["noc"] = _normalise_noc_series(part["noc"])
    part["year"] = pd.to_numeric(part["year"], errors="coerce")
    part["season"] = _normalise_season_series(part["season"])
    part = part.dropna(subset=["noc", "year", "season"])
    part["year"] = part["year"].astype("int64")
    return part.drop_duplicates().reset_index(drop=True)


def _attach_participants(panel: pd.DataFrame, participants: pd.DataFrame | None) -> pd.DataFrame:
    """Add zero-medal rows for participants absent from an aggregated table.

    Only editions already present in ``panel`` are used, so an entry list for a
    future Games cannot accidentally create a fake historical edition.
    """
    if participants is None:
        return panel
    present = panel[["season", "year"]].drop_duplicates()
    part = participants.merge(present, on=["season", "year"], how="inner")
    ignored = len(participants) - len(part)
    if ignored:
        warnings.warn(
            f"Ignored {ignored} participant rows for editions not present in the "
            "historical data (prepare_country_edition_panel only fills in "
            "coverage for historical editions).",
            UserWarning,
            stacklevel=2,
        )
    keys = ["season", "year", "noc"]
    merged = part[keys].drop_duplicates().merge(panel, on=keys, how="left")
    merged["total"] = pd.to_numeric(merged["total"], errors="coerce").fillna(0.0)
    for colour in ("silver", "bronze"):
        if colour in merged.columns:
            merged[colour] = pd.to_numeric(merged[colour], errors="coerce").fillna(0.0)
    if "gold" in merged.columns and merged["gold"].notna().any():
        merged["gold"] = pd.to_numeric(merged["gold"], errors="coerce").fillna(0.0)
    if "gold" in merged.columns:
        # Keep gold entirely missing if the source had no gold information.
        if not panel["gold"].notna().any():
            merged["gold"] = np.nan
    merged["country"] = merged["country"].astype("string")
    merged["country"] = merged["country"].fillna(merged["noc"])
    merged["host"] = pd.to_numeric(merged["host"], errors="coerce").fillna(0).astype("int64")
    return merged


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_country_edition_panel(
    data: pd.DataFrame,
    *,
    participants: pd.DataFrame | None = None,
    country_names: Mapping[str, str] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalise raw or aggregated Olympic data to a country/edition panel.

    Parameters
    ----------
    data:
        Either raw athlete/event rows or an already aggregated table.  See the
        module docstring / :func:`describe_schema` for required columns.
    participants:
        Optional historical participation table (``noc, year, season``).  Use
        this when ``data`` is an aggregated medal table that omits countries
        with zero medals.  Without it, MAE is optimistic.
    country_names:
        Optional NOC -> display-name mapping, or a DataFrame with
        ``noc, country``.

    Returns
    -------
    DataFrame
        Columns: season, year, noc, country, gold, silver, bronze, total, host.
        Sorted by season, year, noc.  One row per participating NOC/edition.

    Notes
    -----
    Team medals are counted once per NOC/event/medal in raw mode.  The 1906
    Intercalated Games are excluded.  If gold information is absent, the gold
    column is NaN and ``predicted_gold`` will be NaN rather than a fabricated 0.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise OlympicDataError("The supplied Olympic DataFrame is empty.")

    frame = _canonicalise_columns(data)
    columns = set(frame.columns)
    raw_shape = {"noc", "year", "season", "medal", "event"} <= columns
    panel_shape = "total" in columns or set(MEDAL_COLOURS) <= columns

    if raw_shape and not panel_shape:
        panel = _prepare_raw(frame)
    elif panel_shape:
        panel = _prepare_panel(frame)
    elif raw_shape:
        panel = _prepare_raw(frame)
    else:
        raise OlympicDataError(
            "Unrecognised Olympic data shape. Supply either raw athlete/event "
            "columns (NOC, Year, Season, Event, Medal) or an aggregated table "
            "(noc, year, season, total or gold/silver/bronze). "
            "See prediction.describe_schema()."
        )

    if panel.empty:
        raise OlympicDataError(
            "No usable Olympic rows remained after cleaning. Check NOC/year/"
            "season values and that medals are Gold/Silver/Bronze."
        )

    prep_attrs = dict(panel.attrs)
    panel = _attach_participants(panel, _normalise_participants(participants))
    panel = _apply_country_names(panel, country_names)
    panel = panel.sort_values(["season", "year", "noc"]).reset_index(drop=True)
    ordered = ["season", "year", "noc", "country", "gold", "silver", "bronze", "total", "host"]
    panel = panel[[*ordered, *[c for c in panel.columns if c not in ordered]]]
    panel.attrs.update(prep_attrs)

    if participants is not None and panel["total"].eq(0).sum() == 0:
        warnings.warn(
            "Zero-medal rows were expected from `participants` but none exist; "
            "check that participants covers the same editions as the data.",
            UserWarning,
            stacklevel=2,
        )
    return panel


def _drop_excluded_editions(panel: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove non-recognised editions (currently 1906) and count them."""
    if panel.empty:
        return panel, 0
    keys = list(zip(panel["season"].tolist(), panel["year"].tolist()))
    mask = np.array([key in EXCLUDED_EDITIONS for key in keys], dtype=bool)
    return panel.loc[~mask].copy(), int(mask.sum())


def _prepare_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate athlete/event rows to one row per participating NOC/edition."""
    required = {"noc", "year", "season", "medal", "event"}
    missing = required - set(frame.columns)
    if missing:
        raise OlympicDataError(
            f"Raw athlete/event data is missing required columns: {sorted(missing)}. "
            "Required: NOC, Year, Season, Event, Medal."
        )

    raw = frame.copy()
    raw["noc"] = _normalise_noc_series(raw["noc"])
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw["season"] = _normalise_season_series(raw["season"])
    raw["medal"] = raw["medal"].astype("string").str.strip().str.title()

    n_before = len(raw)
    raw = raw.dropna(subset=["noc", "year", "season", "event"])
    raw["year"] = raw["year"].astype("int64")
    n_invalid = n_before - len(raw)

    raw, n_excluded = _drop_excluded_editions(raw)

    # One participation row per NOC/edition, including NOCs with zero medals.
    participation = raw[["season", "year", "noc"]].drop_duplicates()

    medals = raw[raw["medal"].isin(["Gold", "Silver", "Bronze"])].copy()
    medal_rows_before = len(medals)
    # Team events have one row per athlete. Count one medal per NOC/event/medal.
    medals = medals.drop_duplicates(subset=["season", "year", "event", "noc", "medal"])
    n_team_duplicates = medal_rows_before - len(medals)

    counts = (
        medals.groupby(["season", "year", "noc", "medal"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Gold", "Silver", "Bronze"], fill_value=0)
        .rename(columns={"Gold": "gold", "Silver": "silver", "Bronze": "bronze"})
        .reset_index()
    )
    counts["total"] = counts[list(MEDAL_COLOURS)].sum(axis=1)

    panel = participation.merge(
        counts, on=["season", "year", "noc"], how="left", validate="one_to_one"
    )
    for colour in (*MEDAL_COLOURS, "total"):
        panel[colour] = pd.to_numeric(panel[colour], errors="coerce").fillna(0).clip(lower=0)

    panel = _apply_country_names(panel, _country_names_from_team(raw))
    panel["host"] = [
        1 if _host_noc(season, int(year)) == noc else 0
        for season, year, noc in zip(panel["season"], panel["year"], panel["noc"])
    ]

    panel.attrs["source"] = "raw_athlete_events"
    panel.attrs["prepare_warnings"] = [
        f"Dropped {n_invalid} raw rows with invalid NOC/Year/Season/Event.",
        f"Excluded {n_excluded} rows from the 1906 Intercalated Games.",
        f"Collapsed {n_team_duplicates} duplicate athlete rows into team medals.",
    ]
    panel.attrs["duplicate_rows_removed"] = int(n_team_duplicates)
    panel.attrs["excluded_1906_rows"] = int(n_excluded)
    return panel


def _prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise an already aggregated country/edition table."""
    required = {"noc", "year"}
    missing = required - set(frame.columns)
    if missing:
        raise OlympicDataError(
            f"Aggregated data is missing required columns: {sorted(missing)}. "
            "Required: noc, year, season, and total or gold/silver/bronze."
        )

    panel = frame.copy()

    # Allow a missing season to be derived from Games, or default to Summer.
    if "season" not in panel.columns:
        if "games" in panel.columns:
            games = panel["games"].astype("string").str.lower()
            derived = pd.Series(pd.NA, index=panel.index, dtype="string")
            derived[games.str.contains("summer", na=False)] = SUMMER
            derived[games.str.contains("winter", na=False)] = WINTER
            if derived.notna().any():
                panel["season"] = derived
            else:
                panel["season"] = SUMMER
                warnings.warn(
                    "No season column and no Summer/Winter in Games; assuming Summer.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            panel["season"] = SUMMER
            warnings.warn(
                "No season column supplied; assuming Summer. Pass season explicitly "
                "if this is wrong.",
                UserWarning,
                stacklevel=2,
            )

    panel["noc"] = _normalise_noc_series(panel["noc"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["season"] = _normalise_season_series(panel["season"])

    n_before = len(panel)
    panel = panel.dropna(subset=["noc", "year", "season"])
    panel["year"] = panel["year"].astype("int64")
    n_invalid = n_before - len(panel)
    panel, n_excluded = _drop_excluded_editions(panel)

    has_gold = "gold" in panel.columns and panel["gold"].notna().any()
    if "total" not in panel.columns:
        if not set(MEDAL_COLOURS) <= set(panel.columns):
            raise OlympicDataError(
                "Aggregated data needs either 'total' or all of gold, silver, bronze."
            )
        for colour in MEDAL_COLOURS:
            panel[colour] = pd.to_numeric(panel[colour], errors="coerce").fillna(0)
        panel["total"] = panel[list(MEDAL_COLOURS)].sum(axis=1)
    else:
        panel["total"] = pd.to_numeric(panel["total"], errors="coerce")

    for colour in MEDAL_COLOURS:
        if colour in panel.columns:
            panel[colour] = pd.to_numeric(panel[colour], errors="coerce")

    if not has_gold:
        panel["gold"] = np.nan
    if "host" in panel.columns:
        panel["host"] = _to_bool_series(panel["host"])

    aggregations: dict[str, Any] = {"total": "sum"}
    for colour in ("silver", "bronze"):
        if colour in panel.columns and panel[colour].notna().any():
            aggregations[colour] = "sum"
    if has_gold:
        aggregations["gold"] = "sum"
    if "country" in panel.columns and panel["country"].notna().any():
        aggregations["country"] = "first"
    if "host" in panel.columns and panel["host"].notna().any():
        aggregations["host"] = "max"

    duplicates = int(panel.duplicated(["season", "year", "noc"]).sum())
    panel = (
        panel.groupby(["season", "year", "noc"], as_index=False, sort=False)
        .agg(aggregations)
    )

    if not has_gold:
        panel["gold"] = np.nan
    if "silver" not in panel.columns:
        panel["silver"] = np.nan
    if "bronze" not in panel.columns:
        panel["bronze"] = np.nan

    panel["total"] = pd.to_numeric(panel["total"], errors="coerce").fillna(0).clip(lower=0)
    for colour in MEDAL_COLOURS:
        panel[colour] = pd.to_numeric(panel[colour], errors="coerce").clip(lower=0)

    # Host: respect a supplied flag where possible, otherwise use the map.
    derived = pd.Series(
        [
            1 if _host_noc(season, int(year)) == noc else 0
            for season, year, noc in zip(panel["season"], panel["year"], panel["noc"])
        ],
        index=panel.index,
        dtype="int64",
    )
    if "host" in panel.columns and panel["host"].notna().any():
        supplied = _to_bool_series(panel["host"])
        panel["host"] = supplied.fillna(derived).astype("int64")
    else:
        panel["host"] = derived

    panel["country"] = panel["country"].astype("string") if "country" in panel.columns else pd.Series(pd.NA, index=panel.index, dtype="string")

    panel.attrs["source"] = "aggregated_panel"
    panel.attrs["prepare_warnings"] = [
        f"Dropped {n_invalid} rows with invalid NOC/Year/Season.",
        f"Excluded {n_excluded} rows from the 1906 Intercalated Games.",
        f"Summed {duplicates} duplicate (season, year, noc) rows.",
    ]
    panel.attrs["duplicate_rows_removed"] = duplicates
    panel.attrs["excluded_1906_rows"] = int(n_excluded)
    return panel

# ---------------------------------------------------------------------------
# Leak-free feature construction
# ---------------------------------------------------------------------------


def _rolling_by_group(
    series: pd.Series,
    groups: pd.Series,
    window: int,
    min_periods: int,
    how: str,
) -> pd.Series:
    """Rolling mean/max/sum within groups, returned aligned to the input index."""
    rolling = series.groupby(groups, sort=False).rolling(window, min_periods=min_periods)
    if how == "mean":
        out = rolling.mean()
    elif how == "max":
        out = rolling.max()
    elif how == "sum":
        out = rolling.sum()
    elif how == "apply":
        out = rolling.apply(_slope, raw=True)
    else:  # pragma: no cover - internal guard
        raise ValueError(f"Unknown rolling aggregation {how!r}")
    out.index = out.index.droplevel(0)
    return out.sort_index()


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of values against 0..n-1 (medals per edition)."""
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < 2 or not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator <= 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denominator)


def _feature_columns(has_gold: bool, use_host: bool) -> list[str]:
    features = [
        name for name in BASE_FEATURES
        if name != "gold_share3" or has_gold
    ]
    if use_host:
        features.append(HOST_FEATURE)
    return features


def build_feature_frame(
    panel: pd.DataFrame,
    season: str = SUMMER,
    target_year: int | None = None,
) -> pd.DataFrame:
    """Build leakage-free features for historical editions and a target year.

    For every (country, edition) row, all features are computed from earlier
    editions only.  The target-year row is appended as a placeholder with
    ``participated == 0`` so it can be scored; it is never used for training.

    Parameters
    ----------
    panel:
        Output of :func:`prepare_country_edition_panel`.
    season:
        'Summer' or 'Winter'.  Seasons are modelled separately: medal scales
        and participation differ substantially.
    target_year:
        Optional future edition to append (must be later than the last
        historical edition).

    Returns
    -------
    DataFrame
        Historical rows (``participated`` 1/0) plus optional target rows.
        Count-stage features are ``momentum_last_vs_mean``, ``trend_ratio``,
        ``depth_ratio``, ``gold_share3``, ``n_prior`` and (optionally) ``host``,
        applied on top of the ``log(prev_total + 1)`` offset.  The occurrence
        stage additionally uses ``log_last`` and ``was_zero_last``.  The raw
        lags (``prev_total``, ``prev3_mean``, etc.) are retained for inspection
        and for the persistence baselines.
    """
    season = _normalise_season_value(season)
    required = {"season", "year", "noc", "total", "gold", "host"}
    missing = required - set(panel.columns)
    if missing:
        raise OlympicDataError(
            f"Feature construction needs prepared panel columns; missing {sorted(missing)}. "
            "Call prepare_country_edition_panel() first."
        )

    hist = panel.loc[panel["season"] == season].copy()
    if hist.empty:
        raise InsufficientDataError(
            f"No {season} rows are available. Available seasons: "
            f"{sorted(panel['season'].dropna().unique().tolist())}."
        )

    hist["noc"] = _normalise_noc_series(hist["noc"])
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce")
    hist["total"] = pd.to_numeric(hist["total"], errors="coerce")
    hist = hist.dropna(subset=["noc", "year"])
    hist["year"] = hist["year"].astype("int64")
    hist = hist.drop_duplicates(["year", "noc"])

    if hist.empty:
        raise InsufficientDataError(f"No usable {season} rows after cleaning.")

    years = sorted(int(y) for y in hist["year"].unique())
    has_gold = bool(hist["gold"].notna().any())

    if target_year is not None:
        target_year = int(target_year)
        if target_year in years:
            raise ValueError(
                f"target_year={target_year} is already in the historical data. "
                "Forecasting it would be in-sample; choose a later edition or use "
                "chronological_backtest() to evaluate it."
            )
        if target_year <= years[-1]:
            raise ValueError(
                f"target_year={target_year} must be later than the last historical "
                f"edition ({years[-1]})."
            )
        grid_years = [*years, target_year]
    else:
        grid_years = years

    countries = sorted(hist["noc"].dropna().unique().tolist())
    first_year = hist.groupby("noc", sort=False)["year"].min().rename("first_year")

    grid = pd.MultiIndex.from_product(
        [countries, grid_years], names=["noc", "year"]
    ).to_frame(index=False)
    grid = grid.merge(first_year, left_on="noc", right_index=True, how="left")
    # A country cannot be predicted before it first appears.
    grid = grid.loc[grid["year"] >= grid["first_year"]].copy()

    hist_columns = ["season", "year", "noc", "total", "gold", "silver", "bronze", "host"]
    available_hist = [col for col in hist_columns if col in hist.columns]
    grid = grid.merge(
        hist[available_hist], on=["year", "noc"], how="left", suffixes=("", "_hist")
    )
    grid["season"] = season
    grid["participated"] = grid["total"].notna().astype("int64")

    for colour in (*MEDAL_COLOURS, "total"):
        if colour not in grid.columns:
            grid[colour] = 0.0
        if colour == "gold" and not has_gold:
            grid[colour] = np.nan
        else:
            grid[colour] = pd.to_numeric(grid[colour], errors="coerce").fillna(0.0)

    country_map = {}
    if "country" in hist.columns:
        country_map = (
            hist.dropna(subset=["country"])
            .drop_duplicates("noc")
            .set_index("noc")["country"]
            .to_dict()
        )
    grid["country"] = grid["noc"].map(country_map).fillna(grid["noc"]).astype("string")

    # Host: rows supplied by the panel keep their flag; target rows use the map.
    grid["host"] = pd.to_numeric(grid["host"], errors="coerce")
    if target_year is not None:
        host_noc = _host_noc(season, target_year)
        target_mask = grid["year"] == target_year
        grid.loc[target_mask, "host"] = [
            1 if host_noc == noc else 0
            for noc in grid.loc[target_mask, "noc"].tolist()
        ]
    grid["host"] = grid["host"].fillna(0).clip(0, 1).astype("int64")

    grid = grid.sort_values(["noc", "year"]).reset_index(drop=True)
    grouped = grid.groupby("noc", sort=False)

    # --- all features come from shifted values, i.e. strictly earlier rows ---
    prev_total = grouped["total"].shift(1).fillna(0.0)
    prev_gold = grouped["gold"].shift(1)
    prev_gold = prev_gold.fillna(0.0) if has_gold else prev_gold

    grid["prev_total"] = prev_total
    grid["prev_gold"] = prev_gold
    grid["prev3_mean"] = _rolling_by_group(prev_total, grid["noc"], 3, 1, "mean").fillna(0.0)
    grid["prev3_max"] = _rolling_by_group(prev_total, grid["noc"], 3, 1, "max").fillna(0.0)
    grid["trend"] = _rolling_by_group(prev_total, grid["noc"], 4, 2, "apply").fillna(0.0)

    # Bounded features applied on top of the log(prev_total + 1) offset.
    grid["momentum_last_vs_mean"] = (
        (grid["prev_total"] - grid["prev3_mean"]) / (grid["prev_total"] + 1.0)
    ).clip(-1.0, 1.0)
    grid["trend_ratio"] = (
        grid["trend"] / (grid["prev3_mean"] + 1.0)
    ).clip(-1.0, 1.0)
    grid["depth_ratio"] = (
        np.log1p(grid["prev3_max"]) - np.log1p(grid["prev_total"])
    ).clip(0.0, 5.0)
    grid["was_zero_last"] = (grid["prev_total"] < 0.5).astype("float64")
    grid["log_last"] = np.log1p(grid["prev_total"])

    prior_participation = grouped["participated"].shift(1).fillna(0)
    grid["n_prior"] = (
        prior_participation.groupby(grid["noc"], sort=False).cumsum().astype("int64")
    )
    grid["last_year"] = grouped["year"].shift(1)

    if has_gold:
        gold_recent = _rolling_by_group(prev_gold, grid["noc"], 3, 1, "sum")
        total_recent = _rolling_by_group(prev_total, grid["noc"], 3, 1, "sum")
        grid["gold_share3"] = (
            gold_recent / total_recent.where(total_recent > 0)
        ).fillna(0.0).clip(0.0, 1.0)
    else:
        grid["gold_share3"] = np.nan

    grid["seen_before"] = grid["n_prior"] > 0
    return grid


# ---------------------------------------------------------------------------
# Target-year candidate rows
# ---------------------------------------------------------------------------


def _normalise_target_participants(
    participants: pd.DataFrame | Sequence[str] | None,
) -> tuple[list[str] | None, dict[str, str]]:
    """Return (NOC list, display-name map) for a target entry list."""
    if participants is None:
        return None, {}
    names: dict[str, str] = {}
    if isinstance(participants, pd.DataFrame):
        frame = _canonicalise_columns(participants)
        if "noc" not in frame.columns:
            raise OlympicDataError("target_participants needs a 'noc' column.")
        codes = _normalise_noc_series(frame["noc"]).dropna().astype(str).tolist()
        if "country" in frame.columns:
            for noc, name in zip(
                _normalise_noc_series(frame["noc"]), frame["country"]
            ):
                if pd.notna(noc) and pd.notna(name):
                    names[str(noc)] = str(name)
    elif isinstance(participants, (list, tuple, set, pd.Series)):
        codes = _normalise_noc_series(pd.Series(list(participants))).dropna().astype(str).tolist()
    else:
        raise OlympicDataError(
            "target_participants must be a DataFrame with 'noc' or a sequence of NOC codes."
        )
    codes = list(dict.fromkeys(codes))
    if not codes:
        raise OlympicDataError("target_participants was supplied but contains no NOC codes.")
    return codes, names


def _target_candidates(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    season: str,
    target_year: int,
    target_participants: pd.DataFrame | Sequence[str] | None,
    active_window: int | None,
    country_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Select and complete the rows that will be forecast for ``target_year``."""
    target = features.loc[features["year"] == target_year].copy()
    if target.empty:
        raise InsufficientDataError(
            f"No candidate rows could be built for {season} {target_year}."
        )

    codes, names = _normalise_target_participants(target_participants)
    if country_names:
        names = {**{str(k).upper(): str(v) for k, v in country_names.items()}, **names}

    if codes is not None:
        have = set(target["noc"].astype(str))
        missing = [code for code in codes if code not in have]
        if missing:
            rows: list[dict[str, Any]] = []
            for code in missing:
                row: dict[str, Any] = {column: 0 for column in target.columns}
                row.update({
                    "season": season,
                    "year": target_year,
                    "noc": code,
                    "country": names.get(code, code),
                    "first_year": target_year,
                    "participated": 0,
                    "host": 1 if _host_noc(season, target_year) == code else 0,
                    "total": 0.0,
                    "gold": np.nan if not target["gold"].notna().any() else 0.0,
                    "silver": 0.0,
                    "bronze": 0.0,
                    "prev_total": 0.0,
                    "prev3_mean": 0.0,
                    "prev3_max": 0.0,
                    "trend": 0.0,
                    "prev_gold": np.nan,
                    "n_prior": 0,
                    "last_year": np.nan,
                    "gold_share3": np.nan,
                    "momentum_last_vs_mean": 0.0,
                    "trend_ratio": 0.0,
                    "depth_ratio": 0.0,
                    "was_zero_last": 1.0,
                    "log_last": 0.0,
                    "seen_before": False,
                })
                rows.append(row)
            target = pd.concat([target, pd.DataFrame(rows)], ignore_index=True)
        target = target.loc[target["noc"].astype(str).isin(codes)]
        target["country"] = target["noc"].map(names).fillna(target["country"])
        target = target.drop_duplicates("noc")
    else:
        history = panel.loc[panel["season"] == season]
        editions = sorted(int(y) for y in history["year"].unique())
        recent = set(editions[-int(active_window):]) if active_window else set(editions)
        active = set(history.loc[history["year"].isin(recent), "noc"].astype(str))
        target = target.loc[target["noc"].astype(str).isin(active) & (target["n_prior"] > 0)]

    if target.empty:
        raise InsufficientDataError(
            f"No eligible {season} {target_year} candidates remain. "
            "Check target_participants/active_window or supply more history."
        )
    target = target.copy().reset_index(drop=True)
    target["seen_before"] = target["n_prior"] > 0
    return target

# ---------------------------------------------------------------------------
# Transparent count model
# ---------------------------------------------------------------------------


@dataclass
class _PoissonModel:
    """Ridge-regularised Poisson GLM with a log link, fitted by IRLS.

    This is the **count stage** of the hurdle model.  It is fitted only on
    country-edition rows that actually won at least one medal, so it estimates
    ``E[medals | medals > 0]`` rather than mixing zero-medal countries into the
    same intercept.
    """

    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    alpha: float
    n_train_rows: int
    n_train_editions: int
    trained_through_year: int
    offset_shrink: float = OFFSET_SHRINK

    def _offset(self, frame: pd.DataFrame) -> np.ndarray:
        anchored = pd.to_numeric(
            frame.get("prev_total", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        return np.log(np.maximum(anchored, 0.0) + self.offset_shrink)

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=list(self.features))
        values = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        standardised = (values.to_numpy(dtype=float) - self.mean) / self.std
        return np.column_stack([np.ones(len(frame)), standardised])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        eta = np.clip(
            self._offset(frame) + self._design(frame) @ self.beta, -20.0, 8.0
        )
        return np.clip(np.exp(eta), 0.0, None)

    def model_card(self, stage: str = "count|medal") -> pd.DataFrame:
        rows = [{
            "stage": stage,
            "feature": "intercept",
            "coefficient": float(self.beta[0]),
            "std_error": float(self.se[0]),
            "effect_per_1sd": float(np.exp(self.beta[0])),
            "effect_type": "rate_ratio",
            "train_mean": np.nan,
            "train_sd": np.nan,
            "note": (
                "Multiplier on the anchor (prev_total + "
                f"{self.offset_shrink:g}); coefficients adjust that anchor, "
                "they do not create an unanchored exponential trend."
            ),
        }]
        for index, name in enumerate(self.features, start=1):
            coefficient = float(self.beta[index])
            mean = float(self.mean[index - 1])
            std = float(self.std[index - 1])
            row = {
                "stage": stage,
                "feature": name,
                "coefficient": coefficient,
                "std_error": float(self.se[index]),
                "effect_per_1sd": float(np.exp(coefficient)),
                "effect_type": "rate_ratio",
                "train_mean": mean,
                "train_sd": std,
                "note": (
                    "Multiplicative change in the conditional medal count per "
                    "+1 SD."
                ),
            }
            if name == HOST_FEATURE:
                # The host flag may be unscaled (see UNSCALED_FEATURES), so
                # report the effect users care about: changing host from 0 to 1.
                # In general a one-unit change is 1/std standard deviations.
                row["effect_at_1"] = float(np.exp(coefficient / std))
                row["note"] = (
                    "Multiplicative change in the conditional medal count when "
                    "the host flag changes from 0 to 1 (effect_at_1). "
                    "effect_per_1sd is the per-standard-deviation coefficient."
                )
            rows.append(row)
        return pd.DataFrame(rows)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_poisson_ridge(
    design: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    offset: np.ndarray | None = None,
    max_iter: int = 200,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Poisson ridge regression and return (coefficients, approx. SEs).

    The intercept is not penalised.  Iteratively reweighted least squares is
    the standard GLM fitting algorithm and is only a few lines here, which
    keeps the method inspectable and avoids a statsmodels dependency.
    """
    n_rows, n_columns = design.shape
    if offset is None:
        offset = np.zeros(n_rows, dtype=float)
    offset = np.asarray(offset, dtype=float)
    beta = np.zeros(n_columns, dtype=float)
    anchor_mean = float(np.mean(np.exp(np.clip(offset, -20.0, 8.0))))
    beta[0] = np.log(max(float(np.mean(y)), 1e-3) / max(anchor_mean, 1e-6))
    penalty = float(max(alpha, 0.0)) * np.eye(n_columns)
    penalty[0, 0] = 0.0  # do not shrink the intercept

    for _ in range(max_iter):
        eta = np.clip(offset + design @ beta, -20.0, 8.0)
        mu = np.exp(eta)
        weights = np.clip(mu, 1e-6, None)
        working = eta + (y - mu) / weights

        weighted_design = design.T * weights
        lhs = weighted_design @ design + penalty
        rhs = weighted_design @ (working - offset)
        lhs.flat[:: n_columns + 1] += 1e-9  # numerical jitter
        try:
            new_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - safety net
            new_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if not np.all(np.isfinite(new_beta)):  # pragma: no cover - safety net
            new_beta = beta
        if np.max(np.abs(new_beta - beta)) < tolerance:
            beta = new_beta
            break
        beta = new_beta

    eta = np.clip(offset + design @ beta, -20.0, 8.0)
    mu = np.exp(eta)
    information = (design.T * np.clip(mu, 1e-6, None)) @ design + penalty
    covariance = np.linalg.pinv(information)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if not np.all(np.isfinite(beta)):  # pragma: no cover
        raise OlympicDataError("Poisson model fitting produced non-finite coefficients.")
    return beta, se


def _fit_logistic_ridge(
    design: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ridge-penalised logistic regression and return (coefficients, SEs).

    Used for the occurrence stage: ``y`` is 1 when the country won at least one
    medal and 0 otherwise.  IRLS/Newton updates keep the implementation
    inspectable and avoid a scikit-learn dependency.  The intercept is not
    penalised; feature coefficients are, which keeps a nearly separable feature
    (e.g. "won nothing last time") finite.
    """
    n_rows, n_columns = design.shape
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    beta = np.zeros(n_columns, dtype=float)
    base_rate = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    beta[0] = float(np.log(base_rate / (1.0 - base_rate)))
    penalty = float(max(alpha, 0.0)) * np.eye(n_columns)
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        probability = _sigmoid(design @ beta)
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        working = design @ beta + (y - probability) / weights
        weighted_design = design.T * weights
        lhs = weighted_design @ design + penalty
        rhs = weighted_design @ working
        lhs.flat[:: n_columns + 1] += 1e-9
        try:
            new_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - safety net
            new_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if not np.all(np.isfinite(new_beta)):  # pragma: no cover - safety net
            new_beta = beta
        if np.max(np.abs(new_beta - beta)) < tolerance:
            beta = new_beta
            break
        beta = new_beta

    probability = _sigmoid(design @ beta)
    weights = np.clip(probability * (1.0 - probability), 1e-6, None)
    information = (design.T * weights) @ design + penalty
    covariance = np.linalg.pinv(information)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if not np.all(np.isfinite(beta)):  # pragma: no cover
        raise OlympicDataError(
            "Occurrence model fitting produced non-finite coefficients."
        )
    return beta, se


@dataclass
class _HurdleModel:
    """Two-part hurdle model for country medal totals.

    Medal counts have many zeros (most NOCs win nothing), so a single count
    model spends its intercept on the zero mass and then under-predicts the
    medal-winning countries.  The hurdle model splits the problem:

    ``P(medal > 0)``          logistic occurrence stage on every training row.
    ``E[medals | medals>0]``  Poisson count stage on medal-winning rows only,
                              anchored on ``log(prev_total + 1)``.

    The forecast is their product, ``E[total] = P(medal) * E[count|medal]``.
    Both stages use only earlier editions, and both are fitted with the same
    ridge penalty.
    """

    count_model: _PoissonModel
    occurrence_features: tuple[str, ...]
    occurrence_mean: np.ndarray
    occurrence_std: np.ndarray
    occurrence_beta: np.ndarray
    occurrence_se: np.ndarray
    alpha: float
    n_train_rows: int
    n_train_editions: int
    n_train_positive: int
    trained_through_year: int
    n_unanchored_positive_excluded: int = 0
    offset_shrink: float = OFFSET_SHRINK

    def _occurrence_design(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=list(self.occurrence_features))
        values = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        standardised = (values.to_numpy(dtype=float) - self.occurrence_mean) / self.occurrence_std
        return np.column_stack([np.ones(len(frame)), standardised])

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """Return P(at least one medal) for every row."""
        return _sigmoid(self._occurrence_design(frame) @ self.occurrence_beta)

    def predict_count(self, frame: pd.DataFrame) -> np.ndarray:
        """Return E[medals | at least one medal] for every row."""
        return self.count_model.predict(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the expected total medal count (probability x conditional count)."""
        return np.clip(
            self.predict_probability(frame) * self.predict_count(frame), 0.0, None
        )

    def model_card(self) -> pd.DataFrame:
        """Coefficient table for both stages, with a ``stage`` column."""
        occurrence_rows = [{
            "stage": "occurrence",
            "feature": "intercept",
            "coefficient": float(self.occurrence_beta[0]),
            "std_error": float(self.occurrence_se[0]),
            "effect_per_1sd": float(np.exp(self.occurrence_beta[0])),
            "effect_type": "odds_ratio",
            "train_mean": np.nan,
            "train_sd": np.nan,
            "note": (
                "Baseline odds of winning at least one medal for an average "
                "training row (all features at their means)."
            ),
        }]
        for index, name in enumerate(self.occurrence_features, start=1):
            coefficient = float(self.occurrence_beta[index])
            mean = float(self.occurrence_mean[index - 1])
            std = float(self.occurrence_std[index - 1])
            row = {
                "stage": "occurrence",
                "feature": name,
                "coefficient": coefficient,
                "std_error": float(self.occurrence_se[index]),
                "effect_per_1sd": float(np.exp(coefficient)),
                "effect_type": "odds_ratio",
                "train_mean": mean,
                "train_sd": std,
                "note": "Change in the odds of winning at least one medal per +1 SD.",
            }
            if name == HOST_FEATURE:
                # One-unit change is 1/std standard deviations. With the host
                # flag unscaled this is exactly exp(coefficient).
                row["effect_at_1"] = float(np.exp(coefficient / std))
                row["note"] = (
                    "Change in the odds of winning at least one medal when the "
                    "host flag changes from 0 to 1 (effect_at_1). "
                    "effect_per_1sd is the per-standard-deviation coefficient."
                )
            occurrence_rows.append(row)
        return pd.concat(
            [self.count_model.model_card(stage="count|medal"),
             pd.DataFrame(occurrence_rows)],
            ignore_index=True,
        )


def _occurrence_columns(
    features: Sequence[str],
    available_columns: Iterable[str] | None = None,
) -> list[str]:
    """Pick occurrence-stage features from the training frame.

    ``log_last`` and ``was_zero_last`` are built by ``build_feature_frame`` but
    are not count-stage features, so they are selected from the training
    columns rather than from the count feature list.  ``gold_share3`` and
    ``host`` are only used when the count model enables them.
    """
    available = set(available_columns if available_columns is not None else features)
    enabled = set(features)
    columns = [name for name in OCCURRENCE_FEATURES if name in available]
    for optional in ("gold_share3", HOST_FEATURE):
        if optional in enabled and optional in available:
            columns.append(optional)
    return columns


def _fit_component(
    training_rows: pd.DataFrame,
    features: Sequence[str],
    alpha: float,
) -> _PoissonModel:
    """Fit the Poisson count component on rows with a positive medal total."""
    values = training_rows.reindex(columns=list(features))
    values = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(training_rows["total"], errors="coerce").fillna(0.0).to_numpy(float)
    y = np.clip(y, 0.0, None)

    mean = values.to_numpy(dtype=float).mean(axis=0)
    std = values.to_numpy(dtype=float).std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    for index, name in enumerate(features):
        if name in UNSCALED_FEATURES:
            std[index] = 1.0
    standardised = (values.to_numpy(dtype=float) - mean) / std
    design = np.column_stack([np.ones(len(values)), standardised])
    anchored = pd.to_numeric(
        training_rows.get("prev_total", pd.Series(0.0, index=training_rows.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    offset = np.log(np.maximum(anchored, 0.0) + OFFSET_SHRINK)

    beta, se = _fit_poisson_ridge(design, y, alpha=alpha, offset=offset)
    return _PoissonModel(
        features=tuple(features),
        mean=mean,
        std=std,
        beta=beta,
        se=se,
        alpha=float(alpha),
        n_train_rows=int(len(training_rows)),
        n_train_editions=int(training_rows["year"].nunique()),
        trained_through_year=int(training_rows["year"].max()),
        offset_shrink=float(OFFSET_SHRINK),
    )


def _fit_model(
    training_rows: pd.DataFrame,
    features: Sequence[str],
    alpha: float,
) -> _HurdleModel:
    """Fit the two-part hurdle model on a chronological training window.

    The occurrence stage uses every row (including zero-medal countries and
    debutants in the training window); the count stage uses only rows with
    ``total > 0``.  Both see the same earlier-edition feature frame, so there is
    no leakage between stages.
    """
    if training_rows.empty:
        raise InsufficientDataError("No training rows are available for this edition.")
    features = list(features)
    y_total = pd.to_numeric(training_rows["total"], errors="coerce").fillna(0.0)
    positive = training_rows.loc[y_total > 0].copy()
    if positive.empty:
        raise InsufficientDataError(
            "No medal-winning rows exist in the training window, so the count "
            "stage cannot be estimated."
        )

    # Fit the count stage on medal-winning rows that have a previous total to
    # anchor on.  First-time medal hauls (including successor states appearing
    # for the first time) have prev_total == 0; leaving them in teaches the
    # anchored intercept to be far too large for established teams.  They are
    # still part of the occurrence stage below, so they are not discarded.
    anchored_previous = pd.to_numeric(
        positive.get("prev_total", pd.Series(0.0, index=positive.index)),
        errors="coerce",
    ).fillna(0.0)
    anchored = positive.loc[anchored_previous > 0]
    excluded_unanchored = 0
    if len(anchored) >= MIN_ANCHORED_COUNT_ROWS:
        excluded_unanchored = int(len(positive) - len(anchored))
        positive = anchored

    count_model = _fit_component(positive, features, alpha)

    occurrence_features = _occurrence_columns(features, training_rows.columns)
    if not occurrence_features:
        raise InsufficientDataError(
            "No occurrence-stage features are available in the training frame."
        )
    occurrence_values = training_rows.reindex(columns=occurrence_features)
    occurrence_values = occurrence_values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    occurrence_mean = occurrence_values.to_numpy(dtype=float).mean(axis=0)
    occurrence_std = occurrence_values.to_numpy(dtype=float).std(axis=0, ddof=0)
    occurrence_std = np.where(occurrence_std < 1e-8, 1.0, occurrence_std)
    for index, name in enumerate(occurrence_features):
        if name in UNSCALED_FEATURES:
            occurrence_std[index] = 1.0
    occurrence_design = np.column_stack([
        np.ones(len(training_rows)),
        (occurrence_values.to_numpy(dtype=float) - occurrence_mean) / occurrence_std,
    ])
    occurrence_y = (y_total.to_numpy(float) > 0).astype(float)
    occurrence_beta, occurrence_se = _fit_logistic_ridge(
        occurrence_design, occurrence_y, alpha=alpha
    )

    return _HurdleModel(
        count_model=count_model,
        occurrence_features=tuple(occurrence_features),
        occurrence_mean=occurrence_mean,
        occurrence_std=occurrence_std,
        occurrence_beta=occurrence_beta,
        occurrence_se=occurrence_se,
        alpha=float(alpha),
        n_train_rows=int(len(training_rows)),
        n_train_editions=int(training_rows["year"].nunique()),
        n_train_positive=int(len(positive)),
        trained_through_year=int(training_rows["year"].max()),
        n_unanchored_positive_excluded=int(excluded_unanchored),
        offset_shrink=float(OFFSET_SHRINK),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

MEDAL_PREDICTION_THRESHOLD = 0.5

METRIC_DEFINITIONS: dict[str, tuple[str, str]] = {
    "n_countries": (
        "Number of country-edition observations scored (all participating NOCs, "
        "including zero-medal countries when the data contains them).",
        "context",
    ),
    "n_debutants": (
        "Countries with no prior participation in this season; they receive the "
        "intercept-only estimate (and a 0 persistence baseline).",
        "context",
    ),
    "n_medal_winners": ("Observations with actual_total > 0.", "context"),
    "n_zero_medal": ("Observations with actual_total == 0.", "context"),
    "mae_model": (
        "Mean absolute error: average |predicted - actual| medals per country. "
        "An MAE of 3 means predictions are off by 3 medals per country on average. "
        "It is dominated by the many zero-medal countries, so read it with the "
        "split metrics below.",
        "lower",
    ),
    "mae_baseline_last": (
        "MAE of the persistence baseline: predict the country's medal total from "
        "the immediately previous edition.",
        "lower",
    ),
    "mae_baseline_mean3": (
        "MAE of the baseline that predicts the mean of the previous three editions.",
        "lower",
    ),
    "mae_improvement_pct": (
        "Percentage reduction in MAE versus the persistence baseline: "
        "(baseline MAE - model MAE) / baseline MAE x 100. Positive is better; "
        "negative means the model lost to the simple baseline.",
        "higher",
    ),
    "rmse_model": (
        "Root mean squared error; punishes large misses (usually big medal teams) "
        "more than MAE.",
        "lower",
    ),
    "median_ae_model": (
        "Median absolute error; robust to a few huge misses. For sparse data this "
        "is often 0 because most countries win no medals.",
        "lower",
    ),
    "bias_model": (
        "Mean signed error (actual - predicted). Positive means the model "
        "under-predicts on average; negative means it over-predicts.",
        "closer to 0",
    ),
    "mae_model_medal_winners": (
        "MAE among countries that actually won at least one medal. This is the "
        "most useful accuracy view for the medal table.",
        "lower",
    ),
    "mae_model_zero_medal": (
        "MAE among countries that won no medal. Shows how noisy predictions are "
        "for the long tail of small NOCs.",
        "lower",
    ),
    "mae_baseline_medal_winners": (
        "Persistence-baseline MAE among countries that actually won a medal. "
        "Use it to see whether the model really improves medal-winner accuracy.",
        "lower",
    ),
    "mae_baseline_zero_medal": (
        "Persistence-baseline MAE among countries that won no medal. Predicting "
        "the previous total is usually excellent here because it is often 0.",
        "lower",
    ),
    "rmse_baseline_last": (
        "RMSE of the persistence baseline (previous edition's total).",
        "lower",
    ),
    "median_ae_baseline_last": (
        "Median absolute error of the persistence baseline; robust to a few huge "
        "misses.",
        "lower",
    ),
    "zero_medal_false_positive_rate_baseline": (
        "Share of actual zero-medal countries where the persistence baseline "
        "expects at least the medal threshold (normally 0 because it repeats the "
        "previous zero).",
        "lower",
    ),
    "zero_medal_false_positive_rate_model": (
        "Share of actual zero-medal countries that the model expects to win at "
        f"least {MEDAL_PREDICTION_THRESHOLD} medal. Lower is better.",
        "lower",
    ),
    "brier_medal_occurrence_model": (
        "Brier score for the occurrence stage: mean squared error between "
        "P(medal > 0) and the actual 0/1 medal outcome. Lower is better; 0 is "
        "perfect. This scores the hurdle's first stage directly instead of only "
        "looking at the final medal total.",
        "lower",
    ),
    "brier_medal_occurrence_baseline": (
        "Brier score of the persistence occurrence rule (probability 1 if the "
        "country medalled in the previous edition, otherwise 0). Compare the "
        "model against this to see whether the probability stage adds value.",
        "lower",
    ),
    "spearman_model": (
        "Spearman rank correlation between predicted and actual totals. Measures "
        "whether the ordering of countries is right, even if magnitudes are off. "
        "In the POOLED row this is the mean of the per-edition values.",
        "higher",
    ),
    "spearman_baseline_last": (
        "Spearman rank correlation of the persistence baseline; the model's "
        "ranking should be read against this. In the POOLED row this is the mean "
        "of the per-edition values.",
        "higher",
    ),
    "top10_overlap_model": (
        "Number of countries shared by the actual top 10 and predicted top 10 "
        "(k = min(10, number of scored countries)). In the POOLED row this is the "
        "mean of the per-edition values, because ranking rows mixed from several "
        "editions in one table is not meaningful.",
        "higher",
    ),
    "top10_overlap_baseline_last": (
        "Number of countries shared by the actual top 10 and the persistence "
        "baseline's top 10. In the POOLED row this is the mean of the per-edition "
        "values.",
        "higher",
    ),
}


def explain_metrics() -> pd.DataFrame:
    """Return plain-language metric definitions for the app/README."""
    return pd.DataFrame(
        [
            {"metric": metric, "meaning": meaning, "better_when": better}
            for metric, (meaning, better) in METRIC_DEFINITIONS.items()
        ]
    )


def _safe_mae(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def _mae_improvement(baseline: float, model: float) -> float:
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(model):
        return float("nan")
    return float((baseline - model) / baseline * 100.0)


def _spearman(first: pd.Series, second: pd.Series) -> float:
    if len(first) < 2:
        return float("nan")
    ranks_first = first.rank(method="average")
    ranks_second = second.rank(method="average")
    if ranks_first.std(ddof=0) == 0 or ranks_second.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_first.to_numpy(float), ranks_second.to_numpy(float))[0, 1])


def _top_k_overlap(
    frame: pd.DataFrame,
    predicted_col: str = "predicted_total",
    k: int = 10,
) -> float:
    k = min(k, len(frame))
    if k == 0:
        return float("nan")
    actual_top = set(frame.nlargest(k, "actual_total")["noc"].astype(str))
    predicted_top = set(frame.nlargest(k, predicted_col)["noc"].astype(str))
    return float(len(actual_top & predicted_top))


def compute_metrics(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual_total",
    predicted_col: str = "predicted_total",
    baseline_col: str = "baseline_last",
    baseline_mean3_col: str = "baseline_mean3",
) -> dict[str, float]:
    """Compute backtest metrics for one test edition or pooled folds.

    The function requires ``noc`` plus the actual/predicted columns.  It returns
    NaNs rather than inventing values when a group (e.g. zero-medal countries)
    is empty.  See :data:`METRIC_DEFINITIONS` for interpretations.

    Notes
    -----
    MAE, RMSE, bias and Brier scores can be pooled across editions.  Spearman
    and top-10 overlap cannot meaningfully be pooled, because a pooled table
    mixes different medal tables; for a multi-edition comparison average those
    metrics over editions (``_metrics_table`` does this for the POOLED row).
    """
    required = {"noc", actual_col, predicted_col}
    missing = required - set(frame.columns)
    if missing:
        raise OlympicDataError(f"compute_metrics is missing columns: {sorted(missing)}")
    data = frame.dropna(subset=[actual_col, predicted_col]).copy()
    if data.empty:
        return {
            "n_countries": 0, "n_debutants": 0, "n_medal_winners": 0,
            "n_zero_medal": 0, "mae_model": float("nan"),
            "mae_baseline_last": float("nan"), "mae_baseline_mean3": float("nan"),
            "mae_improvement_pct": float("nan"), "rmse_model": float("nan"),
            "rmse_baseline_last": float("nan"), "median_ae_model": float("nan"),
            "median_ae_baseline_last": float("nan"), "bias_model": float("nan"),
            "mae_model_medal_winners": float("nan"),
            "mae_baseline_medal_winners": float("nan"),
            "mae_model_zero_medal": float("nan"),
            "mae_baseline_zero_medal": float("nan"),
            "zero_medal_false_positive_rate_model": float("nan"),
            "zero_medal_false_positive_rate_baseline": float("nan"),
            "brier_medal_occurrence_model": float("nan"),
            "brier_medal_occurrence_baseline": float("nan"),
            "spearman_model": float("nan"), "spearman_baseline_last": float("nan"),
            "top10_overlap_model": float("nan"), "top10_overlap_baseline_last": float("nan"),
        }

    actual = data[actual_col].to_numpy(float)
    predicted = data[predicted_col].to_numpy(float)
    error = actual - predicted
    zero_mask = actual == 0
    medal_mask = actual > 0

    has_baseline = baseline_col in data.columns
    baseline = data[baseline_col].fillna(0).to_numpy(float) if has_baseline else np.full(len(data), np.nan)
    baseline_error = actual - baseline

    has_mean3 = baseline_mean3_col in data.columns
    baseline3 = data[baseline_mean3_col].fillna(0).to_numpy(float) if has_mean3 else np.full(len(data), np.nan)
    mae_mean3 = float(np.mean(np.abs(actual - baseline3))) if has_mean3 else float("nan")

    mae_model = float(np.mean(np.abs(error)))
    mae_baseline = float(np.mean(np.abs(baseline_error))) if has_baseline else float("nan")
    zero_predicted = predicted[zero_mask] >= MEDAL_PREDICTION_THRESHOLD
    zero_predicted_baseline = baseline[zero_mask] >= MEDAL_PREDICTION_THRESHOLD

    medal_binary = (actual > 0).astype(float)
    if "predicted_medal_probability" in data.columns:
        probability = (
            data["predicted_medal_probability"].astype(float).clip(0.0, 1.0).to_numpy()
        )
        brier_model = float(np.mean((medal_binary - probability) ** 2))
    else:
        brier_model = float("nan")
    if has_baseline:
        baseline_probability = (baseline > 0).astype(float)
        brier_baseline = float(np.mean((medal_binary - baseline_probability) ** 2))
    else:
        brier_baseline = float("nan")

    return {
        "n_countries": int(len(data)),
        "n_debutants": int((~data["seen_before"].fillna(True).astype(bool)).sum())
        if "seen_before" in data.columns else 0,
        "n_medal_winners": int(medal_mask.sum()),
        "n_zero_medal": int(zero_mask.sum()),
        "mae_model": mae_model,
        "mae_baseline_last": mae_baseline,
        "mae_baseline_mean3": mae_mean3,
        "mae_improvement_pct": _mae_improvement(mae_baseline, mae_model),
        "rmse_model": float(np.sqrt(np.mean(error ** 2))),
        "rmse_baseline_last": float(np.sqrt(np.mean(baseline_error ** 2))) if has_baseline else float("nan"),
        "median_ae_model": float(np.median(np.abs(error))),
        "median_ae_baseline_last": float(np.median(np.abs(baseline_error))) if has_baseline else float("nan"),
        "bias_model": float(np.mean(error)),
        "mae_model_medal_winners": _safe_mae(actual, predicted, medal_mask),
        "mae_baseline_medal_winners": _safe_mae(actual, baseline, medal_mask) if has_baseline else float("nan"),
        "mae_model_zero_medal": _safe_mae(actual, predicted, zero_mask),
        "mae_baseline_zero_medal": _safe_mae(actual, baseline, zero_mask) if has_baseline else float("nan"),
        "zero_medal_false_positive_rate_model": float(np.mean(zero_predicted)) if np.any(zero_mask) else float("nan"),
        "zero_medal_false_positive_rate_baseline": float(np.mean(zero_predicted_baseline)) if np.any(zero_mask) and has_baseline else float("nan"),
        "brier_medal_occurrence_model": brier_model,
        "brier_medal_occurrence_baseline": brier_baseline,
        "spearman_model": _spearman(data[actual_col], data[predicted_col]),
        "spearman_baseline_last": _spearman(data[actual_col], data[baseline_col]) if has_baseline else float("nan"),
        "top10_overlap_model": _top_k_overlap(data),
        "top10_overlap_baseline_last": _top_k_overlap(data, predicted_col=baseline_col) if has_baseline else float("nan"),
    }


# ---------------------------------------------------------------------------
# Data-quality diagnostics
# ---------------------------------------------------------------------------


def data_quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Report coverage and sufficiency before trusting a forecast.

    The most important warning is "zero_medal_rows_present": if an aggregated
    input contains only medalling countries, the long tail is missing and MAE
    will look better than it really is.
    """
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise OlympicDataError("data_quality_report needs a non-empty prepared panel.")

    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, status: str, comment: str) -> None:
        rows.append({
            "metric": metric,
            "value": value,
            "status": status,
            "comment": comment,
        })

    total_rows = len(panel)
    countries = panel["noc"].nunique()
    seasons = sorted(panel["season"].dropna().unique().tolist())
    zero_rows = int((pd.to_numeric(panel["total"], errors="coerce").fillna(0) == 0).sum())
    medal_rows = total_rows - zero_rows

    add("rows", total_rows, "info", "One row per participating NOC per edition.")
    add("seasons", ", ".join(seasons), "info", "Summer and Winter are modelled separately.")
    add("countries", int(countries), "info", "Distinct NOC codes in the historical panel.")
    add("first_year", int(panel["year"].min()), "info", "Earliest edition present.")
    add("last_year", int(panel["year"].max()), "info", "Latest edition present.")
    add("zero_medal_rows", zero_rows, "ok" if zero_rows > 0 else "warning",
        "Zero-medal participants are present." if zero_rows > 0 else
        "No zero-medal rows found. If this is an aggregated medal table, missing "
        "countries make MAE optimistic; supply participants= to fill them in.")
    add("zero_medal_share", round(zero_rows / total_rows, 4) if total_rows else float("nan"),
        "info", "Most Olympic countries win no medal, so overall MAE is dominated by zeros.")
    add("medal_winning_rows", medal_rows, "info", "Rows with at least one medal.")

    for season in seasons:
        subset = panel.loc[panel["season"] == season]
        editions = sorted(int(y) for y in subset["year"].unique())
        status = "ok" if len(editions) >= MIN_TRAIN_EDITIONS + 1 else "warning"
        add(f"editions[{season}]", len(editions), status,
            f"{editions[0]}-{editions[-1]}; need at least {MIN_TRAIN_EDITIONS + 1} "
            f"editions for a {MIN_TRAIN_EDITIONS}-edition minimum training window plus one test year."
            if editions else "No editions.")
        sparse = subset.groupby("noc").size()
        add(f"countries_with_one_edition[{season}]", int((sparse == 1).sum()), "info",
            "Countries with a single appearance are the sparsest group; their "
            "features are mostly zeros and their forecasts are uncertain.")

    has_gold = bool(panel["gold"].notna().any())
    add("gold_available", has_gold, "ok" if has_gold else "warning",
        "Gold medals present." if has_gold else
        "No gold medals in the input; predicted_gold will be NaN, not fabricated.")
    add("host_rows", int(pd.to_numeric(panel["host"], errors="coerce").fillna(0).sum()),
        "info", "Historical host-country observations used to estimate the host flag.")

    removed = panel.attrs.get("duplicate_rows_removed")
    add("duplicate_rows_removed", removed if removed is not None else 0, "info",
        "Duplicate (season, year, noc) rows summed during preparation."
        if removed is not None else "Not tracked for this panel.")
    excluded = panel.attrs.get("excluded_1906_rows")
    add("excluded_1906_rows", excluded if excluded is not None else 0, "info",
        "1906 Intercalated Games excluded as a non-recognised edition."
        if excluded else "No 1906 rows present.")

    report = pd.DataFrame(rows)
    return report

# ---------------------------------------------------------------------------
# Chronological validation
# ---------------------------------------------------------------------------


def chronological_backtest(
    panel: pd.DataFrame,
    *,
    season: str = SUMMER,
    test_years: Sequence[int] | None = None,
    n_test_editions: int = 3,
    min_train_editions: int = MIN_TRAIN_EDITIONS,
    min_history_year: int | None = DEFAULT_MIN_HISTORY_YEAR,
    active_window: int | None = DEFAULT_ACTIVE_WINDOW,
    alpha: float = 1.0,
    use_host: bool = True,
) -> BacktestResult:
    """Expanding-window, strictly chronological backtest.

    For each test edition Y the model is fitted on editions earlier than Y and
    scored on every NOC that actually participated in Y (including zero-medal
    NOCs when present).  The candidate list for Y is built from data before Y
    only, so defunct teams cannot leak in.  No random split is used anywhere.

    Parameters
    ----------
    panel:
        Prepared country/edition panel.
    season:
        'Summer' or 'Winter'.
    test_years:
        Explicit test editions.  Defaults to the last ``n_test_editions``
        eligible editions.
    n_test_editions:
        Number of most recent eligible editions to test when ``test_years`` is
        not given.
    min_train_editions:
        Minimum number of earlier editions required in each fold.
    min_history_year:
        Training-window start.  Features still use all earlier editions; this
        only excludes very old editions from model fitting.  ``None`` uses all.
    active_window:
        Candidate NOCs must have appeared in one of the last N editions before
        the target year.  Removes defunct teams such as URS/GDR without looking
        ahead.
    alpha:
        Ridge penalty for the Poisson model.
    use_host:
        Include the estimated host flag.  Set False to test sensitivity.

    Returns
    -------
    BacktestResult
        ``rows`` (country-level errors) and ``folds`` (training-window metadata).
    """
    season = _normalise_season_value(season)
    if min_train_editions < 1:
        raise ValueError("min_train_editions must be at least 1.")
    if n_test_editions < 1:
        raise ValueError("n_test_editions must be at least 1.")

    season_history = panel.loc[panel["season"] == season].copy()
    if season_history.empty:
        raise InsufficientDataError(f"No {season} rows are available for backtesting.")
    season_history["year"] = pd.to_numeric(season_history["year"], errors="coerce")
    season_history = season_history.dropna(subset=["year"])
    editions = sorted(int(y) for y in season_history["year"].unique())

    def training_editions_for(year: int) -> list[int]:
        return [
            edition
            for edition in editions
            if edition < year
            and (min_history_year is None or edition >= int(min_history_year))
        ]

    if test_years is None:
        eligible = [
            year for year in editions
            if len(training_editions_for(year)) >= min_train_editions
        ]
        if not eligible:
            raise InsufficientDataError(
                f"Not enough {season} editions for a chronological backtest. "
                f"Found {len(editions)} editions ({editions[0]}-{editions[-1]}) "
                f"with min_history_year={min_history_year}; need at least "
                f"{min_train_editions + 1} editions. Add more history, lower "
                "min_train_editions, or set min_history_year=None."
            )
        selected_years = eligible[-int(n_test_editions):]
    else:
        selected_years = sorted(int(year) for year in test_years)
        unknown = [year for year in selected_years if year not in editions]
        if unknown:
            raise InsufficientDataError(
                f"test_years {unknown} are not present in the {season} data."
            )
        for year in selected_years:
            available = training_editions_for(year)
            if len(available) < min_train_editions:
                raise InsufficientDataError(
                    f"Cannot test {year}: only {len(available)} training editions "
                    f"before it (min_history_year={min_history_year}); require "
                    f"{min_train_editions}."
                )

    row_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    notes: list[str] = [
        "Chronological validation only: every fold trains on editions before its "
        "test edition and is scored on all participants of that edition. No "
        "random split or future information is used.",
    ]

    for year in selected_years:
        prior_panel = season_history.loc[season_history["year"] < year].copy()
        if prior_panel.empty:
            raise InsufficientDataError(f"No history exists before test year {year}.")

        features_frame = build_feature_frame(prior_panel, season, target_year=year)
        training = features_frame.loc[features_frame["participated"] == 1].copy()
        if min_history_year is not None:
            training = training.loc[training["year"] >= int(min_history_year)]
        training_editions = sorted(int(y) for y in training["year"].unique())
        if len(training_editions) < min_train_editions:
            raise InsufficientDataError(
                f"Test year {year}: only {len(training_editions)} usable training "
                f"editions after min_history_year={min_history_year}; require "
                f"{min_train_editions}."
            )

        has_gold = bool(features_frame["gold"].notna().any())
        model_features = _feature_columns(has_gold=has_gold, use_host=use_host)
        model = _fit_model(training, model_features, alpha=alpha)

        candidates = _target_candidates(
            features_frame,
            prior_panel,
            season,
            year,
            target_participants=None,
            active_window=active_window,
        )
        predictions = model.predict(candidates)
        medal_probability = model.predict_probability(candidates)
        count_if_medal = model.predict_count(candidates)
        candidate_frame = candidates[["noc", "n_prior", "seen_before"]].copy()
        candidate_frame["predicted_total"] = predictions
        candidate_frame["predicted_medal_probability"] = medal_probability
        candidate_frame["predicted_count_if_medal"] = count_if_medal
        candidate_frame["predicted_gold"] = (
            predictions * candidates["gold_share3"].fillna(0.0).to_numpy(float)
            if has_gold else np.nan
        )

        actual = (
            season_history.loc[season_history["year"] == year,
                               ["noc", "country", "host", "total", "gold"]]
            .rename(columns={"total": "actual_total", "gold": "actual_gold"})
            .copy()
        )

        evaluation = actual.merge(
            candidate_frame, on="noc", how="left", validate="one_to_one"
        )
        evaluation["predicted_total"] = pd.to_numeric(
            evaluation["predicted_total"], errors="coerce"
        ).fillna(0.0).clip(lower=0)
        evaluation["predicted_medal_probability"] = pd.to_numeric(
            evaluation["predicted_medal_probability"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        evaluation["predicted_count_if_medal"] = pd.to_numeric(
            evaluation["predicted_count_if_medal"], errors="coerce"
        ).fillna(0.0).clip(lower=0)
        evaluation["predicted_gold"] = pd.to_numeric(
            evaluation["predicted_gold"], errors="coerce"
        ).fillna(0.0).clip(lower=0)
        evaluation["seen_before"] = evaluation["seen_before"].fillna(False).astype(bool)
        evaluation["n_prior_editions"] = (
            pd.to_numeric(evaluation["n_prior"], errors="coerce").fillna(0).astype(int)
        )

        baselines = features_frame.loc[
            features_frame["year"] == year, ["noc", "prev_total", "prev3_mean"]
        ].rename(columns={"prev_total": "baseline_last", "prev3_mean": "baseline_mean3"})
        evaluation = evaluation.merge(baselines, on="noc", how="left", validate="one_to_one")
        evaluation["baseline_last"] = pd.to_numeric(
            evaluation["baseline_last"], errors="coerce"
        ).fillna(0.0).clip(lower=0)
        evaluation["baseline_mean3"] = pd.to_numeric(
            evaluation["baseline_mean3"], errors="coerce"
        ).fillna(0.0).clip(lower=0)

        evaluation["country"] = evaluation["country"].astype("string").fillna(evaluation["noc"])
        evaluation["error"] = evaluation["actual_total"] - evaluation["predicted_total"]
        evaluation["abs_error"] = evaluation["error"].abs()
        evaluation["baseline_abs_error"] = (
            evaluation["actual_total"] - evaluation["baseline_last"]
        ).abs()
        evaluation["baseline_mean3_abs_error"] = (
            evaluation["actual_total"] - evaluation["baseline_mean3"]
        ).abs()
        evaluation["season"] = season
        evaluation["test_year"] = int(year)
        evaluation["trained_through_year"] = model.trained_through_year
        evaluation["n_train_editions"] = model.n_train_editions
        evaluation["n_train_rows"] = model.n_train_rows

        row_frames.append(evaluation)

        fold_rows.append({
            "season": season,
            "test_year": int(year),
            "n_train_editions": model.n_train_editions,
            "trained_through_year": model.trained_through_year,
            "n_train_rows": model.n_train_rows,
            "n_count_training_rows": int(model.n_train_positive),
            "n_unanchored_positive_excluded": int(model.n_unanchored_positive_excluded),
            "n_candidates": int(len(candidates)),
            "n_actual_participants": int(len(actual)),
            "n_debutants": int((~evaluation["seen_before"]).sum()),
            "alpha": float(alpha),
            "use_host": bool(use_host),
            "min_history_year": int(min_history_year) if min_history_year is not None else np.nan,
            "active_window": int(active_window) if active_window is not None else np.nan,
            "host_observations_in_training": int(training["host"].sum()) if use_host else 0,
        })
        notes.append(
            f"Test {season} {year}: trained through {model.trained_through_year} "
            f"({model.n_train_editions} editions, {model.n_train_rows} rows); "
            f"{len(candidates)} candidates, {len(actual)} participants, "
            f"{int((~evaluation['seen_before']).sum())} debutants scored at 0."
        )

    rows = pd.concat(row_frames, ignore_index=True)
    rows = rows.sort_values(["test_year", "actual_total", "noc"],
                            ascending=[True, False, True]).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    if use_host:
        low_host = folds.loc[folds["host_observations_in_training"] < 5, "test_year"].tolist()
        if low_host:
            notes.append(
                "Host effect is weakly identified in fold(s) "
                f"{low_host} (fewer than 5 host observations in training); treat "
                "the host coefficient as uncertain."
            )
    excluded_total = int(folds["n_unanchored_positive_excluded"].sum())
    if excluded_total:
        notes.append(
            "The count stage was fitted on anchored medal-winning rows only "
            f"({excluded_total} first-time/successor medal-winning rows across "
            "all folds had prev_total == 0 and were excluded from that stage). "
            "The occurrence stage still uses every training row."
        )
    return BacktestResult(rows=rows, folds=folds, notes=notes)


def _metrics_table(backtest: BacktestResult, season: str) -> pd.DataFrame:
    """Build per-edition and pooled metric rows from backtest output."""
    metric_rows: list[dict[str, Any]] = []
    fold_lookup = backtest.folds.set_index("test_year")

    for year in sorted(backtest.rows["test_year"].unique()):
        group = backtest.rows.loc[backtest.rows["test_year"] == year]
        row = compute_metrics(group)
        row["season"] = season
        row["test_year"] = int(year)
        fold = fold_lookup.loc[int(year)]
        row["trained_through_year"] = int(fold["trained_through_year"])
        row["n_train_editions"] = int(fold["n_train_editions"])
        metric_rows.append(row)

    pooled = compute_metrics(backtest.rows)
    # Rank metrics are only meaningful within one edition: a pooled table mixes
    # e.g. the 2008 and 2016 medal tables, so a global top-10 across pooled rows
    # is dominated by whichever edition happens to have larger totals.  Report
    # the mean of the per-edition values instead, and say so in the definitions.
    rank_metrics = (
        "spearman_model",
        "spearman_baseline_last",
        "top10_overlap_model",
        "top10_overlap_baseline_last",
    )
    for metric in rank_metrics:
        values = [row[metric] for row in metric_rows if np.isfinite(row[metric])]
        pooled[metric] = float(np.mean(values)) if values else float("nan")
    pooled["season"] = season
    pooled["test_year"] = "POOLED"
    pooled["trained_through_year"] = int(backtest.folds["trained_through_year"].max())
    pooled["n_train_editions"] = float("nan")
    metric_rows.append(pooled)

    columns = [
        "season", "test_year", "n_countries", "n_debutants", "n_medal_winners",
        "n_zero_medal", "mae_model", "mae_baseline_last", "mae_baseline_mean3",
        "mae_improvement_pct", "rmse_model", "rmse_baseline_last",
        "median_ae_model", "median_ae_baseline_last", "bias_model",
        "mae_model_medal_winners", "mae_baseline_medal_winners",
        "mae_model_zero_medal", "mae_baseline_zero_medal",
        "zero_medal_false_positive_rate_model",
        "zero_medal_false_positive_rate_baseline",
        "brier_medal_occurrence_model", "brier_medal_occurrence_baseline",
        "spearman_model", "spearman_baseline_last",
        "top10_overlap_model", "top10_overlap_baseline_last",
        "trained_through_year", "n_train_editions",
    ]
    table = pd.DataFrame(metric_rows)
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan
    return table[columns]


# ---------------------------------------------------------------------------
# Main forecasting entry point
# ---------------------------------------------------------------------------


def _residual_dispersion(backtest: BacktestResult) -> float:
    """Estimate Poisson overdispersion from chronological residuals.

    phi = sum((actual - predicted)^2) / sum(max(predicted, 0.5)).  phi > 1 means
    errors are larger than a pure Poisson model would imply; it widens the
    approximate intervals.  The clip avoids absurd intervals from tiny folds.
    """
    residuals = (
        pd.to_numeric(backtest.rows["actual_total"], errors="coerce")
        - pd.to_numeric(backtest.rows["predicted_total"], errors="coerce")
    ).to_numpy(float)
    denominator = float(np.maximum(
        pd.to_numeric(backtest.rows["predicted_total"], errors="coerce").to_numpy(float),
        0.5,
    ).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    phi = float(np.sum(residuals ** 2) / denominator)
    return float(np.clip(phi, 0.25, 50.0))


def forecast_medals(
    history: pd.DataFrame | None = None,
    *,
    target_year: int | None = None,
    season: str = SUMMER,
    participants: pd.DataFrame | None = None,
    target_participants: pd.DataFrame | Sequence[str] | None = None,
    country_names: Mapping[str, str] | pd.DataFrame | None = None,
    n_test_editions: int = 3,
    min_train_editions: int = MIN_TRAIN_EDITIONS,
    min_history_year: int | None = DEFAULT_MIN_HISTORY_YEAR,
    active_window: int | None = DEFAULT_ACTIVE_WINDOW,
    alpha: float = 1.0,
    use_host: bool = True,
    top_n: int | None = None,
    data_path: str | os.PathLike[str] | None = None,
) -> ForecastResult:
    """Forecast a future Olympic medal table with chronological validation.

    Parameters
    ----------
    history:
        Raw athlete/event rows or an aggregated country/edition table.  If
        omitted, :func:`load_history` is called (``data_path`` or data_loader).
    target_year:
        Edition to predict.  Defaults to the latest available edition + 4.
        Must be later than the latest historical edition.
    season:
        'Summer' (default) or 'Winter'.  Medals are not comparable across
        seasons, so they are modelled separately.
    participants:
        Optional historical participation table used to fill in missing
        zero-medal NOCs in an aggregated input.  Not the target entry list.
    target_participants:
        Optional entry list for the target edition (DataFrame with ``noc`` or a
        sequence of NOC codes).  If omitted, countries active in the last
        ``active_window`` editions are forecast.
    country_names:
        Optional NOC -> name mapping.
    n_test_editions / min_train_editions / min_history_year / active_window /
    alpha / use_host:
        Validation and model settings.  Defaults are fixed in advance; the test
        editions are never used for hyper-parameter tuning.
    top_n:
        Optionally truncate the returned predictions (ranking is still global).
    data_path:
        CSV/Parquet path used when ``history`` is omitted.

    Returns
    -------
    ForecastResult
        See the class docstring for the DataFrame schema and caveats.

    Raises
    ------
    InsufficientDataError
        If the supplied data cannot support a chronological backtest.  The
        message says exactly how many editions are needed.
    """
    if top_n is not None and int(top_n) < 1:
        raise ValueError("top_n must be a positive integer or None.")
    season = _normalise_season_value(season)

    if history is None:
        history = load_history(data_path)

    panel = prepare_country_edition_panel(
        history,
        participants=participants,
        country_names=country_names,
    )
    season_panel = panel.loc[panel["season"] == season].copy()
    if season_panel.empty:
        raise InsufficientDataError(
            f"No {season} rows are available. Seasons present: "
            f"{sorted(panel['season'].dropna().unique().tolist())}."
        )
    editions = sorted(int(y) for y in season_panel["year"].unique())

    target_was_implicit = target_year is None
    if target_year is None:
        target_year = int(editions[-1]) + 4
    target_year = int(target_year)
    if target_year <= editions[-1]:
        raise ValueError(
            f"target_year={target_year} must be later than the latest historical "
            f"edition ({editions[-1]}). Use chronological_backtest() to evaluate "
            "past editions."
        )

    if len(editions) < min_train_editions + 1:
        raise InsufficientDataError(
            f"{season} data has only {len(editions)} editions ({editions[0]}-"
            f"{editions[-1]}). Need at least {min_train_editions + 1} for a "
            f"{min_train_editions}-edition training window plus a test edition."
        )

    backtest = chronological_backtest(
        panel,
        season=season,
        n_test_editions=n_test_editions,
        min_train_editions=min_train_editions,
        min_history_year=min_history_year,
        active_window=active_window,
        alpha=alpha,
        use_host=use_host,
    )
    metrics = _metrics_table(backtest, season)
    dispersion = _residual_dispersion(backtest)

    # Final model uses all editions prior to the target year.
    features_frame = build_feature_frame(panel, season, target_year=target_year)
    training = features_frame.loc[features_frame["participated"] == 1].copy()
    if min_history_year is not None:
        training = training.loc[training["year"] >= int(min_history_year)]
    if training["year"].nunique() < min_train_editions:
        raise InsufficientDataError(
            f"Final model has only {training['year'].nunique()} usable training "
            f"editions after min_history_year={min_history_year}; need "
            f"{min_train_editions}."
        )

    has_gold = bool(panel.loc[panel["season"] == season, "gold"].notna().any())
    model_features = _feature_columns(has_gold=has_gold, use_host=use_host)
    model = _fit_model(training, model_features, alpha=alpha)

    candidates = _target_candidates(
        features_frame,
        panel,
        season,
        target_year,
        target_participants=target_participants,
        active_window=active_window,
        country_names=country_names if isinstance(country_names, Mapping) else None,
    )
    predicted_total = model.predict(candidates)
    medal_probability = model.predict_probability(candidates)
    count_if_medal = model.predict_count(candidates)
    standard_error = np.sqrt(dispersion * np.maximum(predicted_total, 0.5))
    lower_80 = np.maximum(predicted_total - Z80 * standard_error, 0.0)
    upper_80 = predicted_total + Z80 * standard_error

    gold_share = pd.to_numeric(candidates["gold_share3"], errors="coerce").fillna(0.0)
    predicted_gold = predicted_total * gold_share.to_numpy(float) if has_gold else np.full(len(candidates), np.nan)
    seen_before = candidates["n_prior"].to_numpy() > 0

    note_values: list[str] = []
    for is_seen, is_host in zip(seen_before, candidates["host"].to_numpy()):
        parts: list[str] = []
        if not is_seen:
            parts.append("No prior participation in this season; intercept-only estimate.")
        if is_host:
            parts.append("Host country; host effect estimated from historical data.")
        note_values.append(" ".join(parts))

    predictions = pd.DataFrame({
        "noc": candidates["noc"].astype(str).to_numpy(),
        "country": candidates["country"].astype(str).to_numpy(),
        "season": season,
        "target_year": target_year,
        "predicted_total": np.round(predicted_total, 3),
        "predicted_total_rounded": np.rint(predicted_total).astype(int),
        "medal_probability": np.round(medal_probability, 4),
        "count_if_medal": np.round(count_if_medal, 3),
        "predicted_gold": np.round(predicted_gold, 3),
        "lower_80": np.round(lower_80, 3),
        "upper_80": np.round(upper_80, 3),
        "last_total": pd.to_numeric(candidates["prev_total"], errors="coerce").fillna(0.0).to_numpy(),
        "last_year": pd.to_numeric(candidates["last_year"], errors="coerce").to_numpy(),
        "mean_last3": pd.to_numeric(candidates["prev3_mean"], errors="coerce").fillna(0.0).to_numpy(),
        "trend": pd.to_numeric(candidates["trend"], errors="coerce").fillna(0.0).to_numpy(),
        "n_prior_editions": candidates["n_prior"].fillna(0).astype(int).to_numpy(),
        "host": candidates["host"].fillna(0).astype(bool).to_numpy(),
        "seen_before": seen_before,
        "note": note_values,
    })
    predictions["last_year"] = predictions["last_year"].astype("Int64")
    predictions = predictions.sort_values(
        ["predicted_total", "predicted_gold", "last_total", "noc"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    predictions.insert(0, "rank", np.arange(1, len(predictions) + 1))
    if top_n is not None:
        predictions = predictions.head(int(top_n)).reset_index(drop=True)

    # --- notes: all statements are computed from this run, none are assumed ---
    notes: list[str] = []
    test_year_text = ", ".join(str(int(y)) for y in backtest.folds["test_year"].tolist())
    notes.append(
        f"Forecast for {season} {target_year}. Chronological backtest editions: "
        f"{test_year_text}; each fold trained only on earlier editions."
    )
    pooled = metrics.loc[metrics["test_year"] == "POOLED"].iloc[0]
    improvement = pooled["mae_improvement_pct"]
    if np.isfinite(improvement):
        if improvement >= 0:
            notes.append(
                f"Pooled backtest MAE: model {pooled['mae_model']:.2f} medals vs "
                f"persistence {pooled['mae_baseline_last']:.2f} "
                f"({improvement:+.1f}% improvement); vs last-3-editions mean "
                f"{pooled['mae_baseline_mean3']:.2f}."
            )
        else:
            notes.append(
                f"The model did NOT beat the persistence baseline on pooled MAE: "
                f"model {pooled['mae_model']:.2f} vs {pooled['mae_baseline_last']:.2f} "
                f"({improvement:+.1f}%). Treat the forecast as an experimental "
                "estimate and prefer the baseline if a simpler number is needed."
            )
    notes.append(
        "The forecast is a two-part hurdle model: P(medal > 0) from a logistic "
        "occurrence stage multiplied by E[medals | medal > 0] from a Poisson "
        "count stage fitted on medal-winning rows only. This keeps the many "
        "zero-medal countries from dragging down the medal-winner estimates."
    )
    notes.append(
        f"The count stage used {model.n_train_positive} anchored medal-winning "
        f"training rows (previous total > 0). {model.n_unanchored_positive_excluded} "
        "medal-winning rows with no previous total (first appearances and "
        "successor states) were excluded from that stage so they could not "
        "inflate the anchor; the occurrence stage still uses every training row."
    )
    brier_model = pooled.get("brier_medal_occurrence_model", np.nan)
    brier_baseline = pooled.get("brier_medal_occurrence_baseline", np.nan)
    if np.isfinite(brier_model) and np.isfinite(brier_baseline):
        direction = "better than" if brier_model < brier_baseline else "not better than"
        notes.append(
            f"Occurrence-stage Brier score: model {brier_model:.4f} vs persistence "
            f"rule {brier_baseline:.4f} ({direction} the persistence rule; lower is better)."
        )
    notes.append(
        "80% intervals are approximate: point estimate +/- 1.28 * sqrt(phi * "
        f"predicted), with residual overdispersion phi={dispersion:.2f} estimated "
        "from the backtest. They do not cover model uncertainty, sport-level "
        "shocks or selection changes."
    )
    if not has_gold:
        notes.append(
            "The input has no gold-medal column, so predicted_gold is NaN. No gold "
            "medals were invented."
        )
    else:
        notes.append(
            "predicted_gold is a heuristic: the country's gold share over its last "
            "three prior editions multiplied by predicted_total. It is not a "
            "separately validated gold model."
        )
    zero_rows = int((pd.to_numeric(panel["total"], errors="coerce").fillna(0) == 0).sum())
    if zero_rows == 0:
        notes.append(
            "WARNING: the prepared panel contains no zero-medal rows. If this is a "
            "medal-only table, zero-medal countries are missing and backtest MAE is "
            "optimistic. Supply participants= to fill them in."
        )
    if target_participants is None:
        notes.append(
            f"target_participants was not supplied. Candidates are NOCs active in "
            f"the last {active_window} {season} edition(s) before {target_year}; "
            "this removes defunct teams but may omit a returning NOC. Supply the "
            "real entry list for a production forecast."
        )
    else:
        notes.append(
            "Forecast candidates were restricted to the supplied "
            f"target_participants ({len(candidates)} NOCs); debutants with no "
            "history receive an intercept-only estimate."
        )
    if (season, target_year) not in HOST_NOC_BY_EDITION:
        notes.append(
            f"No host NOC is documented for {season} {target_year}; the host flag "
            "is 0 for every candidate."
        )
    if use_host:
        host_card = model.model_card()
        host_rows = host_card.loc[host_card["feature"] == "host"]
        if not host_rows.empty:
            parts: list[str] = []
            for _, row in host_rows.iterrows():
                kind = (
                    "count-stage multiplier"
                    if row["stage"] == "count|medal"
                    else "occurrence-stage odds ratio"
                )
                actual_effect = row.get("effect_at_1", np.nan)
                if np.isfinite(actual_effect):
                    parts.append(
                        f"{kind} {float(actual_effect):.2f} for host flag 0 -> 1 "
                        f"(log coefficient {float(row['coefficient']):+.3f}, "
                        f"approx. SE {float(row['std_error']):.3f}; "
                        f"effect_per_1sd {float(row['effect_per_1sd']):.2f})"
                    )
                else:
                    parts.append(
                        f"{kind} {float(row['effect_per_1sd']):.2f} per +1 SD "
                        f"(log coefficient {float(row['coefficient']):+.3f}, "
                        f"approx. SE {float(row['std_error']):.3f})"
                    )
            notes.append(
                "Host effect is estimated, not hard-coded: "
                + "; ".join(parts)
                + ". It relies on "
                f"{int((panel.loc[panel['season'] == season, 'host'] == 1).sum())} "
                "historical host observations."
            )
    else:
        notes.append("use_host=False: no host adjustment is applied in this run.")
    notes.append(
        "NOC codes are not merged across political eras (URS, GDR, EUN, TCH, YUG "
        "etc. keep separate histories), and sport-level shocks are not modelled. "
        "These are estimates, not guaranteed medal counts."
    )
    notes.append(
        "1906 Intercalated Games are excluded. Zero-medal countries are retained "
        "when present; debutants are scored at 0 in backtests so metrics are not "
        "flattered."
    )

    quality = data_quality_report(panel)
    return ForecastResult(
        predictions=predictions,
        backtest=backtest.rows,
        backtest_folds=backtest.folds,
        metrics=metrics,
        model_card=model.model_card(),
        data_quality=quality,
        notes=notes,
        target_year=target_year,
        season=season,
    )


# ---------------------------------------------------------------------------
# Loading and CLI
# ---------------------------------------------------------------------------


def _read_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    data_path = Path(path).expanduser()
    if not data_path.exists():
        raise OlympicDataError(f"Olympic data file not found: {data_path}")
    suffix = data_path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(data_path, low_memory=False)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(data_path)
        if suffix in {".json"}:
            return pd.read_json(data_path)
    except Exception as exc:  # pragma: no cover - depends on external file
        raise OlympicDataError(f"Could not read {data_path}: {exc}") from exc
    raise OlympicDataError(
        f"Unsupported data file type {suffix!r}. Use CSV, Parquet or JSON."
    )


def _extract_frame(obj: Any, loader_name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if isinstance(item, pd.DataFrame):
                return item
    if isinstance(obj, dict):
        for item in obj.values():
            if isinstance(item, pd.DataFrame):
                return item
    raise OlympicDataError(
        f"data_loader.{loader_name}() returned {type(obj).__name__}, not a "
        "DataFrame (or a tuple/list containing one)."
    )


def load_history(data_path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load Olympic history from a path or from ``data_loader``.

    Resolution order: explicit ``data_path``, ``OLYMPIQ_DATA_PATH`` environment
    variable, then ``data_loader``.  Supported loader function names include
    ``load_data``, ``load_olympic_data``, ``load_olympics``, ``get_data``,
    ``load_athlete_events``, ``load_panel`` and ``load``.

    The returned object may be raw athlete/event rows or an aggregated panel;
    use :func:`prepare_country_edition_panel` or pass it straight to
    :func:`forecast_medals`.
    """
    if data_path is None:
        data_path = os.environ.get("OLYMPIQ_DATA_PATH")
    if data_path:
        return _read_table(data_path)

    try:
        module = importlib.import_module("data_loader")
    except ImportError as exc:
        raise OlympicDataError(
            "No Olympic data source found. Pass a CSV path, set "
            "OLYMPIQ_DATA_PATH, or create data_loader.py exposing load_data() "
            "that returns a DataFrame. Expected columns: either "
            "(NOC, Year, Season, Event, Medal) or "
            "(noc, year, season, total/gold/silver/bronze). "
            "See prediction.describe_schema()."
        ) from exc

    loader_names = (
        "load_data", "load_olympic_data", "load_olympics", "get_data",
        "load_athlete_events", "load_panel", "load",
    )
    for name in loader_names:
        loader = getattr(module, name, None)
        if not callable(loader):
            continue
        try:
            obj = loader()
        except TypeError:
            fallback_path = (
                getattr(module, "DATA_PATH", None)
                or getattr(module, "DEFAULT_DATA_PATH", None)
                or getattr(module, "ATHLETE_EVENTS_PATH", None)
            )
            if not fallback_path:
                raise
            obj = loader(fallback_path)
        except FileNotFoundError as exc:
            raise OlympicDataError(
                f"data_loader.{name}() could not find the dataset: {exc}. "
                "Place athlete_events.csv in the expected location or set "
                "OLYMPIQ_DATA_PATH."
            ) from exc
        return _extract_frame(obj, name)

    raise OlympicDataError(
        "data_loader.py was imported but exposes none of the supported loaders: "
        f"{', '.join(loader_names)}."
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chronological Olympic medal forecasting (OLYMPIQ Person 3)."
    )
    parser.add_argument("--data", help="CSV/Parquet path; defaults to data_loader.")
    parser.add_argument("--season", default=SUMMER, choices=[SUMMER, WINTER])
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--backtest-editions", type=int, default=3)
    parser.add_argument("--min-train-editions", type=int, default=MIN_TRAIN_EDITIONS)
    parser.add_argument("--min-history-year", default=str(DEFAULT_MIN_HISTORY_YEAR),
                        help="Modern-era training start; 'none' uses all history.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--no-host", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--participants",
                        help="Optional target entry list CSV with a noc column.")
    args = parser.parse_args(argv)

    min_history: int | None
    if str(args.min_history_year).strip().lower() in {"none", "null", ""}:
        min_history = None
    else:
        min_history = int(args.min_history_year)

    target_participants = (
        _read_table(args.participants) if args.participants else None
    )

    try:
        history = load_history(args.data)
        result = forecast_medals(
            history,
            season=args.season,
            target_year=args.target_year,
            n_test_editions=args.backtest_editions,
            min_train_editions=args.min_train_editions,
            min_history_year=min_history,
            alpha=args.alpha,
            use_host=not args.no_host,
            top_n=args.top,
            target_participants=target_participants,
        )
    except OlympicDataError as exc:
        print(f"ERROR: {exc}")
        return 2

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)
    print(f"\n=== {result.season} {result.target_year}: top {len(result.predictions)} forecasts ===")
    print(result.predictions.to_string(index=False))
    print("\n=== Backtest metrics (chronological; POOLED is the overall row) ===")
    print(result.metrics.to_string(index=False))
    print("\n=== Per-fold training windows ===")
    print(result.backtest_folds.to_string(index=False))
    print("\n=== Model card ===")
    print(result.model_card.to_string(index=False))
    print("\n=== Data quality ===")
    print(result.data_quality.to_string(index=False))
    print("\n=== Notes / limitations ===")
    for note in result.notes:
        print(f"- {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
