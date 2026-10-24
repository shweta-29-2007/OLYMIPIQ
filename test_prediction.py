"""Tests for prediction.py.

Run with:  python -m unittest -v test_prediction

The suite uses small, hand-built panels so it runs in under a few seconds and
does not depend on the external Kaggle CSV.  It checks the behaviours the
OLYMPIQ app relies on: schema adaptation, leakage-free features, strictly
chronological validation, honest metrics (including zeros/debutants), and the
documented output contract.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import prediction as p


SUMMER = "Summer"


def _medal_split(total: int) -> tuple[int, int, int]:
    """Split a total into plausible gold/silver/bronze counts."""
    gold = total // 3
    silver = (total - gold) // 2
    bronze = total - gold - silver
    return gold, silver, bronze


def synthetic_panel() -> pd.DataFrame:
    """A small aggregated panel with 12 editions, zeros and a late entrant."""
    years = [1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004]
    totals = {
        "USA": [80, 90, 85, 95, 90, 100, 110, 95, 105, 100, 95, 100],
        "URS": [60, 65, 70, 75, 80, 85, 60, 55, 0, 0, 0, 0],
        "GBR": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "FRA": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "JPN": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
        "BRA": [0, 0, 2, 3, 5, 6, 8, 9, 10, 12, 15, 17],
        "KEN": [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5],
        "IND": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "CHN": [0, 0, 0, 0, 0, 0, 15, 20, 25, 30, 35, 40],
    }
    names = {
        "USA": "United States", "URS": "Soviet Union", "GBR": "Great Britain",
        "FRA": "France", "JPN": "Japan", "BRA": "Brazil", "KEN": "Kenya",
        "IND": "India", "CHN": "China",
    }
    rows = []
    for noc, series in totals.items():
        for year, total in zip(years, series):
            gold, silver, bronze = _medal_split(total)
            rows.append({
                "noc": noc, "year": year, "season": SUMMER,
                "gold": gold, "silver": silver, "bronze": bronze,
                "total": total, "country": names[noc],
            })
    return pd.DataFrame(rows)


def synthetic_raw() -> pd.DataFrame:
    """Raw athlete/event rows, including a duplicate team medal and 1906."""
    rows = [
        # Two athletes in the same event/medal -> one team medal.
        ("USA", 1960, SUMMER, "100m", "Gold"),
        ("USA", 1960, SUMMER, "100m", "Gold"),
        ("USA", 1960, SUMMER, "200m", "Silver"),
        ("URS", 1960, SUMMER, "100m", "Silver"),
        ("GBR", 1960, SUMMER, "100m", np.nan),
        ("FRA", 1906, SUMMER, "100m", "Gold"),
    ]
    return pd.DataFrame(rows, columns=["NOC", "Year", "Season", "Event", "Medal"])


class PreparationTests(unittest.TestCase):
    def test_raw_rows_are_aggregated_and_1906_excluded(self):
        panel = p.prepare_country_edition_panel(synthetic_raw())
        self.assertNotIn(1906, panel["year"].tolist())

        usa = panel.loc[(panel["noc"] == "USA") & (panel["year"] == 1960)].iloc[0]
        self.assertEqual(usa["total"], 2)
        self.assertEqual(usa["gold"], 1)
        self.assertEqual(usa["silver"], 1)

        gbr = panel.loc[(panel["noc"] == "GBR") & (panel["year"] == 1960)].iloc[0]
        self.assertEqual(gbr["total"], 0, "participant-only rows must be retained")
        self.assertEqual(panel.attrs["excluded_1906_rows"], 1)
        self.assertGreaterEqual(panel.attrs["duplicate_rows_removed"], 1)

    def test_aggregated_panel_shape_and_host_map(self):
        panel = p.prepare_country_edition_panel(synthetic_panel())
        self.assertEqual(
            list(panel.columns),
            ["season", "year", "noc", "country", "gold", "silver", "bronze", "total", "host"],
        )
        self.assertTrue((panel["total"] >= 0).all())
        # Factual host flags, not a hard-coded medal bonus.
        self.assertEqual(
            int(panel.loc[(panel["year"] == 1980) & (panel["noc"] == "URS"), "host"].iloc[0]), 1
        )
        self.assertEqual(
            int(panel.loc[(panel["year"] == 1980) & (panel["noc"] == "USA"), "host"].iloc[0]), 0
        )

    def test_participants_fill_missing_zero_rows(self):
        aggregated = synthetic_panel().loc[synthetic_panel()["total"] > 0].copy()
        participants = synthetic_panel()[["noc", "year", "season"]]
        panel = p.prepare_country_edition_panel(aggregated, participants=participants)
        india_1960 = panel.loc[(panel["noc"] == "IND") & (panel["year"] == 1960)]
        self.assertEqual(len(india_1960), 1)
        self.assertEqual(float(india_1960["total"].iloc[0]), 0.0)


class FeatureLeakageTests(unittest.TestCase):
    def test_future_totals_do_not_change_earlier_features(self):
        panel = p.prepare_country_edition_panel(synthetic_panel())
        original = p.build_feature_frame(panel, SUMMER, target_year=2008)
        changed = panel.copy()
        changed.loc[changed["year"] >= 2004, "total"] = 999
        changed.loc[changed["year"] >= 2004, "gold"] = 333
        rebuilt = p.build_feature_frame(changed, SUMMER, target_year=2008)

        feature_columns = [
            "prev_total", "prev3_mean", "prev3_max", "trend",
            "momentum_last_vs_mean", "trend_ratio", "depth_ratio",
            "was_zero_last", "log_last", "n_prior", "gold_share3",
        ]
        keys = ["noc", "year"]
        before = original.loc[original["year"] <= 2004, keys + feature_columns].reset_index(drop=True)
        after = rebuilt.loc[rebuilt["year"] <= 2004, keys + feature_columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

        # But the 2008 features must react to 2004 (legitimate earlier data).
        before_2008 = original.loc[original["year"] == 2008].set_index("noc")["prev_total"]
        after_2008 = rebuilt.loc[rebuilt["year"] == 2008].set_index("noc")["prev_total"]
        self.assertFalse(before_2008.equals(after_2008))

    def test_target_rows_are_marked_as_not_participating(self):
        panel = p.prepare_country_edition_panel(synthetic_panel())
        features = p.build_feature_frame(panel, SUMMER, target_year=2008)
        target = features.loc[features["year"] == 2008]
        self.assertFalse(target.empty)
        self.assertTrue((target["participated"] == 0).all())
        # 2008 host is China.
        self.assertEqual(int(target.loc[target["noc"] == "CHN", "host"].iloc[0]), 1)


class ChronologicalBacktestTests(unittest.TestCase):
    def test_every_fold_trains_only_on_earlier_editions(self):
        panel = p.prepare_country_edition_panel(synthetic_panel())
        result = p.chronological_backtest(
            panel, season=SUMMER, test_years=[1996, 2000, 2004],
            min_history_year=1960, min_train_editions=4, active_window=2,
        )
        self.assertEqual(result.folds["test_year"].tolist(), [1996, 2000, 2004])
        self.assertTrue(
            (result.folds["trained_through_year"] < result.folds["test_year"]).all()
        )
        self.assertTrue(
            (result.rows["trained_through_year"] < result.rows["test_year"]).all()
        )
        self.assertEqual(sorted(result.rows["test_year"].unique()), [1996, 2000, 2004])
        # The model must score every actual participant in the test edition.
        for year in [1996, 2000, 2004]:
            actual = panel.loc[(panel["season"] == SUMMER) & (panel["year"] == year), "noc"].nunique()
            scored = result.rows.loc[result.rows["test_year"] == year, "noc"].nunique()
            self.assertEqual(actual, scored)

    def test_backtest_outputs_probability_and_brier(self):
        panel = p.prepare_country_edition_panel(synthetic_panel())
        result = p.chronological_backtest(
            panel, season=SUMMER, test_years=[2004],
            min_history_year=1960, min_train_editions=4, active_window=2,
        )
        self.assertIn("predicted_medal_probability", result.rows.columns)
        self.assertIn("predicted_count_if_medal", result.rows.columns)
        probability = result.rows["predicted_medal_probability"]
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertTrue((result.rows["predicted_total"] >= 0).all())
        metrics = p.compute_metrics(result.rows)
        self.assertTrue(np.isfinite(metrics["brier_medal_occurrence_model"]))
        self.assertTrue(np.isfinite(metrics["brier_medal_occurrence_baseline"]))

    def test_insufficient_history_raises(self):
        tiny = synthetic_panel()
        tiny = tiny.loc[tiny["year"].isin([2000, 2004])]
        panel = p.prepare_country_edition_panel(tiny)
        with self.assertRaises(p.InsufficientDataError):
            p.chronological_backtest(
                panel, season=SUMMER, test_years=[2004],
                min_history_year=1960, min_train_editions=4,
            )


class MetricTests(unittest.TestCase):
    def test_known_mae_brier_and_group_splits(self):
        frame = pd.DataFrame({
            "noc": ["A", "B", "C", "D"],
            "actual_total": [0.0, 0.0, 5.0, 10.0],
            "predicted_total": [0.2, 0.4, 4.0, 8.0],
            "predicted_medal_probability": [0.1, 0.2, 0.9, 1.0],
            "baseline_last": [0.0, 0.0, 6.0, 9.0],
            "baseline_mean3": [0.0, 1.0, 5.0, 9.0],
            "seen_before": [True, True, True, False],
        })
        metrics = p.compute_metrics(frame)
        self.assertEqual(metrics["n_countries"], 4)
        self.assertEqual(metrics["n_zero_medal"], 2)
        self.assertEqual(metrics["n_medal_winners"], 2)
        self.assertEqual(metrics["n_debutants"], 1)
        self.assertAlmostEqual(metrics["mae_model"], (0.2 + 0.4 + 1.0 + 2.0) / 4)
        self.assertAlmostEqual(metrics["mae_baseline_last"], (0 + 0 + 1 + 1) / 4)
        self.assertAlmostEqual(metrics["mae_model_zero_medal"], 0.3)
        self.assertAlmostEqual(metrics["mae_model_medal_winners"], 1.5)
        expected_brier = np.mean((np.array([0, 0, 1, 1]) - np.array([0.1, 0.2, 0.9, 1.0])) ** 2)
        self.assertAlmostEqual(metrics["brier_medal_occurrence_model"], float(expected_brier))
        baseline_brier = np.mean((np.array([0, 0, 1, 1]) - np.array([0, 0, 1, 1])) ** 2)
        self.assertAlmostEqual(metrics["brier_medal_occurrence_baseline"], float(baseline_brier))

    def test_metric_documentation_covers_every_returned_metric(self):
        frame = pd.DataFrame({
            "noc": ["A", "B"],
            "actual_total": [0.0, 3.0],
            "predicted_total": [0.1, 2.0],
            "predicted_medal_probability": [0.1, 0.8],
            "baseline_last": [0.0, 3.0],
        })
        metrics = p.compute_metrics(frame)
        documented = set(p.explain_metrics()["metric"])
        missing = set(metrics) - documented
        self.assertFalse(missing, f"metrics missing definitions: {sorted(missing)}")


class ForecastContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = p.prepare_country_edition_panel(synthetic_panel())
        cls.result = p.forecast_medals(
            synthetic_panel(), season=SUMMER, target_year=2008,
            n_test_editions=3, min_history_year=1960,
            min_train_editions=4, active_window=2,
        )

    def test_predictions_have_documented_columns_and_are_consistent(self):
        expected = {
            "rank", "noc", "country", "season", "target_year",
            "predicted_total", "predicted_total_rounded", "medal_probability",
            "count_if_medal", "predicted_gold", "lower_80", "upper_80",
            "last_total", "last_year", "mean_last3", "trend",
            "n_prior_editions", "host", "seen_before", "note",
        }
        self.assertTrue(expected <= set(self.result.predictions.columns))
        pred = self.result.predictions
        self.assertTrue((pred["predicted_total"] >= 0).all())
        self.assertTrue(((pred["medal_probability"] >= 0) & (pred["medal_probability"] <= 1)).all())
        self.assertTrue((pred["count_if_medal"] >= 0).all())
        product = pred["medal_probability"] * pred["count_if_medal"]
        self.assertTrue(np.allclose(pred["predicted_total"], product, atol=1e-2))
        self.assertTrue((pred["lower_80"] <= pred["predicted_total"] + 1e-9).all())
        self.assertTrue((pred["upper_80"] >= pred["predicted_total"] - 1e-9).all())
        self.assertEqual(pred["rank"].tolist(), list(range(1, len(pred) + 1)))

    def test_metrics_table_has_per_edition_and_pooled_rows(self):
        metrics = self.result.metrics
        self.assertIn("POOLED", metrics["test_year"].tolist())
        self.assertIn("brier_medal_occurrence_model", metrics.columns)
        pooled = metrics.loc[metrics["test_year"] == "POOLED"].iloc[0]
        self.assertTrue(np.isfinite(pooled["mae_model"]))
        self.assertTrue(np.isfinite(pooled["brier_medal_occurrence_model"]))

    def test_model_card_has_both_stages(self):
        card = self.result.model_card
        self.assertEqual(set(card["stage"]), {"count|medal", "occurrence"})
        self.assertIn("effect_type", card.columns)
        self.assertTrue((card["std_error"] >= 0).all())

    def test_host_effect_is_reported_on_the_natural_0_to_1_scale(self):
        """The sparse host flag must not be standardised into a huge effect."""
        card = self.result.model_card
        hosts = card.loc[card["feature"] == "host"]
        self.assertFalse(hosts.empty)
        for _, row in hosts.iterrows():
            self.assertEqual(float(row["train_sd"]), 1.0)
            self.assertGreater(float(row["effect_at_1"]), 0.0)
            # exp(coefficient / sd) is the exact 0->1 effect for a standardised
            # feature; with sd=1 it is simply exp(coefficient).
            self.assertAlmostEqual(
                float(row["effect_at_1"]),
                float(np.exp(row["coefficient"] / row["train_sd"])),
                places=10,
            )
            self.assertLess(float(row["effect_at_1"]), 10.0)

    def test_pooled_rank_metrics_are_means_of_per_edition_values(self):
        """Ranking rows mixed from several editions is not meaningful."""
        metrics = self.result.metrics
        per_edition = metrics.loc[metrics["test_year"] != "POOLED"]
        pooled = metrics.loc[metrics["test_year"] == "POOLED"].iloc[0]
        for metric in (
            "spearman_model",
            "spearman_baseline_last",
            "top10_overlap_model",
            "top10_overlap_baseline_last",
        ):
            self.assertAlmostEqual(
                float(pooled[metric]), float(per_edition[metric].mean()), places=10
            )

    def test_debutant_is_kept_and_not_given_fabricated_history(self):
        result = p.forecast_medals(
            synthetic_panel(), season=SUMMER, target_year=2008,
            target_participants=["USA", "GBR", "BRA", "KEN", "IND", "CHN", "DEB"],
            n_test_editions=3, min_history_year=1960,
            min_train_editions=4, active_window=2,
        )
        row = result.predictions.loc[result.predictions["noc"] == "DEB"]
        self.assertEqual(len(row), 1)
        self.assertFalse(bool(row["seen_before"].iloc[0]))
        self.assertEqual(int(row["n_prior_editions"].iloc[0]), 0)
        self.assertGreaterEqual(float(row["predicted_total"].iloc[0]), 0.0)
        self.assertTrue(np.isfinite(float(row["medal_probability"].iloc[0])))

    def test_host_flag_and_note(self):
        host_row = self.result.predictions.loc[self.result.predictions["noc"] == "CHN"]
        self.assertEqual(len(host_row), 1)
        self.assertTrue(bool(host_row["host"].iloc[0]))
        self.assertTrue(any("Host effect is estimated" in note for note in self.result.notes))

    def test_zero_medal_country_is_retained(self):
        india = self.result.predictions.loc[self.result.predictions["noc"] == "IND"]
        self.assertEqual(len(india), 1)
        self.assertGreaterEqual(float(india["predicted_total"].iloc[0]), 0.0)

    def test_forecast_is_deterministic(self):
        again = p.forecast_medals(
            synthetic_panel(), season=SUMMER, target_year=2008,
            n_test_editions=3, min_history_year=1960,
            min_train_editions=4, active_window=2,
        )
        pd.testing.assert_frame_equal(
            self.result.predictions, again.predictions, check_exact=False, rtol=1e-10, atol=1e-10
        )
        pd.testing.assert_frame_equal(
            self.result.metrics, again.metrics, check_exact=False, rtol=1e-10, atol=1e-10
        )

    def test_schema_and_metric_help_are_exported(self):
        schema = p.describe_schema()
        output_columns = set(schema.loc[schema["table"] == "predictions_output", "column"])
        self.assertIn("medal_probability", output_columns)
        self.assertIn("count_if_medal", output_columns)
        # Every column that app.py receives must be documented exactly once.
        self.assertEqual(set(self.result.predictions.columns), output_columns)
        documented_backtest = set(schema.loc[schema["table"] == "backtest_output", "column"])
        for column in (
            "actual_total", "predicted_total", "baseline_last", "baseline_mean3",
            "error", "abs_error", "seen_before", "trained_through_year",
        ):
            self.assertIn(column, documented_backtest)
        documented = set(p.explain_metrics()["metric"])
        self.assertIn("brier_medal_occurrence_model", documented)

    def test_forecast_rejects_past_target_year(self):
        with self.assertRaises(ValueError):
            p.forecast_medals(
                synthetic_panel(), season=SUMMER, target_year=2004,
                min_history_year=1960, min_train_editions=4,
            )


class RealDatasetIntegrationTests(unittest.TestCase):
    """Integration checks against the repository's authentic dataset.

    These are skipped automatically when ``data/olympics.csv`` is absent (for
    example on a clean clone before the dataset is downloaded), so the fast unit
    suite still works everywhere.  Set ``OLYMIPIQ_DATA_PATH`` to test another
    file.  No accuracy number in the suite is hard-coded: every assertion reads
    the metrics computed from the file.
    """

    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get("OLYMIPIQ_DATA_PATH", "data/olympics.csv"))
        if not path.exists():
            raise unittest.SkipTest(f"Integration dataset not found at {path}")
        cls.history = p.load_history(path)
        cls.panel = p.prepare_country_edition_panel(cls.history)
        summer = cls.panel.loc[cls.panel["season"] == SUMMER]
        cls.target_year = int(summer["year"].max()) + 4
        cls.result = p.forecast_medals(
            cls.history,
            season=SUMMER,
            target_year=cls.target_year,
            n_test_editions=2,
            min_history_year=1960,
            min_train_editions=4,
            active_window=2,
        )

    def test_real_panel_retains_zero_medal_countries(self):
        self.assertGreater(len(self.panel), 1000)
        self.assertGreater(int((self.panel["total"] == 0).sum()), 0)
        self.assertTrue(
            {"season", "year", "noc", "country", "total", "host"} <= set(self.panel.columns)
        )

    def test_real_backtest_is_chronological_and_scored_on_participants(self):
        folds = self.result.backtest_folds
        rows = self.result.backtest
        self.assertTrue((folds["trained_through_year"] < folds["test_year"]).all())
        self.assertTrue((rows["trained_through_year"] < rows["test_year"]).all())
        for year in folds["test_year"]:
            actual = self.panel.loc[
                (self.panel["season"] == SUMMER) & (self.panel["year"] == year), "noc"
            ].nunique()
            scored = rows.loc[rows["test_year"] == year, "noc"].nunique()
            self.assertEqual(actual, scored)

    def test_real_metrics_and_baseline_are_finite(self):
        pooled = self.result.metrics.loc[
            self.result.metrics["test_year"] == "POOLED"
        ].iloc[0]
        self.assertGreater(int(pooled["n_countries"]), 0)
        self.assertTrue(np.isfinite(pooled["mae_model"]))
        self.assertTrue(np.isfinite(pooled["mae_baseline_last"]))
        self.assertTrue(np.isfinite(pooled["brier_medal_occurrence_model"]))
        # The honest model-vs-baseline note must always be present.
        self.assertTrue(any("persistence baseline" in note for note in self.result.notes))

    def test_real_host_flag_uses_documented_host_noc(self):
        expected = p.HOST_NOC_BY_EDITION.get((SUMMER, self.target_year))
        if expected is None:
            self.skipTest(f"No documented host NOC for {SUMMER} {self.target_year}")
        targets = self.result.predictions
        if expected not in set(targets["noc"]):
            self.skipTest(f"Host NOC {expected} is not in the candidate list")
        self.assertTrue(bool(targets.loc[targets["noc"] == expected, "host"].iloc[0]))
        self.assertEqual(int(targets["host"].sum()), 1)

    def test_data_loader_schema_matches_prediction_contract(self):
        """Person 2's raw and aggregated outputs must produce the same panel."""
        try:
            import data_loader
        except ImportError:  # pragma: no cover - only on an incomplete checkout
            self.skipTest("data_loader module is not available")

        raw = data_loader.load_data()
        summary = data_loader.OlympicDataLoader().get_country_medal_summary()
        raw_panel = p.prepare_country_edition_panel(raw)
        agg_panel = p.prepare_country_edition_panel(summary)
        self.assertEqual(len(raw_panel), len(agg_panel))
        merged = raw_panel.merge(
            agg_panel,
            on=["season", "year", "noc"],
            suffixes=("_raw", "_agg"),
            validate="one_to_one",
        )
        self.assertTrue(np.allclose(merged["total_raw"], merged["total_agg"]))
        for colour in ("gold", "silver", "bronze"):
            self.assertTrue(
                np.allclose(
                    merged[f"{colour}_raw"].fillna(-1),
                    merged[f"{colour}_agg"].fillna(-1),
                )
            )
        self.assertGreater(int((agg_panel["total"] == 0).sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
