from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable


CANONICAL_FEATURE_KEYS = (
    "avg_fitness",
    "active_ratio",
    "avg_energy",
    "avg_damage",
    "graph_density",
    "risk_level",
    "emergence_score",
)


def compute_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    """Compute exact ROC-AUC using Mann-Whitney U statistic (zero external dependencies)."""
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return None
    concordant = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return concordant / (len(pos) * len(neg))


@dataclass(frozen=True)
class FitnessObservation:
    """One pre-cycle state snapshot paired with observed outcome targets."""

    run_id: str
    cycle: int
    features: dict[str, float]
    target_avg_fitness: float
    target_is_collapse: int = 0
    horizon: int = 1


class FitnessExperiment:
    """Collect real H-cycle-ahead observations and score persistence & prevalence baselines."""

    def __init__(self) -> None:
        self._observations: list[FitnessObservation] = []
        self._initial_populations: dict[str, float] = {}

    @property
    def observations(self) -> tuple[FitnessObservation, ...]:
        return tuple(self._observations)

    def record(
        self,
        run_id: str,
        cycle: int,
        features: dict[str, float],
        target_avg_fitness: float,
        target_is_collapse: int | None = None,
        horizon: int = 1,
        initial_population: float | None = None,
        target_active_population: float | None = None,
    ) -> FitnessObservation:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if cycle < 0:
            raise ValueError("cycle must be non-negative")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if not isfinite(target_avg_fitness) or not 0.0 <= target_avg_fitness <= 1.0:
            raise ValueError("target average fitness must be finite and between 0 and 1")
        if not features or any(
            not isinstance(value, (int, float)) or not isfinite(value)
            for value in features.values()
        ):
            raise ValueError("features must be non-empty and finite")
        if not 0.0 <= features.get("avg_fitness", -1.0) <= 1.0:
            raise ValueError("pre-cycle avg_fitness must be between 0 and 1")
        if "active_ratio" in features and not (0.0 <= features["active_ratio"] <= 1.0):
            raise ValueError("active_ratio must be between 0 and 1")

        # Track N0 (initial active population for this run)
        if initial_population is not None:
            self._initial_populations[run_id] = float(initial_population)
        elif run_id not in self._initial_populations:
            if "total_agents" in features:
                self._initial_populations[run_id] = float(features["total_agents"])
            elif "active_population" in features:
                self._initial_populations[run_id] = float(features["active_population"])

        # Determine collapse label C_{t+H}:
        # C_{t+H} = 1 if target_avg_fitness < 0.20 OR N_{t+H} < 0.20 * N0, else 0
        if target_is_collapse is not None:
            if target_is_collapse not in (0, 1):
                raise ValueError("target_is_collapse must be 0 or 1")
            collapse_label = int(target_is_collapse)
        else:
            collapse_label = 0
            if target_avg_fitness < 0.20:
                collapse_label = 1
            elif (
                target_active_population is not None
                and run_id in self._initial_populations
            ):
                n0 = self._initial_populations[run_id]
                if n0 > 0 and target_active_population < (0.20 * n0):
                    collapse_label = 1

        observation = FitnessObservation(
            run_id=run_id,
            cycle=cycle,
            features=dict(features),
            target_avg_fitness=target_avg_fitness,
            target_is_collapse=collapse_label,
            horizon=horizon,
        )
        self._observations.append(observation)
        return observation

    def split_by_run(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> dict[str, list[str]]:
        """
        Split datasets strictly by simulation run to prevent data leakage.
        Snapshots from the same simulation run will never appear across multiple partitions.
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0 (got {total_ratio})")
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")

        run_ids = list(dict.fromkeys(item.run_id for item in self._observations))
        if len(run_ids) < 2:
            raise ValueError("at least 2 distinct simulation runs are required to split")

        shuffled = list(run_ids)
        random.Random(seed).shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        return {
            "train": sorted(shuffled[:n_train]),
            "val": sorted(shuffled[n_train : n_train + n_val]),
            "test": sorted(shuffled[n_train + n_val :]),
        }

    def export_dataset(
        self,
        filepath: str | Path,
        split_ratios: dict[str, float] | None = None,
        seed: int = 42,
    ) -> dict[str, int]:
        """
        Export observations to JSON Lines partitioned strictly by simulation run.
        """
        ratios = split_ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0 (got {total_ratio})")

        partitions = self.split_by_run(
            train_ratio=ratios.get("train", 0.70),
            val_ratio=ratios.get("val", 0.15),
            test_ratio=ratios.get("test", 0.15),
            seed=seed,
        )
        run_to_partition = {
            run_id: part_name
            for part_name, run_list in partitions.items()
            for run_id in run_list
        }

        counts = {"train": 0, "val": 0, "test": 0}
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for obs in self._observations:
                part = run_to_partition.get(obs.run_id, "train")
                counts[part] += 1
                record = {
                    "run_id": obs.run_id,
                    "cycle": obs.cycle,
                    "horizon": obs.horizon,
                    "features": obs.features,
                    "targets": {
                        "target_avg_fitness": obs.target_avg_fitness,
                        "target_is_collapse": obs.target_is_collapse,
                    },
                    "partition": part,
                }
                f.write(json.dumps(record) + "\n")
        return counts

    def import_dataset(self, filepath: str | Path) -> int:
        """
        Import observations from a validated JSON Lines file.
        Rejects malformed records and out-of-range features/targets.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"dataset file not found: {path}")

        imported_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    run_id = str(data["run_id"])
                    cycle = int(data["cycle"])
                    horizon = int(data.get("horizon", 1))
                    features = {str(k): float(v) for k, v in data["features"].items()}
                    targets = data["targets"]
                    target_avg_fitness = float(targets["target_avg_fitness"])
                    target_is_collapse = int(targets["target_is_collapse"])
                except (KeyError, ValueError, TypeError) as err:
                    raise ValueError(f"malformed record at line {line_no}: {err}") from err

                self.record(
                    run_id=run_id,
                    cycle=cycle,
                    features=features,
                    target_avg_fitness=target_avg_fitness,
                    target_is_collapse=target_is_collapse,
                    horizon=horizon,
                )
                imported_count += 1
        return imported_count

    def evaluate_model(
        self,
        predictions: list[dict[str, Any]],
        held_out_run_ids: Iterable[str],
    ) -> dict[str, Any]:
        """
        Evidence-based model evaluation on held-out runs.
        Estimates prevalence baseline strictly from the non-held-out training partition.
        """
        held_out = set(held_out_run_ids)
        if not held_out:
            raise ValueError("at least one held-out run ID is required")

        train_obs = [o for o in self._observations if o.run_id not in held_out]
        test_obs = [o for o in self._observations if o.run_id in held_out]
        if not train_obs:
            raise ValueError("training partition required to estimate prevalence baseline")
        if not test_obs:
            raise ValueError("no test observations found for given held-out run IDs")

        # Training-set prevalence baseline (no data leakage from test set)
        p_base = sum(o.target_is_collapse for o in train_obs) / len(train_obs)
        brier_prevalence_baseline = p_base * (1.0 - p_base)

        # Align predictions to test observations by (run_id, cycle, horizon)
        pred_map = {
            (p["run_id"], p["cycle"], p.get("horizon", 1)): p
            for p in predictions
        }

        reg_errors, base_reg_errors = [], []
        pred_probs, true_labels = [], []
        false_negatives = 0
        total_collapses = 0

        for obs in test_obs:
            key = (obs.run_id, obs.cycle, obs.horizon)
            if key not in pred_map:
                raise ValueError(f"missing prediction for observation {key}")
            pred = pred_map[key]

            # Regression MAE & RMSE
            y_hat = float(pred["pred_avg_fitness"])
            y_true = obs.target_avg_fitness
            reg_errors.append(abs(y_hat - y_true))
            base_reg_errors.append(abs(y_true - obs.features.get("avg_fitness", 0.0)))

            # Classification Probabilities & Safety
            p_hat = float(pred.get("pred_collapse_prob", 0.0))
            c_true = obs.target_is_collapse
            pred_probs.append(p_hat)
            true_labels.append(c_true)

            if c_true == 1:
                total_collapses += 1
                if p_hat < 0.50:
                    false_negatives += 1

        n_test = len(test_obs)
        mae_model = sum(reg_errors) / n_test
        rmse_model = sqrt(sum(e * e for e in reg_errors) / n_test)
        mae_base = sum(base_reg_errors) / n_test
        delta_mae = (mae_base - mae_model) / mae_base if mae_base > 0.0 else 0.0

        brier_model = sum((p - c) ** 2 for p, c in zip(pred_probs, true_labels)) / n_test
        has_both = (0 in true_labels) and (1 in true_labels)
        roc_auc = compute_roc_auc(true_labels, pred_probs) if has_both else None

        # Baseline false negatives (if prevalence predictor thresholded at 0.5)
        baseline_false_neg = total_collapses if p_base < 0.50 else 0

        return {
            "regression": {
                "model_mae": mae_model,
                "model_rmse": rmse_model,
                "persistence_mae": mae_base,
                "delta_mae": delta_mae,
                "beats_baseline": mae_model < mae_base,
            },
            "classification": {
                "model_brier": brier_model,
                "prevalence_brier_baseline": brier_prevalence_baseline,
                "roc_auc": roc_auc,
                "has_both_classes": has_both,
            },
            "safety": {
                "actual_collapses": total_collapses,
                "model_false_negatives": false_negatives,
                "baseline_false_negatives": baseline_false_neg,
                "zero_false_negatives": (false_negatives == 0),
            },
            "sample_counts": {
                "train_samples": len(train_obs),
                "test_samples": n_test,
                "held_out_runs": sorted(held_out),
            },
        }

    def evaluate_persistence_baseline(
        self,
        held_out_run_ids: Iterable[str],
    ) -> dict[str, Any]:
        """
        Evaluate baseline performance on complete, held-out runs.
        Regression: persistence baseline (predicts current fitness).
        Classification: prevalence probability baseline and zero-rule classifier.
        """
        held_out = set(held_out_run_ids)
        if not held_out:
            raise ValueError("at least one held-out run ID is required")

        non_held_out_runs = {item.run_id for item in self._observations} - held_out
        test = [item for item in self._observations if item.run_id in held_out]
        train = [item for item in self._observations if item.run_id not in held_out]
        unknown_runs = held_out - {item.run_id for item in self._observations}
        if unknown_runs:
            raise ValueError(f"unknown held-out run IDs: {sorted(unknown_runs)}")
        if not non_held_out_runs:
            raise ValueError("at least one separate non-held-out run is required")

        # Regression: MAE against current average fitness
        absolute_errors = [
            abs(item.target_avg_fitness - item.features["avg_fitness"])
            for item in test
        ]
        per_run_errors: dict[str, list[float]] = {}
        for item, error in zip(test, absolute_errors):
            per_run_errors.setdefault(item.run_id, []).append(error)

        # Classification Baselines
        train_collapses = sum(item.target_is_collapse for item in train)
        p_base = (train_collapses / len(train)) if train else 0.0

        test_collapses = sum(item.target_is_collapse for item in test)
        p_test = (test_collapses / len(test)) if test else 0.0

        # Primary Brier benchmark: constant probability using training prevalence
        brier_baseline = p_base * (1.0 - p_base)

        # Secondary Brier benchmark: always-zero classifier
        brier_zero = sum((0.0 - item.target_is_collapse) ** 2 for item in test) / len(test)

        # ROC-AUC: only report when test set contains both classes {0, 1}
        test_labels = [item.target_is_collapse for item in test]
        has_both_classes = (0 in test_labels) and (1 in test_labels)
        roc_auc = (
            compute_roc_auc(test_labels, [p_base] * len(test_labels))
            if has_both_classes
            else None
        )

        return {
            "target": "next_cycle_average_fitness",
            "baseline": "predict_current_average_fitness",
            "metric": "mae",
            "mae": sum(absolute_errors) / len(absolute_errors),
            "observation_count": len(test),
            "non_held_out_run_count": len(non_held_out_runs),
            "held_out_run_count": len({item.run_id for item in test}),
            "per_run_mae": {
                run_id: sum(errors) / len(errors)
                for run_id, errors in per_run_errors.items()
            },
            "held_out_run_ids": sorted({item.run_id for item in test}),
            "classification_baselines": {
                "train_prevalence": p_base,
                "test_prevalence": p_test,
                "prevalence_brier_baseline": brier_baseline,
                "zero_classifier_brier": brier_zero,
                "test_collapse_count": test_collapses,
                "has_both_classes": has_both_classes,
                "roc_auc": roc_auc,
                "roc_auc_note": (
                    "Reported only when evaluation set contains both classes {0, 1}"
                    if not has_both_classes
                    else "Evaluated with both classes present"
                ),
            },
        }
