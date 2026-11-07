from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
except Exception:  # pragma: no cover - optional dependency
    ExtraTreesRegressor = None
    GradientBoostingRegressor = None
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None

try:
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover - optional dependency
    LinearRegression = None

try:
    from sklearn.neighbors import KNeighborsRegressor
except Exception:  # pragma: no cover - optional dependency
    KNeighborsRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


@dataclass
class SimulationResult:
    """Serializable output for an ML-driven environment run."""
    metrics: dict[str, Any]
    model_predictions: dict[str, float]
    ensemble_score: float
    scenario: str


class SimulatedEnvironment:
    """
    ML-driven simulation layer for the frontier civilization.

    This keeps the project as a large simulated environment instead of a static
    biological taxonomy. Each civilization cycle produces a feature snapshot that
    is scored by lightweight models such as Random Forest and XGBoost when
    those libraries are available. The ensemble output acts like a world-state
    prediction/score for the next simulated decisions.
    """

    FEATURE_KEYS = (
        "node_count",
        "total_agents",
        "avg_fitness",
        "emergence_score",
        "memory_alive",
        "uptime_s",
        "knowledge_total",
        "risk_level",
    )

    def __init__(self, model_names: Iterable[str] | None = None):
        self.model_names = tuple(model_names or self._default_model_names())
        self.models: dict[str, Any] = {}
        self.training_history: list[dict[str, Any]] = []
        self._initialize_models()

    @staticmethod
    def _default_model_names() -> tuple[str, ...]:
        available = []
        preferred_order = (
            "extra_trees",
            "hist_gradient_boosting",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
            "linear_regression",
        )
        for name in preferred_order:
            if name == "extra_trees" and ExtraTreesRegressor is not None:
                available.append(name)
            elif name == "hist_gradient_boosting" and HistGradientBoostingRegressor is not None:
                available.append(name)
            elif name == "random_forest" and RandomForestRegressor is not None:
                available.append(name)
            elif name == "gradient_boosting" and GradientBoostingRegressor is not None:
                available.append(name)
            elif name == "xgboost" and XGBRegressor is not None:
                available.append(name)
            elif name == "lightgbm" and LGBMRegressor is not None:
                available.append(name)
            elif name == "catboost" and CatBoostRegressor is not None:
                available.append(name)
            elif name == "linear_regression" and LinearRegression is not None:
                available.append(name)
        return tuple(available) or ("random_forest",)

    def _initialize_models(self):
        for model_name in self.model_names:
            model = self._build_model(model_name)
            if model is not None:
                self.models[model_name] = model

        if not self.models:
            self.models["fallback_rule_based"] = None

    def _build_model(self, model_name: str):
        if model_name == "random_forest":
            if RandomForestRegressor is None:
                return None
            return RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
        if model_name == "extra_trees":
            if ExtraTreesRegressor is None:
                return None
            return ExtraTreesRegressor(n_estimators=250, random_state=42)
        if model_name == "gradient_boosting":
            if GradientBoostingRegressor is None:
                return None
            return GradientBoostingRegressor(random_state=42)
        if model_name == "hist_gradient_boosting":
            if HistGradientBoostingRegressor is None:
                return None
            return HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, random_state=42)
        if model_name == "linear_regression":
            if LinearRegression is None:
                return None
            return LinearRegression()
        if model_name == "k_neighbors":
            if KNeighborsRegressor is None:
                return None
            return KNeighborsRegressor(n_neighbors=7)
        if model_name == "xgboost":
            if XGBRegressor is None:
                return None
            return XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
        if model_name == "lightgbm":
            if LGBMRegressor is None:
                return None
            return LGBMRegressor(n_estimators=250, learning_rate=0.08, max_depth=8, random_state=42)
        if model_name == "catboost":
            if CatBoostRegressor is None:
                return None
            return CatBoostRegressor(iterations=250, learning_rate=0.08, depth=8, verbose=False, random_seed=42)
        return None

    def _generate_seed_dataset(self, rows: int = 32):
        dataset = []
        labels = []
        for _ in range(rows):
            node_count = random.randint(2, 8)
            total_agents = random.randint(4, 120)
            avg_fitness = random.uniform(0.2, 1.0)
            emergence = round(avg_fitness * node_count * 0.1 + random.uniform(0.1, 0.7), 4)
            memory_alive = random.randint(10, 500)
            uptime = random.uniform(5.0, 180.0)
            knowledge_total = random.randint(0, 200)
            risk_level = random.uniform(0.0, 1.0)
            dataset.append([
                node_count,
                total_agents,
                avg_fitness,
                emergence,
                memory_alive,
                uptime,
                knowledge_total,
                risk_level,
            ])
            labels.append(emergence + random.uniform(-0.05, 0.08))
        return dataset, labels

    def _ensure_trained(self):
        if self.training_history:
            for name, model in self.models.items():
                if name == "fallback_rule_based" or model is None:
                    continue
                if hasattr(model, "n_estimators"):
                    X = [[row[key] for key in self.FEATURE_KEYS] for row in self.training_history]
                    y = [row["target"] for row in self.training_history]
                    if len(X) >= 2:
                        model.fit(X, y)
            return

        X, y = self._generate_seed_dataset()
        for name, model in self.models.items():
            if name == "fallback_rule_based" or model is None:
                continue
            model.fit(X, y)

    def _state_snapshot(self, civilization):
        stats = civilization.status()
        node_metrics = stats.get("nodes", [])
        avg_fit = stats.get("avg_fitness", 0.0)
        total_agents = stats.get("total_agents", 0)
        node_count = stats.get("node_count", len(node_metrics))
        knowledge_total = stats.get("biological_profile", {}).get("domain_count", 0)
        risk_level = min(1.0, max(0.0, (1.0 - avg_fit) * 0.8 + (total_agents / max(1, total_agents + 40)) * 0.2))
        return {
            "node_count": node_count,
            "total_agents": total_agents,
            "avg_fitness": avg_fit,
            "emergence_score": stats.get("emergence_score", 0.0),
            "memory_alive": stats.get("memory_alive", 0),
            "uptime_s": stats.get("uptime_s", 0.0),
            "knowledge_total": knowledge_total,
            "risk_level": risk_level,
        }

    def _feature_vector(self, snapshot: dict[str, Any]) -> list[float]:
        return [
            float(snapshot.get("node_count", 0)),
            float(snapshot.get("total_agents", 0)),
            float(snapshot.get("avg_fitness", 0.0)),
            float(snapshot.get("emergence_score", 0.0)),
            float(snapshot.get("memory_alive", 0)),
            float(snapshot.get("uptime_s", 0.0)),
            float(snapshot.get("knowledge_total", 0)),
            float(snapshot.get("risk_level", 0.0)),
        ]

    def simulate(self, civilization: Any) -> SimulationResult:
        """Forecast the world-state outcome for the current civilization state."""
        self._ensure_trained()
        snapshot = self._state_snapshot(civilization)
        predictions: dict[str, float] = {}
        for name, model in self.models.items():
            if model is None:
                continue
            vector = self._feature_vector(snapshot)
            prediction = float(model.predict([vector])[0])
            predictions[name] = round(prediction, 4)

        if not predictions:
            ensemble = 0.0
        else:
            ensemble = round(sum(predictions.values()) / len(predictions), 4)

        result = SimulationResult(
            metrics={
                "scenario": "frontier_civilization",
                "prediction_window": "next_cycle",
                **snapshot,
            },
            model_predictions=predictions,
            ensemble_score=ensemble,
            scenario="frontier_civilization",
        )
        self.training_history.append({**snapshot, "target": ensemble})
        if len(self.training_history) > 200:
            self.training_history = self.training_history[-200:]
        return result


__all__ = ["SimulatedEnvironment", "SimulationResult"]
