"""XGBoost lap-time predictor.

Predicts FUEL-ADJUSTED lap time as a function of (tyre compound, tyre age,
lap number, team, track, year). The model effectively learns:
    "How much pace does a Medium tyre have at lap 18 of its life on Monaco?"

This is the heart of the strategy engine: given a candidate pit lap,
we ask this model how each subsequent lap would unfold on the new tyre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["compound", "team", "track"]
NUMERIC_COLS = ["tyre_life", "lap_number", "year"]
TARGET_COL = "fuel_adjusted_lap_time"

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "tyre_model.json"


@dataclass
class TrainReport:
    mae_train: float
    mae_test: float
    r2_test: float
    n_train: int
    n_test: int

    def summary(self) -> str:
        return (
            f"Train MAE: {self.mae_train:.3f}s  |  "
            f"Test MAE: {self.mae_test:.3f}s  |  "
            f"Test R^2: {self.r2_test:.3f}  |  "
            f"n_train={self.n_train}, n_test={self.n_test}"
        )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas 'category' so XGBoost handles them."""
    out = df.copy()
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("category")
    return out


def train(df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42):
    """Train XGBoost on a feature DataFrame from `features.build_feature_matrix`."""
    df = _prepare(df)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        enable_categorical=True,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    report = TrainReport(
        mae_train=mean_absolute_error(y_train, model.predict(X_train)),
        mae_test=mean_absolute_error(y_test, model.predict(X_test)),
        r2_test=r2_score(y_test, model.predict(X_test)),
        n_train=len(X_train),
        n_test=len(X_test),
    )
    return model, report


def save(model, path: Path = MODEL_PATH, *, metadata: dict | None = None) -> None:
    model.save_model(str(path))
    if metadata is not None:
        (path.with_suffix(".meta.json")).write_text(json.dumps(metadata, indent=2))


def load(path: Path = MODEL_PATH):
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def predict_lap_time(
    model,
    *,
    compound: str,
    tyre_life: float,
    lap_number: int,
    team: str,
    track: str,
    year: int,
) -> float:
    """Single-row inference helper used by the strategy engine."""
    row = pd.DataFrame([{
        "tyre_life": tyre_life,
        "lap_number": lap_number,
        "year": year,
        "compound": compound,
        "team": team,
        "track": track,
    }])
    for col in CATEGORICAL_COLS:
        row[col] = row[col].astype("category")
    return float(model.predict(row)[0])
