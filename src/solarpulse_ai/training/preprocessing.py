"""Training-fitted preprocessing pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from solarpulse_ai.training.contracts import TrainingContract


def build_preprocessor(contract: TrainingContract, *, scale: bool) -> ColumnTransformer:
    """Build deterministic transformations whose statistics are fitted by Pipeline.fit."""
    numeric_columns = [*contract.numerical, *contract.boolean]
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if contract.categorical:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical, list(contract.categorical)))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
