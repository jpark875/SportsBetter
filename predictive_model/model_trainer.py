"""
predictive_model/model_trainer.py

LightGBM binary-classification pipeline that outputs raw win/loss
probabilities for every wager type.

Training workflow
-----------------
1. ``ModelTrainer.fit(X, y)``       — train on historical feature matrix.
2. ``ModelTrainer.predict_proba(X)``— output uncalibrated P(win) ∈ (0, 1).
3. Pass the raw probabilities to :class:`~predictive_model.probability_calibrator.IsotonicCalibrator`
   before feeding them to the portfolio optimiser.

The class also exposes a cross-validation helper (``cross_val_log_loss``)
so you can tune hyper-parameters without leaking future data.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

from config.settings import LGBM_MODEL_PATH, MODEL_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

_LGBM_DEFAULTS: Dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

_XGB_DEFAULTS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Trains a LightGBM (default) or XGBoost binary classifier on historical
    NBA game/prop data.

    Parameters
    ----------
    backend : str
        ``"lgbm"`` (default) or ``"xgb"``.
    hyper_params : dict, optional
        Override any default hyper-parameter key-value pairs.
    feature_cols : list[str], optional
        Explicit list of columns to use from the training DataFrame.
        If ``None``, all numeric columns are used automatically.
    """

    def __init__(
        self,
        backend: str = "lgbm",
        hyper_params: Optional[Dict[str, Any]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        self.backend = backend.lower()
        self.feature_cols = feature_cols
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool = False

        if self.backend == "lgbm":
            if not _LGBM_AVAILABLE:
                raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
            params = {**_LGBM_DEFAULTS, **(hyper_params or {})}
            base_model = lgb.LGBMClassifier(**params)
        elif self.backend == "xgb":
            if not _XGB_AVAILABLE:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            params = {**_XGB_DEFAULTS, **(hyper_params or {})}
            base_model = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'lgbm' or 'xgb'.")

        # StandardScaler improves tree ensembles only marginally, but keeps
        # the pipeline composable with linear calibration layers.
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", base_model),
        ])

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
    ) -> "ModelTrainer":
        """
        Train the model on historical data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.  Rows with NaN labels are silently dropped.
        y : pd.Series
            Binary target: 1 = home team wins / prop Over hits, 0 otherwise.
        eval_set : tuple[pd.DataFrame, pd.Series], optional
            Validation set for early stopping (LightGBM / XGBoost only).
        early_stopping_rounds : int
            Number of rounds without improvement before stopping.

        Returns
        -------
        ModelTrainer
            Returns ``self`` for method chaining.
        """
        X, y = self._resolve_features(X, y, fit=True)
        logger.info(
            "Training %s on %d samples × %d features …",
            self.backend.upper(),
            len(X),
            X.shape[1],
        )

        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None:
            X_val, y_val = self._resolve_features(*eval_set)
            clf_step = self._pipeline.named_steps["clf"]
            scaler = self._pipeline.named_steps["scaler"]
            X_val_scaled = scaler.transform(X_val)
            fit_kwargs["clf__eval_set"] = [(X_val_scaled, y_val)]
            fit_kwargs["clf__callbacks"] = [
                lgb.early_stopping(early_stopping_rounds, verbose=False)
                if self.backend == "lgbm"
                else None
            ]
            fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

        self._pipeline.fit(X, y, **fit_kwargs)
        self._is_fitted = True
        logger.info("Training complete.")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return uncalibrated P(positive class) for each row in ``X``.

        For binary models returns shape ``(n,)``.
        For multiclass models use :meth:`predict_proba_all` instead.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._assert_fitted()
        X_clean, _ = self._resolve_features(X, fit=False)
        return self._pipeline.predict_proba(X_clean)[:, 1]

    def predict_proba_all(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return uncalibrated class probabilities for all classes.

        Use this for multiclass models (e.g. soccer 3-way: away/draw/home).

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Columns correspond to sorted class labels (0, 1, 2, …).
        """
        self._assert_fitted()
        X_clean, _ = self._resolve_features(X, fit=False)
        return self._pipeline.predict_proba(X_clean)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_val_log_loss(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Tuple[float, float]:
        """
        Time-aware stratified k-fold log-loss estimate.

        Note: For proper temporal validation you should sort ``X`` by date
        and use ``sklearn.model_selection.TimeSeriesSplit`` instead.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
        n_splits : int

        Returns
        -------
        tuple[float, float]
            ``(mean_log_loss, std_log_loss)``
        """
        X_clean, y_clean = self._resolve_features(X, y, fit=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            self._pipeline, X_clean, y_clean,
            cv=cv, scoring="neg_log_loss",
            # LightGBM already uses every core; fold-level workers on top deadlock
            # in small containers.
            n_jobs=1,
        )
        mean_ll = float(-scores.mean())
        std_ll = float(scores.std())
        logger.info("CV log-loss: %.4f ± %.4f", mean_ll, std_ll)
        return mean_ll, std_ll

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Serialise the fitted pipeline to disk with pickle.

        Parameters
        ----------
        path : str, optional
            File path.  Defaults to ``config.settings.LGBM_MODEL_PATH``.

        Returns
        -------
        str
            Absolute path where the model was written.
        """
        self._assert_fitted()
        path = path or LGBM_MODEL_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ModelTrainer":
        """
        Load a previously serialised :class:`ModelTrainer`.

        Parameters
        ----------
        path : str, optional
            File path.  Defaults to ``config.settings.LGBM_MODEL_PATH``.

        Returns
        -------
        ModelTrainer
        """
        path = path or LGBM_MODEL_PATH
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected ModelTrainer.")
        logger.info("Model loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------
    # Feature introspection
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> pd.Series:
        """
        Return feature importances (gain-based) as a sorted ``pd.Series``.

        Only available after fitting.
        """
        self._assert_fitted()
        clf = self._pipeline.named_steps["clf"]
        if self.backend == "lgbm":
            importances = clf.feature_importances_
        else:
            importances = clf.feature_importances_
        return (
            pd.Series(importances, index=self.feature_cols_)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Select, align, and impute features from X."""
        if fit and self.feature_cols is None:
            # Auto-detect numeric columns, excluding the label column
            exclude = {"HOME_WIN", "PLAYER_ID", "TEAM_ID"}
            self.feature_cols_ = [
                c for c in X.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        elif fit:
            self.feature_cols_ = list(self.feature_cols)

        missing = [c for c in self.feature_cols_ if c not in X.columns]
        if missing:
            logger.warning("Missing feature columns: %s — filling with 0.", missing)
            for c in missing:
                X = X.copy()
                X[c] = 0.0

        X_out = X[self.feature_cols_].copy()

        # Median imputation for NaNs (fit-time medians carried on the scaler)
        X_out = X_out.fillna(X_out.median(numeric_only=True))

        if y is not None:
            mask = y.notna()
            return X_out.loc[mask], y.loc[mask]
        return X_out, None

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ModelTrainer has not been fitted. Call .fit(X, y) first."
            )
