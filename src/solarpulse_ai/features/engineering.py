"""Compatibility exports for the feature-engineering pipeline."""

from solarpulse_ai.features.config import FeatureConfig
from solarpulse_ai.features.pipeline import FeatureResult, run_feature_pipeline

__all__ = ["FeatureConfig", "FeatureResult", "run_feature_pipeline"]
