"""
src/predict.py
==============
Prediction engine for Instagram Reel Reach.

Loads the fitted Scikit-Learn pipeline, transforms raw planned inputs,
enforces non-negativity (clipping at 0), and computes an empirical
80% prediction range derived from test-set error ratio distributions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "linear_regression.pkl")
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metadata.json")

_CACHED_PIPELINE = None
_CACHED_METADATA = None


def get_pipeline():
    """Loads and caches the serialized model pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python -m src.train' first.")
        _CACHED_PIPELINE = joblib.load(MODEL_PATH)
    return _CACHED_PIPELINE


def get_metadata() -> Dict[str, Any]:
    """Loads and caches model metadata and interval factors."""
    global _CACHED_METADATA
    if _CACHED_METADATA is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _CACHED_METADATA = json.load(f)
        else:
            # Safe defaults matching Stage 2 empirical calibration
            _CACHED_METADATA = {
                "data_source": "synthetic",
                "prediction_interval": {
                    "ratio_p10": 0.6489,
                    "ratio_p90": 1.3912,
                    "test_coverage_pct": 79.9
                }
            }
    return _CACHED_METADATA


def format_reach_number(num: int) -> str:
    """Formats numbers with K/M notation for readable ranges (e.g., 12.5K)."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M".replace(".0M", "M")
    elif num >= 10_000:
        return f"{num / 1_000:.1f}K".replace(".0K", "K")
    elif num >= 1_000:
        return f"{num:,.0f}"
    return str(num)


def predict_reach(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts expected Reel reach from pre-posting characteristics.
    
    Returns:
        dict: {
            "point_estimate": int,
            "range_low": int,
            "range_high": int,
            "pct_vs_average": float,
            "formatted_point": str,
            "formatted_range": str,
            "range_display": str
        }
    """
    pipeline = get_pipeline()
    metadata = get_metadata()
    
    # Ensure inputs are formatted into a single-row DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Predict raw reach
    raw_prediction = float(pipeline.predict(df_input)[0])
    
    # Clip at 0: Reach cannot be negative (Linear Regression can mathematically go negative)
    point_estimate = max(0, int(round(raw_prediction)))
    
    # Empirical 80% Prediction Interval:
    # Uses 10th and 90th percentiles of actual/predicted ratios from test set
    interval_info = metadata.get("prediction_interval", {})
    ratio_p10 = float(interval_info.get("ratio_p10", 0.65))
    ratio_p90 = float(interval_info.get("ratio_p90", 1.39))
    
    range_low = max(0, int(round(point_estimate * ratio_p10)))
    range_high = max(point_estimate, int(round(point_estimate * ratio_p90)))
    
    # Compute percentage lift over account historical average
    avg_reach = float(input_dict.get("average_previous_reach", 1.0))
    if avg_reach > 0:
        pct_vs_avg = round(((point_estimate - avg_reach) / avg_reach) * 100.0, 1)
    else:
        pct_vs_avg = 0.0

    low_str = format_reach_number(range_low)
    high_str = format_reach_number(range_high)
    range_display = f"expected range {low_str} – {high_str}"

    return {
        "point_estimate": point_estimate,
        "range_low": range_low,
        "range_high": range_high,
        "pct_vs_average": pct_vs_avg,
        "formatted_point": f"{point_estimate:,}",
        "formatted_range": f"{range_low:,} – {range_high:,}",
        "range_display": range_display,
        "confidence_level": interval_info.get("confidence_level", "80%")
    }


if __name__ == "__main__":
    sample = {
        "followers": 15000,
        "average_previous_reach": 6500,
        "average_engagement_rate": 4.2,
        "account_age_days": 400,
        "posts_last_30_days": 15,
        "reel_duration": 22.0,
        "caption_length": 140,
        "hashtag_count": 6,
        "content_category": "Technology",
        "posting_hour": 19,
        "posting_day": "Friday",
        "has_trending_audio": 1,
        "is_original_content": 1
    }
    result = predict_reach(sample)
    print("Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
