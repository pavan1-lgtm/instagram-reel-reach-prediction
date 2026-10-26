"""
tests/test_preprocessing.py
===========================
Unit tests for data loading, validation, and cleaning logic.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    load_reels_data,
    clean_raw_reels_data,
    ReelFeatureEngineer,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN
)


def test_missing_column_raises_error(tmp_path):
    """Uploading a CSV missing mandatory pre-posting columns must raise a ValueError."""
    bad_df = pd.DataFrame({
        "followers": [1000],
        # Missing average_previous_reach, reel_duration, etc.
    })
    csv_file = tmp_path / "bad_reels.csv"
    bad_df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError) as excinfo:
        load_reels_data(str(csv_file))
    assert "missing required columns" in str(excinfo.value)


def test_cleaning_drops_impossible_values():
    """Negative durations and hours outside [0, 23] must be dropped or sanitized."""
    dirty_data = pd.DataFrame([
        # Row 1: Valid
        {
            "account_id": "acc_001", "followers": 5000, "average_previous_reach": 2000,
            "average_engagement_rate": 3.5, "account_age_days": 100, "posts_last_30_days": 10,
            "reel_duration": 25.0, "caption_length": 100, "hashtag_count": 5,
            "content_category": "Fitness", "posting_hour": 18, "posting_day": "Monday",
            "has_trending_audio": 1, "is_original_content": 1, "actual_reach": 2500
        },
        # Row 2: Impossible hour 25 (must be dropped)
        {
            "account_id": "acc_002", "followers": 5000, "average_previous_reach": 2000,
            "average_engagement_rate": 3.5, "account_age_days": 100, "posts_last_30_days": 10,
            "reel_duration": 25.0, "caption_length": 100, "hashtag_count": 5,
            "content_category": "Fitness", "posting_hour": 25, "posting_day": "Monday",
            "has_trending_audio": 1, "is_original_content": 1, "actual_reach": 2500
        },
        # Row 3: Negative duration (must be dropped)
        {
            "account_id": "acc_003", "followers": 5000, "average_previous_reach": 2000,
            "average_engagement_rate": 3.5, "account_age_days": 100, "posts_last_30_days": 10,
            "reel_duration": -10.0, "caption_length": 100, "hashtag_count": 5,
            "content_category": "Fitness", "posting_hour": 14, "posting_day": "Monday",
            "has_trending_audio": 1, "is_original_content": 1, "actual_reach": 2500
        },
        # Row 4: Negative hashtags (must be sanitized to 0)
        {
            "account_id": "acc_004", "followers": 5000, "average_previous_reach": 2000,
            "average_engagement_rate": 3.5, "account_age_days": 100, "posts_last_30_days": 10,
            "reel_duration": 20.0, "caption_length": 100, "hashtag_count": -3,
            "content_category": "Fitness", "posting_hour": 14, "posting_day": "Monday",
            "has_trending_audio": 1, "is_original_content": 1, "actual_reach": 2500
        }
    ])

    clean_df, report = clean_raw_reels_data(dirty_data, verbose=False)
    
    # Rows 2 and 3 should be dropped -> 2 rows remain
    assert len(clean_df) == 2
    assert report["initial_rows"] == 4
    assert report["final_rows"] == 2
    # Row 4 negative hashtags should be replaced with 0
    assert (clean_df["hashtag_count"] >= 0).all()


def test_feature_engineering_transformer():
    """Verifies that ratio, caption density (+1 Laplace), and diurnal bins are properly computed."""
    sample = pd.DataFrame([{
        "account_id": "acc_01",
        "followers": 10000,
        "average_previous_reach": 5000,
        "reel_duration": 20.0,
        "caption_length": 100,
        "hashtag_count": 0,  # Test Laplace smoothing (no division by zero)
        "posting_hour": 19,
        "content_category": "Education",
        "posting_day": "Friday"
    }])

    fe = ReelFeatureEngineer(drop_posting_hour=True)
    out = fe.transform(sample)

    assert "reach_to_follower_ratio" in out.columns
    assert "caption_density" in out.columns
    assert "posting_time_category" in out.columns
    assert "posting_hour" not in out.columns

    # 5000 / 10000 = 0.5
    assert np.isclose(out["reach_to_follower_ratio"].iloc[0], 0.5)
    # 100 / (0 + 1) = 100.0
    assert np.isclose(out["caption_density"].iloc[0], 100.0)
    # Hour 19 -> Evening
    assert out["posting_time_category"].iloc[0] == "Evening"
