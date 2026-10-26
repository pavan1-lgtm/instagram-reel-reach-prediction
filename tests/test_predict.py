"""
tests/test_predict.py
=====================
Unit tests for the prediction engine (src/predict.py).
"""

import pytest
from src.predict import predict_reach


def test_predict_reach_valid_bounds():
    """Verifies that predicted reach is non-negative and range_low <= point_estimate <= range_high."""
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

    assert result["point_estimate"] >= 0
    assert result["range_low"] >= 0
    assert result["range_high"] >= result["point_estimate"]
    assert result["range_low"] <= result["point_estimate"]
    assert "formatted_point" in result
    assert "formatted_range" in result


def test_predict_reach_never_negative_for_small_accounts():
    """Even for tiny accounts with negative linear combinations, reach must be clipped at 0."""
    small_sample = {
        "followers": 5,
        "average_previous_reach": 1,
        "average_engagement_rate": 0.1,
        "account_age_days": 10,
        "posts_last_30_days": 0,
        "reel_duration": 5.0,
        "caption_length": 10,
        "hashtag_count": 0,
        "content_category": "Gaming",
        "posting_hour": 3,  # 3 AM (Night)
        "posting_day": "Tuesday",
        "has_trending_audio": 0,
        "is_original_content": 0
    }

    result = predict_reach(small_sample)
    assert result["point_estimate"] >= 0
    assert result["range_low"] >= 0
    assert result["range_high"] >= 0
