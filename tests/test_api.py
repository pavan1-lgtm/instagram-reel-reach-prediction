"""
tests/test_api.py
=================
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_index_serves_html():
    """GET / must return HTTP 200 with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Reel Reach Predictor" in response.text


def test_get_metadata_endpoint():
    """GET /api/meta must return metadata including data_source and allowed categories."""
    response = client.get("/api/meta")
    assert response.status_code == 200
    data = response.json()
    assert "data_source" in data
    assert "allowed_categories" in data
    assert "allowed_days" in data
    assert "default_example" in data
    assert "prediction_interval" in data


def test_get_model_info_endpoint():
    """GET /api/model-info must return top coefficients with human readable labels."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "top_coefficients" in data
    assert len(data["top_coefficients"]) >= 5


def test_predict_endpoint_valid_payload():
    """POST /api/predict with valid payload returns HTTP 200 and expected response schema."""
    payload = {
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
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "point_estimate" in data
    assert "range_low" in data
    assert "range_high" in data
    assert data["range_low"] <= data["point_estimate"] <= data["range_high"]
    assert "suggestions" in data
    assert len(data["suggestions"]) > 0


def test_predict_endpoint_validation_negative_followers():
    """POST /api/predict with negative followers must return 422 with readable field message."""
    payload = {
        "followers": -500,  # Invalid
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
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "errors" in data
    assert "followers" in data["errors"]
    assert "Follower count must be at least 1" in data["errors"]["followers"]


def test_predict_endpoint_validation_impossible_hour():
    """POST /api/predict with hour 25 must return 422 with readable field message."""
    payload = {
        "followers": 15000,
        "average_previous_reach": 6500,
        "average_engagement_rate": 4.2,
        "account_age_days": 400,
        "posts_last_30_days": 15,
        "reel_duration": 22.0,
        "caption_length": 140,
        "hashtag_count": 6,
        "content_category": "Technology",
        "posting_hour": 25,  # Invalid
        "posting_day": "Friday",
        "has_trending_audio": 1,
        "is_original_content": 1
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "errors" in data
    assert "posting_hour" in data["errors"]
    assert "Posting hour must be between 0 and 23" in data["errors"]["posting_hour"]
