"""
app/main.py
===========
FastAPI backend for Instagram Reel Reach Prediction.

Provides:
- GET /: Serves glassmorphism frontend (app/static/index.html)
- Mounts /static for CSS, JS, and local assets (100% offline, zero CDN)
- GET /api/meta: Dataset metadata, fields, categories, ranges, and data_source
- GET /api/model-info: Top 5 model coefficients with plain-English interpretations
- POST /api/predict: Validated inference endpoint with friendly field-level errors
"""

import os
import sys
import json
from typing import Literal, Dict, Any, List
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

# Ensure root is in path for src imports
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predict import predict_reach, get_metadata
from src.suggestions import generate_reach_suggestions

app = FastAPI(
    title="Instagram Reel Reach Predictor API",
    description="Supervised ML prediction service estimating expected reach with empirical 80% intervals.",
    version="1.0.0",
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

# Mount static directory for CSS/JS
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==============================================================================
# PYDANTIC REQUEST & RESPONSE MODELS
# ==============================================================================
class ReelPredictionRequest(BaseModel):
    followers: int = Field(..., ge=1, le=100_000_000, description="Follower count (must be >= 1)")
    average_previous_reach: int = Field(..., ge=1, le=100_000_000, description="Historical average reach per reel")
    average_engagement_rate: float = Field(..., ge=0.0, le=100.0, description="Engagement rate in percent (0-100)")
    account_age_days: int = Field(180, ge=1, le=30000, description="Account age in days")
    posts_last_30_days: int = Field(12, ge=0, le=300, description="Posts in the last 30 days")
    reel_duration: float = Field(..., ge=1.0, le=90.0, description="Reel duration in seconds (1-90)")
    caption_length: int = Field(150, ge=0, le=2200, description="Caption length in characters (0-2200)")
    hashtag_count: int = Field(6, ge=0, le=30, description="Hashtag count (0-30)")
    content_category: Literal["Education", "Entertainment", "Fitness", "Gaming", "Technology"]
    posting_hour: int = Field(..., ge=0, le=23, description="Posting hour in 24-hr clock (0-23)")
    posting_day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    has_trending_audio: int = Field(0, ge=0, le=1, description="Trending audio enabled (0 or 1)")
    is_original_content: int = Field(1, ge=0, le=1, description="Original content created by account (0 or 1)")


# ==============================================================================
# FRIENDLY VALIDATION ERROR HANDLER
# Returns structured field-level messages instead of raw Pydantic schemas.
# ==============================================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    field_errors = {}
    for err in exc.errors():
        field_name = str(err["loc"][-1])
        msg = err["msg"]
        
        # User-friendly humanizations
        if "greater than or equal to 1" in msg and field_name == "followers":
            field_errors[field_name] = "Follower count must be at least 1."
        elif "greater than or equal to 1" in msg and field_name == "average_previous_reach":
            field_errors[field_name] = "Average previous reach must be at least 1."
        elif "less than or equal to 23" in msg and field_name == "posting_hour":
            field_errors[field_name] = "Posting hour must be between 0 and 23."
        elif "greater than or equal to 0" in msg and field_name == "posting_hour":
            field_errors[field_name] = "Posting hour must be between 0 and 23."
        elif "less than or equal to 90" in msg and field_name == "reel_duration":
            field_errors[field_name] = "Reel duration cannot exceed 90 seconds."
        elif "greater than or equal to 1" in msg and field_name == "reel_duration":
            field_errors[field_name] = "Reel duration must be at least 1 second."
        elif "less than or equal to 30" in msg and field_name == "hashtag_count":
            field_errors[field_name] = "Instagram allows a maximum of 30 hashtags."
        elif "Input should be 'Education'" in msg or "permitted:" in msg:
            field_errors[field_name] = f"Invalid selection for {field_name}."
        else:
            field_errors[field_name] = msg.replace("Value error, ", "").capitalize()

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid input parameters",
            "errors": field_errors
        }
    )


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the single-page Glassmorphism frontend."""
    if not os.path.exists(INDEX_PATH):
        return JSONResponse({"status": "error", "message": "Frontend static file index.html not found."})
    return FileResponse(INDEX_PATH)


@app.get("/api/meta")
async def get_metadata_endpoint():
    """
    Returns app metadata, valid category/day choices, field ranges,
    and data_source ('synthetic' vs 'real') from models/metadata.json.
    """
    meta = get_metadata()
    return {
        "project_title": meta.get("project_title", "Instagram Reel Reach Predictor"),
        "data_source": meta.get("data_source", "synthetic"),
        "training_date": meta.get("training_date", "Recent"),
        "allowed_categories": meta.get("allowed_categories", [
            "Education", "Entertainment", "Fitness", "Gaming", "Technology"
        ]),
        "allowed_days": meta.get("allowed_days", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]),
        "default_example": meta.get("default_example", {
            "followers": 12500,
            "average_previous_reach": 8200,
            "average_engagement_rate": 5.8,
            "account_age_days": 450,
            "posts_last_30_days": 16,
            "reel_duration": 24.0,
            "caption_length": 85,
            "hashtag_count": 6,
            "content_category": "Education",
            "posting_hour": 19,
            "posting_day": "Saturday",
            "has_trending_audio": 1,
            "is_original_content": 1
        }),
        "field_ranges": meta.get("field_ranges", {}),
        "prediction_interval": meta.get("prediction_interval", {
            "confidence_level": "80%",
            "test_coverage_pct": 79.9
        }),
        "metrics": meta.get("metrics", {})
    }


@app.get("/api/model-info")
async def get_model_info_endpoint():
    """Returns top 5 coefficients with plain-English labels and interpretation."""
    meta = get_metadata()
    return {
        "primary_model": "Ordinary Least Squares Linear Regression",
        "rationale": "Chosen for oral defensibility and direct interpretability of signed feature weights.",
        "top_coefficients": meta.get("top_coefficients", []),
        "vif_top": meta.get("vif_top", []),
    }


@app.post("/api/predict")
async def predict_endpoint(payload: ReelPredictionRequest):
    """
    Inference endpoint:
    - Rebuilds engineered features
    - Computes point estimate (clipped at 0)
    - Computes empirical 80% prediction range
    - Computes percentage vs. account historical average
    - Generates data-grounded improvement suggestions
    """
    input_dict = payload.model_dump()
    
    # Run prediction
    prediction = predict_reach(input_dict)
    
    # Generate suggestions
    suggestions = generate_reach_suggestions(input_dict)
    
    return {
        "status": "success",
        "point_estimate": prediction["point_estimate"],
        "range_low": prediction["range_low"],
        "range_high": prediction["range_high"],
        "formatted_point": prediction["formatted_point"],
        "formatted_range": prediction["formatted_range"],
        "range_display": prediction["range_display"],
        "pct_vs_average": prediction["pct_vs_average"],
        "confidence_level": prediction["confidence_level"],
        "suggestions": suggestions,
    }
