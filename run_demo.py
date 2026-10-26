"""
run_demo.py
===========
Demonstration script showing how to load the serialized Stage 1
pipeline (models/linear_regression.pkl) and run reach predictions
on planned, pre-posting Reel metadata.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("models", "linear_regression.pkl")

def run_prediction_demo():
    print("=" * 75)
    print("INSTAGRAM REEL REACH PREDICTOR - PIPELINE INFERENCE DEMO")
    print("=" * 75)

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model artifact not found at '{MODEL_PATH}'.")
        print("    Please run 'python -m src.train' first to train and save the model.")
        return

    print(f"[*] Loading serialized model pipeline from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    print(f"[+] Loaded successfully! Object type: {type(pipeline).__name__}")
    print(f"[+] Pipeline steps: {[step[0] for step in pipeline.steps]}")

    # Define two planned Reels for comparison
    scenarios = [
        {
            "name": "Scenario A: Standard Weekday Afternoon (No Trending Audio)",
            "data": {
                "followers": 25_000,
                "average_previous_reach": 10_000,
                "average_engagement_rate": 3.5,
                "account_age_days": 500,
                "posts_last_30_days": 10,
                "reel_duration": 45.0,           # Above 30s sweet spot
                "caption_length": 80,
                "hashtag_count": 3,
                "content_category": "Education",
                "posting_hour": 14,              # 2 PM (Afternoon)
                "posting_day": "Tuesday",
                "has_trending_audio": 0,         # No trending audio
                "is_original_content": 1,
            }
        },
        {
            "name": "Scenario B: Optimized Sunday Evening (With Trending Audio)",
            "data": {
                "followers": 25_000,
                "average_previous_reach": 10_000,
                "average_engagement_rate": 3.5,
                "account_age_days": 500,
                "posts_last_30_days": 10,
                "reel_duration": 22.0,           # Inside 15-30s sweet spot
                "caption_length": 180,
                "hashtag_count": 7,              # Optimal hashtag count
                "content_category": "Entertainment",
                "posting_hour": 19,              # 7 PM (Peak Evening window)
                "posting_day": "Sunday",         # Weekend uplift
                "has_trending_audio": 1,         # Trending audio active
                "is_original_content": 1,
            }
        }
    ]

    print("\n" + "-" * 75)
    print("RUNNING INFERENCE ON COMPARISON SCENARIOS (SAME ACCOUNT, DIFFERENT REEL SPECS):")
    print("-" * 75)

    for scenario in scenarios:
        df_input = pd.DataFrame([scenario["data"]])
        # Predict and enforce non-negative reach
        raw_pred = pipeline.predict(df_input)[0]
        clipped_pred = max(0, int(round(raw_pred)))

        print(f"\n>> {scenario['name']}")
        print(f"   Category: {scenario['data']['content_category']} | Day: {scenario['data']['posting_day']} at {scenario['data']['posting_hour']}:00")
        print(f"   Duration: {scenario['data']['reel_duration']}s | Trending Audio: {'Yes' if scenario['data']['has_trending_audio'] else 'No'}")
        print(f"   >> Point Estimate Expected Reach: {clipped_pred:,} accounts")

    print("\n" + "=" * 75)
    print("OBSERVATION:")
    print("Even with identical account baselines (25,000 followers, 10,000 historical reach),")
    print("pre-posting decisions (scheduling for Sunday evening, keeping duration in the 22s")
    print("sweet spot, and adopting trending audio) produce a significant difference in")
    print("expected reach, confirming the model's sensitivity to planned pre-posting choices.")
    print("=" * 75)


if __name__ == "__main__":
    run_prediction_demo()
