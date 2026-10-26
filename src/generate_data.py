"""
src/generate_data.py
====================
Synthetic Dataset Generator for Instagram Reel Reach Prediction.

IMPORTANT NOTICE:
This dataset is 100% SYNTHETIC. It is generated for educational, methodological,
and viva demonstration purposes. It demonstrates the statistical modeling workflow
and does NOT reflect private internal algorithms of Meta or Instagram.

Generator Design:
- ~400 accounts, ~10 Reels each (~4,000 Reels total).
- Log-normally distributed followers and correlated historical reach.
- Multiplicative documented effects (duration sweet-spot, timing uplifts, audio,
  originality, engagement rate, hashtag plateau, category benchmarks).
- Controlled log-normal noise (sigma = 0.25).
- Realistic anomalies deliberately injected for data cleaning demonstration:
  missing values (~2%), duplicates, impossible values (e.g. negative duration,
  posting hour 25), and extreme outliers.
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# TRANSPARENT EFFECT SIZES & GENERATION CONFIGURATION
# College students can point to this single dictionary during viva to explain
# exactly how the underlying data distribution and effect sizes were modeled.
# ==============================================================================
DATA_GENERATION_CONFIG = {
    "random_seed": 42,
    "num_accounts": 400,
    "reels_per_account": 10,
    
    # 1. Follower & Account baselines
    "followers_log_mean": 9.4,       # Median followers ~ exp(9.4) ≈ 12,000
    "followers_log_sigma": 1.15,      # Spread: small creators to large accounts
    "min_reach_ratio": 0.15,          # Baseline reach is 15% - 75% of follower count
    "max_reach_ratio": 0.75,
    "engagement_rate_mean": 3.8,      # Average engagement rate in %
    "engagement_rate_std": 1.4,
    
    # 2. Documented Empirical Reel Uplifts (Multipliers)
    # 15-30s sweet spot: +15% reach boost; over 60s or under 10s: slight penalty
    "duration_sweet_spot": {"min_sec": 15, "max_sec": 30, "uplift": 1.15, "penalty": 0.88},
    
    # Peak audience activity: Evening (17:00-21:00) gives +12%; Weekends give +10%
    "evening_hours": [17, 18, 19, 20, 21],
    "evening_uplift": 1.12,
    "weekend_days": ["Saturday", "Sunday"],
    "weekend_uplift": 1.10,
    
    # Trending audio & originality uplift
    "trending_audio_uplift": 1.18,    # +18% boost when using viral audio
    "original_content_uplift": 1.14,  # +14% boost for non-reposted original reels
    
    # Account engagement rate: every 1% above 3.5% yields ~3% more baseline reach
    "engagement_rate_slope": 0.03,
    
    # Content category baseline differences (relative to Entertainment benchmark)
    "category_uplifts": {
        "Entertainment": 1.05,
        "Technology": 1.02,
        "Fitness": 0.98,
        "Education": 0.96,
        "Gaming": 0.92,
    },
    
    # Hashtag returns: optimal around 5-8 tags (+6%). Diminishing returns/spam penalty > 14
    "hashtag_optimal_min": 4,
    "hashtag_optimal_max": 8,
    "hashtag_optimal_uplift": 1.06,
    "hashtag_excess_penalty": 0.96,
    
    # Log-normal noise: sigma=0.25 (realistic real-world viral variance)
    "noise_sigma": 0.25,
    
    # 3. Deliberately injected mess for cleaning demonstration
    "missing_rate": 0.02,             # ~2% missing across various columns
    "num_duplicates": 15,             # 15 exact duplicate rows
    "num_impossible_values": 18,      # E.g. hour=25, duration=-10s
    "num_extreme_outliers": 8,        # E.g. duration=4800s, reach 40x spike
}


def generate_synthetic_reels_dataset(config: dict = DATA_GENERATION_CONFIG) -> pd.DataFrame:
    """
    Generates a realistic synthetic DataFrame of accounts and reels with
    deliberately injected dirty data for cleaning practice.
    """
    rng = np.random.default_rng(config["random_seed"])
    num_accounts = config["num_accounts"]
    reels_per_account = config["reels_per_account"]
    total_reels = num_accounts * reels_per_account

    # -------------------------------------------------------------------------
    # STEP 1: Generate Account-Level Attributes
    # -------------------------------------------------------------------------
    account_ids = [f"acc_{i+1:04d}" for i in range(num_accounts)]
    
    # Followers: log-normal distribution to simulate realistic creator distribution
    followers = np.exp(rng.normal(
        config["followers_log_mean"], 
        config["followers_log_sigma"], 
        num_accounts
    )).astype(int)
    followers = np.clip(followers, 800, 1_200_000)

    # Historical reach ratio: accounts typically reach between 15% to 75% of followers
    reach_ratios = rng.uniform(config["min_reach_ratio"], config["max_reach_ratio"], num_accounts)
    average_previous_reach = np.round(followers * reach_ratios).astype(int)
    average_previous_reach = np.maximum(average_previous_reach, 100)

    # Engagement rate (%)
    avg_engagement = rng.normal(config["engagement_rate_mean"], config["engagement_rate_std"], num_accounts)
    avg_engagement = np.clip(np.round(avg_engagement, 2), 0.5, 14.0)

    # Account age (days) and activity in last 30 days
    account_age_days = rng.integers(60, 2500, size=num_accounts)
    posts_last_30_days = rng.poisson(lam=12, size=num_accounts)
    posts_last_30_days = np.clip(posts_last_30_days, 1, 45)

    account_df = pd.DataFrame({
        "account_id": account_ids,
        "followers": followers,
        "average_previous_reach": average_previous_reach,
        "average_engagement_rate": avg_engagement,
        "account_age_days": account_age_days,
        "posts_last_30_days": posts_last_30_days,
    })

    # -------------------------------------------------------------------------
    # STEP 2: Generate Reel-Level Characteristics for each account
    # -------------------------------------------------------------------------
    categories = list(config["category_uplifts"].keys())
    category_weights = [0.30, 0.22, 0.18, 0.18, 0.12]  # Entertainment most common
    
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_weights = [0.13, 0.13, 0.13, 0.14, 0.16, 0.16, 0.15]  # Slight weekend tilt

    reels_list = []
    
    for _, acc in account_df.iterrows():
        for _ in range(reels_per_account):
            # Reel duration (seconds): mean 25s, std 14s, clipped between 5s and 90s
            duration = float(np.clip(rng.normal(25.0, 12.0), 5.0, 90.0))
            duration = round(duration, 1)

            # Caption length (chars): median ~160 chars
            caption_len = int(np.clip(rng.normal(160, 75), 10, 650))

            # Hashtag count: average ~6 hashtags
            hashtags = int(np.clip(rng.poisson(lam=6), 0, 25))

            category = str(rng.choice(categories, p=category_weights))
            posting_day = str(rng.choice(days_of_week, p=day_weights))
            
            # Hourly distribution: peaks at 18:00 - 21:00
            hour_probs = np.array([
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02, # 0-5
                0.03, 0.04, 0.05, 0.05, 0.05, 0.05, # 6-11
                0.06, 0.06, 0.05, 0.05, 0.06, 0.07, # 12-17
                0.08, 0.09, 0.07, 0.04, 0.02, 0.01  # 18-23
            ])
            hour_probs = hour_probs / hour_probs.sum()
            posting_hour = int(rng.choice(range(24), p=hour_probs))

            # Audio & Originality
            has_trending_audio = int(rng.binomial(1, 0.35))
            is_original_content = int(rng.binomial(1, 0.82))

            # -----------------------------------------------------------------
            # MULTIPLIER CALCULATION
            # actual_reach = average_previous_reach * multiplier * noise
            # -----------------------------------------------------------------
            mult = 1.0

            # 1. Duration sweet-spot
            if config["duration_sweet_spot"]["min_sec"] <= duration <= config["duration_sweet_spot"]["max_sec"]:
                mult *= config["duration_sweet_spot"]["uplift"]
            elif duration > 60.0 or duration < 10.0:
                mult *= config["duration_sweet_spot"]["penalty"]

            # 2. Timing uplifts
            if posting_hour in config["evening_hours"]:
                mult *= config["evening_uplift"]
            if posting_day in config["weekend_days"]:
                mult *= config["weekend_uplift"]

            # 3. Audio & Originality
            if has_trending_audio == 1:
                mult *= config["trending_audio_uplift"]
            if is_original_content == 1:
                mult *= config["original_content_uplift"]

            # 4. Engagement rate effect
            eng_diff = acc["average_engagement_rate"] - config["engagement_rate_mean"]
            mult *= (1.0 + config["engagement_rate_slope"] * eng_diff)

            # 5. Category uplift
            mult *= config["category_uplifts"].get(category, 1.0)

            # 6. Hashtags uplift / diminishing returns
            if config["hashtag_optimal_min"] <= hashtags <= config["hashtag_optimal_max"]:
                mult *= config["hashtag_optimal_uplift"]
            elif hashtags > 14:
                mult *= config["hashtag_excess_penalty"]

            # 7. Unobservable random noise (log-normal)
            noise = rng.lognormal(mean=0.0, sigma=config["noise_sigma"])

            # Compute actual reach
            actual_reach = int(round(acc["average_previous_reach"] * mult * noise))
            actual_reach = max(actual_reach, 25)

            reels_list.append({
                "account_id": acc["account_id"],
                "followers": acc["followers"],
                "average_previous_reach": acc["average_previous_reach"],
                "average_engagement_rate": acc["average_engagement_rate"],
                "account_age_days": acc["account_age_days"],
                "posts_last_30_days": acc["posts_last_30_days"],
                "reel_duration": duration,
                "caption_length": caption_len,
                "hashtag_count": hashtags,
                "content_category": category,
                "posting_hour": posting_hour,
                "posting_day": posting_day,
                "has_trending_audio": has_trending_audio,
                "is_original_content": is_original_content,
                "actual_reach": actual_reach
            })

    df = pd.DataFrame(reels_list)

    # -------------------------------------------------------------------------
    # STEP 3: Inject Realistic Dirty Data for Preprocessing Practice
    # -------------------------------------------------------------------------
    # A. Duplicates: duplicate a small subset of rows
    dup_indices = rng.choice(df.index, size=config["num_duplicates"], replace=False)
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # B. Missing values: inject ~2% missing in selected columns
    cleanable_cols = [
        "average_engagement_rate", "posts_last_30_days", 
        "caption_length", "hashtag_count", "content_category"
    ]
    for col in cleanable_cols:
        mask = rng.random(len(df)) < config["missing_rate"]
        df.loc[mask, col] = np.nan

    # C. Impossible values
    # - Negative durations
    neg_duration_idx = rng.choice(df.index, size=6, replace=False)
    df.loc[neg_duration_idx, "reel_duration"] = rng.choice([-15.0, -8.5, -2.0, -30.0], size=6)
    
    # - Impossible posting hours (e.g., 25, 27)
    imp_hour_idx = rng.choice(df.index, size=6, replace=False)
    df.loc[imp_hour_idx, "posting_hour"] = rng.choice([24, 25, 28, 99], size=6)

    # - Negative hashtag count
    neg_tag_idx = rng.choice(df.index, size=6, replace=False)
    df.loc[neg_tag_idx, "hashtag_count"] = -5

    # D. Extreme Outliers
    # - Unrealistic reel duration (e.g. 5,000 seconds = 83 minutes for a Reel)
    outlier_dur_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[outlier_dur_idx, "reel_duration"] = 5200.0

    # - Glitched reach (50x spike, impossible data entry anomaly)
    outlier_reach_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[outlier_reach_idx, "actual_reach"] = df.loc[outlier_reach_idx, "actual_reach"] * 45

    # Shuffle rows to distribute messy records randomly
    df = df.sample(frac=1.0, random_state=config["random_seed"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reels_dataset.csv")

    print("[*] Generating synthetic Instagram Reel dataset...")
    df_generated = generate_synthetic_reels_dataset(DATA_GENERATION_CONFIG)
    df_generated.to_csv(output_path, index=False)
    
    print(f"[+] Dataset successfully created at: {output_path}")
    print(f"[+] Total rows generated: {len(df_generated)}")
    print(f"[+] Total unique accounts: {df_generated['account_id'].nunique()}")
    print(f"[+] Missing values present: {df_generated.isna().sum().sum()}")
    print("[i] Dataset is marked as SYNTHETIC for educational and viva demonstration.")
