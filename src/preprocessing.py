"""
src/preprocessing.py
====================
Data Loading, Cleaning, Feature Engineering, and Preprocessing Pipelines
for Instagram Reel Reach Prediction.

VIVA DEFENSE PRINCIPLES:
1. No Data Leakage: Only pre-posting features are processed. Imputers, scalers,
   and encoders are fit STRICTLY on training folds inside an sklearn Pipeline.
2. Robustness to Messy Data: Functions handle missing values, duplicates,
   impossible values, and extreme outliers with documented justifications.
3. Interchangeable Data Source: The loader works identically on synthetic
   and real user-exported CSVs conforming to the template schema.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Expected columns for pre-posting features + target + account_id
RAW_FEATURE_COLUMNS = [
    "account_id",
    "followers",
    "average_previous_reach",
    "average_engagement_rate",
    "account_age_days",
    "posts_last_30_days",
    "reel_duration",
    "caption_length",
    "hashtag_count",
    "content_category",
    "posting_hour",
    "posting_day",
    "has_trending_audio",
    "is_original_content",
]
TARGET_COLUMN = "actual_reach"


# ==============================================================================
# 1. DATA LOADER
# ==============================================================================
def load_reels_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw Reel data from CSV. Works identically for synthetic data
    or real Instagram Insights exports matching data/real_reels_template.csv.
    """
    df = pd.read_csv(file_path)
    # Basic validation
    missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing_cols}")
    return df


# ==============================================================================
# 2. DATA CLEANING & VALIDATION
# ==============================================================================
def clean_raw_reels_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw reels dataset before model splitting:
    1. Removes duplicate rows.
    2. Drops or fixes physically impossible values:
       - posting_hour outside [0, 23]
       - reel_duration <= 0 or reel_duration > 300 seconds (Instagram Reel hard limits)
       - hashtag_count < 0
       - actual_reach <= 0 (if target present)
    3. Handles extreme measurement outliers using justified Winsorization (capping).
       Justification: OLS Linear Regression minimizes squared residuals (SSE).
       A single 50x glitch or 5,000s duration exerts extreme leverage on slopes.
       Capping at the 99.5th percentile neutralizes high leverage while retaining
       the observation.
    4. Note: Missing values (NaN) are preserved here so they can be imputed
       strictly inside the sklearn Pipeline on the training fold (zero leakage).

    Returns:
        cleaned_df, cleaning_report_dict
    """
    initial_rows = len(df)
    report = {"initial_rows": initial_rows, "actions_taken": []}
    cleaned = df.copy()

    # Step A: Deduplication
    dups_count = cleaned.duplicated().sum()
    if dups_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        report["actions_taken"].append(f"Removed {dups_count} exact duplicate rows.")

    # Step B: Filter Impossible Values
    # 1. Posting hour must be between 0 and 23
    invalid_hour_mask = (cleaned["posting_hour"] < 0) | (cleaned["posting_hour"] > 23)
    num_invalid_hours = invalid_hour_mask.sum()
    if num_invalid_hours > 0:
        cleaned = cleaned[~invalid_hour_mask].reset_index(drop=True)
        report["actions_taken"].append(f"Dropped {num_invalid_hours} rows with impossible posting_hour (<0 or >23).")

    # 2. Reel duration must be positive and <= 300 seconds
    invalid_dur_mask = (cleaned["reel_duration"] <= 0) | (cleaned["reel_duration"] > 300)
    num_invalid_dur = invalid_dur_mask.sum()
    if num_invalid_dur > 0:
        cleaned = cleaned[~invalid_dur_mask].reset_index(drop=True)
        report["actions_taken"].append(f"Dropped {num_invalid_dur} rows with impossible reel_duration (<=0s or >300s).")

    # 3. Hashtag count cannot be negative
    invalid_tags_mask = cleaned["hashtag_count"] < 0
    num_invalid_tags = invalid_tags_mask.sum()
    if num_invalid_tags > 0:
        # Here we fix negative hashtags by replacing with 0
        cleaned.loc[invalid_tags_mask, "hashtag_count"] = 0
        report["actions_taken"].append(f"Fixed {num_invalid_tags} rows with negative hashtag_count by setting to 0.")

    # 4. Target validity (if present)
    if TARGET_COLUMN in cleaned.columns:
        invalid_target_mask = cleaned[TARGET_COLUMN] <= 0
        num_invalid_target = invalid_target_mask.sum()
        if num_invalid_target > 0:
            cleaned = cleaned[~invalid_target_mask].reset_index(drop=True)
            report["actions_taken"].append(f"Dropped {num_invalid_target} rows with non-positive {TARGET_COLUMN}.")

    # Step C: Outlier Treatment via Justified Capping (Winsorizing at 99.5th percentile)
    # Reel reach and duration glitches
    if TARGET_COLUMN in cleaned.columns:
        reach_cap = cleaned[TARGET_COLUMN].quantile(0.995)
        outlier_reach_count = (cleaned[TARGET_COLUMN] > reach_cap).sum()
        cleaned[TARGET_COLUMN] = np.clip(cleaned[TARGET_COLUMN], a_min=None, a_max=reach_cap)
        report["actions_taken"].append(
            f"Winsorized {outlier_reach_count} extreme reach outliers at 99.5th percentile ({reach_cap:.0f}). "
            "Justification: Prevents severe leverage in squared-error linear regression."
        )

    final_rows = len(cleaned)
    report["final_rows"] = final_rows
    report["rows_dropped_total"] = initial_rows - final_rows

    if verbose:
        print("=" * 70)
        print("DATA CLEANING REPORT")
        print("=" * 70)
        print(f"Initial row count: {initial_rows}")
        print(f"Final row count:   {final_rows} (Retained {final_rows/initial_rows*100:.1f}%)")
        print("Actions taken:")
        for action in report["actions_taken"]:
            print(f"  • {action}")
        print("=" * 70)

    return cleaned, report


# ==============================================================================
# 3. FEATURE ENGINEERING TRANSFORMER
# ==============================================================================
class ReelFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer.
    Computes domain-informed pre-posting features:
    1. reach_to_follower_ratio = average_previous_reach / followers
       (Measures account pull efficiency; guarded against division by zero).
    2. caption_density = caption_length / (hashtag_count + 1)
       (Chars per tag slot. The +1 is Laplace smoothing to prevent ZeroDivisionError).
    3. posting_time_category = Derived diurnal human activity bins:
       - Morning: 5 - 11
       - Afternoon: 12 - 16
       - Evening: 17 - 21
       - Night: 22 - 4
       (Linear models cannot learn cyclic hours without trigonometric terms;
        discrete diurnal categories represent human routine uplifts cleanly).
    """

    def __init__(self, drop_posting_hour: bool = True):
        self.drop_posting_hour = drop_posting_hour

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. reach_to_follower_ratio: safe division
        followers_safe = np.maximum(X_out["followers"].fillna(1.0).values, 1.0)
        prev_reach_safe = np.maximum(X_out["average_previous_reach"].fillna(0.0).values, 0.0)
        X_out["reach_to_follower_ratio"] = prev_reach_safe / followers_safe

        # 2. caption_density: caption chars per tag slot (+1 avoids division by zero)
        caption_safe = np.maximum(X_out["caption_length"].fillna(0.0).values, 0.0)
        tags_safe = np.maximum(X_out["hashtag_count"].fillna(0.0).values, 0.0)
        X_out["caption_density"] = caption_safe / (tags_safe + 1.0)

        # 3. posting_time_category derived from posting_hour
        def categorize_hour(hour):
            if pd.isna(hour):
                return "Evening"  # Default fallback
            hour = int(hour)
            if 5 <= hour <= 11:
                return "Morning"
            elif 12 <= hour <= 16:
                return "Afternoon"
            elif 17 <= hour <= 21:
                return "Evening"
            else:
                return "Night"

        X_out["posting_time_category"] = X_out["posting_hour"].apply(categorize_hour)

        # Drop posting_hour if requested to prevent redundant collinearity
        if self.drop_posting_hour and "posting_hour" in X_out.columns:
            X_out = X_out.drop(columns=["posting_hour"])

        # Drop account_id if present (metadata, not a predictive feature)
        if "account_id" in X_out.columns:
            X_out = X_out.drop(columns=["account_id"])

        return X_out


# ==============================================================================
# 4. COLUMN TRANSFORMER & PREPROCESSING PIPELINE BUILDER
# ==============================================================================
def get_feature_column_types(drop_posting_hour: bool = True):
    """
    Returns list of numeric and categorical column names post-feature engineering.
    """
    numeric_features = [
        "followers",
        "average_previous_reach",
        "average_engagement_rate",
        "account_age_days",
        "posts_last_30_days",
        "reel_duration",
        "caption_length",
        "hashtag_count",
        "has_trending_audio",
        "is_original_content",
        "reach_to_follower_ratio",
        "caption_density",
    ]
    if not drop_posting_hour:
        numeric_features.append("posting_hour")

    categorical_features = [
        "content_category",
        "posting_day",
        "posting_time_category",
    ]
    return numeric_features, categorical_features


def build_preprocessor_pipeline(drop_posting_hour: bool = True) -> Pipeline:
    """
    Builds a complete Scikit-Learn Preprocessing Pipeline:
    1. Custom ReelFeatureEngineer: generates ratios and diurnal categories.
    2. ColumnTransformer:
       - Numeric: Median Imputation -> StandardScaler
       - Categorical: Most Frequent Imputation -> OneHotEncoder(drop='first', handle_unknown='ignore')
    
    GUARANTEES ZERO DATA LEAKAGE:
    Medians, means, variances, and category levels are strictly fit on training data.
    """
    numeric_features, categorical_features = get_feature_column_types(drop_posting_hour)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full_preprocessing_pipeline = Pipeline(steps=[
        ("feature_engineer", ReelFeatureEngineer(drop_posting_hour=drop_posting_hour)),
        ("column_transformer", col_transformer),
    ])

    return full_preprocessing_pipeline
