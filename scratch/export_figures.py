"""
Save EDA plots to notebooks/figures for documentation and viva presentation slides.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("src"))
from preprocessing import load_reels_data, clean_raw_reels_data, TARGET_COLUMN

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
fig_dir = os.path.join("notebooks", "figures")
os.makedirs(fig_dir, exist_ok=True)

raw_df = load_reels_data(os.path.join("data", "reels_dataset.csv"))
clean_df, _ = clean_raw_reels_data(raw_df, verbose=False)

# 1. Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(clean_df[TARGET_COLUMN], kde=True, ax=axes[0], color="#2b5c8f", bins=40)
axes[0].set_title("Distribution of Raw Reach (actual_reach)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Actual Reach (People Reached)")
sns.histplot(np.log1p(clean_df[TARGET_COLUMN]), kde=True, ax=axes[1], color="#e07a5f", bins=40)
axes[1].set_title("Distribution of Log-Transformed Reach: log1p(actual_reach)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("log1p(Actual Reach)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "01_reach_distribution.png"))
plt.close()

# 2. Correlation Matrix
numeric_cols = [
    "followers", "average_previous_reach", "average_engagement_rate",
    "account_age_days", "posts_last_30_days", "reel_duration",
    "caption_length", "hashtag_count", "has_trending_audio",
    "is_original_content", "actual_reach"
]
corr_matrix = clean_df[numeric_cols].corr()
plt.figure(figsize=(10, 7.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Pre-Posting Features & Target", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "02_correlation_heatmap.png"))
plt.close()

# 3. Categorical Uplifts
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.boxplot(data=clean_df, x="content_category", y=TARGET_COLUMN, ax=axes[0], palette="Set2", showfliers=False)
axes[0].set_title("Reach by Content Category", fontweight="bold")
axes[0].tick_params(axis="x", rotation=25)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.barplot(data=clean_df, x="posting_day", y=TARGET_COLUMN, order=day_order, ax=axes[1], color="#3d85c6", errorbar=None)
axes[1].set_title("Average Reach by Posting Day", fontweight="bold")
axes[1].tick_params(axis="x", rotation=25)

hourly_reach = clean_df.groupby("posting_hour")[TARGET_COLUMN].mean().reset_index()
sns.lineplot(data=hourly_reach, x="posting_hour", y=TARGET_COLUMN, ax=axes[2], marker="o", color="#e65100", linewidth=2.5)
axes[2].set_title("Average Reach by Posting Hour (0-23)", fontweight="bold")
axes[2].set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "03_reach_uplifts_by_category_day_hour.png"))
plt.close()

# 4. Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.scatterplot(data=clean_df, x="followers", y=TARGET_COLUMN, hue="has_trending_audio", alpha=0.6, ax=axes[0], palette={0: "#7f7f7f", 1: "#d95f02"})
axes[0].set_title("Followers vs Actual Reach (by Trending Audio)", fontweight="bold")
sns.scatterplot(data=clean_df, x="average_previous_reach", y=TARGET_COLUMN, hue="is_original_content", alpha=0.6, ax=axes[1], palette={0: "#7570b3", 1: "#1b9e77"})
axes[1].set_title("Average Previous Reach vs Actual Reach (by Original Content)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "04_scatter_account_vs_reach.png"))
plt.close()

print(f"[+] All 4 EDA figures successfully exported to: {fig_dir}")
