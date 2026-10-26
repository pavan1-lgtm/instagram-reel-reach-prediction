/**
 * app.js
 * ======
 * Client-side Controller for Reel Reach Predictor
 *
 * Responsibilities:
 * - Dynamic metadata loading (/api/meta) and synthetic data badge activation
 * - Top-5 model coefficient horizontal visualization (/api/model-info)
 * - Real-time posting hour slider formatting (12-hour AM/PM)
 * - Segmented toggle control for Trending Audio and Original Content
 * - "Load Example" instant filler
 * - Asynchronous prediction fetch (/api/predict)
 * - Shimmer skeleton loading state
 * - Smooth count-up animation for point estimate (reduced-motion compliant)
 * - Empirical 80% range bar visualizer
 * - Inline SVG suggestion chip rendering
 * - Field-level error validation messaging
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM References
  const form = document.getElementById("prediction-form");
  const btnLoadExample = document.getElementById("btn-load-example");
  const dataSourceBadge = document.getElementById("data-source-badge");
  const dataSourceText = document.getElementById("data-source-text");
  const coefContainer = document.getElementById("coef-container");

  // Form inputs
  const inputFollowers = document.getElementById("input-followers");
  const inputAvgReach = document.getElementById("input-avg-reach");
  const inputEngagement = document.getElementById("input-engagement");
  const inputPosts30d = document.getElementById("input-posts-30d");
  const inputAccountAge = document.getElementById("input-account-age");
  const inputDuration = document.getElementById("input-duration");
  const selectCategory = document.getElementById("select-category");
  const selectDay = document.getElementById("select-day");
  const inputHashtags = document.getElementById("input-hashtags");
  const inputCaption = document.getElementById("input-caption");
  const rangeHour = document.getElementById("range-posting-hour");
  const hourDisplay = document.getElementById("hour-display");

  // Segmented Toggles
  const btnAudioNo = document.getElementById("btn-audio-no");
  const btnAudioYes = document.getElementById("btn-audio-yes");
  const inputAudio = document.getElementById("input-audio");

  const btnOrigNo = document.getElementById("btn-orig-no");
  const btnOrigYes = document.getElementById("btn-orig-yes");
  const inputOriginal = document.getElementById("input-original");

  // Result card states
  const stateEmpty = document.getElementById("state-empty");
  const stateLoading = document.getElementById("state-loading");
  const stateSuccess = document.getElementById("state-success");
  const stateError = document.getElementById("state-error");
  const errorMsgText = document.getElementById("error-message-text");

  // Result output elements
  const resultReachNumber = document.getElementById("result-reach-number");
  const resultRangeText = document.getElementById("result-range-text");
  const resultBenchmark = document.getElementById("result-benchmark");
  const benchmarkText = document.getElementById("benchmark-text");
  const rangeFill = document.getElementById("range-fill");
  const rangePin = document.getElementById("range-pin");
  const barLabelLow = document.getElementById("bar-label-low");
  const barLabelMid = document.getElementById("bar-label-mid");
  const barLabelHigh = document.getElementById("bar-label-high");
  const suggestionsContainer = document.getElementById("suggestions-container");

  let defaultExample = null;

  // ============================================================================
  // 1. FORMATTING HELPERS
  // ============================================================================
  function formatHour(val) {
    const h = parseInt(val, 10);
    if (h === 0) return "0:00 (12 AM)";
    if (h < 12) return `${h}:00 (${h} AM)`;
    if (h === 12) return "12:00 (12 PM)";
    return `${h}:00 (${h - 12} PM)`;
  }

  function formatK(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1).replace(".0", "") + "M";
    if (num >= 10000) return (num / 1000).toFixed(1).replace(".0", "") + "K";
    return Number(num).toLocaleString();
  }

  // Check reduced motion preference
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function animateCountUp(targetVal) {
    if (prefersReducedMotion) {
      resultReachNumber.textContent = Number(targetVal).toLocaleString();
      return;
    }

    const duration = 750; // ms
    const startTime = performance.now();
    const startVal = 0;

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic curve
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      const currentVal = Math.round(startVal + (targetVal - startVal) * easedProgress);

      resultReachNumber.textContent = currentVal.toLocaleString();

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        resultReachNumber.textContent = Number(targetVal).toLocaleString();
      }
    }
    requestAnimationFrame(step);
  }

  // ============================================================================
  // 2. INITIALIZE METADATA & MODEL INFO
  // ============================================================================
  async function loadMetadata() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) return;
      const meta = await res.json();

      defaultExample = meta.default_example;

      // Update synthetic badge if data_source is synthetic
      if (meta.data_source === "synthetic") {
        dataSourceBadge.style.display = "inline-flex";
        dataSourceText.textContent = "Demo model trained on synthetic data";
      } else {
        dataSourceBadge.style.display = "none";
      }
    } catch (err) {
      console.warn("Could not fetch /api/meta:", err);
    }
  }

  async function loadModelInfo() {
    try {
      const res = await fetch("/api/model-info");
      if (!res.ok) return;
      const data = await res.json();
      const topCoefs = data.top_coefficients || [];

      if (topCoefs.length === 0) {
        coefContainer.innerHTML = "<p style='color:var(--text-muted);font-size:0.8rem;'>Model coefficient data unavailable.</p>";
        return;
      }

      // Find max absolute coefficient for relative bar widths
      const maxAbs = Math.max(...topCoefs.map(c => Math.abs(c.coefficient)));

      coefContainer.innerHTML = topCoefs.map(item => {
        const pct = Math.min(100, Math.round((Math.abs(item.coefficient) / maxAbs) * 100));
        const sign = item.coefficient > 0 ? "+" : "-";
        const formatted = `${sign}${Math.abs(item.coefficient).toLocaleString()}`;
        return `
          <div class="coef-item">
            <div class="coef-meta">
              <span>${item.label}</span>
              <span style="color:var(--accent-cyan); font-family:monospace;">${formatted}</span>
            </div>
            <div class="coef-bar-track" title="${item.meaning}">
              <div class="coef-bar-fill" style="width: ${pct}%;"></div>
            </div>
            <p style="font-size:0.72rem; color:var(--text-muted);">${item.meaning}</p>
          </div>
        `;
      }).join("");
    } catch (err) {
      console.warn("Could not fetch /api/model-info:", err);
    }
  }

  // ============================================================================
  // 3. UI INTERACTIONS & TOGGLES
  // ============================================================================
  // Posting Hour slider
  rangeHour.addEventListener("input", (e) => {
    hourDisplay.textContent = formatHour(e.target.value);
  });

  // Trending Audio Segmented Toggle
  btnAudioNo.addEventListener("click", () => {
    btnAudioNo.classList.add("active");
    btnAudioYes.classList.remove("active");
    inputAudio.value = "0";
  });
  btnAudioYes.addEventListener("click", () => {
    btnAudioYes.classList.add("active");
    btnAudioNo.classList.remove("active");
    inputAudio.value = "1";
  });

  // Original Content Segmented Toggle
  btnOrigNo.addEventListener("click", () => {
    btnOrigNo.classList.add("active");
    btnOrigYes.classList.remove("active");
    inputOriginal.value = "0";
  });
  btnOrigYes.addEventListener("click", () => {
    btnOrigYes.classList.add("active");
    btnOrigNo.classList.remove("active");
    inputOriginal.value = "1";
  });

  // Load Example Button
  btnLoadExample.addEventListener("click", () => {
    const ex = defaultExample || {
      followers: 12500,
      average_previous_reach: 8200,
      average_engagement_rate: 5.8,
      account_age_days: 450,
      posts_last_30_days: 16,
      reel_duration: 24.0,
      caption_length: 85,
      hashtag_count: 6,
      content_category: "Education",
      posting_hour: 19,
      posting_day: "Saturday",
      has_trending_audio: 1,
      is_original_content: 1
    };

    inputFollowers.value = ex.followers;
    inputAvgReach.value = ex.average_previous_reach;
    inputEngagement.value = ex.average_engagement_rate;
    inputPosts30d.value = ex.posts_last_30_days;
    inputAccountAge.value = ex.account_age_days;
    inputDuration.value = ex.reel_duration;
    selectCategory.value = ex.content_category;
    selectDay.value = ex.posting_day;
    inputHashtags.value = ex.hashtag_count;
    inputCaption.value = ex.caption_length;
    rangeHour.value = ex.posting_hour;
    hourDisplay.textContent = formatHour(ex.posting_hour);

    // Audio toggle
    if (ex.has_trending_audio === 1) {
      btnAudioYes.click();
    } else {
      btnAudioNo.click();
    }

    // Original toggle
    if (ex.is_original_content === 1) {
      btnOrigYes.click();
    } else {
      btnOrigNo.click();
    }

    clearErrors();
  });

  // ============================================================================
  // 4. ERROR CLEARING & DISPLAY
  // ============================================================================
  function clearErrors() {
    document.querySelectorAll(".form-group").forEach(el => el.classList.remove("has-error"));
    document.querySelectorAll(".field-error-msg").forEach(el => {
      el.textContent = "";
      el.style.display = "none";
    });
    stateError.style.display = "none";
  }

  function displayFieldErrors(errors) {
    stateError.style.display = "block";
    errorMsgText.textContent = "Please fix the highlighted fields below.";

    for (const [field, message] of Object.entries(errors)) {
      let group = null;
      let errSpan = null;

      if (field === "followers") {
        group = document.getElementById("group-followers");
        errSpan = document.getElementById("err-followers");
      } else if (field === "average_previous_reach") {
        group = document.getElementById("group-avg-reach");
        errSpan = document.getElementById("err-average_previous_reach");
      } else if (field === "average_engagement_rate") {
        group = document.getElementById("group-engagement");
        errSpan = document.getElementById("err-average_engagement_rate");
      } else if (field === "posts_last_30_days") {
        group = document.getElementById("group-posts-30d");
        errSpan = document.getElementById("err-posts_last_30_days");
      } else if (field === "account_age_days") {
        group = document.getElementById("group-account-age");
        errSpan = document.getElementById("err-account_age_days");
      } else if (field === "reel_duration") {
        group = document.getElementById("group-duration");
        errSpan = document.getElementById("err-reel_duration");
      } else if (field === "content_category") {
        group = document.getElementById("group-category");
        errSpan = document.getElementById("err-content_category");
      } else if (field === "posting_day") {
        group = document.getElementById("group-day");
        errSpan = document.getElementById("err-posting_day");
      } else if (field === "hashtag_count") {
        group = document.getElementById("group-hashtags");
        errSpan = document.getElementById("err-hashtag_count");
      } else if (field === "caption_length") {
        group = document.getElementById("group-caption");
        errSpan = document.getElementById("err-caption_length");
      } else if (field === "posting_hour") {
        group = document.getElementById("group-hour");
        errSpan = document.getElementById("err-posting_hour");
      }

      if (group && errSpan) {
        group.classList.add("has-error");
        errSpan.textContent = message;
        errSpan.style.display = "block";
      }
    }
  }

  // ============================================================================
  // 5. ASYNC FORM SUBMISSION & PREDICTION FLOW
  // ============================================================================
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    // Construct payload
    const payload = {
      followers: parseInt(inputFollowers.value, 10),
      average_previous_reach: parseInt(inputAvgReach.value, 10),
      average_engagement_rate: parseFloat(inputEngagement.value),
      account_age_days: parseInt(inputAccountAge.value, 10) || 180,
      posts_last_30_days: parseInt(inputPosts30d.value, 10) || 12,
      reel_duration: parseFloat(inputDuration.value),
      content_category: selectCategory.value,
      posting_day: selectDay.value,
      hashtag_count: parseInt(inputHashtags.value, 10) || 0,
      caption_length: parseInt(inputCaption.value, 10) || 0,
      posting_hour: parseInt(rangeHour.value, 10),
      has_trending_audio: parseInt(inputAudio.value, 10),
      is_original_content: parseInt(inputOriginal.value, 10),
    };

    // Client-side quick check for empty fields
    const missing = {};
    if (isNaN(payload.followers)) missing.followers = "Followers is required.";
    if (isNaN(payload.average_previous_reach)) missing.average_previous_reach = "Average reach is required.";
    if (isNaN(payload.average_engagement_rate)) missing.average_engagement_rate = "Engagement rate is required.";
    if (isNaN(payload.reel_duration)) missing.reel_duration = "Duration is required.";

    if (Object.keys(missing).length > 0) {
      displayFieldErrors(missing);
      return;
    }

    // Set UI to loading state
    stateEmpty.style.display = "none";
    stateSuccess.style.display = "none";
    stateError.style.display = "none";
    stateLoading.style.display = "flex";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      stateLoading.style.display = "none";

      if (response.status === 422) {
        displayFieldErrors(data.errors || { general: data.detail });
        return;
      }

      if (!response.ok) {
        stateError.style.display = "block";
        errorMsgText.textContent = data.detail || "Server error occurred during prediction.";
        return;
      }

      // Success Display
      stateSuccess.style.display = "flex";

      // 1. Point Estimate with Count-Up
      animateCountUp(data.point_estimate);

      // 2. Expected Range Text
      resultRangeText.textContent = data.range_display || `expected range ${formatK(data.range_low)} – ${formatK(data.range_high)}`;

      // 3. Comparison vs Historical Average
      const pct = data.pct_vs_average;
      const pctSign = pct >= 0 ? `+${pct}%` : `${pct}%`;
      benchmarkText.textContent = `about ${pctSign} vs. your historical average reach`;
      if (pct < 0) {
        resultBenchmark.classList.add("negative");
      } else {
        resultBenchmark.classList.remove("negative");
      }

      // 4. Update Range Bar Position
      // Anchor the scale between 0 and range_high * 1.3
      const maxRangeScale = Math.max(data.range_high * 1.25, payload.average_previous_reach * 1.25);
      const lowPct = Math.max(0, Math.min(95, (data.range_low / maxRangeScale) * 100));
      const highPct = Math.max(lowPct + 5, Math.min(100, (data.range_high / maxRangeScale) * 100));
      const pinPct = Math.max(0, Math.min(100, (data.point_estimate / maxRangeScale) * 100));

      rangeFill.style.left = `${lowPct}%`;
      rangeFill.style.width = `${highPct - lowPct}%`;
      rangePin.style.left = `${pinPct}%`;

      barLabelLow.textContent = `Low: ${formatK(data.range_low)}`;
      barLabelMid.textContent = `Est: ${formatK(data.point_estimate)}`;
      barLabelHigh.textContent = `High: ${formatK(data.range_high)}`;

      // 5. Render Improvement Suggestions Chips
      const suggestions = data.suggestions || [];
      if (suggestions.length === 0) {
        suggestionsContainer.innerHTML = "<p style='color:var(--text-secondary);font-size:0.8rem;'>No specific suggestions.</p>";
      } else {
        suggestionsContainer.innerHTML = suggestions.map(item => {
          const isStrong = item.status === "strong";
          const icon = isStrong
            ? `<svg class="chip-icon strong" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`
            : `<svg class="chip-icon warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
          const badgeText = isStrong ? "Strong" : "Consider";

          return `
            <div class="suggestion-chip ${item.status}">
              ${icon}
              <div class="chip-content">
                <h4>
                  ${item.title}
                  <span class="chip-tag ${item.status}">${badgeText}</span>
                </h4>
                <p>${item.detail}</p>
              </div>
            </div>
          `;
        }).join("");
      }

    } catch (err) {
      console.error("Prediction error:", err);
      stateLoading.style.display = "none";
      stateError.style.display = "block";
      errorMsgText.textContent = "Network error. Please ensure the FastAPI server is running.";
    }
  });

  // Run on startup
  loadMetadata();
  loadModelInfo();
});
