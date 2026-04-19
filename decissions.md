# Cardiac Drift App — Key Decisions

## What We Are Building
A system that predicts cardiac drift on the bike 10 minutes before it happens and alerts the user in real time. Cardiac drift = HR rising steadily while power output remains stable. The goal is prediction (leading indicator), not detection (drift already visible).

## My Profile and skills
- Product Owner background, learning to code
- Basic Python knowledge, picking up pandas
- Good SQL knowledge
- Some Javascript and web development knowledge
- No prior ML or mobile app experience
- Never worked with BLE or binary files
- Strong product thinking and user story background
- Good with architectural and systems thinking, in my current job I'm involved in many architectural decissions of the web applications we build

## Data Source
- Raw .fit files exported manually from Garmin Connect
- .fit files are binary (not XML/TCX) and require the `fitparse` Python library to decode
- No Garmin Connect API — official API requires business approval, unofficial wrappers are fragile
- TCX/GPX rejected: power data is bolted on via extensions, not native, and data richness is lower than .fit

## Core Technical Decisions

### Feature Engineering
- All features must be relative/delta — never absolute HR or power values
- Absolute values are fitness-dependent and break the model as the athlete improves
- Baseline computed from minutes 5–15 of each ride (per-ride recalibration)
- All features expressed as deviation from that ride's own baseline
- Most important feature: HR/power ratio delta from baseline

### Key Features to Engineer
- HR_delta (current HR minus baseline HR)
- HR_slope over 5, 10, 15 minute rolling windows
- Power coefficient of variation (CV) over rolling windows
- HR/power ratio delta from baseline
- Time in zone (cumulative fatigue proxy)

### Labeling Strategy
- Drift onset label applied in hindsight to historical .fit files
- Rule: HR delta > 6–8 bpm over 15 min AND power CV < 5% over same window
- Prediction target: will drift_onset be True in the next 10 minutes?
- This creates a binary classification problem per second of ride data

### Model Choice
- XGBoost classifier (not deep learning — insufficient data, harder to debug)
- Walk-forward cross-validation only — random splits are invalid for time-series data
- Validate across fitness eras (early rides vs later rides) to prove generalization
- SHAP analysis to understand which features drive predictions

### Handling Fitness Progression
- Model must never see absolute HR or power — only deltas
- Stratify training data by fitness era when validating
- Retrain model every 2–3 months as fitness improves significantly
- Per-ride baseline recalibration at inference time handles day-to-day variation automatically

### Architecture
- ML inference runs on the phone, not on the Garmin device
- Garmin device is the sensor source and alert display target only
- Garmin Connect IQ has insufficient compute for meaningful ML on-device
- Phone app reads BLE data from chest strap and power meter directly
- Inference runs every 30 seconds on a 15-minute circular buffer of sensor data
- Alert triggers when drift probability exceeds 75%

### Wahoo
- Wahoo ELEMNT has no public app SDK
- Only real-time path for Wahoo is BLE phone interception (same as Garmin)

## Data Requirements
- Minimum 30–50 labeled rides for a functional XGBoost model
- 100–200 rides for production quality
- HR must be from a chest strap — optical HR is too noisy for training labels
- Power must be from a direct power meter — estimated power is insufficient
- Temperature and humidity per ride are important drift triggers

## 10-Week Roadmap

| Week | Focus | Key Concept |
|------|-------|-------------|
| 1 | Parse .fit files, explore data, first chart | Binary files, fitparse, DataFrames |
| 2 | Data cleaning, resampling, missing values | Data quality, forward-fill, ride filtering |
| 3 | Feature engineering — rolling windows, deltas | Relative features, rolling statistics |
| 4 | Labeling drift onset from historical rides | Supervised labeling, prediction horizon |
| 5 | Build full training dataset, class imbalance | Combining rides, imbalanced classification |
| 6 | XGBoost training, walk-forward validation, SHAP | ML workflow, temporal validation, interpretability |
| 7 | Real-time inference pipeline, circular buffer | Training vs inference, model serving |
| 8 | Phone app + BLE integration | BLE, GATT profiles, hardware-software bridge |
| 9 | Connect inference to alerts, threshold tuning | End-to-end integration, UX of alerts |
| 10 | Real ride testing, debugging, iteration | Production gap, debugging live systems |

## Learning Approach
- Claude delivers code with explanation, I attempt tasks first then review
- After each task: I must be able to read it line by line, modify it, and roughly rebuild it
- One concept per session, one session per week task
- Learning log updated at end of each week and uploaded to project knowledge
- Claude quizzes understanding before giving answers


## Week 1 Findings — Data Exploration

### .fit File Structure
- Messages stream in a repeating pattern: record → hrv → gps_metadata
- During auto-pauses: record messages stop but hrv and gps_metadata continue
- Event messages mark pause boundaries (timer: stop_all / start) and are 
  essential for identifying gaps in the data

### Key Message Types
- `record`: one per second of active riding — primary data source
- `event`: 49 total — marks start/stop/pause boundaries
- `hrv`: inter-beat intervals in seconds — richer than averaged HR in record
- All others are metadata (session, device_info, etc.)

### Data Quality Issues Confirmed
- Timestamp gaps: auto-pauses create discontinuities — confirmed via 
  event messages, not missing records
- Sensor dropout: 8 records have null heart_rate despite active riding — 
  distinct from pause gaps, needs interpolation not exclusion
- Inconsistent fields: not all records carry the same columns — pandas 
  fills missing fields with NaN automatically

### DataFrame Baseline
- 6708 record messages → 6708 rows, 14 columns
- timestamp correctly parsed as datetime64[us] by pandas automatically
- All numeric columns as float64 except temperature (int64)

## Data Quality Findings (Week 2)
- Missing power (22 records): always the first record after a pause resume — 
  power meter takes ~1 second to reconnect via ANT+. Treated with ffill(limit=1).
- Missing HR (8 records): scattered sensor dropouts, some in clusters up to 3 
  consecutive seconds. Treated with ffill(limit=3).
- Pause gaps: detected via timestamp gaps > 3 seconds. Last record before each 
  gap marked is_pause=True.

## Current Status
Week 2 complete. Clean DataFrame produced with:
- Timestamp as index, verified monotonic increasing
- is_pause column marking gap boundaries
- HR and power forward-filled with conservative limits
Week 3 next: feature engineering — rolling windows, HR delta, HR/power ratio delta.

## Findings and learnings (Week 3 - Baseline computation and data manipulation)
The goal of the week was to perform calculations based on the data. 

### Baseline calculation
Calculate the power and hr baseline, meaning the mean hr and mean power for a certain interval. In this case was minutes 5 to 15. 

For that I created a function which extracts the baseline_start and baseline_end based on a time interval. We learned also that pandas would give us the next timestamp when extracting a certain value in case the timestamp doesn's exist. 

### Delta features

In this task we needed to calculate and add a new column to the dataFrame which includes the delta between the hr and power value and the baseline one. 

### Rolling Slope and mean

We needed to calculate also the slope and mean of certain rolling intervals. We decided to go with 5 minutes intervals (or rather said 300 records intervals), for that the rolling() function from pandas is used. 

In the case of the rolling mean was easy as it was to apply rolling(300).mean() to a column. 

In the case of the slope was more complicated as we needed to import numpy as np and use the polyfit function. For that we created this function

```
def get_slope (window):
    return np.polyfit(np.arange(len(window)),window,1)[0]
```

A window needs to be passed as a parameter, and the window is the rolling window. So the arange window give indexes 0 to 299 if you pass a 300 window and uses them as x axis, the values of the rolling window (heart_rate or power) as y axis and 1 is the polynomial. [0] is meant because we want just the first value of the array returned by the function, which is the slope itself.

This is the part of the week I struggled most with. And the learning system was failing to explain the things, for the next sessions we need to dig deeper when new libraries or functions are used, and they need to be explained properly. Otherwise I don't learn the topics properly. 

### HR / Power Ratio Delta

The last task was to calculate the HR / Power ratio and it's delta with the baseline ratio. 

This was kind of an easy task, the only thing to consider where the points in the data with power values of 0 (where I'm not pedaling) which produce a inf value. In this case I used the df.replace() funtion with [np.inf, -np.inf],np.nan . And Claude recommended me to remove the inplace = True , even if I proved that there was no None being returned but the actual values. 