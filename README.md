# Football-Injury-Risk
---

## **📌 Overview**
Football players are exposed to continuous physical stress due to intense training loads, high-speed movements, frequent accelerations/decelerations, and physical contact during matches. These factors contribute to a high incidence of injuries, which can significantly affect player performance, team outcomes, and club finances.

This project focuses on predicting the risk of injury for football players using historical performance metrics, injury history and demographic data. By identifying early warning signs of elevated injury risk, coaches, medical staff, and analysts can make proactive decisions regarding training intensity, player rotation, and personalized recovery plans. 



## **📂 Dataset**
- **Source:** Kaggle – football-datasets: https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets/data
- **Datasets in the source:**
   - Player injury histories (dates, injury_reason, days_missed, games_missed)
   - Player performance stats (minutes, goals, cards, subbed in/out)
   - Player profiles (age, height, position)
   - Market values, transfer histories, teammate relationships
   - Team and competition context for players
- **Tables used based on predicition goal:**
  - player_injuries
  - player_performances
  - player_profiles
    
Seasons filtered to 2000–2025.
- **Predicition Probelm:**
  - Regression: Predict days missed or games lost due to injury.
  - → Target: days_missed missed next season.

## **🧹 Data Processing**
- **Season Normalization**
Season strings come in inconsistent formats (YY/YY, YYYY/YY, YY/YYYY).
A custom parser converts all formats to a **season_id (start year)**.
- **Injury Data**
  - Converted dates to datetime
  - Removed invalid dates (before 2000)
  - Outlier removal: days_missed > 365
  - Normalized injury_reason and mapped to **11 injury groups**:

    - muscle, ligament, tendon, joint_cartilage, bone_fracture
    - contusion_bruise, spine_back, surgery_postop
    - illness_infection, other_medical, unknown


  - Pivoted to create days_missed_ per injury type*

- **Performance Data**

  - Filtered to players who actually played (nb_on_pitch > 0)
  - Aggregated per player-season (to handle transfers)
  - Filled missing values (e.g., goals = 0)

- **Profile Data**

  - Kept: date_of_birth, position, height
  - height=0 → treated as missing → imputed with median per position_group
  - Created simplified position_group (Attack, Midfield, Defense, GK)

- **Final Merge**
Combined injury, profile, and performance tables, then:

  - Added age = season_id - birth_year
  - Removed rows missing height or position group


## **🎯 Target Definition**
For each player-season:

  - next_season_days = total_days_missed in next season
  - Filter to rows where the next season exists
  - Log-transform:

```python
log_next_season_days = log1p(next_season_days)
```
## **🧠 Feature Engineering**
**Numerical Features**
- Performance: nb_on_pitch, nb_in_group, subed_in/out, goals, assists, yellow_cards, goals_conceded
- Profile: height, age
- Injury history: all days_missed_* columns + total_days_missed

**Categorical Features**

- position_group → one‑hot encoded using DictVectorizer

**Multicollinearity**

- Removed nb_in_group due to high correlation with nb_on_pitch (≈0.9)


## **🔎 Exploratory Data Analysis**
Findings:
<img width="1650" height="1187" alt="output_84_0" src="https://github.com/user-attachments/assets/897a63f9-8e90-494c-a4a3-d4fc579ebc30" />
<img width="988" height="731" alt="output_91_0" src="https://github.com/user-attachments/assets/260baf34-96a5-423e-9ef1-4549d470d7fb" />

- Past injuries are the strongest predictors of future injuries
  
- Muscle & unknown injuries are most common
- Some injury types correlate moderately with future risk
- Performance stats have limited predictive power individually


## **📑 Train/Validation/Test Split**
Temporal split:

- Train: 2000–2018
- Validation: 2019–2021
- Test: 2022–2024

This prevents data leakage across seasons.

## **🤖 Models Evaluated**
**1. Ridge Regression**

- MAE: 13.94
- Spearman: 0.316
- Top‑10% injury capture: 0.511

**2. LightGBM**
Best tuned parameters:

- num_leaves=50, max_depth=6, n_estimators=200
- learning_rate=0.05, subsample=0.8, colsample_bytree=0.8

Validation:

- MAE: ~15.0
- Spearman: 0.337 (best ranking ability)
- Top‑10% capture: 0.57

**3. Neural Network (Zero-Inflated Architecture)**
Architecture:

- Shared MLP (256 → 128 → Dropout)
- Two heads:

    - Probability (sigmoid)
    - Regression (linear)



Training:

- Adam (lr=0.001), BCE + MAE loss, 10 epochs

Validation:

- MAE: 13.71 (best)
- Spearman: 0.333
- Top‑10% capture: 0.563


## **🏆 Best Model**
Depends on business goal:
- For lowest MAE (best numerical prediction):
  - 👉 Neural Network (Zero‑Inflated)
- For ranking players by risk (identifying highest-risk players):
  - 👉 LightGBM

## **🔮 Inference Pipeline**
The exported model (nn_pipeline.pkl) includes:

- Feature vectorizer
- Neural network weights
- Feature order rules
- Preprocessing (imputations, one-hot encoding)

Outputs:

- **Predicted days missed next season**
- **Risk category:**

  - 0–1 → No Risk
  - ≤7 → Low
  - ≤21 → Medium
  - ≤60 → High
  - 60 → Severe






## **📦 Exported Artifacts**

- full_dataset.csv — cleaned dataset
- nn_pipeline.pkl — saved final model


## **📘 Conclusion**
This project builds a robust injury prediction pipeline with:

- High-quality feature engineering
- Temporal modeling & evaluation
- Multiple model families (linear, tree-based, deep learning)
- Strong predictive performance on real football data
- API deployment
- Player risk dashboards
- Club analytics workflows

## ** Limitations**
In injury research, true workload usually refers to:

- GPS distance
- High-speed running
- Accelerations / decelerations
- Training load (session RPE × duration)
- Acute:chronic workload ratios
The kaggle datasets do not have these types of information. In addition, we use seasonal aggregation  data because we don't have match level details and information about environmental information such as stadium grass status and weather. And we only club performance data because of not having seasonal national team data.

