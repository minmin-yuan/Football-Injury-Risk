# Football-Injury-Risk
---

## **Problem Description**
Football players are exposed to continuous physical stress due to intense training loads, high-speed movements, frequent accelerations/decelerations, and physical contact during matches. These factors contribute to a high incidence of injuries, which can significantly affect player performance, team outcomes, and club finances.

This project focuses on predicting the risk of injury for football players using historical performance metrics, injury history and demographic data. By identifying early warning signs of elevated injury risk, coaches, medical staff, and analysts can make proactive decisions regarding training intensity, player rotation, and personalized recovery plans. 



## **Dataset**
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
  - Regression: Predict days missed next season due to injury.
  - → Target: days_missed missed next season.

## **🧹 Data Processing**
- **Season Normalization**
Season strings come in inconsistent formats (YY/YY, YYYY/YY, YY/YYYY).
A custom parser converts all formats to a **season_id (start year)**.
- **Injury Data**
  - Converted dates to datetime
  - Removed dates before 2000
  - Outlier removal: days_missed > 365
  - Normalized injury_reason and mapped to **11 injury groups**:

    - muscle, ligament, tendon, joint_cartilage, bone_fracture
    - contusion_bruise, spine_back, surgery_postop
    - illness_infection, other_medical, unknown


  - Pivoted to create days_missed_ per injury type*

- **Performance Data**

  - Filtered to players who actually played (nb_on_pitch > 0)
  - Aggregated per player-season (to handle players who have multiple records per season due to transfers)
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
## **Feature Engineering**
**Numerical Features**
- Performance: nb_on_pitch, nb_in_group, subed_in/out, goals, assists, yellow_cards, goals_conceded
- Profile: height, age
- Injury history: all days_missed_* columns + total_days_missed

**Categorical Features**

- position_group → one‑hot encoded using DictVectorizer
  <img width="988" height="731" alt="output_91_0" src="https://github.com/user-attachments/assets/260baf34-96a5-423e-9ef1-4549d470d7fb" />
There is no obvious separation between the groups.
- All four positions have: Similar spread of values, clustering around the same log-days and presence of extreme values (outliers). There appears to be a few points slightly higher than the others for Goalkeepers.

## **Exploratory Data Analysis**
Findings:
<img width="1650" height="1187" alt="output_84_0" src="https://github.com/user-attachments/assets/897a63f9-8e90-494c-a4a3-d4fc579ebc30" />

- Past injuries are the strongest predictors of future injuries
- Muscle & unknown injuries are most common
- Some injury types correlate moderately with future risk
- Performance stats have limited predictive power individually
 <img width="636" height="391" alt="output_85_19" src="https://github.com/user-attachments/assets/a827b4a5-7bb9-4112-ab70-191d29256869" />
 <img width="636" height="391" alt="output_85_18" src="https://github.com/user-attachments/assets/2fc5ba57-fc39-4713-a2f5-1e1ddad3499f" />

  **Multicollinearity**
- Removed nb_in_group due to high correlation with nb_on_pitch (≈0.9)


## **📑 Train/Validation/Test Split**
Temporal split:

- Train: 2000–2018
- Validation: 2019–2021
- Test: 2022–2024

This prevents data leakage across seasons.

## **Model Training**
Model Selection: We use ridge regression for interpretability, LightGBM for peak predictive performance on tabular data, and a two-stage neural network to explicitly model latent injury vulnerability and exposure-driven risk.
- Ridge Regression provides a strong and interpretable baseline to understand linear relationships and feature effects.
- LightGBM represents a high-performance tree-based model that captures non-linear patterns common in sports performance data.
 -Neural Network (Zero-Inflated) is specifically designed to handle the large number of zero-injury cases and model both injury risk and severity.
  
## **🤖 Models Evaluated**
Evaluation Criterion: This injury-risk prediction task involves zero-inflated, highly skewed data, where most players miss 0 days, while a small subset miss many days. Standard regression metrics alone are insufficient, so multiple complementary metrics were used.

- Mean Absolute Error (MAE)
Chosen for its interpretability in days missed and robustness to outliers compared to MSE. It reflects typical prediction error without being dominated by extreme injuries.
- Spearman Rank Correlation
Measures whether the model correctly ranks players by injury risk, which is crucial for screening and prioritization even if exact day counts are imperfect.
- Top-10% Capture Rate
Evaluates the model’s ability to identify high-risk players, aligning with real-world medical and squad-management decisions where resources focus on the most vulnerable players.

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


## ** Model Comparision**
| Model | MAE ↓   | Spearman ↑ | Top-10% capture ↑ |
| ----- | ------- | ---------- | ----------------- |
| LGBM  | 14.9988 | 0.3370     | 0.5703            |
| Ridge | 13.9409 | 0.3159     | 0.5112            |
| NN    | 13.7139| 0.3325     | 0.5625          |

-Neural Network (NN) achieved the lowest MAE (13.71), indicating the most accurate prediction of injury days on average.

-LGBM showed the best ranking ability (highest Spearman and Top-10% capture), making it strong for identifying high-risk players.

-Ridge performed competitively on MAE but lagged in ranking metrics, reflecting its linear limitation.

## **Hyperparameter tuning**
We chose LGBM and Neural Network to do hyperparamter tuning as they have overal better performance.
| Model                 | MAE ↓     | Spearman ↑       | Top-10% capture ↑ |
| --------------------- | --------- | ---------------- | ----------------- |
| **LGBM (full train)** | **↓13.19** | **↓ ~0.28–0.30** | **↓ ~0.43**      |
| **NN (full train)**   | **13.77** | **0.331**        | **0.552**         |

When trained on the same full data, the NN generalizes better than LGBM. 

**Final Selecion: Neural Network**
- Delivers the best overall error performance (MAE) while remaining competitive in ranking metrics

- Models non-linear interactions between workload, age, position, and injury history

- Zero-inflated design aligns with injury reality (many zero-injury seasons)

- Offers better predictive stability across risk levels rather than only extreme cases

## ** Inference Pipeline**
After training on full training data(combining training and validation set), the exported model (nn_pipeline.pkl) includes:

- Feature vectorizer
- Neural network weights
- Feature order rules

Outputs:

- **Predicted days missed next season**
- **Risk category:**

  - 0–1 → No Risk
  - ≤7 → Low
  - ≤21 → Medium
  - ≤60 → High
  - 60 → Severe






## **📦 Model Deployment**
- **Exported Artifacts**

  - full_dataset.csv — cleaned dataset
  - nn_pipeline.pkl — saved final model

- **
### Local Deployment 
#### 1. Prepare environment

**Option A — Conda**

```bash
conda create -n football python=3.11 -y
conda activate football
pip install -r requirements.txt
```
or
```bash
conda env create -f environment.yml
conda activate football-injury-risk
```
**Option B — pip + venv**
create and activate a venv
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

Windows (PowerShell)
```bash
.venv\Scripts\activate
```

install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Prepare the trained model
To train and save the model locally:
```bash
python train.py
```
#### 3. Run the service
- **Run the Flask service locally**
```bash
python app.py
```
Defalut port:9696\
Access locally: http://127.0.0.1:9696/predict

- **Run the service (production-like) with Gunicorn**
```bash
gunicorn --bind 0.0.0.0:9696 predict:app --workers 4
```
#### 4.Test the API:**
```
python predict-test.py
```
or browser
```
http://127.0.0.1:9696/
```
- **Output: injured days prediction and risk level**
<img width="1542" height="580" alt="Snip20260119_1" src="https://github.com/user-attachments/assets/edad8416-8e2e-4e89-845f-5efed5338a99" />


  
### Containerized Deployment (Docker)
#### Build image & Run container
Dockerfile: Provided in the repository.
```
docker build -t football-risk-app .
```
```
docker run -p 9696:9696 football-risk-app
```
- screenshots of docker running and test
#### Test Docker:**
```
python predict-test.py
```
or browser
```
http:/

<img width="1712" height="714" alt="Snip20260119_2" src="https://github.com/user-attachments/assets/d6124fc7-3951-4c57-b2b3-36f52b4b6544" />
<img width="1110" height="90" alt="Snip20260119_3" src="https://github.com/user-attachments/assets/cea304bd-b505-40af-a98e-80354c91a4cd" />

### Cloud Deployment (Render)
- The service can be deployed to the cloud for remote access:
- Instructions for Render deployment:
  1. push project to github repo
  2. Sign up(use github account is the easist) and log in to Render
  3. Once logged in, click New → Web Service
  4. Connect to github repo and select football-injury-risk repo
  5. Click Create Web Service, Render will automatically build docker image, install dependencies from requirements.txt and start
  the Gunicorn server.
  6. After a minute or two, you’ll get a public URL like: https://netflix-churn-prediction.onrender.com
- Service URL: https://football-injury-risk.onrender.com/ (already deployed and can be accessd)
- screenshots and record of Render deployment
  
  

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

