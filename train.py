
import pandas as pd
import numpy as np
import kagglehub
import os
import re
from sklearn.feature_extraction import DictVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle


path = kagglehub.dataset_download("xfkzujqjvx97n/football-datasets")
injuries_file = os.path.join(path, "player_injuries", "player_injuries.csv")
performance_file = os.path.join(path,"player_performances","player_performances.csv")
profile_file = os.path.join(path, "player_profiles","player_profiles.csv")

injuries = pd.read_csv(injuries_file)
performance = pd.read_csv(performance_file)
profile = pd.read_csv(profile_file)

injuries= injuries[injuries["days_missed"]<=365] #remove outliers


def _two_digit_to_year(two: int, pivot: int = 49) -> int:
    """
    Map a 2-digit year to a full year using a pivot:
    - two <= pivot -> 2000s (e.g., 25 -> 2025)
    - two > pivot  -> 1900s (e.g., 73 -> 1973)
    Default pivot=49 covers modern football datasets well.
    """
    return 2000 + two if two <= pivot else 1900 + two

def season_name_to_start_year(season_str: str) -> int:
    """
    Return the START YEAR for a season string.

    Supported formats:
      - 'YY/YY'       -> e.g., '22/23' -> 2022
      - 'YYYY/YY'     -> e.g., '1909/10' -> 1909 (rollover handled)
      - 'YY/YYYY'     -> e.g., '99/2000' -> 1999
      - 'YYYY/YYYY'   -> e.g., '2023/2024' -> 2023
      - 'YYYY'        -> e.g., '1984' -> 1984 (treated as single-year season)
      - 'YY'          -> e.g., '84' -> 1984 based on pivot

    Notes:
    - Century rollover: '99/00' -> start 1999, end 2000 (start_year=1999).
    - Whitespace is ignored.
    - Raises ValueError for unsupported formats.
    """
    if season_str is None:
        raise ValueError("season_str is None")

    s = str(season_str).strip()
    if not s:
        raise ValueError("Empty season string")

    if "/" in s:
        left, right = [p.strip() for p in s.split("/", 1)]

        # YYYY/YYYY
        if re.fullmatch(r"\d{4}", left) and re.fullmatch(r"\d{4}", right):
            return int(left)

        # YYYY/YY
        if re.fullmatch(r"\d{4}", left) and re.fullmatch(r"\d{2}", right):
            return int(left)

        # YY/YYYY
        if re.fullmatch(r"\d{2}", left) and re.fullmatch(r"\d{4}", right):
            end_year = int(right)
            end_two = end_year % 100
            start_two = int(left)
            # If start_two > end_two, the start is in the previous century
            start_century = (end_year - end_two) - (100 if start_two > end_two else 0)
            return start_century + start_two

        # YY/YY
        if re.fullmatch(r"\d{2}", left) and re.fullmatch(r"\d{2}", right):
            return _two_digit_to_year(int(left))

        raise ValueError(f"Unrecognized season format: {season_str}")

    # Single token (no slash)
    # YYYY
    if re.fullmatch(r"\d{4}", s):
        return int(s)

    # YY
    if re.fullmatch(r"\d{2}", s):
        return _two_digit_to_year(int(s))

    raise ValueError(f"Invalid season format: {season_str}")



def safe_start_year(s):
    try:
        return season_name_to_start_year(s)
    except Exception:
        return np.nan


performance["season_id"] = performance["season_name"].apply(safe_start_year)
performance = performance.dropna(subset=["season_id"])
performance["season_id"] = performance["season_id"].astype(int)


injuries["season_id"] = injuries["season_name"].apply(safe_start_year)
injuries = injuries.dropna(subset=["season_id"])
injuries["season_id"] = injuries["season_id"].astype(int)

# performance_2000_2025 
performance_2000_2025 = performance.loc[performance["season_id"].between(2000, 2025)].copy()
performance_2000_2025 = performance_2000_2025.drop(columns="season_name")

# Filter to 2000–2025
injuries_2000_2025 = injuries.loc[injuries["season_id"].between(2000, 2025)].copy()


injuries_2000_2025["from_date"] = pd.to_datetime(injuries_2000_2025["from_date"])
injuries_2000_2025["end_date"] = pd.to_datetime(injuries_2000_2025["end_date"])



injuries_2000_2025 = injuries_2000_2025.drop(columns="season_name")

injuries_2000_2025 = injuries_2000_2025[
    injuries_2000_2025["from_date"].between("2000-01-01", "2025-12-31")
]
profile["date_of_birth"] = pd.to_datetime(profile["date_of_birth"])


profile_df = profile[["player_id","date_of_birth", "position", "height"]]

profile_df['height'] = profile_df['height'].replace(0, np.nan)
performance_df = performance_2000_2025[["player_id", "season_id","nb_in_group","nb_on_pitch","goals","assists","subed_in","subed_out","yellow_cards", "goals_conceded"]]


performance_df = performance_df [performance_df["nb_on_pitch"]>0] #Keep only players who were actually exposed to injury risk

performance_df["goals"] = performance_df["goals"].fillna(0)

injuries_2000_2025["injury_reason_norm"] = (
    injuries_2000_2025["injury_reason"]
    .str.lower()
    .str.strip()
)


def map_injury_group(text):
    if pd.isna(text):
        return "unknown"

    # Muscle
    if any(k in text for k in [
        "muscle", "strain", "tear", "fiber", "hamstring", "calf",
        "adductor", "groin", "quadriceps", "fatigue", "stiffness",
        "sore"
    ]):
        return "muscle"

    # Ligament
    if any(k in text for k in [
        "ligament", "acl", "mcl", "cruciate", "syndesm",
        "collateral", "sprain"
    ]):
        return "ligament"

    # Tendon
    if any(k in text for k in [
        "tendon", "achilles", "patellar", "tendinitis", "tendinopathy", "heel"
    ]):
        return "tendon"

    # Joint / cartilage
    if any(k in text for k in [
        "meniscus", "cartilage", "joint", "capsular", "patella",
        "kneecap", "hip", "knee"
    ]):
        return "joint_cartilage"

    # Bone / fracture
    if any(k in text for k in [
        "fracture", "broken", "crack", "stress", "bone",
        "splinter", "fissure", "toe", "shin"
    ]):
        return "bone_fracture"

    # Contusion / bruise / impact
    if any(k in text for k in [
        "bruise", "contusion", "knock", "impact", "dead leg"
    ]):
        return "contusion_bruise"

    # Spine / back
    if any(k in text for k in [
        "back", "spine", "lumbar", "cervical", "disc",
        "vertebra"
    ]):
        return "spine_back"

    # Surgery
    if any(k in text for k in [
        "surgery", "arthroscopy", "operation", "post"
    ]):
        return "surgery_postop"

    # Illness / infection
    if any(k in text for k in [
        "flu", "virus", "infection", "covid", "corona",
        "pneumonia", "ill", "fever", "influenza"
    ]):
        return "illness_infection"

    # Medical / other
    if any(k in text for k in [
        "heart", "lung", "kidney", "cancer", "tumor",
        "blood", "stroke", "depression", "appendectomy", "eye", "head"
    ]):
        return "other_medical"

    return "unknown"

injuries_2000_2025["injury_group"] = injuries_2000_2025["injury_reason_norm"].apply(map_injury_group)

df = performance_df.copy()

df = df.merge(profile_df, on='player_id', how='left')

df["age"] = df["season_id"] - df["date_of_birth"].dt.year

df = df.drop(columns=["date_of_birth"])

position_groups = {
    'Attack - Centre-Forward': 'Attack',
    'Attack - Left Winger': 'Attack',
    'Attack - Right Winger': 'Attack',
    'Attack - Second Striker': 'Attack',
    'Attack':"Attack",
    'Midfield': 'Midfield',
    'Midfield - Attacking Midfield': 'Midfield',
    'Midfield - Central Midfield': 'Midfield',
    'Midfield - Defensive Midfield': 'Midfield',
    'Midfield - Left Midfield': 'Midfield',
    'Midfield - Right Midfield': 'Midfield',
    'Defender': 'Defense',
    'Defender - Centre-Back': 'Defense',
    'Defender - Left-Back': 'Defense',
    'Defender - Right-Back': 'Defense',
    'Defender - Sweeper': 'Defense',
    'Goalkeeper': 'Goalkeeper'
}

df['position_group'] = df['position'].map(position_groups)


df= df.drop(columns="position")

agg_dict = {
    "nb_in_group": "sum",
    "nb_on_pitch":"sum",
    "goals": "sum",
    "assists": "sum",
    "subed_in":"sum",
    "subed_out":"sum",
    'yellow_cards': "sum",
    "goals_conceded":"sum",
    "position_group": "first",
    "height":"first",
    "age":"first"
}

df = df.groupby(['player_id', 'season_id']).agg(agg_dict).reset_index()

inj_group_agg = (
    injuries_2000_2025
    .groupby(["player_id", "season_id", "injury_group"])
    .agg(
        injuries_count=("injury_group", "count"),
        total_days_missed=("days_missed", "sum")
    )
    .reset_index()
)


inj_group_agg = inj_group_agg[inj_group_agg["total_days_missed"]<=365]

inj_group_wide = (
    inj_group_agg
    .pivot_table(
        index=["player_id", "season_id"],
        columns="injury_group",
        values="total_days_missed",
        fill_value=0
    )
    .reset_index()
)

inj_group_wide.columns = [
    f"days_missed_{c}" if c not in ["player_id", "season_id"] else c
    for c in inj_group_wide.columns
]


df = df.merge(
    inj_group_wide,
    on=["player_id", "season_id"],
    how="left"
)


# Fill missing heights (NaN) with median per position group
df['height'] = df.groupby('position_group')['height'].transform(lambda x: x.fillna(x.median()))

df = df.dropna(subset=["position_group", "height"])

df['age'] = df['age'].fillna(df['age'].median())

days_missed_cols = [c for c in df.columns if c.startswith("days_missed_")]


df['total_days_missed'] = df[days_missed_cols].sum(axis=1)
df[days_missed_cols] = df[days_missed_cols].fillna(0)
df['total_days_missed'] = df['total_days_missed'].fillna(0)

df = df[df["total_days_missed"]<=365]


# Create a set of all existing (player_id, season_id) pairs
player_season_set = set(zip(df['player_id'], df['season_id']))


df['has_next_season'] = df.apply(
    lambda row: (row['player_id'], row['season_id'] + 1) in player_season_set,
    axis=1
)


df['next_season_days'] = df.groupby('player_id')['total_days_missed'].shift(-1)

df_clean = df[df['has_next_season']].copy()
df_clean.drop(columns='has_next_season', inplace=True)


df_clean["next_season_days"] = df_clean["next_season_days"].fillna(0)

df_clean['log_next_season_days'] = np.log1p(df_clean['next_season_days'])

df_clean = df_clean.drop(columns=["player_id", "next_season_days"])

target = "log_next_season_days"



numeric_cols = [
    'nb_on_pitch',
    'subed_in',
    'subed_out',
    'goals',
    'assists',
    'yellow_cards',
    'goals_conceded',
    'height',
    'age',
    'days_missed_bone_fracture',
    'days_missed_contusion_bruise',
    'days_missed_illness_infection',
    'days_missed_joint_cartilage',
    'days_missed_ligament',
    'days_missed_muscle',
    'days_missed_other_medical',
    'days_missed_spine_back',
    'days_missed_surgery_postop',
    'days_missed_tendon',
    'days_missed_unknown',
    'total_days_missed'
]

categorical_cols = ["position_group"]


feature_cols = numeric_cols + categorical_cols



train_mask = df_clean["season_id"] <= 2018
val_mask   = (df_clean["season_id"] >= 2019) & (df_clean["season_id"] <= 2021)
test_mask  = df_clean["season_id"] >= 2022



df_train = df_clean[train_mask]
df_val = df_clean[val_mask]
df_test = df_clean[test_mask]

y_train = df_train[target]
y_val = df_val[target]
y_test = df_test[target]



del df_train[target]
del df_val[target]
del df_test[target]



dv = DictVectorizer(sparse = False)



train_dict = df_train[feature_cols].to_dict(orient = "records")
X_train = dv.fit_transform(train_dict)

val_dict = df_val[feature_cols].to_dict(orient = "records")
X_val = dv.transform(val_dict)

test_dict = df_test[feature_cols].to_dict(orient = "records")
X_test = dv.transform(test_dict)



# X already numpy arrays
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

# Targets
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)  # logged
y_val_tensor   = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

# Binary targets for probability of any days
y_train_bin = torch.tensor((np.expm1(y_train.to_numpy()) > 0).astype(int), dtype=torch.float32).unsqueeze(1)
y_val_bin   = torch.tensor((np.expm1(y_val.to_numpy()) > 0).astype(int), dtype=torch.float32).unsqueeze(1)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor, y_train_bin), batch_size=1024, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor, y_val_bin), batch_size=1024)

df_full_train = pd.concat([df_train, df_val], ignore_index=True)
y_full_train = pd.concat([y_train, y_val], ignore_index=True)


# Convert to dict for DictVectorizer
full_train_dict = df_full_train[feature_cols].to_dict(orient="records")
X_full_train = dv.transform(full_train_dict)

class ZeroInflatedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # head 1: probability of any days
        self.prob_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # head 2: regression (log-days if >0)
        self.reg_head = nn.Linear(128, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        prob = self.prob_head(shared_out)
        log_days_pred = self.reg_head(shared_out)
        # expected log-days = prob * log-days
        expected_log_days = prob * log_days_pred
        return prob, log_days_pred, expected_log_days


X_train_tensor = torch.tensor(X_full_train, dtype=torch.float32)

y_train_log = torch.tensor(
    y_full_train.to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

y_train_bin = torch.tensor(
    (np.expm1(y_full_train.to_numpy()) > 0).astype(np.float32)
).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_test_log = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

y_test_bin = torch.tensor(
    (np.expm1(y_test.to_numpy()) > 0).astype(np.float32)
).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_log, y_train_bin),
    batch_size=1024,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_log, y_test_bin),
    batch_size=1024
)



lr = 0.001
reg_weight = 1.0
epochs = 10
model = ZeroInflatedNN(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Binary loss + regression loss
bce_loss = nn.BCELoss()   # for probability
mae_loss = nn.L1Loss()    # for log-days


for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch_log, y_batch_bin in train_loader:
        optimizer.zero_grad()

        prob, log_pred, expected_log = model(X_batch)

        # Classification loss
        loss_prob = bce_loss(prob, y_batch_bin)

        # Regression loss only for positive cases
        mask = (y_batch_bin > 0).squeeze(1)

        if mask.any():
            loss_reg = mae_loss(log_pred[mask], y_batch_log[mask])
        else:
            loss_reg = torch.zeros(1, device=X_batch.device)

        loss = loss_prob + reg_weight * loss_reg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


FILENAME = "nn_pipeline.pkl"

# Save
with open(FILENAME, "wb") as f:
    pickle.dump({
        "vectorizer": dv,
        "model_state": model.state_dict(),
        "input_dim": X_full_train.shape[1],
        "hidden_dim": 128,
        "reg_weight": 1.0,
        "feature_order": feature_cols
    }, f)

print("Full NN pipeline saved successfully!")



