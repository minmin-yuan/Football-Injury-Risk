import pickle
import torch
import numpy as np
import torch.nn as nn

# =========================
# Model definition (MUST MATCH TRAINING)
# =========================
class ZeroInflatedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.reg_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        prob = self.prob_head(h)
        log_days = self.reg_head(h)
        expected_log_days = prob * log_days
        return prob, log_days, expected_log_days


# =========================
# Load artifacts
# =========================
FILENAME = "nn_pipeline.pkl"

with open(FILENAME, "rb") as f:
    artifacts = pickle.load(f)

dv = artifacts["vectorizer"]
feature_order = artifacts["feature_order"]
input_dim = artifacts["input_dim"]
state_dict = artifacts["model_state"]

# Rebuild model
model = ZeroInflatedNN(input_dim)
model.load_state_dict(state_dict)
model.eval()

print("✅ NN model loaded successfully")


# =========================
# Risk label helper
# =========================
def risk_label(days):
    if days <= 1:
        return "No risk"
    elif days <= 7:
        return "Low"
    elif days <= 21:
        return "Medium"
    elif days <= 60:
        return "High"
    else:
        return "Severe"


# =========================
# Prediction function
# =========================
def predict_player(player_dict):
    # Ensure all expected features exist
    input_dict = {f: player_dict.get(f, 0) for f in feature_order}

    X_vec = dv.transform([input_dict])
    X_tensor = torch.tensor(X_vec, dtype=torch.float32)

    with torch.no_grad():
        _, _, expected_log_days = model(X_tensor)

    predicted_days = float(np.expm1(expected_log_days.numpy().flatten())[0])
    predicted_risk = risk_label(predicted_days)

    return predicted_days, predicted_risk


# =========================
# Example
# =========================
sample_player = {
    'nb_on_pitch': 8,
    'subed_in': 5,
    'subed_out': 3,
    'goals': 7,
    'assists': 4,
    'yellow_cards': 2,
    'goals_conceded': 12,
    'height': 192,
    'age': 24,
    'days_missed_bone_fracture': 0,
    'days_missed_contusion_bruise': 0,
    'days_missed_illness_infection': 0,
    'days_missed_joint_cartilage': 0,
    'days_missed_ligament': 20,
    'days_missed_muscle': 0,
    'days_missed_other_medical': 0,
    'days_missed_spine_back': 10,
    'days_missed_surgery_postop': 3,
    'days_missed_tendon': 0,
    'days_missed_unknown': 0,
    'total_days_missed': 33,
    'position_group': 'Striker'
}

days, risk = predict_player(sample_player)
print(f"Predicted days missed: {days:.2f}")
print(f"Risk level: {risk}")
