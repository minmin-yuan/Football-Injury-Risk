import torch
import torch.nn as nn
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# -----------------------------
# 1️⃣ Define the NN class
# -----------------------------
class ZeroInflatedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        prob = self.prob_head(shared_out)
        log_pred = self.reg_head(shared_out)
        expected_log = prob * log_pred
        return prob, log_pred, expected_log

# -----------------------------
# 2️⃣ Load the trained pipeline
# -----------------------------
FILENAME = "nn_pipeline.pkl"  # path to your saved pipeline

with open(FILENAME, "rb") as f:
    artifacts = pickle.load(f)

dv = artifacts["vectorizer"]
feature_order = artifacts["feature_order"]
input_dim = artifacts["input_dim"]
hidden_dim = artifacts.get("hidden_dim", 128)
reg_weight = artifacts.get("reg_weight", 1.0)

model = ZeroInflatedNN(input_dim=input_dim, hidden_dim=hidden_dim)
model.load_state_dict(artifacts["model_state"])
model.eval()

# -----------------------------
# 3️⃣ Risk label function
# -----------------------------
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

# -----------------------------
# 4️⃣ Prediction helper
# -----------------------------
def predict_player(player_dict):
    # Ensure all features are present, default 0
    input_dict = {f: player_dict.get(f, 0) for f in feature_order}
    
    # Vectorize
    X_vec = dv.transform([input_dict])
    X_tensor = torch.tensor(X_vec, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        _, _, expected_log_days = model(X_tensor)

    predicted_days = np.expm1(expected_log_days.numpy().flatten())[0]
    predicted_risk = risk_label(predicted_days)
    return predicted_days, predicted_risk

# -----------------------------
# 5️⃣ Flask app
# -----------------------------
app = Flask(__name__)

# UI route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        # Collect player input from form
        player_data = {}
        for f in feature_order:
            val = request.form.get(f, 0)
            # Keep categorical as string, numeric as float
            try:
                val = float(val)
            except:
                val = str(val)
            player_data[f] = val

        days, risk = predict_player(player_data)
        prediction_text = f"Predicted days missed: {days:.1f}, Risk level: {risk}"

    return render_template("index.html", prediction=prediction_text)

# JSON API route
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    days, risk = predict_player(data)
    return jsonify({
        "predicted_days_missed": float(f"{days:.1f}"),
        "risk_level": risk
    })

# -----------------------------
# 6️⃣ Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=9696)
