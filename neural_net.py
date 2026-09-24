import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

SEED = 42

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv("sales_data.csv")

df_clean = df[
    df["units_sold"].notna() &
    (df["ad_spend"] > 0) &
    (df["price"] > 0)
].copy()

print(f"Rows after cleaning: {len(df_clean)} (dropped {len(df) - len(df_clean)})")

# ── 2. Feature engineering ───────────────────────────────────────────────────
# price and competitor_price are nearly perfectly collinear (r = 0.994).
# Replace both with the price gap so the model gets pricing signal without
# redundant features inflating input dimensionality.
df_clean["price_gap"] = df_clean["price"] - df_clean["competitor_price"]

# Encode region as dummy variables (drop_first avoids dummy-variable trap)
df_clean = pd.get_dummies(df_clean, columns=["region"], drop_first=True)

FEATURES = ["month", "ad_spend", "price_gap",
            "region_North", "region_South", "region_West"]

X = df_clean[FEATURES].values
y = df_clean["units_sold"].values

# ── 3. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ── 4. Scale features ────────────────────────────────────────────────────────
# Neural networks are sensitive to feature scale. Fit the scaler on training
# data only; apply the same transform to the test set to prevent data leakage.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── 5. Train model ───────────────────────────────────────────────────────────
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    early_stopping=True,       # hold out 10 % of training data as val set
    validation_fraction=0.1,
    n_iter_no_change=20,       # stop if val loss doesn't improve for 20 epochs
    random_state=SEED,
)

model.fit(X_train_s, y_train)
print(f"Training stopped after {model.n_iter_} iterations")

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train_s)
y_pred_test  = model.predict(X_test_s)

metrics = {
    "R²":   (r2_score(y_train, y_pred_train),   r2_score(y_test, y_pred_test)),
    "RMSE": (root_mean_squared_error(y_train, y_pred_train),
             root_mean_squared_error(y_test, y_pred_test)),
    "MAE":  (mean_absolute_error(y_train, y_pred_train),
             mean_absolute_error(y_test, y_pred_test)),
}

print(f"\n{'Metric':<8} {'Train':>10} {'Test':>10}")
print("-" * 30)
for name, (train_val, test_val) in metrics.items():
    print(f"{name:<8} {train_val:>10.4f} {test_val:>10.4f}")

# ── 7. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual vs Predicted (test set)
ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.7, edgecolors="k", linewidths=0.4)
lims = [min(y_test.min(), y_pred_test.min()) - 5,
        max(y_test.max(), y_pred_test.max()) + 5]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
ax.set_xlabel("Actual units_sold")
ax.set_ylabel("Predicted units_sold")
ax.set_title("Actual vs Predicted (test set)")
ax.legend()

# Training loss curve
ax = axes[1]
ax.plot(model.loss_curve_, label="Training loss")
if model.best_validation_score_ is not None:
    ax.plot(model.validation_scores_, label="Validation score (R²)", linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss / Score")
ax.set_title("Learning curve")
ax.legend()

fig.tight_layout()
fig.savefig("nn_results.png", dpi=150)
print("\nPlot saved to nn_results.png")
