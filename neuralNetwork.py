from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Same X, y preparation as Random Forest (reuse the chunked loop above)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                    max_iter=500, random_state=42, batch_size=256)
mlp.fit(X_train, y_train)

# Test
y_pred_mlp = mlp.predict(X_test)
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))

# Save model and scaler
joblib.dump(mlp, "credit_score_mlp_model_large.pkl")
joblib.dump(scaler, "scaler_large.pkl")

# Credit score function
def get_mlp_credit_score(account_id, history_df):
    account_row = accounts_df[accounts_df["account_id"] == account_id].iloc[0]
    features = extract_features(history_df[history_df["account_id"] == account_id], account_row)
    feature_vector = [features["avg_monthly_income"], features["income_variability"],
                      features["spending_ratio"], features["withdrawal_freq"],
                      features["repayment_ratio"], features["loan_count"],
                      features["avg_loan_size"], features["overdraft_freq"],
                      features["account_age_days"], features["current_balance"],
                      features["balance_to_income"]]
    scaled_vector = scaler.transform([feature_vector])
    score = mlp.predict_proba(scaled_vector)[0][1]
    return score * 100

# Test a few scores
for account_id in accounts_df["account_id"].head(5):
    score = get_mlp_credit_score(account_id, history_df)
    print(f"MLP Score for {account_id}: {score:.2f}")