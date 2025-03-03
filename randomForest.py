from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load accounts
accounts_df = pd.read_csv("synthetic_accounts.csv")

# Process credit history in chunks
X = []
y = []
chunk_size = 100000  # Adjust based on your RAM
for chunk in pd.read_csv("synthetic_credit_history.csv", chunksize=chunk_size):
    for account_id in chunk["account_id"].unique():
        if account_id in accounts_df["account_id"].values:
            history_chunk = chunk[chunk["account_id"] == account_id]
            account_row = accounts_df[accounts_df["account_id"] == account_id].iloc[0]
            features = extract_features(history_chunk, account_row)
            X.append([features["avg_monthly_income"], features["income_variability"],
                      features["spending_ratio"], features["withdrawal_freq"],
                      features["repayment_ratio"], features["loan_count"],
                      features["avg_loan_size"], features["overdraft_freq"],
                      features["account_age_days"], features["current_balance"],
                      features["balance_to_income"]])
            is_good_credit = (features["repayment_ratio"] > 0.9 and
                              features["spending_ratio"] < 0.8 and
                              features["overdraft_freq"] < 0.1 and
                              features["current_balance"] > 0)
            y.append(1 if is_good_credit else 0)

X = np.array(X)
y = np.array(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Test
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save model
joblib.dump(rf, "credit_score_rf_model_large.pkl")

# Credit score function
def get_rf_credit_score(account_id, history_df):
    account_row = accounts_df[accounts_df["account_id"] == account_id].iloc[0]
    features = extract_features(history_df[history_df["account_id"] == account_id], account_row)
    feature_vector = [features["avg_monthly_income"], features["income_variability"],
                      features["spending_ratio"], features["withdrawal_freq"],
                      features["repayment_ratio"], features["loan_count"],
                      features["avg_loan_size"], features["overdraft_freq"],
                      features["account_age_days"], features["current_balance"],
                      features["balance_to_income"]]
    score = rf.predict_proba([feature_vector])[0][1]
    return score * 100

# Test a few scores
history_df = pd.read_csv("synthetic_credit_history.csv")
for account_id in accounts_df["account_id"].head(5):
    score = get_rf_credit_score(account_id, history_df)
    print(f"RF Score for {account_id}: {score:.2f}")