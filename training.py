import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Load synthetic data
accounts_df = pd.read_csv("synthetic_accounts.csv")
credit_history_df = pd.read_csv("synthetic_credit_history.csv")

# Convert dates
accounts_df["opening_date"] = pd.to_datetime(accounts_df["opening_date"])
credit_history_df["transaction_date"] = pd.to_datetime(credit_history_df["transaction_date"])


# Feature extraction function
def extract_features(account_id):
    history = credit_history_df[credit_history_df["account_id"] == account_id]
    account = accounts_df[accounts_df["account_id"] == account_id].iloc[0]

    deposits = history[history["transaction_type"] == "deposit"]["amount"].sum()
    withdrawals = history[history["transaction_type"] == "withdrawal"]["amount"].sum()
    loans = history[history["transaction_type"] == "loan_taken"]["loan_amount"].sum()
    repayments = history[history["transaction_type"] == "loan_payment"]["loan_repayment"].sum()
    months_active = max((datetime.now() - account["opening_date"]).days / 30, 1)

    features = {
        "avg_monthly_income": deposits / months_active,
        "spending_ratio": withdrawals / deposits if deposits > 0 else 0,
        "repayment_ratio": repayments / loans if loans > 0 else 1,  # 1 if no loans
        "overdraft_freq": history["overdraft_status"].mean(),
        "account_age_days": (datetime.now() - account["opening_date"]).days,
        "current_balance": account["current_balance"]
    }
    return features


# Prepare training data
X = []
y = []  # 1 = good credit, 0 = bad credit
for account_id in accounts_df["account_id"]:
    features = extract_features(account_id)
    X.append([
        features["avg_monthly_income"],
        features["spending_ratio"],
        features["repayment_ratio"],
        features["overdraft_freq"],
        features["account_age_days"],
        features["current_balance"]
    ])
    # Simulate label: good credit if low spending, high repayment, few overdrafts
    is_good_credit = (features["spending_ratio"] < 0.8 and
                      features["repayment_ratio"] > 0.9 and
                      features["overdraft_freq"] < 0.1)
    y.append(1 if is_good_credit else 0)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Function to get credit score
def get_credit_score(account_id):
    features = extract_features(account_id)
    feature_vector = [
        features["avg_monthly_income"],
        features["spending_ratio"],
        features["repayment_ratio"],
        features["overdraft_freq"],
        features["account_age_days"],
        features["current_balance"]
    ]
    score = model.predict_proba([feature_vector])[0][1]  # Probability of good credit
    return score * 100  # Scale to 0-100


# Test on a few accounts
for account_id in accounts_df["account_id"].head(5):
    score = get_credit_score(account_id)
    print(f"Account {account_id}: Credit Score = {score:.2f}")