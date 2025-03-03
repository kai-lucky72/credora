import pandas as pd
import numpy as np
from datetime import datetime


def extract_features(history_chunk, account_row):
    deposits = history_chunk[history_chunk["transaction_type"] == "deposit"]["amount"]
    withdrawals = history_chunk[history_chunk["transaction_type"] == "withdrawal"]["amount"]
    loans = history_chunk[history_chunk["transaction_type"] == "loan_taken"]["loan_amount"]
    repayments = history_chunk[history_chunk["transaction_type"] == "loan_payment"]["loan_repayment"]
    months_active = max((datetime.now() - pd.to_datetime(account_row["opening_date"])).days / 30, 1)

    features = {
        "avg_monthly_income": deposits.sum() / months_active,
        "income_variability": deposits.std() / deposits.mean() if deposits.mean() > 0 else 0,
        "spending_ratio": withdrawals.sum() / deposits.sum() if deposits.sum() > 0 else 0,
        "withdrawal_freq": len(withdrawals) / months_active,
        "repayment_ratio": repayments.sum() / loans.sum() if loans.sum() > 0 else 1,
        "loan_count": len(loans),
        "avg_loan_size": loans.mean() if len(loans) > 0 else 0,
        "overdraft_freq": history_chunk["overdraft_status"].mean(),
        "account_age_days": (datetime.now() - pd.to_datetime(account_row["opening_date"])).days,
        "current_balance": account_row["current_balance"],
        "balance_to_income": account_row["current_balance"] / deposits.sum() if deposits.sum() > 0 else 0
    }
    return features