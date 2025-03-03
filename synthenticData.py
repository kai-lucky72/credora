from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_synthetic_bank_data(num_accounts=1000, transactions_per_account=50):
    # Accounts table
    accounts = []
    for i in range(num_accounts):
        account_id = f"ACC{i:05d}"  # e.g., ACC00001
        opening_date = fake.date_between(start_date="-2y", end_date="today")
        accounts.append({
            "account_id": account_id,
            "opening_date": opening_date,
            "current_balance": round(random.uniform(10000, 1000000), 2)  # 10K-1M RWF
        })

    # Credit History table
    credit_history = []
    for account in accounts:
        account_id = account["account_id"]
        start_date = account["opening_date"]
        for _ in range(transactions_per_account):
            trans_date = start_date + timedelta(days=random.randint(0, 730))  # 2 years
            trans_type = random.choices(
                ["deposit", "withdrawal", "loan_taken", "loan_payment"],
                weights=[0.4, 0.45, 0.05, 0.1],  # Realistic distribution
                k=1
            )[0]
            amount = (
                random.uniform(20000, 200000) if trans_type == "deposit" else  # Salaries, income
                random.uniform(5000, 50000) if trans_type == "withdrawal" else  # Daily spending
                random.uniform(50000, 1000000) if trans_type == "loan_taken" else  # Loans
                random.uniform(10000, 50000)  # Loan payments
            )
            loan_amount = amount if trans_type == "loan_taken" else 0
            loan_repayment = amount if trans_type == "loan_payment" else 0
            overdraft_status = random.random() < 0.05  # 5% overdraft chance

            credit_history.append({
                "account_id": account_id,
                "transaction_date": trans_date,
                "transaction_type": trans_type,
                "amount": round(amount, 2),
                "loan_amount": round(loan_amount, 2),
                "loan_repayment": round(loan_repayment, 2),
                "overdraft_status": overdraft_status
            })

    return pd.DataFrame(accounts), pd.DataFrame(credit_history)

# Generate and save
accounts_df, credit_history_df = generate_synthetic_bank_data()
accounts_df.to_csv("synthetic_accounts.csv", index=False)
credit_history_df.to_csv("synthetic_credit_history.csv", index=False)

print(f"Generated {len(accounts_df)} accounts and {len(credit_history_df)} credit history records.")