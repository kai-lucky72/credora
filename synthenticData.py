from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta
import csv

fake = Faker()


def generate_synthetic_bank_data(num_accounts=50000, transactions_per_account=100, batch_size=1000):
    # Write accounts and credit history to CSV incrementally
    with open("synthetic_accounts.csv", "w", newline="") as acc_file, \
            open("synthetic_credit_history.csv", "w", newline="") as hist_file:

        # Headers
        acc_writer = csv.DictWriter(acc_file, fieldnames=["account_id", "opening_date", "current_balance"])
        hist_writer = csv.DictWriter(hist_file, fieldnames=["account_id", "transaction_date", "transaction_type",
                                                            "amount", "loan_amount", "loan_repayment",
                                                            "overdraft_status"])
        acc_writer.writeheader()
        hist_writer.writeheader()

        # Process in batches
        for batch_start in range(0, num_accounts, batch_size):
            batch_end = min(batch_start + batch_size, num_accounts)
            print(f"Generating accounts {batch_start} to {batch_end}...")

            # Accounts batch
            accounts = []
            for i in range(batch_start, batch_end):
                account_id = f"ACC{i:05d}"
                opening_date = fake.date_between(start_date="-3y", end_date="today")
                current_balance = round(random.uniform(-50000, 2000000), 2)  # Allow overdrafts
                accounts.append({
                    "account_id": account_id,
                    "opening_date": opening_date.strftime("%Y-%m-%d"),
                    "current_balance": current_balance
                })

            # Write accounts batch
            acc_writer.writerows(accounts)

            # Credit History batch
            for account in accounts:
                account_id = account["account_id"]
                start_date = datetime.strptime(account["opening_date"], "%Y-%m-%d")
                loan_balance = 0
                history_batch = []

                for _ in range(transactions_per_account):
                    trans_date = start_date + timedelta(days=random.randint(0, 1095))  # 3 years
                    trans_type = random.choices(
                        ["deposit", "withdrawal", "loan_taken", "loan_payment"],
                        weights=[0.35, 0.45, 0.05, 0.15], k=1
                    )[0]

                    if trans_type == "deposit":
                        amount = random.choice([random.uniform(20000, 300000), random.uniform(5000, 50000)]) * \
                                 (1 + random.uniform(-0.1, 0.1))
                    elif trans_type == "withdrawal":
                        amount = random.uniform(3000, 100000) * (1 + random.uniform(-0.2, 0.2))
                    elif trans_type == "loan_taken":
                        amount = random.uniform(50000, 2000000)
                        loan_balance += amount
                    else:  # loan_payment
                        amount = min(random.uniform(5000, 100000), loan_balance) if loan_balance > 0 else 0
                        loan_balance -= amount if loan_balance > 0 else 0

                    loan_amount = amount if trans_type == "loan_taken" else 0
                    loan_repayment = amount if trans_type == "loan_payment" else 0
                    overdraft_status = (
                                float(account["current_balance"]) - amount < 0) if trans_type == "withdrawal" else False

                    history_batch.append({
                        "account_id": account_id,
                        "transaction_date": trans_date.strftime("%Y-%m-%d"),
                        "transaction_type": trans_type,
                        "amount": round(amount, 2),
                        "loan_amount": round(loan_amount, 2),
                        "loan_repayment": round(loan_repayment, 2),
                        "overdraft_status": int(overdraft_status)
                    })
                    # Update balance
                    account["current_balance"] = float(account["current_balance"]) + \
                                                 (loan_amount - amount if trans_type in ["loan_taken",
                                                                                         "withdrawal"] else amount)

                # Write history batch for this account
                hist_writer.writerows(history_batch)

    print(f"Generated {num_accounts} accounts and {num_accounts * transactions_per_account} credit history records.")


# Run it
generate_synthetic_bank_data()