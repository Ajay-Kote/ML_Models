"""
augment_sms_data.py

FIX for the OTP false-positive issue: the training data had only ONE
legitimate message containing the word "OTP" out of 5224 rows, so the
model never learned that OTP-style messages CAN be legitimate -- it
just associated the pattern "OTP" with smishing.

This script adds 20 new genuine transactional/OTP messages (label=0,
legitimate) covering banking, e-commerce, delivery, social media, and
ride-hailing -- varied enough that the model learns the general PATTERN
of a legitimate OTP message, not just one specific wording.

Run this ONCE from inside module2_sms/data/:
    python augment_sms_data.py

It appends 16 rows to train.csv, 2 to val.csv, 2 to test.csv (keeping
the original train/val/test split ratio roughly intact), then you
retrain with:
    cd ../model
    python train.py
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "processed")

# All legitimate (label=0). Deliberately varied senders/contexts so the
# model generalizes "this is a normal transactional OTP" rather than
# memorizing one exact phrasing.
NEW_LEGITIMATE_OTP_MESSAGES = [
    "123456 is your OTP for HDFC Bank NetBanking login. Valid for 10 minutes. Do not share with anyone.",
    "Use 908213 as your one-time password to complete your Amazon order. This OTP is valid for 5 minutes.",
    "Your Paytm OTP is 552310. Never share your OTP with anyone, Paytm will never ask for it.",
    "645789 is your verification code for WhatsApp. For your security, do not share this code.",
    "Your SBI OTP for fund transfer of Rs 5000 is 118844. Valid till 10:45 PM. Do not share.",
    "OTP to reset your Gmail password is 774521. If you didn't request this, ignore this message.",
    "Your Swiggy order #4521 OTP for delivery is 3092. Share only with the delivery partner.",
    "213456 is your OTP to login to Zomato. This code is valid for 3 minutes.",
    "Your Flipkart account verification code is 665201. Do not share this OTP with anyone for security reasons.",
    "Use OTP 887744 to verify your mobile number on Google Pay.",
    "Your ICICI Bank OTP for debit card transaction of Rs 2500 at Amazon is 556677.",
    "339021 is your one-time password for Uber login. Valid for 5 minutes.",
    "Your Airtel SIM verification OTP is 445566. Do not share this code with anyone including Airtel staff.",
    "OTP 112233 for your Ola cab booking confirmation. Valid for 10 minutes.",
    "Your bank transaction of Rs 1200 was successful. OTP used: 998877. Thank you for banking with us.",
    "Your Instagram login code is 209384. This code expires in 10 minutes.",
    # held out for val
    "Your Zerodha Kite OTP for login is 667788. Valid for 5 mins, do not share with anyone.",
    "Netflix verification code: 118826. Enter this to confirm your new device.",
    # held out for test
    "Your delivery OTP for order #78219 is 4567. Please share with delivery executive at doorstep.",
    "Your PhonePe OTP for UPI payment of Rs 850 is 331209. Valid for 3 minutes. Do not share.",
]

TRAIN_MESSAGES = NEW_LEGITIMATE_OTP_MESSAGES[:16]
VAL_MESSAGES = NEW_LEGITIMATE_OTP_MESSAGES[16:18]
TEST_MESSAGES = NEW_LEGITIMATE_OTP_MESSAGES[18:20]


def append_rows(csv_path: str, messages: list[str]):
    df = pd.read_csv(csv_path)
    new_rows = pd.DataFrame({"text": messages, "label": [0] * len(messages)})
    before = len(df)
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"{csv_path}: {before} -> {len(df)} rows (+{len(messages)})")


def main():
    append_rows(os.path.join(DATA_DIR, "train.csv"), TRAIN_MESSAGES)
    append_rows(os.path.join(DATA_DIR, "val.csv"), VAL_MESSAGES)
    append_rows(os.path.join(DATA_DIR, "test.csv"), TEST_MESSAGES)
    print("\nDone. Now retrain:")
    print("  cd ../model")
    print("  python train.py")


if __name__ == "__main__":
    main()