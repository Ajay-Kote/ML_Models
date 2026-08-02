# SMS Detection Module (Module 2)

A DistilBERT model that classifies an SMS message as **smishing** or
**legitimate**. This is just the model — no API yet. You'll wire it up with
the other 4 modules later.

```
module2_sms/
├── data/
│   ├── download_data.py     # step 1: get the dataset
│   ├── prepare_data.py      # step 2: clean it and split into train/val/test
│   ├── raw/                 # sms.tsv lands here
│   └── processed/           # train.csv, val.csv, test.csv land here
├── model/
│   ├── train.py             # step 3: train the model
│   ├── predict.py           # step 4: use the trained model
│   └── saved_model/         # created after training
├── explainability/
│   └── lime_explainer.py    # (optional, for later) shows which words drove a prediction
└── requirements.txt
```

## Setup

```bash
cd module2_sms
pip install -r requirements.txt
```

## Step 1 — Download the dataset

```bash
cd data
python download_data.py
```

This downloads the **UCI SMS Spam Collection** — 5,574 real SMS messages
labeled spam/ham — from a public GitHub mirror:
`https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv`

It's spam-vs-ham, not smishing-specific (no single clean public smishing
dataset exists), but it's a solid starting point since scam SMS overlaps a
lot with spam patterns — urgency, fake prizes, shortened links.

**If you want a stronger dataset later:** search Kaggle for "SMS Phishing
Dataset for Machine Learning and Pattern Recognition" and drop it in
`data/raw/`, then update `prepare_data.py` to merge it in.

## Step 2 — Prepare the data

```bash
python prepare_data.py
```

Cleans duplicates, converts labels to 0/1, and splits into:
- `train.csv` (70%)
- `val.csv` (15%)
- `test.csv` (15%)

## Step 3 — Train the model

```bash
cd ../model
python train.py
```

Fine-tunes DistilBERT on your data. Takes a few minutes on GPU, longer on
CPU. **This needs internet access** the first time (to download the base
`distilbert-base-uncased` model from Hugging Face) — if you're on a
restricted network, run this step on Google Colab (free GPU) instead.

At the end it prints accuracy, precision, recall, F1, and ROC-AUC on the
test set, and saves the trained model to `model/saved_model/`.

Optional flags:
```bash
python train.py --epochs 4 --batch_size 16 --learning_rate 2e-5
```

## Step 4 — Test the model

```bash
python predict.py "Your account is suspended, verify immediately at bit.ly/xyz"
```

Or from Python:
```python
from predict import SmishingDetector

detector = SmishingDetector()
result = detector.predict("Congratulations! You won a prize, claim now")
print(result)
# {'text': '...', 'label': 'smishing', 'smishing_probability': 0.94,
#  'confidence': 0.94, 'embedding': [...]}
```

Try it on a batch of test messages too, just to eyeball how it's doing:
```python
messages = [
    "Hey are we still meeting at 6?",
    "URGENT: your bank account will be locked. Verify at bit.ly/xyz now",
    "Reminder: your electricity bill is due tomorrow",
]
for r in detector.predict_batch(messages):
    print(r["label"], "->", r["text"])
```

## (Optional) Step 5 — See why it made a prediction

```bash
cd ../explainability
python lime_explainer.py "Congratulations! You won a prize, claim now at bit.ly/xyz"
```

This highlights which words pushed the prediction toward smishing. Not
needed right now — useful once you're writing the explainability section of
your report.

## What's left to do

- Run the actual training (needs GPU/Colab — I couldn't run it in my
  sandbox since it has no Hugging Face access)
- Note down the test metrics (accuracy/F1/ROC-AUC) for your evaluation section
- Optionally strengthen the dataset with real smishing examples
- Later: build the API + connect it with the other 4 modules
