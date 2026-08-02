"""
predict.py

Loads the trained model (from model/saved_model/) and uses it to classify
new SMS messages as "smishing" or "legitimate".

Usage (command line):
    python predict.py "Your account is suspended, verify now at bit.ly/xyz"

Usage (import into other code):
    from predict import SmishingDetector

    detector = SmishingDetector()
    result = detector.predict("Congratulations! You won a prize, claim now")
    print(result["label"], result["smishing_probability"])
"""

import os
import sys

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MAX_TOKEN_LENGTH = 96


class SmishingDetector:
    def __init__(self, model_dir: str = MODEL_DIR):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"No trained model found at '{model_dir}'.\n"
                f"Run train.py first to create it."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_dir, output_hidden_states=True
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Classifies a single SMS message.

        Returns a dictionary:
            text                  - the input message
            label                 - "smishing" or "legitimate"
            smishing_probability  - float between 0 and 1
            confidence            - how confident the model is in its label
            embedding             - 768-number vector describing the message
                                     (not needed yet, will be used later when
                                     modules are combined together)
        """
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        output = self.model(**tokens)
        probabilities = F.softmax(output.logits, dim=1).squeeze(0)

        smishing_probability = probabilities[1].item()
        confidence = probabilities.max().item()
        label = "smishing" if smishing_probability >= 0.5 else "legitimate"

        # The embedding is the model's internal representation of the message
        # (the [CLS] token from the last layer). Useful later, not needed now.
        last_hidden_layer = output.hidden_states[-1]
        embedding = last_hidden_layer[:, 0, :].squeeze(0).cpu().tolist()

        return {
            "text": text,
            "label": label,
            "smishing_probability": round(smishing_probability, 4),
            "confidence": round(confidence, 4),
            "embedding": embedding,
        }

    def predict_batch(self, texts: list) -> list:
        """Classifies a list of messages, one at a time."""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "<sms text>"')
        sys.exit(1)

    detector = SmishingDetector()
    result = detector.predict(sys.argv[1])

    print(f"Text        : {result['text']}")
    print(f"Prediction  : {result['label']}")
    print(f"Probability : {result['smishing_probability']}")
    print(f"Confidence  : {result['confidence']}")
