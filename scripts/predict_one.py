"""Load best v2 Snappfood-BERT checkpoint and classify one Persian sentence."""
from __future__ import annotations

import argparse
import json
import os
import sys

# Avoid Windows cp1252 crashes when printing Persian
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from hazm import Normalizer
    _normalizer = Normalizer()
except Exception:
    _normalizer = None


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(REPO, "outputs", "v2_parsbert", "final_model.pth")
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
# Labels in training: SAD=0, HAPPY=1
ID2LABEL = {0: "SAD", 1: "HAPPY"}


class CustomParsBERT(nn.Module):
    def __init__(self, model, dropout_prob=0.3, l2_lambda=0.01):
        super().__init__()
        self.bert = model.bert
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(model.config.hidden_size, model.config.num_labels)
        self.l2_lambda = l2_lambda

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            l2_reg = sum(param.norm(2) for param in self.classifier.parameters())
            loss = loss + self.l2_lambda * l2_reg
            return loss, logits
        return logits


def load_model(device: torch.device) -> tuple[CustomParsBERT, AutoTokenizer]:
    if not os.path.isfile(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    print(f"Loading tokenizer/base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = CustomParsBERT(base, dropout_prob=0.3, l2_lambda=0.01)

    print(f"Loading weights: {CKPT}")
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(text: str, model: CustomParsBERT, tokenizer: AutoTokenizer, device: torch.device, max_len: int = 198):
    cleaned = _normalizer.normalize(text) if _normalizer is not None else text
    enc = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
    return {
        "text": text,
        "normalized": cleaned,
        "label": ID2LABEL[pred_id],
        "pred_id": pred_id,
        "prob_sad": float(probs[0].item()),
        "prob_happy": float(probs[1].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default=None, help="Persian review to classify")
    args = parser.parse_args()
    text = args.text
    if text is None:
        text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("No text provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, tokenizer = load_model(device)
    result = predict(text, model, tokenizer, device)

    out_path = os.path.join(os.path.dirname(CKPT), "predict_one_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    try:
        print("---")
        print(f"Input     : {result['text']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Prediction: {result['label']}  (id={result['pred_id']})")
        print(f"P(SAD)    : {result['prob_sad']:.4f}")
        print(f"P(HAPPY)  : {result['prob_happy']:.4f}")
    except UnicodeEncodeError:
        print(f"Prediction: {result['label']} (id={result['pred_id']})")
        print(f"P(SAD)={result['prob_sad']:.4f}  P(HAPPY)={result['prob_happy']:.4f}")
        print("(Persian text omitted due to console encoding; see JSON file)")


if __name__ == "__main__":
    main()
