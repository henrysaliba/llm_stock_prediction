import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# fixed sentiment labels (order must match model output logits)
LABELS = ["negative", "neutral", "positive"]

class FinBert:
    def __init__(self):
        # load tokenizer (text -> token ids)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        # load model (finbert sequence classification head)
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # set model to evaluation mode (no dropout etc.)
        self.model.eval()

    @torch.inference_mode()
    def score(self, text: str):
        # tokenize input text (truncate if >128 tokens)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # forward pass through model
        out = self.model(**enc)
        
        # apply softmax to get probabilities for each label
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).tolist()
        
        # choose label with highest logit
        label = LABELS[int(torch.argmax(out.logits, dim=-1))]
        return label, probs

def run_sentiment(in_csv: str, out_csv: str):
    # ensure output directory exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # if input file is missing or empty -> write empty sentiment file
    if (not os.path.exists(in_csv)) or os.path.getsize(in_csv) == 0:
        print(f"[INFO] no news at {in_csv}. writing empty sentiment file.")
        pd.DataFrame(columns=["ticker","title","link","pubDate","sentiment",
                              "p_negative","p_neutral","p_positive"]).to_csv(out_csv, index=False)
        return out_csv

    # read input csv
    df = pd.read_csv(in_csv)

    # if dataframe empty or missing "title" column -> write empty sentiment file
    if df.empty or "title" not in df.columns:
        print(f"[INFO] news empty or missing 'title'. writing empty sentiment file.")
        pd.DataFrame(columns=["ticker","title","link","pubDate","sentiment",
                              "p_negative","p_neutral","p_positive"]).to_csv(out_csv, index=False)
        return out_csv

    # initialize finbert once
    fb = FinBert()

    # store predictions
    labels, p_neg, p_neu, p_pos = [], [], [], []

    # iterate over all titles (replace NaN with "")
    for t in tqdm(df["title"].fillna(""), desc="Scoring"):
        # run sentiment on title (fallback "blank" if empty)
        lab, probs = fb.score(t if t else "blank")
        labels.append(lab)
        p_neg.append(probs[0]); p_neu.append(probs[1]); p_pos.append(probs[2])

    # add new columns with predictions
    df["sentiment"] = labels
    df["p_negative"] = p_neg
    df["p_neutral"]  = p_neu
    df["p_positive"] = p_pos

    # save output csv
    df.to_csv(out_csv, index=False)
    return out_csv
