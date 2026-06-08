"""
TripAdvisor Sentiment Analysis - Full Pipeline
包含：Baseline / TF-IDF+LR / RoBERTa(+weighted) / 諷刺偵測 / LLM摘要生成
使用前請在同目錄建立 .env 填入 OPENAI_API_KEY
"""

import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 模型儲存目錄
# ──────────────────────────────────────────────
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────────────────────
# 0. Load & Prepare Data
# ──────────────────────────────────────────────
df = pd.read_csv("data/clean/mining_data_clean.csv")

label_map = {"負面": 0, "中立": 1, "正面": 2}
id2label  = {0: "負面", 1: "中立", 2: "正面"}
df["label"] = df["sentiment"].map(label_map)

X = df["text"].astype(str).tolist()
y = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}\n")

results = {}

# ──────────────────────────────────────────────
# 1. Baseline: Majority Class
# ──────────────────────────────────────────────
print("=" * 55)
print("1. BASELINE: Majority Class")
majority = pd.Series(y_train).mode()[0]
y_pred_baseline = [majority] * len(y_test)
f1_baseline = f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)
print(f"Macro-F1: {f1_baseline:.4f}")
print(classification_report(y_test, y_pred_baseline,
      target_names=["負面","中立","正面"], zero_division=0))
results["Baseline (Majority)"] = f1_baseline

# ──────────────────────────────────────────────
# 2. TF-IDF + Logistic Regression
# ──────────────────────────────────────────────
print("=" * 55)
print("2. TF-IDF + Logistic Regression")
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")
print(f"Macro-F1: {f1_lr:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["負面","中立","正面"]))
results["TF-IDF + LR"] = f1_lr

# 儲存 TF-IDF 向量化器與 LR 模型
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(lr,    os.path.join(MODELS_DIR, "lr_classifier.pkl"))
print(f"  → 模型已儲存至 {MODELS_DIR}/tfidf_vectorizer.pkl & lr_classifier.pkl")

# ──────────────────────────────────────────────
# 3 & 4. RoBERTa Fine-tune
# ──────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16
EPOCHS     = 3
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_roberta(use_class_weights=False, run_name="RoBERTa"):
    print("=" * 55)
    print(run_name)

    train_loader = DataLoader(ReviewDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(ReviewDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)
    optimizer   = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    if use_class_weights:
        counts  = np.bincount(y_train)
        weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
        weights = weights / weights.sum() * len(counts)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        print(f"  Class weights: {weights.cpu().numpy().round(3)}")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            preds = torch.argmax(model(input_ids=ids, attention_mask=mask).logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n  Macro-F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["負面","中立","正面"]))
    return f1, model, tokenizer

f1_rob, _, _             = train_roberta(use_class_weights=False, run_name="3. RoBERTa (no weight)")
results["RoBERTa (no weight)"] = f1_rob

f1_rob_w, model_best, tokenizer_best = train_roberta(use_class_weights=True, run_name="4. RoBERTa (weighted) ← 改進方法")
results["RoBERTa (weighted) ★"] = f1_rob_w

# 儲存 RoBERTa (weighted) 模型與 tokenizer
roberta_save_path = os.path.join(MODELS_DIR, "roberta_weighted")
model_best.save_pretrained(roberta_save_path)
tokenizer_best.save_pretrained(roberta_save_path)
print(f"  → RoBERTa (weighted) 已儲存至 {roberta_save_path}/")

# ──────────────────────────────────────────────
# 5. 諷刺/刷星偵測
# ──────────────────────────────────────────────
print("=" * 55)
print("5. 諷刺/刷星評論偵測")

all_loader = DataLoader(ReviewDataset(X, y), batch_size=BATCH_SIZE)
model_best.eval()
all_preds_full = []
with torch.no_grad():
    for batch in all_loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        preds = torch.argmax(model_best(input_ids=ids, attention_mask=mask).logits, dim=1).cpu().numpy()
        all_preds_full.extend(preds)

df["predicted_sentiment"] = [id2label[p] for p in all_preds_full]
df["is_inconsistent"] = (
    (df["rating"] >= 4) &
    (df["predicted_sentiment"].isin(["負面", "中立"]))
)

total        = len(df)
inconsistent = df["is_inconsistent"].sum()
hi_star_neg  = ((df["rating"] >= 4) & (df["predicted_sentiment"] == "負面")).sum()
hi_star_neu  = ((df["rating"] >= 4) & (df["predicted_sentiment"] == "中立")).sum()

print(f"\n  總評論數: {total}")
print(f"  不一致（高星低情感）評論數: {inconsistent} ({inconsistent/total*100:.1f}%)")
print(f"    - 4~5星 但預測為負面: {hi_star_neg} 筆")
print(f"    - 4~5星 但預測為中立: {hi_star_neu} 筆")

inconsistent_df = df[df["is_inconsistent"]][["rating","text","sentiment","predicted_sentiment"]]
inconsistent_df.to_csv("inconsistent_reviews.csv", index=False, encoding="utf-8-sig")
print(f"\n  不一致評論已儲存至 inconsistent_reviews.csv")

print("\n  範例（前5筆）：")
for _, row in inconsistent_df.head(5).iterrows():
    print(f"  ★{row['rating']} | GPT:{row['sentiment']} | 預測:{row['predicted_sentiment']}")
    print(f"  \"{str(row['text'])[:120]}...\"")
    print()

'''
# ──────────────────────────────────────────────
# 6. LLM 摘要生成（GPT-5.4-mini）
# ──────────────────────────────────────────────
print("=" * 55)
print("6. LLM 摘要生成（GPT-5.4-mini）")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_summary(reviews_df, place_name="此景點/餐廳"):
    pos_count = (reviews_df["predicted_sentiment"] == "正面").sum()
    neu_count = (reviews_df["predicted_sentiment"] == "中立").sum()
    neg_count = (reviews_df["predicted_sentiment"] == "負面").sum()
    inc_count = reviews_df["is_inconsistent"].sum()
    total     = len(reviews_df)

    sample_pos = reviews_df[reviews_df["predicted_sentiment"] == "正面"]["text"].head(5).tolist()
    sample_neg = reviews_df[reviews_df["predicted_sentiment"] == "負面"]["text"].head(5).tolist()
    sample_inc = reviews_df[reviews_df["is_inconsistent"]]["text"].head(5).tolist()

    prompt = f"""
你是一個旅遊評論分析師，請根據以下資料為「{place_name}」產生一份中文分析報告。

【統計資料】
- 總評論數: {total}
- 正面評論: {pos_count} 筆 ({pos_count/total*100:.1f}%)
- 中立評論: {neu_count} 筆 ({neu_count/total*100:.1f}%)
- 負面評論: {neg_count} 筆 ({neg_count/total*100:.1f}%)
- 疑似刷星（高星低情感）: {inc_count} 筆 ({inc_count/total*100:.1f}%)

【正面評論範例】
{chr(10).join(f'- {t[:150]}' for t in sample_pos)}

【負面評論範例】
{chr(10).join(f'- {t[:150]}' for t in sample_neg)}

【疑似刷星評論範例】
{chr(10).join(f'- {t[:150]}' for t in sample_inc)}

請輸出：
1. 整體評價摘要（2~3句）
2. 主要優點
3. 主要缺點
4. 刷星風險評估（是否值得注意，並說明原因）
5. 最終建議：「值得前往」或「謹慎考慮」，並給出理由
"""
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=800
    )
    return response.choices[0].message.content

print("\n  正在呼叫 GPT-5.4-mini...\n")
summary = generate_summary(df)
print(summary)

with open("summary_output.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print("\n  摘要已儲存至 summary_output.txt")

# ──────────────────────────────────────────────
# 7. Final Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL RESULTS (Macro-F1)")
print("=" * 55)
for name, score in results.items():
    marker = " ← 最佳" if score == max(results.values()) else ""
    print(f"  {name:35s}: {score:.4f}{marker}")

print(f"\n諷刺偵測: 疑似刷星評論佔 {inconsistent/total*100:.1f}% ({inconsistent}/{total} 筆)")
'''