"""
classify_and_evaluate.py
────────────────────────────────────────────────────────
完整流程：
  1. 用 GPT 對 clean CSV 做情感分類，產生 ground truth
  2. 用訓練好的 TF-IDF + LR / RoBERTa 做預測
  3. 對比 ground truth vs 預測，輸出評估指標與混淆矩陣

用法：
  # 預設跑兩個 clean 檔案，兩個模型
  python classify_and_evaluate.py

  # 只跑單一檔案
  python classify_and_evaluate.py --files data/clean/The_Cat_Cabinet_clean.csv

  # 跳過 GPT 分類（已有 ground truth CSV），直接做評估
  python classify_and_evaluate.py --skip-classify

  # 指定使用的模型
  python classify_and_evaluate.py --model tfidf
"""

import os
import time
import argparse
import concurrent.futures

import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score
)

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
load_dotenv()
API_KEY    = os.getenv("OPENAI_API_KEY")
MODELS_DIR = "saved_models"
LABELS     = ["負面", "中立", "正面"]
ID2LABEL   = {0: "負面", 1: "中立", 2: "正面"}

DEFAULT_FILES = [
    "data/clean/The_Cat_Cabinet_clean.csv",
    "data/clean/StoneGrill_1870_clean.csv",
]

# ──────────────────────────────────────────────
# 1. GPT 分類（ground truth）
# ──────────────────────────────────────────────
def classify_text(client, title, text, gpt_model="gpt-4o-mini", retries=3):
    title_str = str(title) if pd.notna(title) else ""
    text_str  = str(text)  if pd.notna(text)  else ""

    prompt = (
        "請分析以下評論的情感，將其分類為以下三種之一：「正面」(Positive)、「負面」(Negative) 或「中立」(Neutral)。\n"
        "請「只」輸出這三個選項其中之一的中文字（正面、負面、中立），不要有任何其他解釋。\n\n"
        f"標題：{title_str}\n"
        f"評論內容：{text_str}"
    )

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "你是一個精準的情感分析助手。"},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=10,
            )
            result = resp.choices[0].message.content.strip()
            # 正規化：只保留合法標籤
            for label in ["正面", "負面", "中立"]:
                if label in result:
                    return label
            return result  # 回傳原始（可能是錯誤回應）
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return f"Error: {e}"


def gpt_classify_file(file_path: str, client: OpenAI,
                      gpt_model: str = "gpt-4o-mini",
                      max_workers: int = 5) -> pd.DataFrame:
    """讀取 clean CSV，用 GPT 並行分類，儲存 *_gpt_labeled.csv 並回傳 DataFrame。"""
    out_path = file_path.replace("_clean.csv", "_gpt_labeled.csv")

    if os.path.exists(out_path):
        print(f"  [已存在] 跳過 GPT 分類，直接讀取：{out_path}")
        return pd.read_csv(out_path)

    print(f"\n[GPT 分類] {file_path}")
    df = pd.read_csv(file_path)
    results = [""] * len(df)

    title_col = "title" if "title" in df.columns else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                classify_text, client,
                row.get(title_col, "") if title_col else "",
                row["text"],
                gpt_model
            ): idx
            for idx, row in df.iterrows()
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(df), desc=f"  GPT 分類中"
        ):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"Error: {e}"

    df["gpt_sentiment"] = results

    # 過濾掉無效標籤
    valid_mask = df["gpt_sentiment"].isin(LABELS)
    invalid    = (~valid_mask).sum()
    if invalid:
        print(f"  警告：{invalid} 筆 GPT 回應無效，將排除")
        df = df[valid_mask].reset_index(drop=True)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  GPT 分類完成，儲存至：{out_path}  ({len(df)} 筆有效)")
    return df


# ──────────────────────────────────────────────
# 2. 模型預測
# ──────────────────────────────────────────────
def predict_tfidf(texts: list[str]) -> list[str]:
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    lr    = joblib.load(os.path.join(MODELS_DIR, "lr_classifier.pkl"))
    preds = lr.predict(tfidf.transform(texts))
    return [ID2LABEL[p] for p in preds]


def predict_roberta(texts: list[str]) -> list[str]:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 128
    BATCH   = 32
    path    = os.path.join(MODELS_DIR, "roberta_weighted")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    model.eval()

    class _DS(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True,
                            padding="max_length", max_length=MAX_LEN,
                            return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze()}

    loader    = DataLoader(_DS(texts), batch_size=BATCH)
    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            p    = torch.argmax(
                model(input_ids=ids, attention_mask=mask).logits, dim=1
            ).cpu().numpy()
            all_preds.extend(p)
            print(f"  RoBERTa 推論中... {min((i+1)*BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return [ID2LABEL[p] for p in all_preds]


# ──────────────────────────────────────────────
# 3. 不一致偵測（高星低情感）
# ──────────────────────────────────────────────
def detect_inconsistency(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """回傳 bool Series：rating >= 4 但預測為負面或中立（疑似刷星）。"""
    return (df["rating"] >= 4) & (df[pred_col].isin(["負面", "中立"]))


# ──────────────────────────────────────────────
# 4. 評估指標輸出
# ──────────────────────────────────────────────
def print_metrics(y_true: list, y_pred: list, model_name: str, place: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=LABELS)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    m_p,  m_r,  m_f1,  _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    w_p,  w_r,  w_f1,  _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    W = 54
    bar = "=" * W
    print(f"\n{bar}")
    print(f"  [{place}]  {model_name}")
    print(f"  Ground truth: GPT-4o-mini  |  總筆數: {len(y_true)}")
    print(bar)
    print(f"  Accuracy : {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)} 筆正確)")

    print(f"\n  {'類別':<6} {'Precision':>10} {'Recall':>9} {'F1-Score':>9} {'Support':>8}")
    print(f"  {'─'*46}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<6} {prec[i]:>10.4f} {rec[i]:>9.4f} {f1[i]:>9.4f} {int(sup[i]):>8}")
    print(f"  {'─'*46}")
    print(f"  {'macro avg':<6} {m_p:>10.4f} {m_r:>9.4f} {m_f1:>9.4f} {len(y_true):>8}")
    print(f"  {'weighted':<6} {w_p:>10.4f} {w_r:>9.4f} {w_f1:>9.4f} {len(y_true):>8}")

    print(f"\n  混淆矩陣（列=GPT實際, 欄=模型預測）")
    print(f"  {'':10}", end="")
    for lbl in LABELS:
        print(f"  {lbl:>5}", end="")
    print()
    print(f"  {'─'*34}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<10}", end="")
        for j in range(len(LABELS)):
            mark = "*" if i == j else " "
            print(f"  {cm[i][j]:>4}{mark}", end="")
        print()
    print(f"  （* 表示預測正確）")
    print(bar)

    return {
        "place": place, "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(m_f1, 4), "weighted_f1": round(w_f1, 4),
        "macro_precision": round(m_p, 4), "macro_recall": round(m_r, 4),
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPT 分類 + 模型評估")
    parser.add_argument("--files",          nargs="+", default=DEFAULT_FILES,
                        help="要處理的 clean CSV 路徑（可多個）")
    parser.add_argument("--model",          choices=["tfidf", "roberta", "both"],
                        default="both", help="評估哪個模型（預設: both）")
    parser.add_argument("--gpt-model",      default="gpt-4o-mini",
                        help="GPT 模型名稱（預設: gpt-4o-mini）")
    parser.add_argument("--workers",        type=int, default=5,
                        help="GPT 並行 thread 數（預設: 5）")
    parser.add_argument("--skip-classify",  action="store_true",
                        help="跳過 GPT 分類（直接讀已有的 *_gpt_labeled.csv）")
    args = parser.parse_args()

    if not API_KEY:
        raise EnvironmentError("找不到 OPENAI_API_KEY，請確認 .env 檔案")

    client = OpenAI(api_key=API_KEY)

    use_tfidf   = args.model in ("tfidf",  "both")
    use_roberta = args.model in ("roberta", "both")

    all_results = []

    for file_path in args.files:
        place = (os.path.basename(file_path)
                 .replace("_clean.csv", "")
                 .replace("_", " "))

        # ── Step 1: GPT 分類 ──────────────────
        if args.skip_classify:
            labeled_path = file_path.replace("_clean.csv", "_gpt_labeled.csv")
            if not os.path.exists(labeled_path):
                print(f"[跳過分類] 找不到 {labeled_path}，略過此檔案")
                continue
            df = pd.read_csv(labeled_path)
            print(f"\n讀取已有 GPT 標籤：{labeled_path}  ({len(df)} 筆)")
        else:
            df = gpt_classify_file(file_path, client,
                                   gpt_model=args.gpt_model,
                                   max_workers=args.workers)

        y_true = df["gpt_sentiment"].tolist()
        texts  = df["text"].astype(str).tolist()

        # ── Step 2: 模型預測 + 評估 ──────────
        if use_tfidf:
            print(f"\n  [TF-IDF + LR] 預測中 ...")
            df["predicted_tfidf"] = predict_tfidf(texts)
            r = print_metrics(y_true, df["predicted_tfidf"].tolist(),
                              "TF-IDF + LR", place)
            all_results.append(r)

        if use_roberta:
            print(f"\n  [RoBERTa (weighted)] 預測中 ...")
            df["predicted_roberta"] = predict_roberta(texts)
            r = print_metrics(y_true, df["predicted_roberta"].tolist(),
                              "RoBERTa (weighted)", place)
            all_results.append(r)

        # ── 不一致偵測 ────────────────────────
        # 優先用 RoBERTa，否則用 TF-IDF
        main_pred_col = "predicted_roberta" if use_roberta else "predicted_tfidf"
        df["is_inconsistent"] = detect_inconsistency(df, main_pred_col)
        inc = df["is_inconsistent"].sum()
        print(f"\n  疑似刷星（4~5星但負/中立）: {inc} 筆 ({inc/len(df)*100:.1f}%)")

        # ── 儲存含預測結果的完整 CSV ──────────
        out_csv = file_path.replace("_clean.csv", "_evaluated.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  完整結果已儲存：{out_csv}")

    # ── 最終彙總表 ────────────────────────────
    if all_results:
        print(f"\n{'='*54}")
        print("  最終彙總（Macro F1）")
        print(f"{'='*54}")
        summary = pd.DataFrame(all_results)
        print(summary.to_string(index=False))

        # 儲存彙總
        summary.to_csv("logs/evaluation_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n  彙總表已儲存至 logs/evaluation_summary.csv")


if __name__ == "__main__":
    main()
