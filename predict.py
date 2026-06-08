"""
predict.py — TripAdvisor Sentiment Prediction
使用訓練好的模型對新的 review 資料做情感預測。

支援兩種模型：
  --model tfidf   : TF-IDF + Logistic Regression (快速)
  --model roberta : RoBERTa (weighted fine-tuned, 較準確)

評估指標：
  - 若 CSV 含 ground truth 欄位（如 sentiment），自動計算：
      Accuracy / Precision / Recall / F1（per-class + macro + weighted）+ 混淆矩陣
  - 若跑 --model both 且無 ground truth，顯示：
      TF-IDF vs RoBERTa 模型一致性矩陣（Model Agreement Matrix）

用法範例：
  # 預測（無 ground truth）
  python predict.py --model tfidf   --input data/clean/The_Cat_Cabinet_clean.csv
  python predict.py --model roberta --input data/clean/StoneGrill_1870_clean.csv
  python predict.py --model both    --input data/clean/The_Cat_Cabinet_clean.csv

  # 預測並評估（有 ground truth，使用 sentiment 欄位）
  python predict.py --model both    --input data/clean/mining_data_clean.csv --label-col sentiment
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
MODELS_DIR = "saved_models"
LABELS     = ["負面", "中立", "正面"]
ID2LABEL   = {0: "負面", 1: "中立", 2: "正面"}
LABEL2ID   = {"負面": 0, "中立": 1, "正面": 2}


# ──────────────────────────────────────────────
# TF-IDF + LR 預測
# ──────────────────────────────────────────────
def predict_tfidf(texts: list[str]) -> list[str]:
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    lr_path    = os.path.join(MODELS_DIR, "lr_classifier.pkl")

    if not os.path.exists(tfidf_path) or not os.path.exists(lr_path):
        raise FileNotFoundError(
            f"找不到 TF-IDF 模型檔案，請先執行 train_pipeline.py\n"
            f"  預期位置: {tfidf_path}, {lr_path}"
        )

    print("  載入 TF-IDF + LR 模型 ...")
    tfidf = joblib.load(tfidf_path)
    lr    = joblib.load(lr_path)

    X = tfidf.transform(texts)
    preds = lr.predict(X)
    return [ID2LABEL[p] for p in preds]


# ──────────────────────────────────────────────
# RoBERTa 預測
# ──────────────────────────────────────────────
def predict_roberta(texts: list[str]) -> list[str]:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    roberta_path = os.path.join(MODELS_DIR, "roberta_weighted")
    if not os.path.exists(roberta_path):
        raise FileNotFoundError(
            f"找不到 RoBERTa 模型資料夾，請先執行 train_pipeline.py\n"
            f"  預期位置: {roberta_path}/"
        )

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 128
    BATCH   = 32

    print(f"  載入 RoBERTa 模型 (device={DEVICE}) ...")
    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    model     = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(DEVICE)
    model.eval()

    class _DS(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
            }

    loader = DataLoader(_DS(texts), batch_size=BATCH)
    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            done = min((i + 1) * BATCH, len(texts))
            print(f"  RoBERTa 推論中... {done}/{len(texts)}", end="\r")
    print()
    return [ID2LABEL[p] for p in all_preds]


# ──────────────────────────────────────────────
# 不一致偵測（高星低情感）
# ──────────────────────────────────────────────
def detect_inconsistency(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """回傳 bool Series，True 表示高星但預測為負/中立（疑似刷星）"""
    return (df["rating"] >= 4) & (df[pred_col].isin(["負面", "中立"]))


# ──────────────────────────────────────────────
# 預測分佈摘要
# ──────────────────────────────────────────────
def print_summary(df: pd.DataFrame, pred_col: str, model_name: str):
    total  = len(df)
    counts = df[pred_col].value_counts()
    inc    = detect_inconsistency(df, pred_col).sum()

    print(f"\n  [{model_name}] 預測分佈 (共 {total} 筆)")
    print(f"  {'─'*40}")
    for label in LABELS:
        n = counts.get(label, 0)
        print(f"  {label}: {n:4d} 筆 ({n/total*100:5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  疑似刷星（4~5星但負/中立）: {inc} 筆 ({inc/total*100:.1f}%)")


# ──────────────────────────────────────────────
# 評估指標（有 ground truth 時）
# ──────────────────────────────────────────────
def print_metrics(y_true: list[str], y_pred: list[str], model_name: str):
    """計算並印出混淆矩陣、Accuracy、Precision、Recall、F1。"""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix
    )

    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=LABELS)

    # per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    W = 42
    print(f"\n{'='*W}")
    print(f"  [{model_name}] 評估指標（vs. Ground Truth）")
    print(f"{'='*W}")
    print(f"  Accuracy : {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)} 筆預測正確)")

    # per-class table
    print(f"\n  {'類別':<6} {'Precision':>10} {'Recall':>9} {'F1-Score':>9} {'Support':>8}")
    print(f"  {'─'*46}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<6} {precision[i]:>10.4f} {recall[i]:>9.4f} {f1[i]:>9.4f} {int(support[i]):>8}")
    print(f"  {'─'*46}")
    print(f"  {'macro avg':<6} {macro_p:>10.4f} {macro_r:>9.4f} {macro_f1:>9.4f} {len(y_true):>8}")
    print(f"  {'weighted':<6} {weighted_p:>10.4f} {weighted_r:>9.4f} {weighted_f1:>9.4f} {len(y_true):>8}")

    # confusion matrix
    print(f"\n  混淆矩陣（列=實際, 欄=預測）")
    print(f"  {'':8}", end="")
    for lbl in LABELS:
        print(f"  {lbl:>5}", end="")
    print()
    print(f"  {'─'*34}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<8}", end="")
        for j in range(len(LABELS)):
            marker = "*" if i == j else " "
            print(f"  {cm[i][j]:>4}{marker}", end="")
        print()
    print(f"  （* 表示預測正確）")
    print(f"{'='*W}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────
# 模型一致性矩陣（無 ground truth，兩模型互比）
# ──────────────────────────────────────────────
def print_agreement_matrix(df: pd.DataFrame):
    """
    比較 predicted_tfidf 與 predicted_roberta 的預測一致性。
    視 TF-IDF 為 'actual'，RoBERTa 為 'predicted'，產生交叉矩陣。
    """
    from sklearn.metrics import confusion_matrix, cohen_kappa_score

    y_tfidf   = df["predicted_tfidf"].tolist()
    y_roberta = df["predicted_roberta"].tolist()

    agree = sum(a == b for a, b in zip(y_tfidf, y_roberta))
    total = len(y_tfidf)
    rate  = agree / total
    kappa = cohen_kappa_score(y_tfidf, y_roberta)

    cm = confusion_matrix(y_tfidf, y_roberta, labels=LABELS)

    W = 52
    print(f"\n{'='*W}")
    print(f"  模型一致性分析（TF-IDF vs RoBERTa）")
    print(f"{'='*W}")
    print(f"  Agreement Rate : {rate:.4f}  ({agree}/{total} 筆兩模型相同)")
    print(f"  Cohen's Kappa  : {kappa:.4f}  ", end="")
    if   kappa >= 0.80: print("（幾乎完全一致）")
    elif kappa >= 0.60: print("（一致性良好）")
    elif kappa >= 0.40: print("（中等一致性）")
    elif kappa >= 0.20: print("（輕度一致）")
    else:               print("（一致性弱）")

    # 交叉矩陣
    print(f"\n  一致性矩陣（列=TF-IDF預測, 欄=RoBERTa預測）")
    print(f"  {'':8}", end="")
    for lbl in LABELS:
        print(f"  {lbl:>5}", end="")
    print()
    print(f"  {'─'*36}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<8}", end="")
        for j in range(len(LABELS)):
            marker = "*" if i == j else " "
            print(f"  {cm[i][j]:>4}{marker}", end="")
        print()
    print(f"  （* 表示兩模型預測相同）")

    # 分歧分析
    diff_df = df[df["predicted_tfidf"] != df["predicted_roberta"]].copy()
    if len(diff_df) > 0:
        print(f"\n  分歧筆數：{len(diff_df)} 筆，分佈：")
        for _, grp in diff_df.groupby(["predicted_tfidf", "predicted_roberta"]):
            pass
        breakdown = diff_df.groupby(["predicted_tfidf", "predicted_roberta"]).size()
        for (t, r), cnt in breakdown.items():
            print(f"    TF-IDF={t} / RoBERTa={r} : {cnt} 筆")
    print(f"{'='*W}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TripAdvisor Sentiment Predictor")
    parser.add_argument(
        "--model", choices=["tfidf", "roberta", "both"], default="both",
        help="要使用的模型：tfidf / roberta / both (default: both)"
    )
    parser.add_argument(
        "--input", required=True,
        help="輸入的 clean CSV 路徑（需有 text, rating 欄位）"
    )
    parser.add_argument(
        "--output", default=None,
        help="輸出的 CSV 路徑（預設：在輸入檔名加 _predicted）"
    )
    parser.add_argument(
        "--label-col", default=None,
        help="Ground truth 欄位名稱（如 sentiment）。有此欄位時自動計算 Accuracy/F1/混淆矩陣。"
    )
    args = parser.parse_args()

    # ── 讀取資料 ──────────────────────────────
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到輸入檔案: {args.input}")

    df = pd.read_csv(args.input)
    print(f"\n讀取: {args.input}  ({len(df)} 筆)")

    if "text" not in df.columns:
        raise ValueError("輸入 CSV 缺少 'text' 欄位")

    # 自動偵測 ground truth 欄位
    label_col = args.label_col
    if label_col is None and "sentiment" in df.columns:
        label_col = "sentiment"
        print(f"  偵測到 ground truth 欄位：'{label_col}'，將輸出評估指標")
    elif label_col and label_col not in df.columns:
        print(f"  警告：找不到 --label-col '{label_col}'，將跳過評估指標")
        label_col = None

    texts = df["text"].astype(str).tolist()

    # ── 預測 ──────────────────────────────────
    use_tfidf   = args.model in ("tfidf",  "both")
    use_roberta = args.model in ("roberta", "both")

    if use_tfidf:
        step = "1/2" if use_roberta else "1/1"
        print(f"\n[{step}] TF-IDF + LR 預測 ...")
        df["predicted_tfidf"] = predict_tfidf(texts)
        print_summary(df, "predicted_tfidf", "TF-IDF + LR")
        if label_col:
            print_metrics(df[label_col].tolist(), df["predicted_tfidf"].tolist(), "TF-IDF + LR")

    if use_roberta:
        step = "2/2" if use_tfidf else "1/1"
        print(f"\n[{step}] RoBERTa (weighted) 預測 ...")
        df["predicted_roberta"] = predict_roberta(texts)
        print_summary(df, "predicted_roberta", "RoBERTa (weighted)")
        if label_col:
            print_metrics(df[label_col].tolist(), df["predicted_roberta"].tolist(), "RoBERTa (weighted)")

    # ── 模型一致性矩陣（both + 無 ground truth）──
    if use_tfidf and use_roberta and label_col is None:
        print_agreement_matrix(df)

    # ── 不一致偵測 ────────────────────────────
    main_pred_col = "predicted_roberta" if use_roberta else "predicted_tfidf"
    df["is_inconsistent"] = detect_inconsistency(df, main_pred_col)

    # ── 輸出 ──────────────────────────────────
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + "_predicted" + ext

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n結果已儲存至: {args.output}")

    # ── 顯示前 5 筆不一致評論 ─────────────────
    inc_df = df[df["is_inconsistent"]]
    if len(inc_df) > 0:
        print(f"\n前 {min(5, len(inc_df))} 筆疑似刷星評論：")
        for _, row in inc_df.head(5).iterrows():
            pred = row.get(main_pred_col, "?")
            print(f"  [{row['rating']}星 | 預測:{pred}] {str(row['text'])[:100]}...")
    else:
        print("\n未偵測到疑似刷星評論。")

    print("\nDone.")


if __name__ == "__main__":
    main()
