"""
generate_summary.py
────────────────────────────────────────────────────────
使用 predict.py 的預測結果，呼叫 LLM 產生景點/餐廳中文分析報告。

前置條件：
  1. 已執行 predict.py，產生含 predicted_roberta 或 predicted_tfidf 欄位的 CSV
  2. .env 檔中設定 OPENAI_API_KEY

執行方式：
  # 用 RoBERTa 預測結果（預設）
  python generate_summary.py --place "The Cat Cabinet" --data data/clean/The_Cat_Cabinet_predicted.csv

  # 指定用 TF-IDF 預測欄位
  python generate_summary.py --place "StoneGrill 1870" --data data/clean/StoneGrill_1870_predicted.csv --pred-col predicted_tfidf

  # 自訂輸出路徑與模型
  python generate_summary.py --place "The Cat Cabinet" --data data/clean/The_Cat_Cabinet_predicted.csv --out summary_cat.txt --model gpt-4o-mini
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────────────────────────────────────────────
# 參數解析
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LLM 摘要生成（基於 predict.py 的預測結果）")
parser.add_argument("--place",    type=str, default="此景點/餐廳",
                    help="景點或餐廳名稱（顯示在報告中）")
parser.add_argument("--data",     type=str, required=True,
                    help="predict.py 輸出的預測 CSV 路徑（須含 text, rating, is_inconsistent, predicted_* 欄位）")
parser.add_argument("--pred-col", type=str, default="predicted_roberta",
                    choices=["predicted_roberta", "predicted_tfidf"],
                    help="使用哪個預測欄位（預設: predicted_roberta）")
parser.add_argument("--top-n",    type=int, default=5,
                    help="各類別抽取的範例評論數（預設 5）")
parser.add_argument("--out",      type=str, default=None,
                    help="摘要輸出路徑（預設：與輸入 CSV 同目錄，加 _summary.txt）")
parser.add_argument("--model",    type=str, default="gpt-4o-mini",
                    help="OpenAI 模型名稱（預設: gpt-4o-mini）")
parser.add_argument("--max-tokens",   type=int,   default=1000, help="最大輸出 token 數")
parser.add_argument("--temperature",  type=float, default=0.3,  help="生成溫度（0~1）")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────
# 預設輸出路徑
# ──────────────────────────────────────────────────────────
if args.out is None:
    base, _ = os.path.splitext(args.data)
    args.out = base + "_summary.txt"

# ──────────────────────────────────────────────────────────
# 載入 API Key
# ──────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("找不到 OPENAI_API_KEY，請在 .env 檔中設定。")

client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────
# 載入預測資料
# ──────────────────────────────────────────────────────────
if not os.path.exists(args.data):
    raise FileNotFoundError(
        f"找不到輸入檔案: {args.data}\n"
        "請先執行 predict.py 產生預測結果。"
    )

print(f"載入預測結果：{args.data}")
df = pd.read_csv(args.data)
print(f"  共 {len(df)} 筆評論")

# 確認預測欄位存在
pred_col = args.pred_col
if pred_col not in df.columns:
    available = [c for c in df.columns if c.startswith("predicted_")]
    if available:
        pred_col = available[0]
        print(f"  警告：找不到 {args.pred_col}，改用 {pred_col}")
    else:
        raise ValueError(
            f"CSV 中找不到預測欄位（{args.pred_col}）。\n"
            "請先執行 predict.py 產生 predicted_roberta 或 predicted_tfidf 欄位。"
        )

# 確認 is_inconsistent 欄位
if "is_inconsistent" not in df.columns:
    print("  警告：找不到 is_inconsistent 欄位，將依評分+預測自動推算。")
    df["is_inconsistent"] = (df["rating"] >= 4) & (df[pred_col].isin(["負面", "中立"]))


# ──────────────────────────────────────────────────────────
# LLM 摘要生成
# ──────────────────────────────────────────────────────────
def generate_summary(reviews_df: pd.DataFrame, place_name: str,
                     pred_col: str, top_n: int = 5) -> str:
    """
    根據 predict.py 的預測結果產生中文分析報告。

    Parameters
    ----------
    reviews_df : 含 pred_col / is_inconsistent / rating 欄位的 DataFrame
    place_name : 顯示在報告中的景點/餐廳名稱
    pred_col   : 預測欄位名稱（predicted_roberta 或 predicted_tfidf）
    top_n      : 各類別範例評論數量
    """
    total     = len(reviews_df)
    pos_count = (reviews_df[pred_col] == "正面").sum()
    neu_count = (reviews_df[pred_col] == "中立").sum()
    neg_count = (reviews_df[pred_col] == "負面").sum()
    inc_count = reviews_df["is_inconsistent"].sum()

    # 各類別評分分佈
    pos_avg = reviews_df[reviews_df[pred_col] == "正面"]["rating"].mean()
    neu_avg = reviews_df[reviews_df[pred_col] == "中立"]["rating"].mean()
    neg_avg = reviews_df[reviews_df[pred_col] == "負面"]["rating"].mean()

    # 抽取範例評論
    sample_pos = (reviews_df[reviews_df[pred_col] == "正面"]
                  .sample(min(top_n, pos_count), random_state=42)["text"].tolist()
                  if pos_count > 0 else [])
    sample_neu = (reviews_df[reviews_df[pred_col] == "中立"]
                  .sample(min(top_n, neu_count), random_state=42)["text"].tolist()
                  if neu_count > 0 else [])
    sample_neg = (reviews_df[reviews_df[pred_col] == "負面"]
                  .sample(min(top_n, neg_count), random_state=42)["text"].tolist()
                  if neg_count > 0 else [])
    sample_inc = (reviews_df[reviews_df["is_inconsistent"]]
                  .sample(min(top_n, int(inc_count)), random_state=42)["text"].tolist()
                  if inc_count > 0 else [])

    model_label = "RoBERTa" if "roberta" in pred_col else "TF-IDF"

    prompt = f"""
你是一個專業的旅遊/餐廳評論分析師，請根據以下由 {model_label} 情感分析模型所產生的評論資料，為「{place_name}」產生一份詳細的中文分析報告。

【統計資料】
- 總評論數: {total}
- 正面評論: {pos_count} 筆 ({pos_count/total*100:.1f}%)，平均星數 {pos_avg:.1f}
- 中立評論: {neu_count} 筆 ({neu_count/total*100:.1f}%)，平均星數 {neu_avg:.1f}
- 負面評論: {neg_count} 筆 ({neg_count/total*100:.1f}%)，平均星數 {neg_avg:.1f}
- 疑似刷星（高星低情感）: {inc_count} 筆 ({inc_count/total*100:.1f}%)

【正面評論範例（隨機抽樣）】
{chr(10).join(f'- {t[:200]}' for t in sample_pos) if sample_pos else '（無）'}

【中立評論範例（隨機抽樣）】
{chr(10).join(f'- {t[:200]}' for t in sample_neu) if sample_neu else '（無）'}

【負面評論範例（隨機抽樣）】
{chr(10).join(f'- {t[:200]}' for t in sample_neg) if sample_neg else '（無）'}

【疑似刷星評論範例（高評分但情感偏負/中立）】
{chr(10).join(f'- {t[:200]}' for t in sample_inc) if sample_inc else '（無疑似刷星評論）'}

請依照以下格式輸出分析報告：

1. 整體評價摘要（2~3句）
2. 主要優點（條列式，至少3點）
3. 主要缺點（條列式，至少2點，若無明顯缺點請說明）
4. 刷星風險評估（根據疑似刷星比例，說明是否值得注意及原因）
5. 最終建議：「值得前往」或「謹慎考慮」，並給出具體理由
"""

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_completion_tokens=args.max_tokens,
    )
    return response.choices[0].message.content


# ──────────────────────────────────────────────────────────
# 執行摘要生成
# ──────────────────────────────────────────────────────────
total = len(df)
# 若欄位不存在，自動根據 rating + 預測欄位補算
if "is_inconsistent" not in df.columns:
    df["is_inconsistent"] = (df["rating"] >= 4) & (df[pred_col].isin(["負面", "中立"]))
inconsistent = df["is_inconsistent"].sum()
pred_label   = "RoBERTa" if "roberta" in pred_col else "TF-IDF"

print("=" * 55)
print(f"LLM 摘要生成（模型：{args.model}）")
print(f"  景點/餐廳  : {args.place}")
print(f"  預測來源   : {pred_col} ({pred_label})")
print(f"  總評論數   : {total}")
print(f"  正面       : {(df[pred_col]=='正面').sum()} 筆")
print(f"  中立       : {(df[pred_col]=='中立').sum()} 筆")
print(f"  負面       : {(df[pred_col]=='負面').sum()} 筆")
print(f"  疑似刷星   : {inconsistent} 筆 ({inconsistent/total*100:.1f}%)")
print(f"\n  正在呼叫 {args.model}...\n")

summary = generate_summary(df, place_name=args.place, pred_col=pred_col, top_n=args.top_n)
print(summary)

# ──────────────────────────────────────────────────────────
# 儲存摘要
# ──────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write(f"景點/餐廳：{args.place}\n")
    f.write(f"預測來源 ：{pred_col} ({pred_label})\n")
    f.write(f"資料檔案 ：{args.data}\n")
    f.write("=" * 55 + "\n\n")
    f.write(summary)

print(f"\n  摘要已儲存至: {args.out}")
