"""
Preprocessing Script for Prediction Data
將 reviews_The_Cat_Cabinet.csv 和 reviews_StoneGrill_1870-Amsterdam.csv
清理並輸出成可直接送入訓練好模型進行預測的格式。

輸出：
  - The_Cat_Cabinet_clean.csv
  - StoneGrill_1870_clean.csv

欄位：rating (int), text (str), date (str), trip_type (str), date_status (str)
注意：無 sentiment 欄位，因為這批資料是要做預測用的。
"""

import re
import pandas as pd

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"

FILES = {
    "reviews_Emerald_Cave.csv": "Emerald_Cave_clean.csv",
    "reviews_The_Cat_Cabinet.csv":           "The_Cat_Cabinet_clean.csv",
    "reviews_StoneGrill_1870-Amsterdam.csv": "StoneGrill_1870_clean.csv",
}


# ──────────────────────────────────────────────
# 輔助函式
# ──────────────────────────────────────────────

def parse_rating(rating_series: pd.Series) -> pd.Series:
    """把 '4 of 5 bubbles' 等字串轉成整數星數。"""
    def _to_int(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        m = re.match(r"^(\d)", s)
        if m:
            return int(m.group(1))
        return None
    return rating_series.apply(_to_int)


def parse_date_and_trip(date_series: pd.Series):
    """
    原始 date 欄可能長這樣：
        'Apr 2026 • Solo'   (date + trip_type)
        'Apr 2026'          (date only)
        '26-Mar'            (短格式，無法解析)

    另外 StoneGrill 裡有 non-breaking space (\xa0) 取代一般空白，需先正規化。

    回傳三個 Series：date_str, trip_type, date_status
    """
    # 正規化：把 \xa0 換成普通空白，再去除首尾空白
    normalized = date_series.astype(str).str.replace("\xa0", " ", regex=False).str.strip()

    # 以 • 分割
    split = normalized.str.split(r"\s*[•·]\s*", n=1, expand=True, regex=True)
    date_raw = split[0].str.strip()
    trip_raw = split[1].str.strip() if 1 in split.columns else pd.Series([""] * len(split), index=split.index, dtype=str)

    # 嘗試解析日期
    parsed = pd.to_datetime(date_raw, format="%b %Y", errors="coerce")

    date_status = pd.Series(["unknown_format"] * len(date_raw), dtype=str)
    date_status[trip_raw.notna() & (trip_raw != "")] = "date_with_label"
    date_status[(trip_raw.isna() | (trip_raw == "")) & parsed.notna()] = "date_only"

    trip_out = trip_raw.where(trip_raw != "", other=None)
    return date_raw, trip_out, date_status


def clean_text(text_series: pd.Series) -> pd.Series:
    """基本文字清理：去除首尾空白，移除過短（< 3 字元）的評論。"""
    cleaned = text_series.astype(str).str.strip()
    return cleaned


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"Processing: {input_path}")
    print(f"{'='*55}")

    df = pd.read_csv(input_path)

    # ── 1. 刪除不需要的欄位 ──────────────────────
    drop_cols = [c for c in ["reviewer", "title"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  原始筆數: {len(df)}")
    print(f"  原始欄位: {df.columns.tolist()}")

    # ── 2. 清理 rating ──────────────────────────
    df["rating"] = parse_rating(df["rating"])
    invalid_rating = df["rating"].isna().sum()
    if invalid_rating:
        print(f"  [警告] 無法解析的 rating: {invalid_rating} 筆，將刪除")
        df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    print(f"  Rating 分佈: {df['rating'].value_counts().sort_index(ascending=False).to_dict()}")

    # ── 3. 清理 text ────────────────────────────
    df["text"] = clean_text(df["text"])
    short_mask = df["text"].str.len() < 3
    if short_mask.sum():
        print(f"  [警告] 文字過短（< 3 字元）: {short_mask.sum()} 筆，將刪除")
        df = df[~short_mask]

    # ── 4. 處理 date / trip_type ────────────────
    date_str, trip_type, date_status = parse_date_and_trip(df["date"])
    df["date"]        = date_str
    df["trip_type"]   = trip_type
    df["date_status"] = date_status

    unknown   = (date_status == "unknown_format").sum()
    with_trip = (date_status == "date_with_label").sum()
    date_only = (date_status == "date_only").sum()
    print(f"  日期解析: date_with_label={with_trip}, date_only={date_only}, unknown_format={unknown}")

    # ── 5. 重新排列欄位 ─────────────────────────
    df = df[["rating", "text", "date", "trip_type", "date_status"]].reset_index(drop=True)

    # ── 6. 輸出 ─────────────────────────────────
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 清理後筆數: {len(df)}")
    print(f"  [OK] 已儲存至: {output_path}")
    print()
    df.info()
    return df


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs(CLEAN_DIR, exist_ok=True)

    results = {}
    for src, dst in FILES.items():
        src_path = os.path.join(RAW_DIR, src)
        dst_path = os.path.join(CLEAN_DIR, dst)
        results[dst] = preprocess(src_path, dst_path)

    print("\n" + "=" * 55)
    print("DONE -- Clean files saved to data/clean/:")
    for dst, df in results.items():
        print(f"  {dst}  =>  {len(df)} rows, cols: {df.columns.tolist()}")
