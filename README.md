# TripAdvisor Review Sentiment Analyzer

以 TripAdvisor 評論資料為基礎，訓練多個情感分析模型（TF-IDF + LR、RoBERTa），並以 GPT-4o-mini 作為 ground truth 進行評估，最後用 LLM 產生景點/餐廳分析報告。

---

## 專案結構

```
tripadvisor/
├── data/
│   ├── raw/                              # 原始爬取的評論 CSV
│   │   ├── reviews_The_Cat_Cabinet.csv
│   │   ├── reviews_StoneGrill_1870-Amsterdam.csv
│   │   └── reviews_Emerald_Cave.csv
│   └── clean/                            # 清理後 / 預測結果 / 評估結果
│       ├── mining_data_clean.csv         # 訓練資料（含 sentiment 標籤）
│       ├── The_Cat_Cabinet_clean.csv
│       ├── The_Cat_Cabinet_gpt_labeled.csv
│       ├── The_Cat_Cabinet_evaluated.csv
│       ├── The_Cat_Cabinet_predicted.csv
│       ├── StoneGrill_1870_clean.csv
│       ├── StoneGrill_1870_gpt_labeled.csv
│       ├── StoneGrill_1870_evaluated.csv
│       ├── Emerald_Cave_clean.csv
│       ├── Emerald_Cave_gpt_labeled.csv
│       └── Emerald_Cave_evaluated.csv
├── logs/                                 # 執行紀錄與評估彙總
│   ├── evaluation_summary.csv
│   └── *.txt
├── saved_models/                         # 訓練好的模型
│   ├── tfidf_vectorizer.pkl
│   ├── lr_classifier.pkl
│   └── roberta_weighted/
├── train_pipeline.py                     # 訓練所有模型
├── preprocess_for_prediction.py          # 清理 raw CSV → clean CSV
├── predict.py                            # 對 clean CSV 做情感預測
├── classify_and_evaluate.py              # GPT 分類 + 模型評估
├── generate_summary.py                   # 用 LLM 產生景點分析報告
├── classify_reviews.py                   # 互動式 GPT 分類工具（原版）
└── requirements.txt
```

---

## 環境設定

### 安裝套件

```bash
pip install -r requirements.txt
```

### API Key 設定

在專案根目錄建立 `.env` 檔：

```
OPENAI_API_KEY=sk-...
```

> `.env` 已在 `.gitignore` 中，不會被 git 追蹤。

---

## 執行流程

```
raw CSV
  │
  ▼ (1) preprocess_for_prediction.py
clean CSV
  │
  ├──▶ (2) predict.py              → 模型預測（TF-IDF / RoBERTa）
  │
  ├──▶ (3) classify_and_evaluate.py → GPT 分類 + 評估指標 + 混淆矩陣
  │
  └──▶ (4) generate_summary.py     → LLM 分析報告
```

---

## 各程式說明

---

### 1. `preprocess_for_prediction.py` — 資料清理

將 `data/raw/` 中的原始評論 CSV 清理後存入 `data/clean/`。

**清理內容：**
- 移除 `reviewer`、`title` 欄位
- 將 `rating`（如 `"4 of 5 bubbles"`）轉為整數
- 拆解 `date` 欄位（分離日期與旅遊類型，處理 `\xa0` 等特殊字元）
- 標記 `date_status`（`date_with_label` / `date_only` / `unknown_format`）

**用法：**

```bash
# 預設清理 reviews_The_Cat_Cabinet.csv 和 reviews_StoneGrill_1870-Amsterdam.csv
python preprocess_for_prediction.py
```

若要新增其他 raw CSV，修改 `preprocess_for_prediction.py` 中的 `FILES` 字典：

```python
FILES = {
    "reviews_Emerald_Cave.csv": "Emerald_Cave_clean.csv",
    ...
}
```

**輸出欄位：** `rating`, `text`, `date`, `trip_type`, `date_status`

---

### 2. `train_pipeline.py` — 模型訓練

使用 `data/clean/mining_data_clean.csv` 訓練並評估四種方法，最後儲存最佳模型。

**訓練的模型：**
| 編號 | 方法 | 說明 |
|---|---|---|
| 1 | Baseline | 多數類別預測 |
| 2 | TF-IDF + LR | TF-IDF 向量化 + Logistic Regression |
| 3 | RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` fine-tune |
| 4 | RoBERTa (weighted) ★ | 加入 class weight 的改進版（最佳模型） |

**用法：**

```bash
python train_pipeline.py
```

**輸出（`saved_models/`）：**
- `tfidf_vectorizer.pkl` — TF-IDF 向量化器
- `lr_classifier.pkl` — Logistic Regression 分類器
- `roberta_weighted/` — RoBERTa fine-tuned 模型與 tokenizer

> ⚠️ 需要 GPU 或等待時間較長。使用 CPU 時 RoBERTa 訓練約需 20-30 分鐘。

---

### 3. `predict.py` — 情感預測

使用訓練好的模型對 clean CSV 做情感預測（正面 / 中立 / 負面）。

**支援兩種模型：**
- `tfidf`：TF-IDF + Logistic Regression（快速）
- `roberta`：RoBERTa weighted（較準確）
- `both`：兩者都跑

**基本用法：**

```bash
# TF-IDF 預測
python predict.py --model tfidf \
  --input data/clean/The_Cat_Cabinet_clean.csv

# RoBERTa 預測
python predict.py --model roberta \
  --input data/clean/StoneGrill_1870_clean.csv

# 兩個模型都跑（預設）
python predict.py --model both \
  --input data/clean/Emerald_Cave_clean.csv \
  --output data/clean/Emerald_Cave_predicted.csv
```

**搭配 ground truth 評估（若 CSV 有 `sentiment` 欄位）：**

```bash
python predict.py --model both \
  --input data/clean/mining_data_clean.csv \
  --label-col sentiment
```

**輸出說明：**
- `predicted_tfidf`：TF-IDF 預測結果
- `predicted_roberta`：RoBERTa 預測結果
- `is_inconsistent`：疑似刷星（4~5星但預測為負/中立）

**評估指標：**
| 情況 | 輸出指標 |
|---|---|
| 有 ground truth（`--label-col`）| Accuracy / Precision / Recall / F1（per-class + macro + weighted）+ 混淆矩陣 |
| `--model both` 且無 ground truth | 模型一致性矩陣 + Cohen's Kappa |

---

### 4. `classify_and_evaluate.py` — GPT 分類 + 完整評估

以 GPT-4o-mini 對 clean CSV 做情感分類（作為 ground truth），再用訓練好的模型預測，最後輸出完整評估指標。

**用法：**

```bash
# 預設跑 The_Cat_Cabinet 和 StoneGrill 兩個檔案，兩個模型
python classify_and_evaluate.py

# 指定單一檔案
python classify_and_evaluate.py \
  --files data/clean/Emerald_Cave_clean.csv

# 多個檔案
python classify_and_evaluate.py \
  --files data/clean/The_Cat_Cabinet_clean.csv data/clean/Emerald_Cave_clean.csv

# 跳過 GPT 分類（已有 *_gpt_labeled.csv），只跑模型評估
python classify_and_evaluate.py --skip-classify

# 只評估 RoBERTa（跳過 GPT 分類）
python classify_and_evaluate.py --model roberta --skip-classify

# 指定 GPT 模型與並行數
python classify_and_evaluate.py --gpt-model gpt-4o --workers 10
```

**參數說明：**

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--files` | The_Cat_Cabinet, StoneGrill | 要處理的 clean CSV 路徑（可多個） |
| `--model` | `both` | 評估哪個模型：`tfidf` / `roberta` / `both` |
| `--gpt-model` | `gpt-4o-mini` | 用於 ground truth 分類的 GPT 模型 |
| `--workers` | `5` | GPT 並行 thread 數 |
| `--skip-classify` | False | 跳過 GPT 分類，直接讀已有的 `*_gpt_labeled.csv` |

**輸出檔案：**
- `data/clean/*_gpt_labeled.csv`：含 `gpt_sentiment` 欄位的 ground truth 標籤
- `data/clean/*_evaluated.csv`：含 GPT 標籤 + 模型預測 + `is_inconsistent`（疑似刷星）的完整結果
- `logs/evaluation_summary.csv`：所有模型/檔案的指標彙總表

---

### 5. `generate_summary.py` — LLM 分析報告

使用 `predict.py` 的預測結果，呼叫 GPT 產生景點/餐廳的中文分析報告。

**用法：**

```bash
# 基本用法（使用 RoBERTa 預測結果）
python generate_summary.py \
  --place "The Cat Cabinet" \
  --data data/clean/The_Cat_Cabinet_predicted.csv

# 使用 TF-IDF 預測欄位
python generate_summary.py \
  --place "StoneGrill 1870 Amsterdam" \
  --data data/clean/StoneGrill_1870_predicted.csv \
  --pred-col predicted_tfidf

# 自訂輸出路徑與模型
python generate_summary.py \
  --place "Emerald Cave" \
  --data data/clean/Emerald_Cave_predicted.csv \
  --out summary_emerald.txt \
  --model gpt-4o
```

**參數說明：**

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--place` | `此景點/餐廳` | 景點或餐廳名稱（顯示在報告中） |
| `--data` | （必填）| predict.py 輸出的預測 CSV 路徑 |
| `--pred-col` | `predicted_roberta` | 使用哪個預測欄位 |
| `--top-n` | `5` | 各類別抽取的範例評論數 |
| `--out` | 自動（同輸入檔名加 `_summary.txt`）| 摘要輸出路徑 |
| `--model` | `gpt-4o-mini` | OpenAI 模型名稱 |
| `--max-tokens` | `1000` | 最大輸出 token 數 |
| `--temperature` | `0.3` | 生成溫度（0~1，越低越穩定）|

**報告內容：**
1. 整體評價摘要
2. 主要優點（條列式）
3. 主要缺點（條列式）
4. 刷星風險評估
5. 最終建議：值得前往 / 謹慎考慮

> ⚠️ 需先執行 `predict.py` 產生 `*_predicted.csv`，再執行此腳本。  
> 💡 若已執行過 `classify_and_evaluate.py`，也可直接使用 `*_evaluated.csv`（其中包含 `predicted_roberta` 與 `is_inconsistent` 欄位）。

---

## 典型執行流程（新景點）

```bash
# Step 1：清理資料（修改 FILES 字典後執行）
python preprocess_for_prediction.py

# Step 2：用模型預測
python predict.py --model both \
  --input data/clean/Emerald_Cave_clean.csv \
  --output data/clean/Emerald_Cave_predicted.csv

# Step 3：GPT 分類 + 評估（會產生 ground truth 並對比模型）
python classify_and_evaluate.py \
  --files data/clean/Emerald_Cave_clean.csv \
  --model both

# Step 4：產生 LLM 分析報告
python generate_summary.py \
  --place "Emerald Cave" \
  --data data/clean/Emerald_Cave_predicted.csv
```

---

## 評估結果摘要

以下為 GPT-4o-mini 作為 ground truth 的評估結果（Macro F1）：

| 景點 | TF-IDF Accuracy | TF-IDF Macro F1 | RoBERTa Accuracy | RoBERTa Macro F1 |
|---|---|---|---|---|
| The Cat Cabinet | 0.6848 | 0.5935 | **0.8956** | **0.8281** |
| StoneGrill 1870 | 0.8925 | 0.5598 | **0.9725** | **0.7615** |
| Emerald Cave | 0.9302 | 0.4372 | **0.9888** | **0.8425** |

> RoBERTa (weighted) 在所有指標上均優於 TF-IDF + LR。  
> Macro F1 比 Accuracy 更能反映模型在不平衡資料集上的真實表現。

---

## 注意事項

- **資料不平衡**：正面評論通常佔多數（StoneGrill 約 93%），導致 Accuracy 偏高但 Macro F1 偏低，建議以 Macro F1 作為主要指標。
- **中立類別最難分**：兩個模型在中立的 F1 均最低，因為中立語意曖昧且訓練樣本少。
- **GPT-4o-mini 的限制**：用作 ground truth 有一定噪音，尤其是「業者回覆型」評論（如 Emerald Cave 中大量的 business reply 型評論），GPT 可能判為正面，但實際上並非旅客自發評論。
