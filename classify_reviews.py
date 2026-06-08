import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm
import concurrent.futures
import time

def classify_text(client, title, text, model="gpt-5.4-mini", retries=3):
    # 處理可能為空的標題或評論
    title_str = str(title) if pd.notna(title) else ""
    text_str = str(text) if pd.notna(text) else ""
    
    prompt = f"""請分析以下評論的情感，將其分類為以下三種之一：「正面」(Positive)、「負面」(Negative) 或「中立」(Neutral)。
請「只」輸出這三個選項其中之一的中文字（正面、負面、中立），不要有任何其他解釋。

標題：{title_str}
評論內容：{text_str}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一個精準的情感分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_completion_tokens=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2) # 等待後重試
            else:
                return f"Error: {str(e)}"

def process_file(file_path, api_key, max_workers=5):
    print(f"正在處理檔案：{file_path}")
    
    # 初始化 OpenAI Client
    client = OpenAI(api_key=api_key)
    
    # 讀取資料
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        print("不支援的檔案格式，請提供 .csv 或 .json 檔案")
        return

    # 確保 'text' 欄位存在
    if 'text' not in df.columns:
        print("找不到 'text' 欄位，無法進行分析")
        return
        
    results = [""] * len(df)
    
    # 使用 ThreadPoolExecutor 來加速請求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 將任務提交給 executor
        futures = {
            executor.submit(classify_text, client, row.get('title'), row.get('text')): idx
            for idx, row in df.iterrows()
        }
        
        # 使用 tqdm 顯示進度條
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="分類進度"):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                results[idx] = f"Error: {str(e)}"
                
    # 將結果存入 DataFrame
    df['sentiment'] = results
    
    # 儲存結果
    output_path = file_path.rsplit('.', 1)[0] + "_classified.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"處理完成！結果已儲存至：{output_path}\n")


if __name__ == "__main__":
    print("=== OpenAI 評論情感分類器 ===")
    api_key = input("請輸入您的 OpenAI API Key: ")
    
    # 這裡會列出當前目錄下的 CSV 與 JSON 檔案
    files = [f for f in os.listdir('.') if f.endswith('.csv') or f.endswith('.json')]
    
    if not files:
        print("當前目錄下找不到任何 .csv 或 .json 資料集。")
    else:
        print("\n找到以下資料集：")
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
            
        choices = input("\n請輸入要處理的檔案編號 (例如: 0，或輸入 'all' 處理全部): ")
        
        if choices.lower() == 'all':
            selected_files = files
        else:
            try:
                selected_indices = [int(c.strip()) for c in choices.split(',')]
                selected_files = [files[i] for i in selected_indices]
            except:
                print("輸入無效！")
                selected_files = []
                
        # 處理選擇的檔案
        for file in selected_files:
            process_file(file, api_key)
