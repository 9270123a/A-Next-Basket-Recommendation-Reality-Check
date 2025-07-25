import pandas as pd

# 1. 讀取您整理好的 "history" 檔 (假設包含至少 user_id, product_id 欄位)
df = pd.read_csv(r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP.csv")

# 2. 依 product_id 統計出現次數
freq_df = df.groupby("product_id").size().reset_index(name="count")

# 3. 依 count 排序 (由大至小)，方便查看最熱門商品
freq_df = freq_df.sort_values("count", ascending=False)

# 4. 輸出成 CSV：其中就會有 2 欄：product_id, count
freq_df.to_csv("dataset\CRSP_pop.csv", index=False)

print("Done! 已經產生 dataset/CRSP_pop.csv，內容含 product_id,count。")
