import pandas as pd
import json
import os

# 讀取處理好的 CRSP.csv
df = pd.read_csv(r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP.csv")

# 先依 (user_id, order_number) 分組，把同一次 order_number 的商品(product_id)匯集成一個清單
grouped = df.groupby(["user_id", "order_number"])["product_id"].agg(list).reset_index()

# 創建一個字典 {user_id: [ [basket1], [basket2], ... ]}
merged_dict = {}
for user_id, subdf in grouped.groupby("user_id"):
    # 按照 order_number 排序
    subdf = subdf.sort_values("order_number")
    # 取出各個籃子的 product_id list
    baskets = subdf["product_id"].tolist()  # list of lists
    # 注意：json 的 key 只能是 string，因此將 user_id 轉字串
    merged_dict[str(user_id)] = baskets

# 準備輸出檔案路徑
output_dir = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\mergeddataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "CRSP_merged.json")

# 將 merged_dict 寫入 JSON 檔
with open(output_file, "w") as f:
    json.dump(merged_dict, f)

print("Done! Merged dataset saved to:", output_file)
