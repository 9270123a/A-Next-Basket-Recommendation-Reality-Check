import pandas as pd
import json
import random
import argparse
import os


#------------------------------------------------------------#
# 1. 讀取「股票市值特徵」資料，篩選 rank<=100 的 CUSIP
#------------------------------------------------------------#
stock_info_path = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\allstock_year_data.csv"
stock_df = pd.read_csv(stock_info_path)

# 先把「任一年 lag_year_size_rank <= 100」的 CUSIP 篩出
top_stocks = stock_df[stock_df["lag_year_size_rank"] <= 100].copy()
valid_cusips = set(top_stocks["CUSIP"].unique())

# 為每個 CUSIP 做一個排序依據 (示範取最大lag_year_size)
cusip_size_map = (
    stock_df
    .groupby("CUSIP")["lag_year_size"]
    .max()    # 取每檔股票曾出現的最大市值 (可改成最新年度/平均等)
    .fillna(0)
    .to_dict()
)

#------------------------------------------------------------#
# 2. 讀取 Fund_holding，預先只留「CUSIP 在 valid_cusips」的資料
#------------------------------------------------------------#
fund_path = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\Combined_Fund_holdings.csv"
df = pd.read_csv(
    fund_path,
    usecols=["crsp_portno", "report_dt", "cusip"]
)
df = df.dropna(subset=["crsp_portno", "report_dt", "cusip"])

# 只保留在 valid_cusips 裡的紀錄
df = df[df["cusip"].isin(valid_cusips)].copy()
df["report_dt"] = pd.to_datetime(df["report_dt"], errors="coerce")
df = df.dropna(subset=["report_dt"])  # 去除無法轉日期

#------------------------------------------------------------#
# 3. 定義保留前 50 檔的函數 (以 lag_year_size 排序取最大者)
#------------------------------------------------------------#
def keep_top_50_by_size(cusip_list):
    """
    若清單長度 > 50，根據我們在 cusip_size_map 裡的大小排序，保留前 50。
    如果找不到對應就視為 0 (排最後)。
    """
    unique_cusips = list(set(cusip_list))

    # 依照 cusip_size_map 進行排序(由大到小)
    unique_cusips.sort(key=lambda x: cusip_size_map.get(x, 0), reverse=True)

    # 保留前 50
    return unique_cusips[:50]

#------------------------------------------------------------#
# 4. (user_id, date) -> basket，若超過50檔就留最大的50檔
#------------------------------------------------------------#
basket_data = (
    df.groupby(["crsp_portno", "report_dt"])["cusip"]
      .agg(keep_top_50_by_size)
      .reset_index()
      .rename(columns={
          "crsp_portno": "user_id",
          "report_dt":   "trans_date",
          "cusip":       "product_list"
      })
)

#------------------------------------------------------------#
# 5. 篩掉交易日數 <3 或 >50 的使用者
#------------------------------------------------------------#
valid_users = []
for user_id, subdf in basket_data.groupby("user_id"):
    if 3 <= len(subdf) <= 50:
        valid_users.append(user_id)

basket_data = basket_data[basket_data["user_id"].isin(valid_users)]
basket_data = basket_data.sort_values(["user_id", "trans_date"]).reset_index(drop=True)

#------------------------------------------------------------#
# 6. 多行化 (一筆一檔)，最後一次標train，其餘標prior
#------------------------------------------------------------#
rows = []
for user_id, subdf in basket_data.groupby("user_id", sort=False):
    subdf = subdf.sort_values("trans_date")
    last_date = subdf["trans_date"].iloc[-1]

    order_num = 1 
    for idx, row_ in subdf.iterrows():
        basket = row_["product_list"]
        eval_set = "prior"
        if row_["trans_date"] == last_date:
            eval_set = "train"

        for product_id in basket:
            rows.append({
                "user_id":      user_id,
                "order_number": order_num,
                "product_id":   product_id,  # 這裡為 CUSIP
                "eval_set":     eval_set
            })
        order_num += 1

baskets = pd.DataFrame(rows)
print("total transactions:", len(baskets))

#------------------------------------------------------------#
# 7. 保留「prior集中出現次數 >= 3」的商品
#------------------------------------------------------------#
history_baskets = baskets[baskets["eval_set"] == "prior"]
item_count = history_baskets["product_id"].value_counts()
valid_items = set(item_count[item_count >= 3].index)

baskets = baskets[baskets["product_id"].isin(valid_items)].reset_index(drop=True)
print("After item filter, total transactions:", len(baskets))

#------------------------------------------------------------#
# 8. 輸出 item_features：lag_year_size, lag_year_ret 等
#    取最終保留的 CUSIP，對應 stock_df 最後一筆資料
#------------------------------------------------------------#
final_cusips = baskets["product_id"].unique()
features_df = stock_df[stock_df["CUSIP"].isin(final_cusips)].copy()

# 可能同一CUSIP多筆年度資料 -> 取最後一年
features_df = (
    features_df
    .sort_values(["CUSIP", "Year"])
    .groupby("CUSIP", as_index=False)
    .tail(1)
)

item_features = features_df[[
    "CUSIP",
    "lag_year_size",
    "lag_year_ret",
    "lag_year_std",
    "lag_year_skew",
    "lag_year_95VAR"
]].drop_duplicates()

item_features.to_csv("dataset/item_features.csv", index=False)
print("item_features.csv saved, total:", len(item_features))

#------------------------------------------------------------#
# 9. 重新編碼: user_id & product_id (CUSIP) -> 整數
#    以便後續稀疏矩陣或 UPCF 方法可直接使用
#------------------------------------------------------------#
original_baskets = baskets.copy()  # 保留原始 CUSIP & user_id

# 建字典: user_id -> 新編碼
user_dict = {}
user_ind = 0

# 建字典: product_id(CUSIP) -> 新編碼
item_dict = {}
item_ind = 0

# 透過遍歷 baskets，每碰到新 user_id or product_id，就給它分配一個整數
for i in range(len(baskets)):
    u = baskets.at[i, "user_id"]
    p = baskets.at[i, "product_id"]

    if u not in user_dict:
        user_dict[u] = user_ind
        user_ind += 1
    if p not in item_dict:
        item_dict[p] = item_ind
        item_ind += 1

    baskets.at[i, "user_id"]    = user_dict[u]
    baskets.at[i, "product_id"] = item_dict[p]

#------------------------------------------------------------#
# 10. 輸出最終結果
#------------------------------------------------------------#
baskets.to_csv("dataset/CRSP.csv", index=False)
print("Final baskets saved -> dataset/CRSP.csv")

# 輸出使用者對應表
user_mapping_df = pd.DataFrame({
    "original_user_id": list(user_dict.keys()),
    "new_user_id":      list(user_dict.values())
})
user_mapping_df.to_csv("dataset/user_id_mapping.csv", index=False)

# 輸出商品對應表 (CUSIP)
item_mapping_df = pd.DataFrame({
    "original_product_id": list(item_dict.keys()),
    "new_product_id":      list(item_dict.values())
})
item_mapping_df.to_csv("dataset/product_id_mapping.csv", index=False)

# 保存原始ID版資料
original_baskets.to_csv("dataset/CRSP_original_ids.csv", index=False)
print("All done.")

