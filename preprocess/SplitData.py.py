import pandas as pd

# 假設這個檔已經是您處理好的 baskets 資料
df = pd.read_csv(r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP.csv")

# 確保先按照 user_id, order_number 排序
df = df.sort_values(["user_id", "order_number"])

all_users = df["user_id"].unique()

hist_list = []
fut_list = []

for u in all_users:
    sub = df[df["user_id"] == u]
    max_order = sub["order_number"].max()
    #   max_order 表示該使用者最後一次的籃子
    sub_history = sub[sub["order_number"] < max_order]  # 之前籃子
    sub_future = sub[sub["order_number"] == max_order]  # 最後一籃

    hist_list.append(sub_history)
    fut_list.append(sub_future)

# 合併
df_history = pd.concat(hist_list, ignore_index=True)
df_future = pd.concat(fut_list, ignore_index=True)

# 輸出
df_history.to_csv("dataset/CRSP_history.csv", index=False)
df_future.to_csv("dataset/CRSP_future.csv", index=False)

print("Done split into history & future!")
