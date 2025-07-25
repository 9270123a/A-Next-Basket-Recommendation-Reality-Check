import pandas as pd
import json
import os

def csv_to_json_baskets(csv_path, output_json):
    """
    讀取指定 csv_path (user_id, order_number, product_id, eval_set)，
    將同一 user_id 下的 order_number (按照從小到大) 的 product_id 整理成一條籃子序列，
    並在序列最前與最後各加一個 [-1]。
    最後把結果輸出成 output_json。
    """
    df = pd.read_csv(csv_path)

    # 依 user_id、order_number 分組，將同一次 order_number 的所有 product_id 放在同一個 list
    user_dict = {}
    grouped = df.groupby(["user_id","order_number"])
    
    for (uid, order_n), grp in grouped:
        # 取出該 (user, order_number) 下所有商品
        items = grp["product_id"].tolist()

        # 在 user_dict 中放進去
        if uid not in user_dict:
            user_dict[uid] = {}  # 先用一個 {order_n: [items]} 的結構暫存
        user_dict[uid][order_n] = user_dict[uid].get(order_n, []) + items
    
    # 再把 user_dict 組裝成需要的格式：
    #   user_dict[uid] = [[-1], basket1, basket2, ..., [-1]]
    final_dict = {}
    for uid, order_map in user_dict.items():
        # 按 order_number 排序
        order_numbers = sorted(order_map.keys())
        baskets_list = []
        # 將每個 order_number 的 items(去重後) 轉成一個 list
        for o_n in order_numbers:
            items = list(set(order_map[o_n]))  # 如果需要排序，可以再 sorted()
            baskets_list.append(items)
        
        # 在最前最後加上 [-1]
        baskets_list = [[-1]] + baskets_list + [[-1]]
        final_dict[str(uid)] = baskets_list
    
    # 輸出 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_dict, f)
    print(f"Done! 已產出 {output_json}")

if __name__ == "__main__":
    # 假設您有兩個 CSV：一個 history、一個 future
    
    csv_path = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP.csv"
    history_csv = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP_history.csv"
    future_csv  = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP_future.csv"


    # 輸出 JSON 放在哪裡？
    output_folder = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\jsondata"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    history_json = os.path.join(output_folder, "CRSP_history.json")
    future_json  = os.path.join(output_folder, "CRSP_future.json")
    CRSP_json = os.path.join(output_folder,"CRSP.json" )

    # 轉檔
    csv_to_json_baskets(history_csv, history_json)
    csv_to_json_baskets(future_csv,  future_json)
    csv_to_json_baskets(csv_path,  CRSP_json)
