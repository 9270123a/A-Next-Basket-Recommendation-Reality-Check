import pandas as pd
import json
import os

def csv_to_json_baskets(csv_path, output_json):
    """
    將一個 CSV(須包含至少 user_id, order_number, product_id 欄位)
    轉成 JSON 檔，每個 user_id 對應一系列籃子：
        final_dict[user_id] = [ [-1], [一訂單的所有商品], [另一訂單的所有商品], ..., [-1] ]
    並寫出 output_json。
    """

    # 1. 讀入 CSV
    df = pd.read_csv(csv_path)
    
    # 2. 依 (user_id, order_number) 分組，收集商品
    user_dict = {}
    grouped = df.groupby(["user_id", "order_number"])
    for (uid, order_n), grp in grouped:
        items = grp["product_id"].tolist()
        # 用 setdefault 是為了簡化操作
        user_dict.setdefault(uid, {}).setdefault(order_n, []).extend(items)

    # 3. 建立最終的籃子序列
    final_dict = {}
    for uid, order_map in user_dict.items():
        order_numbers = sorted(order_map.keys())
        baskets_list = []
        for o_n in order_numbers:
            # 去重，若不需要去重可直接用 order_map[o_n]
            unique_items = list(set(order_map[o_n]))
            baskets_list.append(unique_items)

        # 在開頭與結尾各加一個 [-1]
        baskets_list = [[-1]] + baskets_list + [[-1]]

        # user_id 改成字串 (json key 都是字串無妨)
        final_dict[str(uid)] = baskets_list

    # 4. 輸出 JSON 檔
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_dict, f)
    print(f"[csv_to_json_baskets] Done. 檔案已輸出至 {output_json}")


if __name__ == "__main__":
    # 這裡你可自行修改路徑
    csv_path = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP.csv"
    output_json = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP.json"
    
    # 執行轉換
    csv_to_json_baskets(csv_path, output_json)
