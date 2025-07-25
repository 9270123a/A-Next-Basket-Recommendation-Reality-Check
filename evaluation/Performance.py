import os
import numpy as np
import math
import pandas as pd
import json
from metrics import *  # 如果 metrics 就在同一支檔案，可省略此行。否則自行 import.
from tqdm import tqdm  # 用於顯示進度條

model_name = "DNNTSP"

def get_repeat_eval(pred_folder, dataset, size, fold_list, file):
    """
    pred_folder: 預測檔所在的資料夾 (雖然範例中還沒實際用到)
    dataset: 資料集名稱 (如 'CRSP')
    size: 評估的 top-K
    fold_list: 要處理的 fold id 清單 (如 [0, 1, 2])
    file: 寫入評估結果的檔案物件
    """
    #=== 1) 讀取檔案路徑（示範使用 CRSP_*）===#
    history_file = r'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\jsondata\CRSP_history.json'
    truth_file   = r'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\jsondata\CRSP_future.json'
    keyset_file  = r'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\keyset\CRSP_keyset_0.json'
    pred_file = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\methods\dnntsp\train\CRSP_pred0.json"

    #=== 2) 載入資料 ===#
    # 2.1 載入 future.json
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)
    # 2.2 載入歷史 CSV，並轉成 user -> set of products 字典 (提升效率)
    df_history = pd.read_csv(r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP_history.csv")
    history_dict = {}
    for row in df_history.itertuples(index=False):
        u = row.user_id
        p = row.product_id
        if u not in history_dict:
            history_dict[u] = set()
        history_dict[u].add(p)

    #=== 3) 準備統計容器 ===#
    a_ndcg = []
    a_recall = []
    a_hit = []
    a_repeat_ratio = []
    a_explore_ratio = []
    a_recall_repeat = []
    a_recall_explore = []
    a_hit_repeat = []
    a_hit_explore = []

    #=== 4) 逐個 fold_id 進行評估 ===#
    for ind in fold_list:
        # 載入 keyset (train/val/test user)
        with open(keyset_file, 'r') as f:
            keyset = json.load(f)
        # 載入預測結果
        with open(pred_file, 'r') as f:
            data_pred = json.load(f)

        # 以下是本 fold 的計算
        ndcg = []
        recall = []
        hit = []
        repeat_ratio = []
        explore_ratio = []
        recall_repeat = []
        recall_explore = []
        hit_repeat = []
        hit_explore = []

        #=== 5) 在測試使用者清單上做迴圈，加 tqdm 進度條 ===#
        test_users = keyset['test']  # 這通常是字串清單
        for user in tqdm(test_users, desc=f"Fold {ind}, K={size}"):
            # 1) 有些使用者可能不在預測結果中
            if user not in data_pred:
                continue

            # 2) 取得預測清單
            pred = data_pred[user]
            # 3) 取得真實籃子 (此處假設 future.json 結構是 [[-1],[basket],[-1]] → 取 index=1)
            truth = data_truth[user][1]

            # 4) 取得該使用者在 history.csv 裡的商品
            # 注意 user 是字串, history_dict 的 key 可能是 int, 所以要轉型
            u_int = int(user)
            repeat_items = history_dict.get(u_int, set())

            #--- 切分 repeat vs. explore
            truth_repeat = list(set(truth) & repeat_items)
            truth_explore = list(set(truth) - repeat_items)

            #--- 計算常用指標
            u_ndcg = get_NDCG(truth, pred, size)
            ndcg.append(u_ndcg)
            u_recall = get_Recall(truth, pred, size)
            recall.append(u_recall)
            u_hit = get_HT(truth, pred, size)
            hit.append(u_hit)

            #--- repeat / explore ratio
            u_repeat_ratio, u_explore_ratio = get_repeat_explore(repeat_items, pred, size)
            repeat_ratio.append(u_repeat_ratio)
            explore_ratio.append(u_explore_ratio)

            if len(truth_repeat) > 0:
                recall_repeat.append(get_Recall(truth_repeat, pred, size))
                hit_repeat.append(get_HT(truth_repeat, pred, size))

            if len(truth_explore) > 0:
                recall_explore.append(get_Recall(truth_explore, pred, size))
                hit_explore.append(get_HT(truth_explore, pred, size))

        #=== 6) 這個 fold 的結果加總 ===#
        a_ndcg.append(np.mean(ndcg))
        a_recall.append(np.mean(recall))
        a_hit.append(np.mean(hit))
        a_repeat_ratio.append(np.mean(repeat_ratio))
        a_explore_ratio.append(np.mean(explore_ratio))
        a_recall_repeat.append(np.mean(recall_repeat))
        a_recall_explore.append(np.mean(recall_explore))
        a_hit_repeat.append(np.mean(hit_repeat))
        a_hit_explore.append(np.mean(hit_explore))

        file.write(f"Fold {ind}, recall={np.mean(recall):.4f}\n")

    #=== 7) 最終統計平均值並輸出 ===#
    print('basket size:', size)
    print('recall, ndcg, hit:', np.mean(a_recall), np.mean(a_ndcg), np.mean(a_hit))
    print('repeat-explore ratio:', np.mean(a_repeat_ratio), np.mean(a_explore_ratio))
    print('repeat-explore recall', np.mean(a_recall_repeat), np.mean(a_recall_explore))
    print('repeat-explore hit:', np.mean(a_hit_repeat), np.mean(a_hit_explore))

    file.write(f"basket size: {size}\n")
    file.write(f"recall, ndcg, hit: {np.mean(a_recall):.4f} {np.mean(a_ndcg):.4f} {np.mean(a_hit):.4f}\n")
    file.write(f"repeat-explore ratio: {np.mean(a_repeat_ratio):.4f} {np.mean(a_explore_ratio):.4f}\n")
    file.write(f"repeat-explore recall: {np.mean(a_recall_repeat):.4f} {np.mean(a_recall_explore):.4f}\n")
    file.write(f"repeat-explore hit: {np.mean(a_hit_repeat):.4f} {np.mean(a_hit_explore):.4f}\n\n")

    return np.mean(a_recall)

if __name__ == '__main__':
    # 可以改用 argparse 讀取 pred_folder, fold_list, output 等參數
    pred_folder = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\pred"
    fold_list = [0]  # 設定要評估的 fold ID
    dataset_list = ["CRSP"]

    #=== 自動檢查／建立輸出結果資料夾 ===#
    eval_folder = "eval_out"
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    #=== 建立輸出檔路徑 ===#
    eval_file = os.path.join(eval_folder, f"eval_results_{model_name}.txt")
    with open(eval_file, "w") as f:
        for dataset in dataset_list:
            f.write(f"############ {dataset} - {model_name} ############ \n")
            # 跑不同 K 值
            get_repeat_eval(pred_folder, dataset, 10, fold_list, f)
            get_repeat_eval(pred_folder, dataset, 20, fold_list, f)

    print(f"評估結果已寫入 {eval_file}")
