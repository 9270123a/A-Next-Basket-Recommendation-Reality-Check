import pandas as pd
import json
import random
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CRSP', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='Fold ID')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    
    # 讀取資料
    data_future = pd.read_csv(f'dataset/{dataset}_future.csv')
    data_history = pd.read_csv(f'dataset/{dataset}_history.csv')
    data = pd.concat([data_history, data_future])
    
    # 建立 user_id 和 product_id 的映射表
    user = list(set(data_future['user_id']))
    user_num = len(user)
    random.shuffle(user)
    
    # 重要變更：確保所有 user_id 都是字串形式
    user = [str(user_id) for user_id in user]
    
    # 建立 user_id 映射，從 0 開始編碼
    # 重要變更：使用字串形式的 user_id 作為鍵
    user_dict = {user[i]: str(i) for i in range(user_num)}  # 使用字串形式作為值
    
    # 取得商品 (product_id) 的最大值，並建立映射表
    product_dict = {str(item): idx for idx, item in enumerate(data['product_id'].unique())}  # 將商品ID也轉為字串
    
    # 切割資料集
    train_user = user[:int(user_num * 4 / 5 * 0.9)]
    val_user = user[int(user_num * 4 / 5 * 0.9):int(user_num * 4 / 5)]
    test_user = user[int(user_num * 4 / 5):]
    
    # 生成 keyset 字典，並進行編碼
    # 重要變更：確保 keyset 中的用戶 ID 也是字串形式
    keyset_dict = {
        'item_num': len(product_dict),  # 商品總數
        'train': [user_dict[u] for u in train_user],  # 使用映射的 user_id（字串形式）
        'val': [user_dict[u] for u in val_user],  # 使用映射的 user_id（字串形式）
        'test': [user_dict[u] for u in test_user]  # 使用映射的 user_id（字串形式）
    }
    
    # 儲存 keyset
    print(f"Keyset generated: {keyset_dict}")
    if not os.path.exists('keyset/'):
        os.makedirs('keyset/')
    keyset_file = f'keyset/{dataset}_keyset_{fold_id}.json'
    with open(keyset_file, 'w') as f:
        json.dump(keyset_dict, f)
    
    # 儲存映射表 (user_id 和 product_id 的映射)
    user_mapping_df = pd.DataFrame({
        "original_user_id": list(user_dict.keys()),  # 原始用戶ID (字串形式)
        "new_user_id": list(user_dict.values())  # 新用戶ID (字串形式)
    })
    user_mapping_df.to_csv(f"keyset/{dataset}_user_mapping_{fold_id}.csv", index=False)
    
    item_mapping_df = pd.DataFrame({
        "original_product_id": list(product_dict.keys()),  # 原始商品ID (字串形式)
        "new_product_id": list(product_dict.values())  # 新商品ID (數字形式)
    })
    item_mapping_df.to_csv(f"keyset/{dataset}_product_mapping_{fold_id}.csv", index=False)
    
    print(f"User and product mappings saved: {dataset}_user_mapping_{fold_id}.csv, {dataset}_product_mapping_{fold_id}.csv")
    
    # 額外輸出檢查訊息，確認生成的數據格式一致性
    print(f"檢查生成的數據格式:")
    print(f"- train 用戶數量: {len(keyset_dict['train'])}")
    print(f"- val 用戶數量: {len(keyset_dict['val'])}")
    print(f"- test 用戶數量: {len(keyset_dict['test'])}")
    print(f"- 用戶 ID 格式示例: {keyset_dict['train'][:3]}")  # 檢查前三個用戶ID格式
    print(f"- 確認所有 ID 都是字串形式: {all(isinstance(uid, str) for uid in keyset_dict['train'] + keyset_dict['val'] + keyset_dict['test'])}")