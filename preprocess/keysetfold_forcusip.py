import pandas as pd
import json
import random
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CRSP', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id

    data_future = pd.read_csv(f'dataset/{dataset}_future.csv')
    data_history = pd.read_csv(f'dataset/{dataset}_history.csv')
    data = pd.concat([data_history, data_future])

    # 對 product_id 做自訂映射 (若尚未做，視您程式流程而定)
    data['product_id'] = data['product_id'].astype(str)
    unique_cusips = data['product_id'].unique()
    cusip_to_int = {cusip: idx for idx, cusip in enumerate(unique_cusips)}
    data['product_id'] = data['product_id'].map(cusip_to_int)

    # 取得 user
    user = list(set(data_future['user_id']))
    user_num = len(user)
    random.shuffle(user)
    user = [str(user_id) for user_id in user]

    train_user = user[:int(user_num*4/5*0.9)]
    val_user = user[int(user_num*4/5*0.9):int(user_num*4/5)]
    test_user = user[int(user_num*4/5):]

    # 注意這裡：將 numpy.int64 轉成 Python int
    item_num = int(data['product_id'].max() + 1)

    keyset_dict = {
        'item_num': item_num,
        'train': train_user,
        'val': val_user,
        'test': test_user
    }

    print(keyset_dict)
    if not os.path.exists('keyset/'):
        os.makedirs('keyset/')
    keyset_file = f'keyset/{dataset}_keyset_{fold_id}.json'
    with open(keyset_file, 'w') as f:
        json.dump(keyset_dict, f)

    print(f"Keyset file saved -> {keyset_file}")
