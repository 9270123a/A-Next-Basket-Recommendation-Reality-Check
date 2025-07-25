import json
import sys
import os
from tqdm import tqdm

# 請根據你專案中的路徑，維持正確 import
from metric import evaluate
from data_container import get_data_loader
from load_config import get_attribute
from util import convert_to_gpu, load_model
from train.train_main import create_model

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CRSP', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--best_model_path', type=str, required=True)
    args = parser.parse_args()

    # ★ 只保留這一組，由 args 物件讀取參數
    dataset = args.dataset
    fold = args.fold_id
    model_path = args.best_model_path  # 從 args 取 best_model_path

    # 以下維持你原先邏輯
    history_path = rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\jsondata\CRSP_history.json'
    future_path = rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\jsondata\CRSP_future.json'
    keyset_path = rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\keyset\CRSP_keyset_0.json'

    pred_path = f'{dataset}_pred{fold}.json'
    truth_path = f'{dataset}_truth{fold}.json'

    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    model = create_model()
    model = load_model(model, model_path)

    data_loader = get_data_loader(
        history_path=history_path,
        future_path=future_path,
        keyset_path=keyset_path,
        data_type='test',
        batch_size=1,
        item_embedding_matrix=model.item_embedding
    )

    model.eval()

    pred_dict = {}
    truth_dict = {}
    test_key = keyset['test']
    user_ind = 0

    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(tqdm(data_loader)):
        pred_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
        # 排序取前100
        pred_list = pred_data.detach().squeeze(0).cpu().numpy().argsort()[::-1][:100].tolist()
        truth_list = truth_data.detach().squeeze(0).cpu().numpy().argsort()[::-1][:100].tolist()

        pred_dict[test_key[user_ind]] = pred_list
        truth_dict[test_key[user_ind]] = truth_list
        user_ind += 1

    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    with open(truth_path, 'w') as f:
        json.dump(truth_dict, f)
