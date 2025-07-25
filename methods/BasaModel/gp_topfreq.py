import pandas as pd
import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CRSP', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    data_history = pd.read_csv(rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP_history.csv')
    data_future = pd.read_csv(rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP_future.csv')

    g_top_file = rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\dataset\CRSP_pop.csv'
    g_top = pd.read_csv(g_top_file)
    g_top_list = g_top['product_id'].to_list()

    # data_history = pd.read_csv(f'../../dataset/{dataset}_history.csv')
    # data_future = pd.read_csv(f'../../dataset/{dataset}_future.csv')

    pred_dict = dict()
    for user, user_data in data_future.groupby('user_id'):
        user_history = data_history[data_history['user_id'].isin([user])]
        history_items = user_history['product_id'].tolist()
        s_pop_dict = dict()
        for item in history_items:
            if item not in s_pop_dict.keys():
                s_pop_dict[item] = 1
            else:
                s_pop_dict[item] += 1
        s_dict = sorted(s_pop_dict.items(), key=lambda d: d[1], reverse=True)
        pred = []
        for item, cnt in s_dict:
            pred.append(item)

        ind = 0
        while(len(pred)<100):
            if g_top_list[ind] not in pred:
                pred.append(g_top_list[ind])
            ind += 1
        pred_dict[user] = pred

    with open(r'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\methods\g-p-gp-topfreq\result\gp_top_pred.json', 'w') as f:
        json.dump(pred_dict, f)



