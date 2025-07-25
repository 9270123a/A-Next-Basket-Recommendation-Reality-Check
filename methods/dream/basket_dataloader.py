
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Set

def get_repeat_candidates(user_baskets: List[List[List[int]]]) -> List[List[int]]:
    """
    針對每位使用者，回傳曾經買過的所有 item（去重後）。
    Args
        user_baskets : list            # len = #users
            [ [basket1, basket2, ...], # 每個 basket 是 item id list
              ... ]
    Returns
        repeat_cand : list[list[int]]  # 與 user_baskets 同長度
    """
    repeat_cand = []
    for baskets in user_baskets:
        seen: Set[int] = set()
        for basket in baskets:
            seen.update(basket)
        repeat_cand.append(list(seen))        # 轉回 list 方便後續 tensor 化
    return repeat_cand


def get_explore_candidates(user_baskets: List[List[List[int]]],
                           total_items: int) -> List[List[int]]:
    """
    針對每位使用者，回傳「尚未買過」的 item 列表 (= 探索候選)。
    Args
        user_baskets : 同上
        total_items :  int             # 資料集中 item 的總數
    Returns
        explore_cand : list[list[int]]
    """
    all_items: Set[int] = set(range(total_items))
    explore_cand = []
    for baskets in user_baskets:
        seen: Set[int] = set()
        for basket in baskets:
            seen.update(basket)
        explore_cand.append(list(all_items - seen))
    return explore_cand


class BasketDataset(Dataset):

    def __init__(self, config, mode='train'):
        # load data.

        train_file = config['train_file']
        tgt_file = config['tgt_file']
        data_config_file = config['data_config_file']
        with open(train_file, 'r') as f:
            data_all = json.load(f)
        with open(tgt_file, 'r') as f:
            tgt_all = json.load(f)
        with open(data_config_file, 'r') as f:
            data_config = json.load(f)

        self.total_num = data_config['item_num'] # item total num
        self.mode = mode
        # Could do some filtering here
        self.data = []
        self.tgt = []
        for user in data_config[mode]:
            self.data.append(data_all[user][1:-1])
            self.tgt.append(tgt_all[user][1])
        self.user_sum = len(self.data)
        self.repeat_candidates = get_repeat_candidates(self.data)
        self.explore_candidates = get_explore_candidates(self.data, self.total_num)

    def __getitem__(self, ind):
        return self.data[ind], self.tgt[ind], \
               self.repeat_candidates[ind], self.explore_candidates[ind]

    def get_batch_data(self, s, e):
        return self.data[s:e], self.tgt[s:e], \
               self.repeat_candidates[s:e], self.explore_candidates[s:e]

    def __len__(self):
        return self.user_sum

def load_datafile(train_file, tgt_file, data_config_file):

    # load data.
    with open(train_file, 'r') as f:
        data = json.load(f)
    with open(tgt_file, 'r') as f:
        tgt = json.load(f)
    with open(data_config_file, 'r') as f:
        data_config = json.load(f)

    # Could do some filtering here
    data_train = []
    tgt_train = []
    for user in data_config['train']:
        data_train.append(data[user])
        tgt_train.append(tgt[user])
    # Valid
    data_val = []
    tgt_val = []
    for user in data_config['meta']:
        data_val.append(data[user])
        tgt_val.append(tgt[user])
    #test
    data_test = []
    tgt_test = []
    for user in data_config['test']:
        data_test.append(data[user])
        tgt_test.append(tgt[user])

    return data_train, tgt_train, data_val, tgt_val, data_test, tgt_test

if __name__ =='__main__':
    train_file = '../jsondata/Instacart_baskets_train.json'
    tgt_file = '../jsondata/Instacart_baskets_test.json'
    config_file = '../jsondata/Instacart_keyset.json'
    dataset_s = BasketDataset(train_file, tgt_file, config_file)
    print(dataset_s.get_batch_data(0,4))