# -*- coding: utf-8 -*-
"""
Fully patched version of **sets2sets_new.py**
===========================================
Changes made
------------
1. **Parameter rename** – avoid accidental shadowing of `k`:
   * `decoding_next_k_step(..., next_k_step, activate_codes_num)`
2. **Loop bounds** – always iterate with `next_k_step` or the real top‑k size.
3. **Safe top‑k** – compute `actual_topk = min(user_topk, output_dim)` before every `topk()` call.
4. **Remove duplicated zero‑dim check** & minor clean‑ups.

Everything else (model definition, training pipeline, CLI) stays the same.
"""

import math
import os
import sys
import time
import json
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ---------------------------------------------------------------------------
# Global hyper‑parameters
# ---------------------------------------------------------------------------
num_iter = 20            # epochs
hidden_size = 32
num_layers = 1

use_embedding = 1        # 1 → use nn.Embedding for basket items
use_linear_reduction = 0 # 1 → project one‑hot into hidden_size via Linear

use_dropout = 0
use_average_embedding = 1

labmda = 10              # weight for set‑ranking loss
MAX_LENGTH = 100         # max sequence length fed into GRU
learning_rate = 1e-3

use_cuda = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderRNN_new(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.reduction = nn.Linear(input_size, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size if (use_embedding or use_linear_reduction) else input_size,
            hidden_size,
            num_layers
        )

    def forward(self, basket: List[int], hidden):
        """`basket` is a list of item IDs (integers)."""

        if use_embedding:
                        # 1) 先把 basket 轉為 idx_tensor
            idx_tensor = torch.LongTensor(basket).view(-1, 1)
            if use_cuda:
                idx_tensor = idx_tensor.cuda()

            # ---- 防呆檢查：空籃、負值、超界 ----
            if idx_tensor.numel() == 0:
                raise ValueError("Encoder got an empty basket! (decoder_input = [])")

            for idx in idx_tensor:
                id_val = idx.item()
                if id_val < 0 or id_val >= self.input_size:          # ← 多了 < 0 檢查
                    raise ValueError(f"illegal item id {id_val}  (embedding size = {self.input_size})")

            # ---- 累加 / 平均 embedding ----
            emb_sum = torch.zeros(1, 1, self.hidden_size, device=idx_tensor.device)
            for idx in idx_tensor:
                emb_sum += self.embedding(idx).view(1, 1, -1)
            if use_average_embedding:
                emb_sum /= self.hidden_size
            embedding = emb_sum                                # (1, 1, hidden_size)

        else:
            # 不用 embedding，直接用 one-hot
            device_ = basket.device if hasattr(basket, "device") else torch.device("cpu")
            one_hot = torch.zeros(self.input_size, device=device_)
            one_hot[basket] = 1
            reduced = (self.reduction(one_hot.unsqueeze(0))
                       if use_linear_reduction else one_hot.unsqueeze(0))
            embedding = reduced.unsqueeze(0)  # shape: (1, 1, hidden_size)

        # 執行 GRU
        output, hidden = self.gru(embedding, hidden)
        return output, hidden

    def initHidden(self):
        tensor = torch.zeros(num_layers, 1, self.hidden_size)
        return tensor.cuda() if use_cuda else tensor

# ---------------------------------------------------------------------------
# Attention Decoder
# ---------------------------------------------------------------------------
class AttnDecoderRNN_new(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout_p: float = 0.2,
                 max_length: int = MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn1 = nn.Linear(hidden_size + output_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_combine5 = nn.Linear(output_size, output_size)

        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    # ---------------------------------------------------------------------
    def forward(self, basket: List[int], hidden, encoder_outputs, history_record):
        """Return softmax distribution over all items."""
        device = hidden.device
        idx_tensor = torch.LongTensor(basket).view(-1, 1).to(device)
        emb_sum = torch.zeros(1, 1, hidden_size, device=device)
        for idx in idx_tensor:
            emb_sum += self.embedding(idx).view(1, 1, -1)
        if use_average_embedding:
            emb_sum /= hidden_size

        if use_dropout:
            emb_sum = self.dropout(emb_sum)

        history_ctx = torch.FloatTensor(history_record).view(1, -1).to(device)

        attn_weights = F.softmax(self.attn(torch.cat((emb_sum[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((emb_sum[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        logits = self.out(output[0])
        mask = (history_ctx != 0).float()
        weight = torch.sigmoid(self.attn_combine5(history_ctx))
        one_vec = torch.ones_like(history_ctx)
        logits = logits * (one_vec - mask * weight) + history_ctx * weight
        probs = F.softmax(logits, dim=1)
        return probs, hidden

    def initHidden(self):
        tensor = torch.zeros(num_layers, 1, self.hidden_size)
        return tensor.cuda() if use_cuda else tensor

# ---------------------------------------------------------------------------
# Loss (unchanged)
# ---------------------------------------------------------------------------
class custom_MultiLabelLoss_torch(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights):
        """
        pred   : (B, items_total)  — 已做 softmax
        target : (B, items_total)  — one‑hot 多標籤
        weights: (B, items_total)  — 逆頻率權重
        """
        # ① 先做帶權 MSE（跟原版相同）
        mse = torch.sum(weights * (pred - target).pow(2))

        # ② 取出正、負樣本分數
        ones_idx  = (target == 1).nonzero(as_tuple=False)   # (m, 2) : [batch,col]
        zeros_idx = (target == 0).nonzero(as_tuple=False)   # (n, 2)

        if len(ones_idx) == 0 or len(zeros_idx) == 0:
            return mse   # 這 batch 全正或全負，跳 ranking loss

        pos = pred.index_select(1, ones_idx[:, 1])   # (B, n₁)
        neg = pred.index_select(1, zeros_idx[:, 1])  # (B, n₀)

        # ③ pairwise difference： broadcasting，無需 repeat
        #    shape → (B, n₁, n₀)
        diff = -(pos.unsqueeze(2) - neg.unsqueeze(1))
        exp_loss = torch.exp(diff).sum()

        # ④ normalize & 加權
        pair_loss = labmda * exp_loss / (pos.size(1) * neg.size(1))
        return mse + pair_loss

# ---------------------------------------------------------------------------
# Helper: safe top‑k decoding
# ---------------------------------------------------------------------------

def decoding_next_k_step(encoder, decoder, input_seq, target_seq, output_size: int,
                         next_k_step: int, activate_codes_num: int):
    """Decode `next_k_step` baskets; return (decoded_indices, prob_rank_list)."""
    device = next(encoder.parameters()).device
    encoder_hidden = encoder.initHidden().to(device)

    # ---------- Encode history ----------
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    history_record = np.zeros(output_size)

    for t, basket in enumerate(input_seq):
        # 跳過起始符 (t == 0) 以及序列尾端 [-1]
        if t == 0 or basket == [-1]:
            continue
        for item in basket:
            history_record[item] += 1
        out, encoder_hidden = encoder(basket, encoder_hidden)
        encoder_outputs[t - 1] = out[0, 0]
    history_record /= max(1, len(input_seq) - 2)

    # ---------- Decode ----------
    decoder_input = input_seq[-2]        # last real basket
    decoder_hidden = encoder_hidden
    decoded_vectors, prob_vectors = [], []
    user_topk = 400                      # hard‑coded upper bound

    for step in range(next_k_step):
        probs, decoder_hidden = decoder(decoder_input, decoder_hidden,
                                        encoder_outputs, history_record)
        output_dim = probs.size(-1)
        if output_dim == 0:              # no candidate item
            return decoded_vectors, prob_vectors

        actual_topk = min(user_topk, output_dim)
        topv, topi = probs.data.topk(actual_topk)

        # decide how many items go into this predicted basket
        vectorized_target = np.zeros(output_size)
        for idx in target_seq[step + 1]:
            vectorized_target[idx] = 1
        pick_num = activate_codes_num if activate_codes_num > 0 else int(vectorized_target.sum())

        basket_pred = [topi[0][i].item() for i in range(min(pick_num, actual_topk))]
        decoded_vectors.append(basket_pred)
        prob_vectors.append([topi[0][i].item() for i in range(actual_topk)])

        decoder_input = basket_pred       # teacher‑forcing with predicted basket

    return decoded_vectors, prob_vectors


def train(input_variable, target_variable, encoder, decoder,
          codes_inverse_freq, encoder_optimizer, decoder_optimizer,
          criterion, output_size, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_variable)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size,
                                  device=next(encoder.parameters()).device)

    # -------- 建歷史頻率 --------
    history_record = np.zeros(output_size)
    for t in range(1, input_length-1):            # 第 0 個是 [-1] 起始符
        for item in input_variable[t]:
            history_record[item] += 1.0 / (input_length-2)

    # -------- Encoder --------
    for t in range(1, input_length-1):
        encoder_output, encoder_hidden = encoder(input_variable[t], encoder_hidden)
        encoder_outputs[t-1] = encoder_output[0,0]

    # -------- Decoder (只解下一籃) --------
    decoder_input  = input_variable[-2]           # 最後一個真實籃
    decoder_hidden = encoder_hidden
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,
                                            encoder_outputs, history_record)

    # -------- Loss --------
    target_vec = np.zeros(output_size)
    for idx in target_variable[1]:
        target_vec[idx] = 1
    target = torch.FloatTensor(target_vec).view(1,-1).to(decoder_output.device)
    weights = torch.FloatTensor(codes_inverse_freq).view(1,-1).to(decoder_output.device)

    loss = criterion(decoder_output, target, weights)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()

# ============================================================
# 2. 小工具：印出剩餘時間
# ============================================================
def asMinutes(s):
    m = math.floor(s/60); s -= m*60
    return f'{m}m {int(s)}s'
def timeSince(start, pct):
    now = time.time(); s = now-start
    return f'{asMinutes(s)} (- {asMinutes(s/pct - s)})'

# ============================================================
# 3. 訓練迴圈（epoch）
# ============================================================
def trainIters(hist_data, fut_data, output_size,
               encoder, decoder, model_name,
               train_keys, val_keys, codes_inv_freq,
               next_k_step, n_epochs, top_k):
    os.makedirs("./models", exist_ok=True)

    start      = time.time()
    best_recall= 0.0
    criterion  = custom_MultiLabelLoss_torch()
    enc_opt    = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    dec_opt    = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        random.shuffle(train_keys)
        running_loss = 0.0

        for uid in tqdm(train_keys, desc=f'Epoch {epoch}'):
            loss = train(hist_data[uid], fut_data[uid],
                         encoder, decoder, codes_inv_freq,
                         enc_opt, dec_opt, criterion, output_size)
            running_loss += loss

        avg_loss = running_loss / len(train_keys)
        print(f'{timeSince(start,(epoch+1)/n_epochs)}  Loss={avg_loss:.4f}')

        # -------- 驗證 --------
        recall, ndcg, hr = evaluate(hist_data, fut_data,
                                    encoder, decoder, output_size,
                                    val_keys, next_k_step, top_k)
        if recall > best_recall:
            best_recall = recall
            torch.save(encoder, f'./models/encoder_{model_name}_best.pt')
            torch.save(decoder, f'./models/decoder_{model_name}_best.pt')
            print(f'  ↳ new best recall={best_recall:.4f}  (model saved)')

# ============================================================
# 4. 反向計數用的權重 (跟原版相同，放這裡方便呼叫)
# ============================================================
def get_codes_frequency_no_vector(history_data, num_dim, key_set):
    freq = np.zeros(num_dim)
    for uid in key_set:
        for basket in history_data[uid]:
            if basket == [-1]: continue
            for item in basket:
                freq[item] += 1
    return freq

def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        # print('postivie is 0')
    else:
        precision = correct / positive
    if 0 == truth:
        recall = 0
        flag = 1
        # print('recall is 0')
    else:
        recall = correct / truth

    if flag == 0 and precision + recall > 0:
        F = 2 * precision * recall / (precision + recall)
    else:
        F = 0
    return precision, recall, F, correct


def get_F_score(prediction, test_Y):
    jaccard_similarity = []
    prec = []
    rec = []

    count = 0
    for idx in range(len(test_Y)):
        pred = prediction[idx]
        T = 0
        P = 0
        correct = 0
        for id in range(len(pred)):
            if test_Y[idx][id] == 1:
                T = T + 1
                if pred[id] == 1:
                    correct = correct + 1
            if pred[id] == 1:
                P = P + 1

        if P == 0 or T == 0:
            continue
        precision = correct / P
        recall = correct / T
        prec.append(precision)
        rec.append(recall)
        if correct == 0:
            jaccard_similarity.append(0)
        else:
            jaccard_similarity.append(2 * precision * recall / (precision + recall))
        count = count + 1

    print(
        'average precision: ' + str(np.mean(prec)))
    print('average recall : ' + str(
        np.mean(rec)))
    print('average F score: ' + str(
        np.mean(jaccard_similarity)))


def get_DCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1

    return dcg


def get_NDCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(min(num_real_item, k))
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_HT(groundtruth, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            return 1
        count += 1

    return 0


def evaluate(history_data, future_data, encoder, decoder, output_size, test_key_set, next_k_step, activate_codes_num):
    #activate_codes_num: pick top x as the basket.
    prec = []
    rec = []
    F = []
    prec1 = []
    rec1 = []
    F1 = []
    prec2 = []
    rec2 = []
    F2 = []
    prec3 = []
    rec3 = []
    F3 = []

    NDCG = []
    n_hit = 0
    count = 0

    for iter in range(len(test_key_set)):
        # training_pair = training_pairs[iter - 1]
        # input_variable = training_pair[0]
        # target_variable = training_pair[1]
        input_variable = history_data[test_key_set[iter]]
        target_variable = future_data[test_key_set[iter]]

        if len(target_variable) < 2 + next_k_step:
            continue
        count += 1
        output_vectors, prob_vectors = decoding_next_k_step(encoder, decoder, input_variable, target_variable,
                                                            output_size, next_k_step, activate_codes_num)

        hit = 0
        for idx in range(len(output_vectors)):
            # for idx in [2]:
            vectorized_target = np.zeros(output_size)
            for ii in target_variable[1 + idx]: #target_variable[[-1], [item, item], .., [-1]]
                vectorized_target[ii] = 1

            vectorized_output = np.zeros(output_size)
            for ii in output_vectors[idx]:
                vectorized_output[ii] = 1

            precision, recall, Fscore, correct = get_precision_recall_Fscore(vectorized_target, vectorized_output)
            prec.append(precision)
            rec.append(recall)
            F.append(Fscore)
            if idx == 0:
                prec1.append(precision)
                rec1.append(recall)
                F1.append(Fscore)
            elif idx == 1:
                prec2.append(precision)
                rec2.append(recall)
                F2.append(Fscore)
            elif idx == 2:
                prec3.append(precision)
                rec3.append(recall)
                F3.append(Fscore)
            # length[idx] += np.sum(target_variable[1 + idx])
            # prob_vectors is the probability
            target_topi = prob_vectors[idx]
            hit += get_HT(vectorized_target, target_topi, activate_codes_num)
            ndcg = get_NDCG(vectorized_target, target_topi, activate_codes_num)
            NDCG.append(ndcg)
        if hit == next_k_step:
            n_hit += 1

    return np.mean(rec), np.mean(NDCG), n_hit / len(test_key_set)


def get_codes_frequency_no_vector(history_data, num_dim, key_set):
    result_vector = np.zeros(num_dim)
    #pid is users id
    for pid in key_set:
        for basket in history_data[pid]:
            if basket == [-1]:
                continue
            for idx in basket:          # <-- 展開 basket
                result_vector[idx] += 1
    return result_vector
def scan_max_id(history, future):
    m = 0
    for seq in list(history.values()) + list(future.values()):
        for basket in seq:
            if basket == [-1]: continue
            m = max(m, max(basket))
    return m


def main(argv):
    dataset = argv[1]
    fold = argv[2]
    topk = int(argv[3])
    training = int(argv[4])

    encoder_pathes = f"./models/encoder_{dataset}{fold}_best.pt"
    decoder_pathes = f"./models/decoder_{dataset}{fold}_best.pt"
    
    directory = './amodels/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    dataset = argv[1]
    ind = argv[2]
    history_file = '../../jsondata/'+dataset+'_history.json'
    future_file = '../../jsondata/'+dataset+'_future.json'
    keyset_file = '../../keyset/'+dataset+'_keyset_'+str(ind)+'.json'
    model_version = dataset+str(ind)
    topk = int(argv[3])
    training = int(argv[4])

    next_k_step = 1
    with open(history_file, 'r') as f:
        history_data = json.load(f)
    with open(future_file, 'r') as f:
        future_data = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    def scan_max_id(hist, fut):
        m = 0
        for seq in list(hist.values()) + list(fut.values()):
            for basket in seq:
                if basket == [-1]:           # 起始符略過
                    continue
                m = max(m, *basket)          # basket 是 list[int]
        return m

    max_id   = scan_max_id(history_data, future_data)
    input_size = max_id + 1                 # 嚴格 = 最大 ID + 1
    print(f"[info] max item id = {max_id} → input_size = {input_size}")

    training_key_set = keyset['train']
    val_key_set = keyset['val']
    test_key_set = keyset['test']

    # weights is inverse personal top frequency. normalized by max freq.
    weights = np.zeros(input_size)
    codes_freq = get_codes_frequency_no_vector(history_data, input_size, future_data.keys())
    max_freq = max(codes_freq)
    for idx in range(len(codes_freq)):
        if codes_freq[idx] > 0:
            weights[idx] = max_freq / codes_freq[idx]
        else:
            weights[idx] = 0

    # Sets2sets model
    encoder = EncoderRNN_new(input_size, hidden_size, num_layers)
    attn_decoder = AttnDecoderRNN_new(hidden_size, input_size, num_layers, dropout_p=0.1)
    if use_cuda:
        encoder = encoder.cuda()
        attn_decoder = attn_decoder.cuda()

    # train mode or test mode
    if training == 1:
        trainIters(history_data, future_data, input_size, encoder, attn_decoder, model_version, training_key_set, val_key_set, weights,
                   next_k_step, num_iter, topk)

    else:
        for i in [10, 20]: #top k
            valid_recall = []
            valid_ndcg = []
            valid_hr = []
            recall_list = []
            ndcg_list = []
            hr_list = []
            print('k = ' + str(i))
            for model_epoch in range(num_iter):
                print('Epoch: ', model_epoch)
                encoder_pathes = f"./models/encoder_{dataset}{fold}_best.pt"
                decoder_pathes = f"./models/decoder_{dataset}{fold}_best.pt"


                encoder_instance = torch.load(encoder_pathes, map_location=torch.device('cpu'))
                decoder_instance = torch.load(decoder_pathes, map_location=torch.device('cpu'))

                recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
                                            val_key_set, next_k_step, i)
                valid_recall.append(recall)
                valid_ndcg.append(ndcg)
                valid_hr.append(hr)
                recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
                                            test_key_set, next_k_step, i)
                recall_list.append(recall)
                ndcg_list.append(ndcg)
                hr_list.append(hr)
            valid_recall = np.asarray(valid_recall)
            valid_ndcg = np.asarray(valid_ndcg)
            valid_hr = np.asarray(valid_hr)
            idx1 = valid_recall.argsort()[::-1][0]
            idx2 = valid_ndcg.argsort()[::-1][0]
            idx3 = valid_hr.argsort()[::-1][0]
            print('max valid recall results:')
            print('Epoch: ', idx1)
            print('recall: ', recall_list[idx1])
            print('ndcg: ', ndcg_list[idx1])
            print('phr: ', hr_list[idx1])
            sys.stdout.flush()

            print('max valid ndcg results:')
            print('Epoch: ', idx2)
            print('recall: ', recall_list[idx2])
            print('ndcg: ', ndcg_list[idx2])
            print('phr: ', hr_list[idx2])
            sys.stdout.flush()

            print('max valid phr results:')
            print('Epoch: ', idx3)
            print('recall: ', recall_list[idx3])
            print('ndcg: ', ndcg_list[idx3])
            print('phr: ', hr_list[idx3])
            sys.stdout.flush()


if __name__ == '__main__':
    main(sys.argv)
