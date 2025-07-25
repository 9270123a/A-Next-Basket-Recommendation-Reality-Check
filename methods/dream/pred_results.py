import json, os, torch
from Explainablebasket import NBRNet

# ===== 絕對路徑 =====
ckpt_path   = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\methods\dream\models\CRSP-recall20-0-0-1--Apr-24-2025_19-56-44.pth"
history_file= r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\jsondata\CRSP_history.json"
keyset_file = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\keyset\CRSP_keyset_0.json"
pred_file   = r"C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\methods\dream\pred\Dream_pred.json"

# ===== 讀資料 =====
with open(history_file) as f: data_history = json.load(f)
with open(keyset_file)  as f: keyset       = json.load(f)

# ===== 從 checkpoint 讀出“當時的” config =====
ckpt = torch.load(ckpt_path, map_location='cpu')
conf = ckpt['config']                     # 完整保留當時 hidden_size、attention…

conf['device']   = torch.device('cpu')
conf['item_num'] = keyset['item_num']     # 若 keyset 一樣就無需改

# ===== 建模並載入權重 =====
model = NBRNet(conf, keyset)
model.load_state_dict(ckpt['state_dict'])   # 不再報 size / missing key
model.eval()

# ===== 產生 Top‑100 推薦 =====
pred = {}
for uid in keyset['test']:
    basket_seq = [data_history[uid][1:-1]]
    cand       = [[i for i in range(conf['item_num'])]]
    scores     = model.forward(basket_seq, cand)[0].detach().numpy()
    pred[uid]  = scores.argsort()[::-1][:100].tolist()

os.makedirs(os.path.dirname(pred_file), exist_ok=True)
with open(pred_file, 'w', encoding='utf-8') as f:
    json.dump(pred, f)
print('Prediction saved →', pred_file)
