# summary_to_csv.py  ── 讀 RAW 字串 → DataFrame → 轉存 CSV + 印表
import re, pandas as pd, numpy as np

RAW = r"""
模型名稱	Size	Recall@k	NDCG@k	PHR@k	Repeat-Explore Ratio	Repeat-Explore Recall	Repeat-Explore Hit
g_top_pred	10	0.1488	0.2023	0.6868	0.432/0.567	0.153/0.088	0.692/0.164
	20	0.2459	0.2721	0.7532	0.401/0.598	0.249/0.17	0.755/0.284
gp_top_pred	10	0.5708	0.6411	0.9836	0.837/0.162	0.595/0.003	0.995/0.003
	20	0.7399	0.7699	0.9877	0.745/0.254	0.77/0.02	0.998/0.031
p_top_pred	10	0.5704	0.6409	0.9836	0.837/0.162	0.595	0.995
	20	0.7368	0.769	0.986	0.745/0.254	0.77	0.998
upcf-racf	10	0.5525	0.627	0.9811	0.765/0.234	0.571/0.058	0.991/0.107
	20	0.687	0.732	0.986	0.637/0.362	0.710/0.138	
Beacon	10	0.188	0.2134	0.7557	0.431/0.568	0.19/0.106	0.756/0.177
	20	0.7399	0.7700	0.9877	0.7450	0.7709	0.9983
Dream	10	0.2946	0.3255	0.8663	0.542/0.457	0.301/0.097	0.869/0.167
	20	0.4776	0.4329	0.9475	0.49/0.509	0.487/0.164	
DNNTSP	10	0.5926	0.6601	0.9852	0.829/0.17	0.616/0.017	0.995/0.022
	20	0.7624	0.7936	0.986	0.734/0.265	0.793/0.029	
SET2SET	10	0.5971	0.7052	0.9877			
	20	0.4953	0.7519	0.9836			
tifuknn	10	0.6006	0.6787	0.986	0.818/0.181	0.626/0.017	0.998/0.022
	20	0.7753	0.8151	0.9885	0.719/0.28	0.805/0.07	0.999/0.085
"""

rows = []
cur_model = None
for line in RAW.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    parts = re.split(r"\t+", line)

    # ---- 跳過標題列 ----
    if parts[0].startswith("模型名稱"):
        continue

    # ---- 取得模型名 ----
    if parts[0]:                 # 有模型名
        cur_model = parts[0].strip()
        raw_tokens = parts[1:]
    else:                        # 行首空白 → 沿用上一個模型
        raw_tokens = parts[1:]

    # ---- 拆出所有 "a/b" 與純數字 ----
    nums = []        # 只含 0-9 . 的 token
    pairs = []       # 含 "/" 的 token
    for t in raw_tokens:
        if "/" in t:
            pairs.append(t)
        elif t.strip() != "":
            nums.append(t)

    # --- 保障長度 ---
    while len(nums) < 4:
        nums.append("nan")
    while len(pairs) < 3:
        pairs.append("nan")

    size  = int(float(nums[0]))              # k
    rec   = float(nums[1])
    ndcg  = float(nums[2])
    phr   = float(nums[3]) if nums[3] != "nan" else np.nan

    def split_pair(s):
        if "/" in s:
            a, b = s.split("/")
            return float(a), float(b)
        return np.nan, np.nan

    rr,   er   = split_pair(pairs[0])
    rrec_r, rrec_e = split_pair(pairs[1])
    rhit_r, rhit_e = split_pair(pairs[2])

    rows.append({
        "model": cur_model,
        "k": size,
        "recall": rec,
        "ndcg": ndcg,
        "phr": phr,
        "repeat_ratio": rr,
        "explore_ratio": er,
        "recall_repeat": rrec_r,
        "recall_explore": rrec_e,
        "hit_repeat": rhit_r,
        "hit_explore": rhit_e,
    })

df = pd.DataFrame(rows)
df.to_csv("model_metrics.csv", index=False, encoding="utf-8-sig")
print("✓ model_metrics.csv 產出完成！共", len(df), "列")