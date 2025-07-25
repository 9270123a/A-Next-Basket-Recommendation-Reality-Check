#!/usr/bin/env python
# plot_model_metrics.py  – 畫 4-3 / 4-4 及 Repeat–Explore 圖
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.font_manager as fm, matplotlib as mpl
import math

# -------- 中文字體 --------
for ft in ["Noto Sans CJK TC","Microsoft YaHei","SimHei","PingFang TC"]:
    if ft in {f.name for f in fm.fontManager.ttflist}:
        mpl.rcParams["font.sans-serif"] = [ft]
        mpl.rcParams["axes.unicode_minus"] = False
        break
# --------------------------

df = pd.read_csv("model_metrics.csv")

# === 圖 A：k = 10 & 20  Recall / NDCG =========================
for k in [10, 20]:
    sub = df[(df.k == k) & df["recall"].notna() & df["ndcg"].notna()]
    if sub.empty:
        print(f"(跳過 k={k} ，recall/ndcg 皆 NaN)")
        continue

    sub = sub.sort_values("recall", ascending=False)
    x   = np.arange(len(sub)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x-w/2, sub["recall"], width=w, label=f"Recall@{k}")
    ax.bar(x+w/2, sub["ndcg"],   width=w, label=f"NDCG@{k}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"], rotation=20, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title(f"圖 4-{3 if k==10 else 4}  模型比較 (k={k})")
    y_max = np.nanmax(sub[["recall", "ndcg"]].to_numpy())
    if math.isfinite(y_max):
        ax.set_ylim(0, y_max * 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"fig_4-{3 if k==10 else 4}_recall_ndcg_k{k}.png", dpi=300)
    print(f"✓ fig_4-{3 if k==10 else 4}_recall_ndcg_k{k}.png 完成")

# === 圖 B：Repeat / Explore Ratio (k=10) ======================
sub = df[(df.k == 10) & df["repeat_ratio"].notna()]
fig, ax = plt.subplots(figsize=(10,4))
ax.bar(sub["model"], sub["repeat_ratio"], label="Repeat", alpha=.7)
ax.bar(sub["model"], sub["explore_ratio"],
       bottom=sub["repeat_ratio"], label="Explore", alpha=.7)
ax.set_xticklabels(sub["model"], rotation=20, ha="right")
ax.set_ylim(0,1.05)
ax.set_ylabel("Ratio")
ax.set_title("Repeat / Explore Ratio (k=10)")
ax.legend()
fig.tight_layout()
fig.savefig("fig_repeat_explore_ratio_k10.png", dpi=300)
print("✓ fig_repeat_explore_ratio_k10.png 完成")

# === 圖 C：Repeat / Explore  Recall & Hit (k=10) ==============
sub = df[df.k == 10]
fig, ax = plt.subplots(figsize=(10,4))
x = np.arange(len(sub)); w = .25
ax.bar(x-w, sub["recall_repeat"],  width=w, label="Recall-Repeat")
ax.bar(x   , sub["recall_explore"],width=w, label="Recall-Explore")
ax.bar(x+w, sub["hit_repeat"],     width=w, label="Hit-Repeat")
ax.set_xticks(x)
ax.set_xticklabels(sub["model"], rotation=20, ha="right")
y_max = np.nanmax(sub[["recall_repeat","recall_explore","hit_repeat"]].to_numpy())
if math.isfinite(y_max):
    ax.set_ylim(0, y_max*1.15)
ax.set_title("Repeat / Explore 指標 (k=10)")
ax.legend()
fig.tight_layout()
fig.savefig("fig_repeat_explore_metrics_k10.png", dpi=300)
print("✓ fig_repeat_explore_metrics_k10.png 完成")
