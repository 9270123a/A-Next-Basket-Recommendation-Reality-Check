import json, argparse, os

def main(hist, out_item, out_user):
    with open(hist) as f:
        data = json.load(f)

    # item_candidate：整份資料出現過的商品
    all_items = sorted({i for baskets in data.values() for b in baskets for i in b})
    os.makedirs(os.path.dirname(out_item), exist_ok=True)
    with open(out_item, 'w') as f:
        json.dump(all_items, f)

    # user_candidate：每人可能推薦的集合（簡單示範：他看過的全部 item）
    user_cand = {u: sorted({i for b in bs for i in b}) for u, bs in data.items()}
    os.makedirs(os.path.dirname(out_user), exist_ok=True)
    with open(out_user, 'w') as f:
        json.dump(user_cand, f)

    print(f"item_candidate → {out_item}  ({len(all_items)} items)")
    print(f"user_candidate → {out_user}  ({len(user_cand)} users)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--hist', required=True)
    ap.add_argument('--out_item', required=True)
    ap.add_argument('--out_user', required=True)
    args = ap.parse_args()
    main(args.hist, args.out_item, args.out_user)
