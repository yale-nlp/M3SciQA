import argparse
import json
from data_utils import extract_json_like_string
import numpy as np
from tqdm import tqdm

def calculate_mrr(target, ranked_list):
    try:
        rank = ranked_list.index(target) + 1
        return 1 / rank
    except ValueError:
        return 0

def calculate_ndcg_at_k(target, ranked_list, k):
    dcg = 0
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i] == target:
            dcg = 1 / np.log2(i + 2)  
            break
    dcg_max = 1  
    return dcg / dcg_max if dcg else 0  

def calculate_recall_at_k(target, ranked_list, k):
    return int(target in ranked_list[:k])

def main(result_path: str, k: int):
    re = {
        "MRR": [],
        "NDCG@k": [],
        "Recall@k": []
    }

    with open(result_path, "r") as f:
        for l in tqdm(f):
            data = json.loads(l)
            raw_response = data['response']
            target = data["reference_s2_id"]
            ranking = json.loads(extract_json_like_string(raw_response))['ranking']
            re["MRR"].append(calculate_mrr(target, ranking))
            re["NDCG@k"].append(calculate_ndcg_at_k(target, ranking, k))
            re["Recall@k"].append(calculate_recall_at_k(target, ranking, k))

    print(f"MRR: {np.mean(re['MRR']):.4f}")
    print(f"NDCG@{k}: {np.mean(re['NDCG@k']):.4f}")
    print(f"Recall@{k}: {np.mean(re['Recall@k']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path of the response of locality questions')
    parser.add_argument('--k', type=int, default=3, help='k in NDCG@k and Recall@k')
    args = parser.parse_args()
    main(args.result_path, args.k)