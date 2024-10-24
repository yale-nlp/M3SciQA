#!/usr/bin/env python3


import time
import json
import requests
import pandas as pd


pd.set_option('display.max_columns', None)


def arxiv2s2(anchor, SEMANTIC_SCHOLAR_API_KEY):
    api_key = SEMANTIC_SCHOLAR_API_KEY
    headers = {'x-api-key': api_key}
    r = requests.post(
                    'https://api.semanticscholar.org/graph/v1/paper/batch',
                    params={'fields': 'paperId'},
                    json={"ids": [f"ARXIV:{anchor}"]},
                    headers=headers
                )
    s2 = r.json()[0]['paperId']
    return s2


def compose_arxiv2s2_map(locality_df, SEMANTIC_SCHOLAR_API_KEY):
    arxiv2s2_map = dict()
    for _, locality in locality_df.iterrows():
        arxiv_id = locality['reference_id']
        if arxiv_id not in arxiv2s2_map:
            print(arxiv_id)
            try:
                s2_id = arxiv2s2(arxiv_id, SEMANTIC_SCHOLAR_API_KEY)
                print(s2_id)
                arxiv2s2_map[arxiv_id] = s2_id
                time.sleep(2)
            except:
                print('null')
                arxiv2s2_map[arxiv_id] = 'null'
            print()
    return arxiv2s2_map


def compose_full_reference_list():
    with open('paper_cluster_S2.json') as f:
            paper_cluster = json.load(f)
    with open('paper_cluster_S2_content_new.json') as f2:
        paper_content = json.load(f2)
    s2_combined = dict()
    for key in paper_cluster:
        s2_combined[key] = list()

        for idx, ref_paper in enumerate(paper_cluster[key]):
            s2_combined[key].append(
                {'id': ref_paper,
                'title': paper_content[key][idx]['title'],
                'abstract': paper_content[key][idx]['abstract']}
            )
    return s2_combined


def compose_full_locality_dataset(locality_df, SEMANTIC_SCHOLAR_API_KEY):
    """composing a locality dataset with full reference lists with s2 ids"""
    s2_combined = compose_full_reference_list()
    
    arxiv2s2_map = compose_arxiv2s2_map(locality_df, SEMANTIC_SCHOLAR_API_KEY)
    # with open('arxive2s2_map.json', 'w') as map_file:
    #     json.dump(arxiv2s2_map, map_file, indent=4)
    # with open('arxive2s2_map.json') as map_file:
    #     arxiv2s2_map = json.load(map_file)

    locality_list = json.loads(locality_df.to_json(orient='records'))
    for index, locality in enumerate(locality_list):
        locality_list[index]['reference_id_s2'] = arxiv2s2_map[locality_df.loc[index, 'reference_id']]
        locality_list[index]['references'] = s2_combined[locality['anchor_id']]

    return locality_list


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file) 
    
    SEMANTIC_SCHOLAR_API_KEY = config['SEMANTIC_SCHOLAR_API_KEY']
    LOCALITY_DATASET_JSONL = config['LOCALITY_DATASET_JSONL']
    FULL_LOCALITY_JSON = config['FULL_LOCALITY_JSON']

    locality_df = pd.read_json(LOCALITY_DATASET_JSONL, lines=True, dtype=str)
    full_locality_dataset = compose_full_locality_dataset(locality_df, SEMANTIC_SCHOLAR_API_KEY)
    with open(FULL_LOCALITY_JSON, 'w') as full_locality_file:
        json.dump(full_locality_dataset, full_locality_file, indent=4)
