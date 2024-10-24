#!/usr/bin/env python3


import os
import json
import copy


def main(intermediate_results):
    datasets = list()
    for path in intermediate_results:
        with open(path) as curr_f:
            datasets.append(json.load(curr_f))

    length_1_references = [[] for _ in range(len(datasets[0]))]
    for dataset in datasets:
        for index, locality in enumerate(dataset):
            if len(locality['references']) == 1:
                assert len(length_1_references[index]) == 0
                length_1_references[index] = locality['references']

    count = 0
    final_locality_dataset = copy.deepcopy(datasets[-1])
    for index, locality in enumerate(final_locality_dataset):
        if length_1_references[index]:
            final_locality_dataset[index]['references'] = length_1_references[index]
        if final_locality_dataset[index]['references']:
            count += 1

    print(json.dumps(final_locality_dataset, indent=4))
    print(count)

    return final_locality_dataset


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file) 

    REPO_ROOT_PATH = config['REPO_ROOT_PATH']
    QWEN_RELATIVE_PATH = config['QWEN_RELATIVE_PATH']
    FULL_LOCALITY_JSON = config['FULL_LOCALITY_JSON']
    FINAL_LOCALITY_JSON = config['FINAL_LOCALITY_JSON']
    INTERMEDIATE_JSONS = config['INTERMEDIATE_JSONS']
    intermediate_results = [os.path.join(REPO_ROOT_PATH, QWEN_RELATIVE_PATH, i_json) for i_json in INTERMEDIATE_JSONS]
    intermediate_results.append(os.path.join(REPO_ROOT_PATH, FULL_LOCALITY_JSON))
    
    final_locality_dataset = main(intermediate_results)
    final_locality_path = os.path.join(REPO_ROOT_PATH, QWEN_RELATIVE_PATH, FINAL_LOCALITY_JSON)
    with open(final_locality_path, 'w') as final_file:
        json.dump(final_locality_dataset, final_file, indent=4)
