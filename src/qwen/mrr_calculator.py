#!/usr/bin/env python3
"""
This script calculates the MRRs for the locality dataset, combined validation dataset, and combined test dataset.
If you have prepared your FINAL LOCALITY DATASET defined by the design doc,
you should be able to run the script directly after making sure your GLOBAL CONSTANTS below are set correctly.
"""


import copy
import json
import pandas as pd


def evaluate_300_locality(final_locality_dataset):
    """EVALUATE OVER THE 300 LOCALITY QUESTIONS ONLY"""
    rr_sum = 0
    total_count = 0
    valid_rank_count = 0
    gt_rank_count = 0  # ground truth in rank

    for anchor in final_locality_dataset:
        total_count += 1
        s2_ids = [item['s2_id'] for item in anchor['references']]
        if s2_ids:
            valid_rank_count += 1
        ground_truth = anchor['reference_id_s2']
        if ground_truth not in s2_ids:
            continue
        gt_rank_count += 1
        reciprocal_rank = 1 / (s2_ids.index(ground_truth) + 1)
        rr_sum += reciprocal_rank
    total_mrr = rr_sum / total_count
    valid_rank_mrr = rr_sum / valid_rank_count
    gt_rank_mrr = rr_sum / gt_rank_count

    return [[total_count, total_mrr], [valid_rank_count, valid_rank_mrr], [gt_rank_count, gt_rank_mrr]]


def evaluate_validation_and_test_dataset(final_locality_dataset, val_df, test_df):
    """EVALUATE OVER VAL AND TEST BY MAPPING BACK TO COMBINED QUESTIONS"""
    # PREPARE OUTPUT DICTIONARIES
    counts_template = {
        'val': 0,
        'test': 0,
        'table': 0,
        'figure': 0,
        'Comparison': 0,
        'Data Extraction': 0,
        'Location': 0,
        'Visual Understanding': 0
    }
    rr_sums = copy.deepcopy(counts_template)
    total_counts = copy.deepcopy(counts_template)
    valid_counts = copy.deepcopy(counts_template)
    gt_counts = copy.deepcopy(counts_template)

    # VALIDATION SET
    for index, combined in val_df.iterrows():
        total_counts['val'] += 1
        curr_anchor = combined['question_anchor']
        curr_locality = None
        for locality in final_locality_dataset:
            if locality['question_anchor'] == curr_anchor:
                curr_locality = locality
                break

        # assert curr_locality is not None
        if curr_locality is None:
            print(index)
            continue
        s2_ids = [item['s2_id'] for item in curr_locality['references']]
        if s2_ids:
            valid_counts['val'] += 1
        ground_truth = curr_locality['reference_id_s2']
        if ground_truth not in s2_ids:
            continue
        reciprocal_rank = 1 / (s2_ids.index(ground_truth) + 1)
        rr_sums['val'] += reciprocal_rank
        gt_counts['val'] += 1

    # TEST SET
    for index, combined in test_df.iterrows():
        total_counts['test'] += 1
        total_counts[combined['modal']] += 1
        total_counts[combined['anchor_reasoning_type']] += 1
        curr_anchor = combined['question_anchor']
        curr_locality = None
        for locality in final_locality_dataset:
            if locality['question_anchor'] == curr_anchor:
                curr_locality = locality
                break

        # assert curr_locality is not None
        if curr_locality is None:
            print(index)
            continue
        s2_ids = [item['s2_id'] for item in curr_locality['references']]
        if s2_ids:
            valid_counts['test'] += 1
            valid_counts[combined['modal']] += 1
            valid_counts[combined['anchor_reasoning_type']] += 1
        ground_truth = curr_locality['reference_id_s2']
        if ground_truth not in s2_ids:
            continue
        reciprocal_rank = 1 / (s2_ids.index(ground_truth) + 1)
        rr_sums['test'] += reciprocal_rank
        rr_sums[combined['modal']] += reciprocal_rank
        rr_sums[combined['anchor_reasoning_type']] += reciprocal_rank

        gt_counts['test'] += 1
        gt_counts[combined['modal']] += 1
        gt_counts[combined['anchor_reasoning_type']] += 1

    total_counts_mrr = {k: (v / total_counts[k]) for k, v in rr_sums.items()}
    valid_counts_mrr = {k: (v / valid_counts[k]) for k, v in rr_sums.items()}
    gt_counts_mrr = {k: (v / gt_counts[k]) for k, v in rr_sums.items() if gt_counts[k] != 0}

    return [[total_counts, total_counts_mrr], [valid_counts, valid_counts_mrr], [gt_counts, gt_counts_mrr]]


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file) 

    FINAL_LOCALITY_JSON = config['FINAL_LOCALITY_JSON']
    VALIDATION_DATASET_JSONL = config['VALIDATION_DATASET_JSONL']
    TEST_DATASET_JSONL = config['TEST_DATASET_JSONL']

    with open(FINAL_LOCALITY_JSON) as tf:
        final_locality_dataset = json.load(tf)

    # EVALUATE OVER THE 300 LOCALITY QUESTIONS ONLY
    [[total_count, total_mrr], [valid_rank_count, valid_rank_mrr], [gt_rank_count, gt_rank_mrr]] = \
        evaluate_300_locality(final_locality_dataset)
    print('EVALUATE OVER THE 300 LOCALITIES ONLY')
    print('\nQ (total count) = ' + str(total_count))
    print('MRR = ' + str(total_mrr))
    print('\nQ (valid rank count) = ' + str(valid_rank_count))
    print('MRR = ' + str(valid_rank_mrr))
    print('\nQ (ground truth in rank count) = ' + str(gt_rank_count))
    print('MRR = ' + str(gt_rank_mrr))

    # EVALUATE OVER VAL AND TEST BY MAPPING BACK TO COMBINED QUESTIONS
    val_df = pd.read_json(VALIDATION_DATASET_JSONL, lines=True, dtype=str)
    test_df = pd.read_json(TEST_DATASET_JSONL, lines=True, dtype=str)

    [[total_counts, total_counts_mrr], [valid_counts, valid_counts_mrr], [gt_counts, gt_counts_mrr]] = \
        evaluate_validation_and_test_dataset(final_locality_dataset, val_df, test_df)
    print('\n\n\nEVALUATE OVER VAL SET AND TEST SET BY MAPPING BACK TO COMBINED QUESTIONS')
    print('\nQ (total count) = ')
    print(json.dumps(total_counts, indent=4))
    print('MRR = ')
    print(json.dumps(total_counts_mrr, indent=4))

    print('\nQ (valid rank count) = ')
    print(json.dumps(valid_counts, indent=4))
    print('MRR = ')
    print(json.dumps(valid_counts_mrr, indent=4))

    print('\nQ (ground truth in rank count) = ')
    print(json.dumps(gt_counts, indent=4))
    print('MRR = ')
    print(json.dumps(gt_counts_mrr, indent=4))
