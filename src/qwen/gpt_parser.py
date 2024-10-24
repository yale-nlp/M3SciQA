#!/usr/bin/env python3


import os
import re
import json
import copy


NUM_LOCALITY = 300
REFORMAT_PROMPT_TEMPLATE = '''
Format the below text chunk into a single python dictionary with keys "question", "answer",  and "rank".
So your response should start with a "{{" and end with a "}}"; any null value should be written as "None".
"rank" should be a list of 40-digit s2_id without duplicates; keep the original ranking order; leave the list empty if no valid s2_id or ranking is provided.

<text chunk>
{text}
</text chunk>
'''


def get_index_and_batch(file_path):
    x = file_path
    try:
        return int(x.rsplit('/')[1].split('|', 1)[0]), int(x.rsplit('.', 1)[0].rsplit('|', 1)[1])
    except:
        return int(x.rsplit('/')[1].split('|', 2)[1]), int(x.rsplit('.', 1)[0].rsplit('|', 1)[1])


def main():
    curr_gpt_files = os.listdir(GPT_DIR)
    gpt_files = [os.path.join(GPT_DIR, f) for f in curr_gpt_files if not f.startswith('.')]
    gpt_files = sorted(gpt_files, key=lambda x: get_index_and_batch(x))


    # process gpt result
    rankings = [[] for _ in range(NUM_LOCALITY)]
    for file in gpt_files:
        index, _ = get_index_and_batch(file)
        with open(file) as gf:
            batch_locality = json.load(gf)
        for r in batch_locality['rank']:
            if not r:
                continue
            pattern = r'[0-9a-fA-F]{40}'
            matches = re.findall(pattern, str(r))
            if matches:
                rankings[index].append(matches[0])


    # compute new locality
    with open(PREV_LOCALITY_JSON) as f:
        prev_locality = json.load(f)
    new_locality = copy.deepcopy(prev_locality)
    for i, loc in enumerate(new_locality):
        new_ref = list()
        for ref in loc['references']:
            if ref['s2_id'] in rankings[i]:
                new_ref.append(ref)
        new_locality[i]['references'] = new_ref
    print(json.dumps(new_locality, indent=4))
    with open(NEW_LOCALITY_JSON, 'w') as f:
        json.dump(new_locality, f, indent=4)


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file) 

    GPT_DIR = config['GPT_OUTPUT_DIRECTORY']
    PREV_LOCALITY_JSON = config['PREV_LOCALITY_JSON']
    NEW_LOCALITY_JSON = config['NEW_LOCALITY_JSON']

    main()
