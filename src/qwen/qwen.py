#!/usr/bin/env python3


import os
import json
import dashscope
import pandas as pd
from http import HTTPStatus


pd.set_option('display.max_columns', None)


BATCH_SIZE = 8
PROMPT_TEMPLATE = """
Answer the question from the figure and the reference papers provided only, {locality_question}
Additionally, rerank the following reference papers according to their relevance to this question. Each reference paper is consists of a s2_id, a title, and an abstract.
{paper_cluster}
Format your answer as a python dictionary with keys "question", "answer",  and "rank". "rank" should be a list of s2_id. 
If no relevant reference papers are provided, return an empty list for "rank".
"""


def main():
    with open(CURRENT_LOCALITY) as f:
        full_locality = json.load(f)


    count = 0
    for index, locality in enumerate(full_locality):
        if len(locality['references']) <= 1:
            continue

        batch = 0
        for i in range(0, len(locality['references']), BATCH_SIZE):
            image_path = locality['evidence_anchor']
            print(image_path)

            end_i = min(len(locality['references']), i + BATCH_SIZE)
            paper_cluster = locality['references'][i:end_i]
            print('references in current batch: ' + str(i) + ': ' + str(end_i))

            prompt = PROMPT_TEMPLATE.format(locality_question=locality['question_anchor'],
                                            paper_cluster=json.dumps(paper_cluster))
            print(prompt)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": os.path.join(IMAGE_ROOT_PATH, image_path)},
                        {"text": prompt}
                    ]
                }
            ]

            response = dashscope.MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages,
                seed=1234
            )

            # The response status_code is HTTPStatus.OK indicate success,
            # otherwise indicate request is failed, you can get error code
            # and message from code and message.
            if response.status_code == HTTPStatus.OK:
                image_name = image_path.rsplit('.', 1)[0].replace('/', '=')
                file_name = str(index) + '|' + image_name + '|batch|' + str(batch) + '.json'
                print('\nwriting to: ' + file_name)
                with open(os.path.join(OUTPUT_DIRECTORY, file_name), 'w') as fw:
                    json.dump(response, fw, indent=4)
            else:
                print(response.code)  # The error code.
                print(json.dumps(response.message, indent=4))  # The error message.

            batch += 1
            print()

        count += 1
    print(count)


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file) 

    dashscope.api_key = config['DASHSCOPE_API_KEY']

    IMAGE_ROOT_PATH = config['IMAGE_ROOT_PATH']
    REPO_ROOT_PATH = config['REPO_ROOT_PATH']
    OUTPUT_DIRECTORY = config['MODEL_OUTPUT_DIRECTORY']
    CURRENT_LOCALITY = os.path.join(REPO_ROOT_PATH, 'full_locality_dataset.json')

    main()
