#!/usr/bin/env python3


import os
import ast
import json
import openai


REFORMAT_PROMPT_TEMPLATE = '''
Format the below text chunk into a valid single python dictionary with keys "question", "answer",  and "rank".
So your response should start with a "{{" and end with a "}}"; any null value should be None.
"rank" should be a list of 40-digit s2_id without duplicates; keep the original ranking order; leave the list empty if no valid s2_id or ranking is provided.

<text chunk>
{text}
</text chunk>
'''


def gpt4(prompt: str, temperature=0.1):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system",
                   "content": "You are the sole expert in the field of reformatting texts."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


def get_index_and_batch(file_path):
    x = file_path
    try:
        return int(x.rsplit('/')[1].split('|', 1)[0]), int(x.rsplit('.', 1)[0].rsplit('|', 1)[1])
    except:
        return int(x.rsplit('/')[1].split('|', 2)[1]), int(x.rsplit('.', 1)[0].rsplit('|', 1)[1])


def main():
    # outputs from model
    output_files = list()
    for d in INPUT_DIRS:
        curr_output_files = os.listdir(d)
        output_files.extend([os.path.join(d, f) for f in curr_output_files if not f.startswith('.')])

    output_files = sorted(output_files,
                        # ordered by (question index, batch number)
                        key=lambda x: get_index_and_batch(x))

    failed_gpt = list()
    for i, o in enumerate(output_files):
        with open(o) as f:
            qwen_response = json.load(f)

        text = qwen_response['output']['choices'][0]['message']['content'][0]['text']

        gpt_prompt = REFORMAT_PROMPT_TEMPLATE.format(text=text)
        print(gpt_prompt)
        file_name = 'gpt|' + o.split('/')[-1]
        print('writing to: ' + file_name)
        gpt_response = gpt4(gpt_prompt)
        bracket_gpt_response = gpt_response[gpt_response.find('{'): gpt_response.rfind('}') + 1]
        print(bracket_gpt_response)
        try:
            gpt_dict = ast.literal_eval(bracket_gpt_response)
            with open(os.path.join(OUTPUT_DIR, file_name), 'w') as fw:
                json.dump(gpt_dict, fw, indent=4)
            print(gpt_dict)
            print('\n\n')
        except:
            failed_gpt.append(o)


    for i in failed_gpt:
        print(i)


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file) 
    
    openai.api_key = config['OPENAI_API_KEY']
    INPUT_DIRS = config['GPT_INPUT_DIRECTORIES']
    OUTPUT_DIR = config['GPT_OUTPUT_DIRECTORY']

    main()
