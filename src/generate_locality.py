from data_utils import arxiv2s2, paper_title_abstract_list
from models_w_vision import *
from tqdm import tqdm
import argparse
import json


LOCALITY_QUESTION_PATH = "../data/locality.jsonl"
RESULTS_PATH = "../results/locality_response/"

def DEFAULT_PROMPT(question: str, reference_title_abstract_list: str):
    prompt = f"""You are given a figure, a question, and a list of paper candidates of titles and abstracts. Your task is to answer the question based on the figure information, then re-rank the list of paper candidates I input to you. 

Provide your answer at the end in a json file of this format using S2_id only:{{"ranking":  [] }}.
Only include the paper if you think they are relevant, do not include papers that irrelevant. Make sure the responded list is in a valid format.
You have to provide a ranking!! Not the paper name directly!!

<question>
{question}
</question>

<paper candidates>
{reference_title_abstract_list}
</paper candidates>"""
    
    return prompt


def main(model_name: str):
    model = globals()[model_name]
    print(type(model))

    print(f"[INFO] Generating responses for locality questions using {model_name}")
    results_path = os.path.join(RESULTS_PATH, f"{model_name}.jsonl")
    print(f"[INFO] Result has been saved to {results_path}")

    with open(LOCALITY_QUESTION_PATH, "r") as f1:
        for idx, l in tqdm(enumerate(f1)):
            if idx < 5:
                continue
            data = json.loads(l)
            anchor_arxiv_id = data['anchor_id']
            reference_arxiv_id = data['reference_id']
            question = data['question_anchor']
            figure_path = os.path.join("../data", data['evidence_anchor'])
            reference_s2_id = arxiv2s2(reference_arxiv_id)
            reference_title_abstract_list = paper_title_abstract_list(anchor_arxiv_id)

            prompt = DEFAULT_PROMPT(question=question, reference_title_abstract_list=reference_title_abstract_list)

            response = model(prompt=prompt, image_path=figure_path)
            with open(results_path, "a") as f2:
                f2.write(json.dumps({
                    "question_anchor": question,
                    "reference_arxiv_id": reference_arxiv_id,
                    "reference_s2_id": reference_s2_id,
                    "response": response
                })+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    models = ['claude_3_haiku', 'claude_3_sonnet', 'claude_3_opus', 'claude_3_5_sonnet', 
              'gpt_4_v', 'gpt_4_o', 'gemini_1_0_pro', 'gemini_1_5_pro', 'qwen2vl_7b']
    parser.add_argument('--model', type=str, choices=models, help='Select a model from the list: ' + ', '.join(models))
    args = parser.parse_args()
    main(args.model)