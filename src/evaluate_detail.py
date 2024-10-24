import argparse
from models_wo_vision import gpt_4
import json
from tqdm import tqdm
import numpy as np
import os

EVALUATION_TEMPERATURE = 0.0

def gpt_4_evaluator(question, reference, candidate):
    def extract_response(text):
        import re
        pattern = r'\{.*?\}'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0) if match else None
    
    prompt = f"""I am testing a model performance on open-ended questions, I want you to help me in checking if the candidate answer has the same meaning with the reference answer given the question. If you think the reference answer and the candidate answer have the same meaning, respond {{"selection": "1"}}; else, respond by {{"selection": "0"}}; if you think the candidate is partially correct, respond by {{"selection": "0.5"}}.

    <QUESTION>
    {question}
    </QUESTION>

    <REFERENCE>
    {reference}
    </REFERENCE>

    <CANDIDATE>
    {candidate}
    </CANDIDATE>

    Again, if you think they have the same meaning, respond {{"selection": "1"}}; if you think they are totally irrelevant, respond by {{"selection": "0"}} only; if you think the candidate is partially correct, respond by {{"selection": "0.5"}}.
    
    Do not use other format.
"""
    response = gpt_4(prompt, temperature=EVALUATION_TEMPERATURE)
    response = extract_response(response)
    response = json.loads(response)['selection']
    return float(response)


def main(result_path: str):
    model_name = os.path.splitext(result_path)[0]
    result_writing_path=f"{model_name}_eval.jsonl"
    print(f"Result with scores is saved to {result_writing_path}")
    score_list = []
    with open(result_path, "r") as f:
        for l in tqdm(f):
            data = json.loads(l)
            question = data['question']
            reference = data['answer']
            candidate = data['response']

            score = gpt_4_evaluator(
                question=question,
                reference=reference,
                candidate=candidate
            )

            data['is_match'] = score
            score_list.append(score)
            with open(result_writing_path, "a") as f2:
                f2.write(json.dumps(data)+"\n")

    print(f"""Mean score is for {result_path}: {np.mean(score_list)}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path of the response of detail questions')
    args = parser.parse_args()
    main(args.result_path)