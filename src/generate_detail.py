from data_utils import document_qa, read_paper
from models_wo_vision import *
from tqdm import tqdm
import argparse
import json


DETAIL_QUESTION_PATH = "../data/combined_val.jsonl"
RESULTS_PATH = "../results/retrieval"
RETRIEVAL_PAPER_PATH = "../retrieval_paper.json"
PAPER_FULL_CONTENT_PATH = "../paper_full_content.json"


def main_api(model_name: str, k: int, chunk_length: int):
    os.makedirs(f"{RESULTS_PATH}@{k}", exist_ok=True)

    model = globals()[model_name]
    print(type(model))

    print(f"[INFO] Generating responses for detail questions using {model_name}")
    results_path = os.path.join(f"{RESULTS_PATH}@{k}", f"{model_name}.jsonl")
    print(f"[INFO] Result has been saved to {results_path}")

    with open(RETRIEVAL_PAPER_PATH, "r") as f:
        retrieval_paper_dict = json.load(f)

    with open(DETAIL_QUESTION_PATH, "r") as f:
        for l in tqdm(f):
            data = json.loads(l)
            ref_id = data['reference_arxiv_id']
            question_anchor = data['question_anchor']
            paper_retrieval = retrieval_paper_dict[question_anchor][:10]
            paper_content_collection = []
            for ref_id in paper_retrieval:
                paper_content_collection.append(read_paper(ref_id, PAPER_FULL_CONTENT_PATH))

            # string for all k papers
            paper_content_collection = [x for x in paper_content_collection if x][:k]
            if paper_content_collection:
                paper_content_all = "\n".join(paper_content_collection)
            else:
                paper_content_all = "no documents available"

            question = data['question']
            answer = data['answer']
            reasoning = data['reference_reasoning_type']

            response = document_qa(
                question=question,
                document=paper_content_all,
                chunk_length=chunk_length,
                response_model=model,
                extract_model=gpt_3_5 # use GPT 3.5 to extract answers 
            )

            with open(results_path, "a") as f2:
                f2.write(json.dumps({
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "reference_reasoning_type": reasoning
                })+"\n")


def main_opensource(model_name: str, k: int, chunk_length: int):
    from functools import partial
    os.makedirs(f"{RESULTS_PATH}@{k}", exist_ok=True)

    if model_name == "qwen2vl_7b":
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch
        model_path = "../pretrained/Qwen2-VL-7B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        call_model_func = partial(qwen2vl_7b, model, processor)
    

    print(f"[INFO] Generating responses for detail questions using {model_name}")
    results_path = os.path.join(f"{RESULTS_PATH}@{k}", f"{model_name}.jsonl")
    print(f"[INFO] Result has been saved to {results_path}")

    with open(RETRIEVAL_PAPER_PATH, "r") as f:
        retrieval_paper_dict = json.load(f)

    with open(DETAIL_QUESTION_PATH, "r") as f:
        for l in tqdm(f):
            data = json.loads(l)
            ref_id = data['reference_arxiv_id']
            question_anchor = data['question_anchor']
            paper_retrieval = retrieval_paper_dict[question_anchor][:10]
            paper_content_collection = []
            for ref_id in paper_retrieval:
                paper_content_collection.append(read_paper(ref_id, PAPER_FULL_CONTENT_PATH))

            # string for all k papers
            paper_content_collection = [x for x in paper_content_collection if x][:k]
            if paper_content_collection:
                paper_content_all = "\n".join(paper_content_collection)
            else:
                paper_content_all = "no documents available"

            question = data['question']
            answer = data['answer']
            reasoning = data['reference_reasoning_type']

            response = document_qa(
                question=question,
                document=paper_content_all,
                chunk_length=chunk_length,
                response_model=call_model_func,
                extract_model=gpt_3_5 # use GPT 3.5 to extract answers 
            )

            with open(results_path, "a") as f2:
                f2.write(json.dumps({
                    "question": question,
                    "answer": answer,
                    "response": response,
                    "reference_reasoning_type": reasoning
                })+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    models = ['gpt_4', 'gpt_3_5', 'gpt_4_o',
              'claude_3_haiku', 'claude_3_sonnet', 'claude_3_opus', 'claude_3_5_sonnet', 
              'together_llama_3_70B', 'together_mistral', 'together_mixtral', 'together_gemma', "qwen2vl_7b"]

    parser.add_argument('--model', type=str, choices=models, help='Select a model from the list: ' + ', '.join(models))
    parser.add_argument('--k', type=int, default=3, help='Number of papers being retrieved')
    parser.add_argument('--chunk_length', type=int, default=10000, help='Length of texts chunks')
    args = parser.parse_args()

    if args.model == "qwen2vl_7b":
        main_opensource(args.model, args.k, args.chunk_length)
    else:
        main_api(args.model, args.k, args.chunk_length)