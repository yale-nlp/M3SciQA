def encode_image(image_path):
  import base64
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def arxiv2s2(anchor):
    import os
    import requests
    api_key = os.environ.get("SEMENTICS_SCHOLAR_API_KEY")
    headers = {'x-api-key': api_key}
    r = requests.post(
                    'https://api.semanticscholar.org/graph/v1/paper/batch',
                    params={'fields': 'paperId'},
                    json={"ids": [f"ARXIV:{anchor}"]},
                    headers=headers
                )
    s2 = r.json()[0]['paperId']
    return s2

def s22arxiv(s2Id: str):
    import requests
    import time
    import os
    api_key = os.environ.get("SEMENTICS_SCHOLAR_API_KEY")
    headers = {'x-api-key': api_key}
    r = requests.post(
                'https://api.semanticscholar.org/graph/v1/paper/batch',
                params={'fields': 'externalIds'},
                json={"ids": [s2Id]},
                headers=headers
            )
    time.sleep(1)
    try:
        return r.json()[0]['externalIds']['ArXiv']
    except:
        return None

def paper_title_abstract_list(anchor_id: str) -> str:
    import json
    with open("../paper_cluster_S2_content.json", "r") as f1, \
         open("../paper_cluster_S2.json") as f2:
        cluster_S2_content = json.load(f1)
        cluster_S2 = json.load(f2)
        
    ls = cluster_S2_content[anchor_id]
    s2_id_list = cluster_S2[anchor_id]
    concat = []
    for paper_info, s2 in zip(ls[:20], s2_id_list[:20]):
        single_info = f"S2_id: {s2}\nTitle: {paper_info['title']}\nAbstract: {paper_info['abstract']}\n"
        concat.append(single_info)
    return "\n".join(concat)

def read_paper(ArXiv_id: str, dataset_path: str) -> str:
    import json
    """Read a paper given an ArXiv Id

    Args:
        ArXiv_id (str): ArXiv id of the paper
        dataset_path (dict): dataset path

    Returns:
        paper_content (str): string for full content of the paper"""
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    if ArXiv_id not in dataset:
        return None
    paper_dict = dataset[ArXiv_id]

    title = f"Title: {paper_dict['title']}\n"
    abstract = f"Abstract: {paper_dict['abstract']}\n"
    # tldr = f"TLDR (summarization): {paper_dict['tldr']}\n"

    text_list = paper_dict["full_text"]
    full_text = []
    for item in text_list:
        if item["section_name"].startswith("Appendix"):
            break
        full_text.append(f'Section name: {item["section_name"]}\nparagraphs: {item["paragraphs"]}')
    full_text = "\n".join(full_text)

    # full_text = "\n".join([f'Section name: {item["section_name"]}\nparagraphs: {item["paragraphs"]}' for item in text_list])

    paper_content = title + abstract + full_text
    return paper_content


def extract_json_like_string(text):
    import re
    pattern = r'\{.*?"ranking": \[.*?\].*?\}'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None

from typing import List, Callable


def document_qa(question: str, document: str, chunk_length: int, response_model: Callable[[str], str], extract_model: Callable[[str], str]) -> str:
    """
    Perform document-level QA by breaking the document into chunks, querying a language model for each chunk,
    and then aggregating the answers.

    Args:
        question (str): question to be answered.
        document (str): concatenation of k documents
        chunk_length (int): length of each chunk
        model (Callable): model used to generate response
    
    Return:
        final_answer (str): aggregated response
    """

    if chunk_length < 50:
        raise ValueError("Chunk length must be greater than 50 to allow overlapping.")

    def create_chunks(text: str, length: int, overlap: int=50) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + length
            chunks.append(text[start:end])
            start = end - overlap  
            start = max(start, 0)
        return chunks

    chunks = create_chunks(document, chunk_length)

    answers = []
    for chunk in chunks:
        prompt = f"""Answer the below question about a scientific paper. The question is composed with 2 parts the second part of the question can be answered by the paper. I will input you a chunk of a paper since the paper is too long. 
        
        Provide your reasoning. Append the answer at the end of the response with json format {{\"answer\": \"\"}}. You should answer the question in a short-answer form. Do not provide long answers. If you do not know the answer, response by {{\"answer\": \"I don't know\"}}
        
<QUESTION>
{question}
</QUESTION>
<CHUNK>
{chunk}
</CHUNK>"""
        response = response_model(prompt=prompt)
        prompt2 = f"""Extract the `answer` field from teh input RESPONSE. Respond only by the extracted answer. No other texts are needed. 

<RESPONSE>
{response}
</RESPONSE>
"""
        extracted_response = extract_model(prompt2)
        answers.append(extracted_response)

    aggregated_answer = "\n".join([f"<ANSWER CANDIDATE>\n{x}\n</ANSWER CANDIDATE>" for x in answers])
    prompt = f"""I input you a set of answer candidate for a question, aggregate the information from all candidates and give me 1 single answer. Note that if one answer candidate is `I don't know`, you can ignore it.
    Answer the question based on the answer candidates. Summarize the answer in a short answer.
    <QUESTION>
    {question}
    </QUESTION>
    {aggregated_answer}
"""
    final_answer = response_model(prompt)

    return final_answer

