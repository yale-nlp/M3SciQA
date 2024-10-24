import os 
from dotenv import load_dotenv
load_dotenv()
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

DEFAULT_SYSTEM_PROMPT = "You are the sole expert in this field and you can understand scientific papers."
DEFAULT_TEMPERATURE = 0.1

def gpt_4(prompt: str, temperature=DEFAULT_TEMPERATURE):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
                    model = "gpt-4-0125-preview",
                    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}],
                    temperature = temperature,
            )
    return response.choices[0].message.content

def gpt_4_o(prompt: str, temperature=DEFAULT_TEMPERATURE):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
                    model = "gpt-4o-2024-05-13",
                    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}],
                    temperature = temperature,
            )
    return response.choices[0].message.content

def gpt_3_5(prompt: str, temperature=DEFAULT_TEMPERATURE):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
                    model = "gpt-3.5-turbo",
                    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}],
                    temperature = temperature,
            )
    return response.choices[0].message.content


def claude_3_haiku(prompt: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            system=DEFAULT_SYSTEM_PROMPT,
            temperature = temperature,
            messages=[
                {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response


def claude_3_sonnet(prompt: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=DEFAULT_SYSTEM_PROMPT,
            temperature = temperature,
            messages=[
                {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response

def claude_3_opus(prompt: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.messages.create(
            model="claude-3-opus-202402290",
            max_tokens=1024,
            system=DEFAULT_SYSTEM_PROMPT,
            temperature = temperature,
            messages=[
                {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response

def claude_3_5_opus(prompt: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=DEFAULT_SYSTEM_PROMPT,
            temperature = temperature,
            messages=[
                {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response

def together_llama_3_70B(prompt, temperature=DEFAULT_TEMPERATURE):
    from together import Together
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    response = client.chat.completions.create(
                    model="meta-llama/Llama-3-70b-chat-hf",
                    messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                )
    res = response.choices[0].message.content
    return res

def together_mistral(prompt, temperature=DEFAULT_TEMPERATURE):
    from together import Together
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    response = client.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                )
    res = response.choices[0].message.content
    return res

def together_mixtral(prompt, temperature=DEFAULT_TEMPERATURE):
    from together import Together
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    response = client.chat.completions.create(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                )
    res = response.choices[0].message.content
    return res

def together_gemma(prompt, temperature=DEFAULT_TEMPERATURE):
    from together import Together
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    response = client.chat.completions.create(
                    model="google/gemma-7b-it",
                    messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                )
    res = response.choices[0].message.content
    return res

def qwen2vl_7b(model, processor, prompt: str, temperature=DEFAULT_TEMPERATURE):
    
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
    
    text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]

    return response