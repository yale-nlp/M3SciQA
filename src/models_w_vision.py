import os 
from dotenv import load_dotenv
load_dotenv()
from data_utils import *
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch



DEFAULT_SYSTEM_PROMPT = "You are the sole expert in understanding scientific literatures and scientifc images."
DEFAULT_TEMPERATURE = 0.1

def claude_3_haiku(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    base64_image = encode_image(image_path)
    message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=DEFAULT_SYSTEM_PROMPT,
            temperature = temperature,
            messages=[
                {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response


def claude_3_sonnet(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    base64_image = encode_image(image_path)
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
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response

def claude_3_opus(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    base64_image = encode_image(image_path)
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
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response

def claude_3_5_sonnet(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    base64_image = encode_image(image_path)
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
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }]
                }
            ]
        )
    final_response = message.content[0].text
    return final_response

def gpt_4_v(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
                    model = "gpt-4-vision-preview",
                    messages = [{
                        "role": "system", "content": DEFAULT_SYSTEM_PROMPT,
                        "role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    temperature = temperature,
            )
    return response.choices[0].message.content

def gpt_4_o(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
                    model = "gpt-4o-2024-05-13",
                    messages = [{
                        "role": "system", "content": DEFAULT_SYSTEM_PROMPT,
                        "role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    temperature = temperature,
            )
    return response.choices[0].message.content

def gemini_1_0_pro(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    from PIL import Image
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    def encode_image(image_path):
        image = Image.open(image_path)
        return image
    
    base64_image = encode_image(image_path)
    model = genai.GenerativeModel(model_name='gemini-1.0-pro-vision-latest',
                                  generation_config={"temperature": temperature})
    
    response = model.generate_content([prompt] + [base64_image], stream=True)
    response.resolve()
    for candidate in response.candidates:
            return [part.text for part in candidate.content.parts]

    return response.text

def gemini_1_5_pro(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):
    from PIL import Image
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    def encode_image(image_path):
        image = Image.open(image_path)
        return image
    
    base64_image = encode_image(image_path)
    model = genai.GenerativeModel(model_name='gemini-1.5-pro',
                                  generation_config={"temperature": temperature})
    
    response = model.generate_content([prompt] + [base64_image], stream=True)
    response.resolve()
    for candidate in response.candidates:
            return [part.text for part in candidate.content.parts]

    return response.text



def qwen2vl_7b(prompt: str, image_path: str, temperature=DEFAULT_TEMPERATURE):

    model_path = "../pretrained/Qwen2-VL-7B-Instruct"

    base64_image = encode_image(image_path)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    processor = AutoProcessor.from_pretrained(model_path)
    
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "video": base64_image,
                        },
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