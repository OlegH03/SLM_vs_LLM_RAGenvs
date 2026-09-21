import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import os
from openai import OpenAI


def load_local_model(model_name, quantize=False):
    if quantize:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=os.getenv("HF_TOKEN")
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    return {"model": model, "tokenizer": tokenizer}


def load_api_model(model_name):
    load_dotenv()
    api_key = os.getenv("KEY_LLM")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    return {"client": client, "model_name": model_name}
