import os
import torch
import platform
import subprocess
import sys
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def init_model():
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        # "baichuan-inc/Baichuan2-13B-Chat",
        "/Data/Sland/Baichuan2-13B",
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        # "baichuan-inc/Baichuan2-13B-Chat"
        "/Data/Sland/Baichuan2-13B",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        # "baichuan-inc/Baichuan2-13B-Chat",
        "/Data/Sland/Baichuan2-13B",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    prompt = input()
    messages = [{"role": "user", "content": prompt}]
    response = model.chat(tokenizer, messages)
    sys.stdout.write(response)

if (__name__ == '__main__'):
    main()