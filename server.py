from flask import Flask, request, jsonify, render_template, make_response
import os
import torch
import platform
import subprocess
import sys
import argparse
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

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

app = Flask(__name__)
@app.route('/chat', methods=['POST'])
def chat():
    global model
    global tokenizer
    message = request.json
    response = model.chat(tokenizer, message)
    return jsonify({"response": response})

@app.route('/ready', methods=['GET'])
def ready():
    return jsonify({"response": "ready"})

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=11000)
    args = parser.parse_args()

    app.run(host='localhost', port=args.port)