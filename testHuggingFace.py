#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:46:58 2025

@author: tim
"""

'''# Use a pipeline as a high-level helper
from transformers import pipeline
import accelerate

pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
'''
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer

model_id = "HuggingFaceTB/SmolLM3-3B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model_id, tokenizer=tokenizer)

messages = [
    {"role": "user", "content": "Give me a brief explanation of gravity in simple terms."},
]
print(pipe(messages))