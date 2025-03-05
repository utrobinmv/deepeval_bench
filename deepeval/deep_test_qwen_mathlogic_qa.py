#!/usr/bin/env python
# coding: utf-8

from src.load_model import DeepEvalOpenAI

model_name = 'arcee-ai-Arcee-Agent-AWQ-4bit-smashed'
model_name = "Qwen2.5-7B-Instruct-AWQ"
base_url = 'http://192.168.45.70:8000/v1'
api_key = 'sk-1234'

model = DeepEvalOpenAI(model=model_name, api_base=base_url, api_key=api_key, temperature=0, max_tokens=1)

from deepeval.benchmarks.mera_mathlogicqa.mathlogicqa import MathLogicQA
from deepeval.benchmarks.mera_mathlogicqa.task import MathLogicQATask

tasks = [MathLogicQATask.Math, MathLogicQATask.Logic]

benchmark = MathLogicQA(tasks=tasks)

results = benchmark.evaluate(model=model) # batch_size=5

print("Overall Score: ", results)


# In[ ]:




