#!/usr/bin/env python
# coding: utf-8

from src.load_model import DeepEvalOpenAI

model_name = 'arcee-ai-Arcee-Agent-AWQ-4bit-smashed'
model_name = "Qwen2.5-7B-Instruct-AWQ"
base_url = 'http://192.168.45.70:8000/v1'
api_key = 'sk-1234'

# model_name = "gpt-4o-2024-05-13"
# base_url = 'http://192.168.45.70:8000/v1'


model = DeepEvalOpenAI(model=model_name, api_base=base_url, api_key=api_key, temperature=0, max_tokens=1024)

from deepeval.benchmarks.mera_rutie.rutie import RutieQA
from deepeval.benchmarks.mera_rutie.task import RutieQATask

tasks = [RutieQATask.Dialog]

#benchmark = RutieQA(tasks=tasks, n_problems_per_task=5)
#benchmark = RutieQA(tasks=tasks, n_shots=20)
benchmark = RutieQA(tasks=tasks, n_shots=10)

results = benchmark.evaluate(model=model)

print("Overall Score: ", results)


# In[ ]:
