#!/usr/bin/env python
# coding: utf-8


#!deepeval set-local-model --model-name="gpt-3.5-turbo" \
#    --base-url="http://192.168.45.70:8000/v1/" \
#    --api-key="sk-1234"

from src.load_model import DeepEvalOpenAI

model_name = 'arcee-ai-Arcee-Agent-AWQ-4bit-smashed'
model_name = "Qwen2.5-7B-Instruct-AWQ"
base_url = 'http://192.168.45.70:8000/v1'
api_key = 'sk-1234'

# model_name = "gpt-4o-2024-05-13"
# base_url = 'http://192.168.45.70:8000/v1'


model = DeepEvalOpenAI(model=model_name, api_base=base_url, api_key=api_key, temperature=0, max_tokens=1024)

from deepeval.benchmarks.mera_rudetox.rudetox import RuDetox

benchmark = RuDetox(n_shots=4, n_problems_per_task=5)
#benchmark = RuDetox(n_shots=3)

results = benchmark.evaluate(model=model)

print("Overall Score: ", results)


# In[ ]:





