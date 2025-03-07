#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.load_model import DeepEvalOpenAI


# In[2]:


model_name = 'arcee-ai-Arcee-Agent-AWQ-4bit-smashed'
model_name = "Qwen2.5-7B-Instruct-AWQ"
base_url = 'http://192.168.45.70:8000/v1'
api_key = 'sk-1234'


# In[3]:


model = DeepEvalOpenAI(model=model_name, api_base=base_url, api_key=api_key, temperature=0, max_tokens=100)


# In[5]:


from deepeval.benchmarks.mera_multitask.multitask import MeraMultiTaskQA
from deepeval.benchmarks.mera_multitask.task import MeraMultiTaskQATask


# In[6]:


list(MeraMultiTaskQATask)


# In[7]:


# tasks = [MeraMultiTaskQATask.RuOpenBookQA, MeraMultiTaskQATask.SimpleAr]
# tasks = [MeraMultiTaskQATask.RuOpenBookQA]
# tasks = [MeraMultiTaskQATask.SimpleAr]
#tasks = [MeraMultiTaskQATask.RussianWinogradSchemaDataset]
#tasks = [MeraMultiTaskQATask.ruWorldTree]
#tasks = [MeraMultiTaskQATask.ruMultiAr]
#tasks = [MeraMultiTaskQATask.RussianCommitmentBank]
tasks = [MeraMultiTaskQATask.ruEthics]

# In[8]:


benchmark = MeraMultiTaskQA(tasks=tasks, n_shots=5, n_problems_per_task=5)
#benchmark = MeraMultiTaskQA(n_problems_per_task=5)
#benchmark = MeraMultiTaskQA()


# In[9]:


results = benchmark.evaluate(model=model) # batch_size=5


# In[11]:


print("Overall Score: ", results)


# In[ ]:




