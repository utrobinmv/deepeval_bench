from typing import List
import json
import requests
import httpx
import copy
from deepeval.models.base_model import DeepEvalBaseLLM


class DeepEvalOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        api_key: str,
        api_base: str = "http://localhost/v1",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 100,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        system_prompt: str = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.system_prompt = system_prompt

        self.generate_kwargs = \
            {'temperature': temperature, # Параметр, который контролирует "креативность" модели. Чем выше значение, тем более случайным будет вывод.
             'max_tokens': max_tokens, # Максимальное количество токенов в ответе.
             'top_p': top_p, # Параметр, который контролирует ядерную выборку (nucleus sampling). Чем ниже значение, тем более сфокусированным будет вывод.
             'frequency_penalty': frequency_penalty, # Штраф за частоту использования токенов. Положительные значения уменьшают вероятность повторения.
             'presence_penalty': presence_penalty # Штраф за новые токены. Положительные значения увеличивают вероятность новых тем.
             }

        # self.temperature = temperature 
        # self.max_tokens = max_tokens
        # self.top_p = top_p
        # self.frequency_penalty = frequency_penalty
        # self.presence_penalty = presence_penalty

    def _sanitize_prompt(self, prompt: str) -> str:
        """
        Очищает и экранирует промпт, если это необходимо.
        """
        try:
            # Попытка сериализовать промпт в JSON
            json.dumps(prompt)
            return prompt
        except (TypeError, ValueError):
            # Если промпт содержит невалидные символы, экранируем их
            return json.dumps(prompt)        
        
    def _pre_config_request(self, prompt: str, kwargs) -> tuple:
        # Очищаем промпт
        safe_prompt = self._sanitize_prompt(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": safe_prompt}],
        }

        generate_kwargs = copy.copy(self.generate_kwargs)

        for key in kwargs.keys():
            if key in generate_kwargs.keys():
                generate_kwargs[key] = kwargs[key]

        data.update(generate_kwargs)

        return headers, data

    def generate(self, prompt: str): # , generate_kwargs: dict = {}) -> str:
        generate_kwargs: dict = {}
        headers, data = self._pre_config_request(prompt, generate_kwargs)

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str, generate_kwargs: dict = {}) -> str:
        
        headers, data = self._pre_config_request(prompt, generate_kwargs)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def generate_samples(self, prompt: str, n: int = 3, temperature: float = None, generate_kwargs: dict = {}) -> List[str]:
        """
        Генерирует несколько текстовых образцов на основе заданного промпта.

        :param prompt: Промпт для генерации текста.
        :param n: Количество образцов для генерации.
        :return: Список сгенерированных текстов.
        """

        headers, data = self._pre_config_request(prompt, generate_kwargs)
        
        if temperature is not None:
            data["temperature"] = temperature
        
        data["n"] = n  # Количество образцов

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        # for choice in response.json()["choices"]:
        #     print(' -- ', choice["message"]["content"])

        return [choice["message"]["content"] for choice in response.json()["choices"]]

    async def a_generate_samples(self, prompt: str, n: int = 3, temperature: float = None) -> List[str]:
        """
        Асинхронно генерирует несколько текстовых образцов на основе заданного промпта.

        :param prompt: Промпт для генерации текста.
        :param n: Количество образцов для генерации.
        :return: Список сгенерированных текстов.
        """
        
        headers, data = self._pre_config_request(prompt, generate_kwargs)
        
        if temperature is not None:
            data["temperature"] = temperature
        
        data["n"] = n  # Количество образцов
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return [choice["message"]["content"] for choice in response.json()["choices"]]
        
    def get_model_name(self):
        return self.model
    
    def load_model(self):
        return self.model
