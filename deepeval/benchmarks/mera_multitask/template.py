from datasets import Dataset

from .task import MeraMultiTaskQATask


class MeraMultiTaskQATemplate:
    n_shot_examples = []

    def __init__(self):
        super().__init__()

    def create_n_shot_examples(self, eval_set: Dataset, n_shot_indeces:list[int]):
        self.n_shot_examples = []
        for index in n_shot_indeces:
            example = eval_set[index]
            example['_index'] = index
            self.n_shot_examples.append(example)

    def generate_output(self, input: str, n_shots: int, index: int = None):
        prompt = ""
        for i in range(n_shots):
            one_shot = self.n_shot_examples[i]
            if index is not None:
                if one_shot['_index'] == index:
                    one_shot = self.n_shot_examples[n_shots]
            prompt += self.format_question(
                one_shot
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer=True):
        instruction = data["instruction"]
        inputs = data["inputs"]

        if not isinstance(inputs, dict):
            inputs = {"inputs": inputs}

        # question = inputs.pop("text")
        # subject = inputs["subject"]

        task = data["task"]

        correct = data["outputs"]
        #meta = data["meta"]

        if task == MeraMultiTaskQATask.ruEthics.value:
            correct = correct["moral"]

        prompt = instruction.format(**inputs)

        # options = ['A', 'B', 'C', 'D']
        # for option in options:
        #     option_
        #     pass

        #options: str = data["options"]
        #formatted_options = "\n".join(options.split(", "))
        #prompt = f"Question: {question}\n{formatted_options}\nAnswer:"
        if task == MeraMultiTaskQATask.SimpleAr.value:
            if not include_answer:
                prompt += f"\nВ качестве ответа напиши только число без дополнительных пояснений."
            prompt += f"\nОтвет:"
        if task == MeraMultiTaskQATask.RussianWinogradSchemaDataset.value:
            prompt += f"\nОтвет:"

        prompt += ""
        if include_answer:
            prompt += " {}\n\n".format(correct)
        return prompt

    @staticmethod
    def format_output(data: dict):
        task = data["task"]
        if task == MeraMultiTaskQATask.ruEthics.value:
            return data["outputs"]["moral"]

        return data["outputs"]
