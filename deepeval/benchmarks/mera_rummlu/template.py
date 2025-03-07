from datasets import Dataset


class RuMMLUQATemplate:
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

        correct = data["outputs"]
        meta = data["meta"]

        prompt = instruction.format(**inputs)

        prompt += ""
        if include_answer:
            prompt += " {}\n\n".format(correct)
        return prompt

    @staticmethod
    def format_output(data: dict):
        return data["outputs"]
