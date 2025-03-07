from datasets import Dataset

from .task import RuModArQATask


class RuModArQATemplate:
    test_set = None
    n_shot_examples = {}

    def __init__(self):
        super().__init__()

    def create_n_shot_examples(self, eval_set: Dataset, n_shot_indeces:dict[list[int]]):
        self.test_set = eval_set

        self.n_shot_examples = {}
        for task in n_shot_indeces.keys():
            self.n_shot_examples[task] = []
            for index in n_shot_indeces[task]:
                example = self.test_set[index]
                example['_index'] = index
                self.n_shot_examples[task].append(example)

    def generate_output(self, input: str, n_shots: int, index: int = None):
        prompt = ""
        data = self.test_set[index]
        task = data['meta']['task_type']

        for i in range(n_shots):
            one_shot = self.n_shot_examples[task][i]
            if index is not None:
                if one_shot['_index'] == index:
                    one_shot = self.n_shot_examples[task][n_shots]
            prompt += self.format_question(
                one_shot
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer=True):
        instruction = data["instruction"]
        inputs = {'inputs': data["inputs"]}
        # question = inputs.pop("text")
        # subject = inputs["subject"]

        correct = data["outputs"]
        meta = data["meta"]
        task = data['meta']['task_type']

        prompt = instruction.format(**inputs)

        # options = ['A', 'B', 'C', 'D']
        # for option in options:
        #     option_
        #     pass

        #options: str = data["options"]
        #formatted_options = "\n".join(options.split(", "))
        #prompt = f"Question: {question}\n{formatted_options}\nAnswer:"
        #prompt += "\nОтвет: "
        if include_answer:
            prompt += " {}\n\n".format(correct)
        # else:
        #     if task == RuModArQATask.Multiple_independent.value:
        #         prompt += " Напиши только число без дополнительных пояснений."
        #     if task == RuModArQATask.Multiple_within_text.value:
        #         prompt += " В качестве ответа напиши только число или числа через запятую без пробелов и без дополнительных пояснений."
            
        return prompt

    @staticmethod
    def format_output(data: dict):
        return data["outputs"]
