from datasets import Dataset


class RutieQATemplate:
    n_shot_dataset = None

    def __init__(self):
        super().__init__()

    def create_n_shot_examples(self, eval_set: Dataset):
        self.n_shot_dataset = eval_set

    def generate_output(self, input: str, n_shots: int, index: int = None):
        prompt = ""
        num_shots = max(index - n_shots, 0)
        for i in range(num_shots):
            one_shot = self.n_shot_dataset[i]
            if index is not None:
                if one_shot['_index'] == index:
                    one_shot = self.n_shot_examples[n_shots]
            prompt += self.format_question(
                one_shot
            )
        prompt += input
        return prompt

    def format_question(self, data: dict, n_shots: int, index: int = None):
        instruction = data["instruction"]
        inputs = data["inputs"]
        # question = inputs.pop("text")
        # subject = inputs["subject"]

        correct = data["outputs"]
        meta = data["meta"]

        context = ""
        num_shots = min(index, n_shots)
        for i in range(num_shots):
            one_shot = self.n_shot_dataset[index - num_shots + i]
            q_str = "\nuser: {question}\n1. {choice1}\n2. {choice2}\nКакой ответ из двух наиболее правильный?"
            q_str += "\nassistant: " + one_shot["outputs"] + "\n"
            context += q_str.format(**one_shot['inputs'])

        inputs['context'] = context

        prompt = instruction.format(**inputs)

        # options = ['A', 'B', 'C', 'D']
        # for option in options:
        #     option_
        #     pass

        #options: str = data["options"]
        #formatted_options = "\n".join(options.split(", "))
        #prompt = f"Question: {question}\n{formatted_options}\nAnswer:"
        #prompt += "\nОтвет: "
        # if include_answer:
        #     prompt += " {}\n\n".format(correct)
        return prompt

    @staticmethod
    def format_output(data: dict):
        return data["outputs"]
