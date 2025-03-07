import datasets
#from datasets import load_dataset
from typing import List, Optional, Dict
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.utils import should_use_batch
#from deepeval.benchmarks.schema import MultipleChoiceSchemaLower
from deepeval.benchmarks.schema import BaseModel, Literal
from deepeval.telemetry import capture_benchmark_run

from .task import MeraMultiTaskQATask
from .template import MeraMultiTaskQATemplate


# class MultipleChoiceSchema(BaseModel):
#     answer: Literal["A", "B", "C", "D"]

class MeraMultiTaskQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[MeraMultiTaskQATask] = None,
        n_shots: int = 5,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        assert n_shots <= 5, "MeraMultiTaskQA only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[MeraMultiTaskQATask] = (
            list(MeraMultiTaskQATask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None

        self.template = MeraMultiTaskQATemplate()

        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode = verbose_mode

        self.dataset = {}
        # if not confinement_instructions:
        #     self.confinement_instructions = (
        #         "Output 'a', 'b', 'c', or 'd'. Full answer not needed."
        #     )
        # else:

        self.confinement_instructions = confinement_instructions

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        with capture_benchmark_run("MeraMultiTaskQA", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []
            #use_batch = should_use_batch(model, batch_size)

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task)
                if (
                    self.n_problems_per_task is not None
                    and self.n_problems_per_task < len(goldens)
                ):
                    goldens = goldens[: self.n_problems_per_task]
                task_correct_predictions = 0
                task_total_predictions = len(goldens)
                overall_total_predictions += len(goldens)

                for idx, golden in enumerate(
                    tqdm(goldens, desc=f"Processing {task.value}")
                ):
                    prediction, score = self.predict(model, golden).values()
                    if score:
                        task_correct_predictions += 1
                        overall_correct_predictions += 1
                    predictions_row.append(
                        (
                            task.value,
                            golden.input,
                            prediction,
                            golden.expected_output,
                            score,
                        )
                    )
                    if self.verbose_mode:
                        self.print_verbose_logs(
                            idx,
                            task.value,
                            golden.input,
                            golden.expected_output,
                            prediction,
                            score,
                        )

                task_accuracy = (
                    task_correct_predictions / task_total_predictions
                )
                print(
                    f"MeraMultiTaskQA Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall MeraMultiTaskQA Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            # Columns: 'Task', 'Input', 'Prediction', 'Score'
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=[
                    "Task",
                    "Input",
                    "Prediction",
                    "Expected Output",
                    "Correct",
                ],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        prompt: dict = self.template.generate_output(
            input=golden.input,
            n_shots=self.n_shots,
            index=golden.index
        )

        # Enforced model generation
        # try:
        #     res: MultipleChoiceSchema = model.generate(
        #         prompt=prompt, schema=MultipleChoiceSchema
        #     )
        #     prediction = res.answer
        # except TypeError:
        if self.confinement_instructions:
            prompt += f"\n\n{self.confinement_instructions}"
        prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = str(prediction)

        # Define Metric
        score = self.scorer.exact_match_score(
            golden.expected_output, prediction
        )
        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self, task: MeraMultiTaskQATask) -> List[Golden]:
        if task.value in self.dataset.keys():
            dataset = self.dataset[task.value]
        else:
            if task == MeraMultiTaskQATask.RuOpenBookQA:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'ruopenbookqa')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.RuOpenBookQA.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['train']

            elif task == MeraMultiTaskQATask.SimpleAr:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'simplear')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.SimpleAr.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['train']

            elif task == MeraMultiTaskQATask.RussianWinogradSchemaDataset:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'rwsd')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.RussianWinogradSchemaDataset.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['validation']

            elif task == MeraMultiTaskQATask.ruWorldTree:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'ruworldtree')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.ruWorldTree.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['train']

            elif task == MeraMultiTaskQATask.ruMultiAr:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'rumultiar')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.ruMultiAr.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['train']

            elif task == MeraMultiTaskQATask.RussianCommitmentBank:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'rcb')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.RussianCommitmentBank.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['validation']

            elif task == MeraMultiTaskQATask.ruEthics:
                dataset = datasets.load_dataset("MERA-evaluation/MERA", 'ruethics')
                def add_taskname(_, idx): return {"task": MeraMultiTaskQATask.ruEthics.value}
                dataset = dataset.map(add_taskname, with_indices=True)
                def add_index(example, idx): return {"index": idx}
                dataset = dataset.map(add_index, with_indices=True)
                dataset = dataset['test']


            self.dataset[task.value] = dataset

        # Construct test set
        n_shot_indeces = [0, 25, 45, 70, 85, 95, 115, 135, 150, 170, 190, 195, 215, 250, 260]
        if task == MeraMultiTaskQATask.RussianWinogradSchemaDataset:
            n_shot_indeces = [0, 25, 45, 70, 85, 95, 115, 135, 150, 170, 190, 195, 115, 150, 200]
        if task == MeraMultiTaskQATask.RussianCommitmentBank:
            n_shot_indeces = [0, 25, 45, 70, 85, 95, 115, 135, 150, 170, 190, 195, 115, 150, 200]
        if task == MeraMultiTaskQATask.ruWorldTree:
            n_shot_indeces = [0, 25, 45, 70, 85, 95, 15, 35, 50, 75, 19, 17, 80, 85, 95]
        self.template.create_n_shot_examples(dataset, n_shot_indeces)

        test_set = dataset.filter(
            lambda data: data["task"] == task.value
        )
        goldens: List[Golden] = []
        for data in test_set:
            index = data['index']
            input = self.template.format_question(data, include_answer=False)
            expected_output = self.template.format_output(data)
            golden = Golden(input=input, expected_output=expected_output, index=index)
            goldens.append(golden)
        return goldens

    def print_verbose_logs(
        self,
        idx: int,
        task_value: str,
        input: str,
        expected_output: str,
        prediction: str,
        score: int,
    ) -> str:
        steps = [
            f"Input:\n{input}",
            f"Score: {score}\nPrediction: {prediction}\nExpected Output: {expected_output}",
        ]
        verbose_logs = ""
        for i in range(len(steps) - 1):
            verbose_logs += steps[i]

            # don't add new line for penultimate step
            if i < len(steps) - 2:
                verbose_logs += " \n \n"

        if self.verbose_mode:
            print("*" * 50)
            print(f"Problem {idx + 1} (Task = {task_value})")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
