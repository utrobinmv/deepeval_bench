from datasets import load_dataset
from typing import List, Optional, Dict, Union
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
#from deepeval.benchmarks.schema import MultipleChoiceSchemaLower
from deepeval.telemetry import capture_benchmark_run
from deepeval.metrics.utils import initialize_model

from .task import RuDetoxTask
from .template import RuDetoxTemplate


class RuDetox(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[RuDetoxTask] = None,
        n_shots: int = 3,
        n_problems_per_task: Optional[int] = None,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        assert n_shots <= 5, "RuDetox only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[RuDetoxTask] = (
            list(RuDetoxTask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.evaluation_model, self.using_native_evaluation_model = (
            initialize_model(evaluation_model)
        )
        self.verbose_mode: bool = verbose_mode
        if not confinement_instructions:
            #self.confinement_instructions = "Output the answer, which should a text segment taken from the context."
            self.confinement_instructions = ""
        else:
            self.confinement_instructions = confinement_instructions

        self.template = RuDetoxTemplate()

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        with capture_benchmark_run("RuDetox", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []

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
                    f"RuDetox Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall RuDetox Accuracy: {overall_accuracy}")

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
        #     res: MultipleChoiceSchemaLower = model.generate(
        #         prompt=prompt, schema=MultipleChoiceSchemaLower
        #     )
        #     prediction = res.answer
        # except TypeError:
        prompt += f"\n\n{self.confinement_instructions}"
        prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        prediction = str(prediction)

        # Define Metric
        score = self.scorer.squad_score(
            golden.input,
            prediction,
            golden.expected_output,
            self.evaluation_model,
            self.using_native_evaluation_model,
        )
        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self, task: RuDetoxTask) -> List[Golden]:
        dataset = load_dataset("MERA-evaluation/MERA", 'rudetox')
        self.dataset = dataset

        train_set = dataset["train"]

        def add_index(example, idx): return {"index": idx}
        train_set = train_set.map(add_index, with_indices=True)

        # Construct test set
        test_set = train_set
        # test_set = dataset["train"].filter(
        #     lambda data: data["title"] == task.value
        # )
        n_shot_indeces = [0, 25, 46, 70, 85, 95, 115, 135, 150, 170, 190, 195, 215, 250, 260]
        self.template.create_n_shot_examples(train_set, n_shot_indeces)

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
