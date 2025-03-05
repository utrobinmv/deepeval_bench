from datasets import load_dataset
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

from .task import UseQATask
from .template import UseQATemplate


class UseQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[UseQATask] = None,
        n_shots: int = 5,
        n_problems_per_task: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        assert n_shots <= 5, "UseQA only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[UseQATask] = (
            list(UseQATask) if tasks is None else tasks
        )
        self.n_problems_per_task: Optional[int] = n_problems_per_task
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None

        self.template = UseQATemplate()

        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode = verbose_mode
        # if not confinement_instructions:
        #     self.confinement_instructions = (
        #         "Output 'a', 'b', 'c', or 'd'. Full answer not needed."
        #     )
        # else:

        self.confinement_instructions = confinement_instructions

    def evaluate(
        self, model: DeepEvalBaseLLM, batch_size: Optional[int] = None
    ) -> Dict:
        with capture_benchmark_run("UseQA", len(self.tasks)):
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
                    f"UseQA Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall UseQA Accuracy: {overall_accuracy}")

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

    def load_benchmark_dataset(self, task: UseQATask) -> List[Golden]:
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("MERA-evaluation/MERA", 'use')
            self.dataset = dataset

        # Construct test set
        train_set = dataset["train"]
        n_shot_indeces = {'text': [1839, 2165, 2094, 1884, 1861, 2360, 1818, 2228, 
                                   2357, 2479, 2466, 2244, 2281, 2527, 1960],
                          'multiple_choice_independent_options': [1037, 1257, 1782, 1435, 
                                                                  991, 1774, 1050, 1185,
                                                                  1605, 999, 1302, 1001,
                                                                  1716, 981, 1172],
                          'multiple_choice_options_within_text': [900, 591, 389, 647,
                                                                  848, 952, 837, 786,
                                                                  547, 778, 814, 387,
                                                                  594, 708, 780],
                          'multiple_choice_based_on_text': [50, 2, 89, 178, 61, 310,
                                                            67, 290, 64, 19, 185,
                                                            121, 87, 56, 231],
                          'matching': [2580, 2618, 2540, 2546, 2587, 2604,
                                       2536, 2610, 2601, 2551, 2616, 2583,
                                       2541, 2543, 2529]
                          }
        
        self.template.create_n_shot_examples(train_set, n_shot_indeces)

        test_set = train_set.filter(
            lambda data: data['meta']["type"] == task.value
        )
        goldens: List[Golden] = []
        for index, data in enumerate(test_set):
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
