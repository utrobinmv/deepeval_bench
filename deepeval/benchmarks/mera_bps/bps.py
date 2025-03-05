from datasets import load_dataset
from typing import List, Optional, Dict
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.utils import should_use_batch
from deepeval.benchmarks.schema import BaseModel, Literal
from deepeval.telemetry import capture_benchmark_run

from .template import BPSTemplate


class MultipleChoice01Schema(BaseModel):
    answer: Literal["0", "1"]

class BPS(DeepEvalBaseBenchmark):
    def __init__(
        self,
        n_shots: int = 5,
        n_problems: Optional[int] = None,
        verbose_mode: bool = False,
        confinement_instructions: Optional[str] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        assert n_shots <= 5, "MamuramuQA only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.n_problems: Optional[int] = n_problems
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None

        self.template = BPSTemplate()

        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode = verbose_mode
        if not confinement_instructions:
            self.confinement_instructions = (
                "Напиши только ответ 0 или 1, без пояснений."
            )
        else:
            self.confinement_instructions = confinement_instructions

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        with capture_benchmark_run("BPS", self.n_problems):
            overall_correct_predictions = 0

            goldens = self.load_benchmark_dataset()

            if (
                self.n_problems is not None
                and self.n_problems < len(goldens)
            ):
                goldens = goldens[: self.n_problems]

            overall_total_predictions = len(goldens)
            predictions_row = []

            # Solving each problem
            for idx, golden in enumerate(
                tqdm(goldens, desc=f"Processing {self.n_problems} problems")
            ):
                prediction, score = self.predict(model, golden).values()
                if score:
                    overall_correct_predictions += 1
                predictions_row.append(
                    (golden.input, prediction, golden.expected_output, score)
                )
                if self.verbose_mode:
                    self.print_verbose_logs(
                        idx,
                        golden.input,
                        golden.expected_output,
                        prediction,
                        score,
                    )

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall BPS Accuracy: {overall_accuracy}")

            self.predictions = pd.DataFrame(
                predictions_row,
                columns=["Input", "Prediction", "Expected Output", "Correct"],
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
        try:
            res: MultipleChoice01Schema = model.generate(
                prompt=prompt, schema=MultipleChoice01Schema
            )
            prediction = res.answer
        except TypeError:
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

    def load_benchmark_dataset(self) -> List[Golden]:
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("MERA-evaluation/MERA", 'bps')
            self.dataset = dataset

        # Construct test set
        train_set = dataset["train"]
        n_shot_indeces = [0, 25, 45, 70, 85, 95, 115, 135, 150, 170, 190, 195, 215, 30, 70]
        self.template.create_n_shot_examples(train_set, n_shot_indeces)

        test_set = train_set
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
            print(f"Problem {idx + 1}")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
