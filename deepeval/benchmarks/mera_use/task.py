from enum import Enum


class UseQATask(Enum):
    Text = "text"
    Multiple_independent = "multiple_choice_independent_options"
    Multiple_within_text = "multiple_choice_options_within_text"
    Multiple_text = "multiple_choice_based_on_text"
    Matching = "matching"
