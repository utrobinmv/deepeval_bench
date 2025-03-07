from enum import Enum


class RuModArQATask(Enum):
    ThreeDigitAddControl = "three_digit_addition_control"
    ThreeDigitAddOne = "three_digit_addition_plus_one"
    ThreeDigitSubControl = "three_digit_subtraction_control"
    ThreeDigitSubOne = "three_digit_subtraction_plus_one"
    TwoDigitMulControl = "two_digit_multiplication_control"
    TwoDigitMulOne = "two_digit_multiplication_plus_one"
