from random import random


def get_error(pct: float):
    src = 2 * random() - 1
    error = src * pct

    return error
