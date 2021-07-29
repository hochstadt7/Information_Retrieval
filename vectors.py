import math


def norm(arr):
    norm_squared = 0
    for a in arr:
        norm_squared += a*a
    return math.sqrt(norm_squared)
