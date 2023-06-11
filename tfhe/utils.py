import numpy
from numpy.typing import NDArray


def int_to_bitarray(x: int) -> NDArray[numpy.bool_]:
    return numpy.array([((x >> i) & 1 != 0) for i in range(16)])


def bitarray_to_int(x: NDArray[numpy.bool_]) -> int:
    int_answer = 0
    for i in range(16):
        int_answer = int_answer | (x[i] << i)
    return int_answer
