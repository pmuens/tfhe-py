import random
import time
import warnings

import numpy
from numpy.typing import NDArray

from tfhe.boot_gates import NAND
from tfhe.keys import (
    empty_ciphertext,
    tfhe_decrypt,
    tfhe_encrypt,
    tfhe_key_pair,
    tfhe_parameters,
)

rng = numpy.random.RandomState(123)

size = 8
bits1 = numpy.array([random.choice([False, True]) for i in range(size)])
bits2 = numpy.array([random.choice([False, True]) for i in range(size)])
EXPECTED = numpy.array([not (b1 and b2) for b1, b2 in zip(bits1, bits2)])


def test() -> None:
    assert (run() == EXPECTED).all()


def run() -> NDArray[numpy.bool_]:
    secret_key, cloud_key = tfhe_key_pair(rng)

    ciphertext1 = tfhe_encrypt(rng, secret_key, bits1)
    ciphertext2 = tfhe_encrypt(rng, secret_key, bits2)

    params = tfhe_parameters(cloud_key)

    result = empty_ciphertext(params, ciphertext1.shape)

    NAND(cloud_key, result, ciphertext1, ciphertext2)

    answer_bits = tfhe_decrypt(secret_key, result)

    return answer_bits


if __name__ == "__main__":
    # FIXME: Ignores overflow detected by Numpy in # pylint: disable=fixme
    #   `lwe_key_switch_translate_from_array` method.
    warnings.filterwarnings("ignore", "overflow encountered in scalar subtract")

    print(f"Expected:\t {EXPECTED}")
    t = time.time()
    res = run()
    print(f"Result:\t\t {res}")
    print(f"Time:\t\t {time.time() - t} seconds")
