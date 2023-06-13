# pylint: disable=redefined-outer-name

import time
import warnings
from typing import Tuple

import numpy

from tfhe.boot_gates import AND, XOR
from tfhe.keys import (
    empty_ciphertext,
    tfhe_decrypt,
    tfhe_encrypt,
    tfhe_key_pair,
    tfhe_parameters,
)

rng = numpy.random.RandomState(123)

# See: https://en.wikipedia.org/wiki/Adder_(electronics)#Half_adder
EXPECTED = [
    {
        "bits": [False, False],  # A = 0  B = 0
        "result": [False, False],  # Carry = 0  Sum = 0
    },
    {
        "bits": [False, True],  # A = 0  B = 1
        "result": [False, True],  # Carry = 0  Sum = 1
    },
    {
        "bits": [True, False],  # A = 1  B = 0
        "result": [False, True],  # Carry = 0  Sum = 1
    },
    {
        "bits": [True, True],  # A = 1  B = 1
        "result": [True, False],  # Carry = 1  Sum = 0
    },
]


def test() -> None:
    for _, item in enumerate(EXPECTED):
        [bit1, bit2] = item["bits"]
        [expected_carry, expected_sum] = item["result"]

        [computed_carry, computed_sum] = run(bit1, bit2)

        assert computed_carry == expected_carry
        assert computed_sum == expected_sum


def run(bit1: bool, bit2: bool) -> Tuple[bool, bool]:
    secret_key, cloud_key = tfhe_key_pair(rng)

    ciphertext1 = tfhe_encrypt(rng, secret_key, numpy.array([bit1]))
    ciphertext2 = tfhe_encrypt(rng, secret_key, numpy.array([bit2]))

    params = tfhe_parameters(cloud_key)

    shape = ciphertext1.shape
    result_carry = empty_ciphertext(params, shape)
    result_sum = empty_ciphertext(params, shape)

    AND(cloud_key, result_carry, ciphertext1, ciphertext2)
    XOR(cloud_key, result_sum, ciphertext1, ciphertext2)

    answer_bits_carry = tfhe_decrypt(secret_key, result_carry)
    answer_bits_sum = tfhe_decrypt(secret_key, result_sum)

    return answer_bits_carry[0], answer_bits_sum[0]


if __name__ == "__main__":
    # FIXME: Ignores overflow detected by Numpy in # pylint: disable=fixme
    #   `lwe_key_switch_translate_from_array` method.
    warnings.filterwarnings("ignore", "overflow encountered in scalar subtract")

    total = 0.0
    for _, item in enumerate(EXPECTED):
        [bit1, bit2] = item["bits"]
        [expected_carry, expected_sum] = item["result"]
        print(f"Expected:\t Carry -> {expected_carry}  Sum -> {expected_sum}")

        t = time.time()
        [computed_carry, computed_sum] = run(bit1, bit2)
        print(f"Result:\t\t Carry -> {computed_carry}  Sum -> {computed_sum}")
        elapsed = time.time() - t
        print(f"Time:\t\t {elapsed} seconds")
        total += elapsed
        print()

    print(f"Avg. Time:\t {total / len(EXPECTED)} seconds")
