import time
import warnings

import numpy

from tfhe.boot_gates import MUX
from tfhe.keys import (
    empty_ciphertext,
    tfhe_decrypt,
    tfhe_encrypt,
    tfhe_key_pair,
    tfhe_parameters,
)
from tfhe.utils import bitarray_to_int, int_to_bitarray

rng = numpy.random.RandomState(123)

EXPECTED = 12345


def test() -> None:
    assert run() == EXPECTED


def run() -> int:
    secret_key, cloud_key = tfhe_key_pair(rng)

    bits2020 = int_to_bitarray(2020)
    bits42 = int_to_bitarray(42)
    bits12345 = int_to_bitarray(12345)

    ciphertext2020 = tfhe_encrypt(rng, secret_key, bits2020)
    ciphertext42 = tfhe_encrypt(rng, secret_key, bits42)
    ciphertext12345 = tfhe_encrypt(rng, secret_key, bits12345)

    params = tfhe_parameters(cloud_key)
    result = empty_ciphertext(params, ciphertext2020.shape)

    MUX(cloud_key, result, ciphertext2020, ciphertext42, ciphertext12345)

    answer_bits = tfhe_decrypt(secret_key, result)
    answer_int = bitarray_to_int(answer_bits)

    return answer_int


if __name__ == "__main__":
    # FIXME: Ignores overflow detected by Numpy in # pylint: disable=fixme
    #   `lwe_key_switch_translate_from_array` method.
    warnings.filterwarnings("ignore", "overflow encountered in scalar subtract")

    print(f"Expected:\t {EXPECTED}")
    t = time.time()
    res = run()
    print(f"Result:\t\t {res}")
    print(f"Time:\t\t {time.time() - t} seconds")
