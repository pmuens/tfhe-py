# pylint: disable=duplicate-code
# pylint: disable=redefined-outer-name
# type: ignore

from typing import Tuple

import numpy
from numpy.typing import NDArray

from tfhe.boot_gates import tfhe_gate_CONSTANT_, tfhe_gate_MUX_, tfhe_gate_XNOR_
from tfhe.keys import (
    TFHECloudKey,
    TFHESecretKey,
    empty_ciphertext,
    tfhe_decrypt,
    tfhe_encrypt,
    tfhe_key_pair,
    tfhe_parameters,
)
from tfhe.lwe import LweSampleArray


def int_to_bitarray(x: int) -> NDArray[numpy.bool_]:
    return numpy.array([((x >> i) & 1 != 0) for i in range(16)])


def bitarray_to_int(x: NDArray[numpy.bool_]) -> int:
    int_answer = 0
    for i in range(16):
        int_answer = int_answer | (x[i] << i)
    return int_answer


def encrypt() -> Tuple[TFHESecretKey, TFHECloudKey, LweSampleArray, LweSampleArray]:
    rng = numpy.random.RandomState(123)

    secret_key, cloud_key = tfhe_key_pair(rng)

    bits1 = int_to_bitarray(2017)
    bits2 = int_to_bitarray(42)

    ciphertext1 = tfhe_encrypt(rng, secret_key, bits1)
    ciphertext2 = tfhe_encrypt(rng, secret_key, bits2)

    return secret_key, cloud_key, ciphertext1, ciphertext2


# elementary full comparator gate that is used to compare the i-th bit:
#   input: ai and bi the i-th bit of a and b
#          lsb_carry: the result of the comparison on the lowest bits
#   algo: if (a==b) return lsb_carry else return b
def encrypted_compare_bit_(
    cloud_key: TFHECloudKey,
    result: LweSampleArray,
    a: LweSampleArray,
    b: LweSampleArray,
    lsb_carry: LweSampleArray,
    tmp: LweSampleArray,
) -> None:
    tfhe_gate_XNOR_(cloud_key, tmp, a, b)
    tfhe_gate_MUX_(cloud_key, result, tmp, lsb_carry, a)


# this function compares two multibit words, and puts the max in result
def encrypted_minimum_(
    cloud_key: TFHECloudKey,
    result: LweSampleArray,
    a: LweSampleArray,
    b: LweSampleArray,
) -> None:
    nb_bits = result.shape[0]

    params = tfhe_parameters(cloud_key)

    tmp1 = empty_ciphertext(params, (1,))
    tmp2 = empty_ciphertext(params, (1,))

    # initialize the carry to 0
    tfhe_gate_CONSTANT_(tmp1, False)

    # run the elementary comparator gate n times
    for i in range(nb_bits):
        encrypted_compare_bit_(cloud_key, tmp1, a[i : i + 1], b[i : i + 1], tmp1, tmp2)

    # tmp1 is the result of the comparaison: 0 if a is larger, 1 if b is larger
    # select the max and copy it to the result
    tfhe_gate_MUX_(cloud_key, result, tmp1, b, a)


def process(
    cloud_key: TFHECloudKey, ciphertext1: LweSampleArray, ciphertext2: LweSampleArray
) -> LweSampleArray:
    # if necessary, the params are inside the key
    params = tfhe_parameters(cloud_key)

    # do some operations on the ciphertexts: here, we will compute the
    # minimum of the two
    result = empty_ciphertext(params, ciphertext1.shape)
    encrypted_minimum_(cloud_key, result, ciphertext1, ciphertext2)

    return result


def verify(secret_key: TFHESecretKey, answer: LweSampleArray) -> None:
    answer_bits = tfhe_decrypt(secret_key, answer)
    int_answer = bitarray_to_int(answer_bits)
    print("Answer:", int_answer)


secret_key, cloud_key, ciphertext1, ciphertext2 = encrypt()
answer = process(cloud_key, ciphertext1, ciphertext2)
verify(secret_key, answer)
