from typing import cast

import numpy
from numpy.typing import NDArray

from .keys import TFHECloudKey
from .lwe import (
    LWESampleArray,
    lwe_add_mul_to,
    lwe_add_to,
    lwe_copy,
    lwe_key_switch,
    lwe_negate,
    lwe_noiseless_trivial,
    lwe_sub_mul_to,
    lwe_sub_to,
)
from .lwe_bootstrapping import tfhe_bootstrap_fft, tfhe_bootstrap_wo_ks_fft
from .numeric_functions import Torus32, mod_switch_to_torus32


def NAND(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped NAND gate
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,1/8) - ca - cb
    nand_const = mod_switch_to_torus32(1, 8)
    lwe_noiseless_trivial(temp_result, nand_const)
    lwe_sub_to(temp_result, ca)
    lwe_sub_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def OR(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped OR gate
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,1/8) + ca + cb
    or_const = mod_switch_to_torus32(1, 8)
    lwe_noiseless_trivial(temp_result, or_const)
    lwe_add_to(temp_result, ca)
    lwe_add_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def AND(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped AND gate
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) + ca + cb
    and_const = mod_switch_to_torus32(-1, 8)
    lwe_noiseless_trivial(temp_result, and_const)
    lwe_add_to(temp_result, ca)
    lwe_add_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def XOR(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped XOR gate
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,1/4) + 2*(ca + cb)
    xor_const = mod_switch_to_torus32(1, 4)
    lwe_noiseless_trivial(temp_result, xor_const)
    lwe_add_mul_to(temp_result, numpy.int32(2), ca)
    lwe_add_mul_to(temp_result, numpy.int32(2), cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def XNOR(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped XNOR gate
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,-1/4) + 2*(-ca-cb)
    xnor_const = mod_switch_to_torus32(-1, 4)
    lwe_noiseless_trivial(temp_result, xnor_const)
    lwe_sub_mul_to(temp_result, numpy.int32(2), ca)
    lwe_sub_mul_to(temp_result, numpy.int32(2), cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def NOT(result: LWESampleArray, ca: LWESampleArray) -> None:
    """
    * Homomorphic bootstrapped NOT gate (doesn't need to be bootstrapped)
    * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
    """
    lwe_negate(result, ca)


def COPY(result: LWESampleArray, ca: LWESampleArray) -> None:
    """
    * Homomorphic bootstrapped COPY gate (doesn't need to be bootstrapped)
    * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
    """
    lwe_copy(result, ca)


def CONSTANT(result: LWESampleArray, vals: bool | NDArray[numpy.int32]) -> None:
    """
    * Homomorphic Trivial Constant gate (doesn't need to be bootstrapped)
    * Takes a boolean value)
    * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    if isinstance(vals, numpy.ndarray):
        mus = cast(NDArray[Torus32], numpy.array([mu if x else -mu for x in vals]))
    else:
        mus = cast(
            NDArray[Torus32],
            numpy.ones(result.shape, numpy.int32) * (mu if vals else -mu),
        )
    lwe_noiseless_trivial(result, mus)


def NOR(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped NOR gate
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) - ca - cb
    nor_const = mod_switch_to_torus32(-1, 8)
    lwe_noiseless_trivial(temp_result, nor_const)
    lwe_sub_to(temp_result, ca)
    lwe_sub_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def ANDNY(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped AndNY Gate: not(a) and b
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) - ca + cb
    and_n_y_const = mod_switch_to_torus32(-1, 8)
    lwe_noiseless_trivial(temp_result, and_n_y_const)
    lwe_sub_to(temp_result, ca)
    lwe_add_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def ANDYN(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped AndYN Gate: a and not(b)
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) + ca - cb
    and_y_n_const = mod_switch_to_torus32(-1, 8)
    lwe_noiseless_trivial(temp_result, and_y_n_const)
    lwe_add_to(temp_result, ca)
    lwe_sub_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def ORNY(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped OrNY Gate: not(a) or b
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,1/8) - ca + cb
    or_n_y_const = mod_switch_to_torus32(1, 8)
    lwe_noiseless_trivial(temp_result, or_n_y_const)
    lwe_sub_to(temp_result, ca)
    lwe_add_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def ORYN(
    bk: TFHECloudKey, result: LWESampleArray, ca: LWESampleArray, cb: LWESampleArray
) -> None:
    """
    * Homomorphic bootstrapped OrYN Gate: a or not(b)
    * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LWESampleArray(in_out_params, result.shape)

    # compute: (0,1/8) + ca - cb
    or_y_n_const = mod_switch_to_torus32(1, 8)
    lwe_noiseless_trivial(temp_result, or_y_n_const)
    lwe_add_to(temp_result, ca)
    lwe_sub_to(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_fft(result, bk.bk_fft, mu, temp_result)


def MUX(
    bk: TFHECloudKey,
    result: LWESampleArray,
    a: LWESampleArray,
    b: LWESampleArray,
    c: LWESampleArray,
) -> None:
    """
    * Homomorphic bootstrapped Mux(a,b,c) = a?b:c = a*b + not(a)*c
    * Takes in input 3 LWE samples (with message space [-1/8,1/8], noise<1/16)
    * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
    """

    mu = mod_switch_to_torus32(1, 8)
    in_out_params = bk.params.in_out_params
    extracted_params = bk.params.tgsw_params.tlwe_params.extracted_lwe_params

    temp_result = LWESampleArray(in_out_params, result.shape)
    temp_result1 = LWESampleArray(extracted_params, result.shape)
    u1 = LWESampleArray(extracted_params, result.shape)
    u2 = LWESampleArray(extracted_params, result.shape)

    # compute "AND(a,b)": (0,-1/8) + a + b
    and_const = mod_switch_to_torus32(-1, 8)
    lwe_noiseless_trivial(temp_result, and_const)
    lwe_add_to(temp_result, a)
    lwe_add_to(temp_result, b)
    # Bootstrap without KeySwitch
    tfhe_bootstrap_wo_ks_fft(u1, bk.bk_fft, mu, temp_result)

    # compute "AND(not(a),c)": (0,-1/8) - a + c
    lwe_noiseless_trivial(temp_result, and_const)
    lwe_sub_to(temp_result, a)
    lwe_add_to(temp_result, c)
    # Bootstrap without KeySwitch
    tfhe_bootstrap_wo_ks_fft(u2, bk.bk_fft, mu, temp_result)

    # Add u1=u1+u2
    mux_const = mod_switch_to_torus32(1, 8)
    lwe_noiseless_trivial(temp_result1, mux_const)
    lwe_add_to(temp_result1, u1)
    lwe_add_to(temp_result1, u2)

    # Key switching
    lwe_key_switch(result, bk.bk_fft.ks, temp_result1)
