from typing import cast

import numpy
from numpy.typing import NDArray

from .keys import TFHECloudKey
from .lwe import (
    LweSampleArray,
    lweAddMulTo,
    lweAddTo,
    lweCopy,
    lweKeySwitch,
    lweNegate,
    lweNoiselessTrivial,
    lweSubMulTo,
    lweSubTo,
)
from .lwe_bootstrapping import tfhe_bootstrap_FFT, tfhe_bootstrap_woKS_FFT
from .numeric_functions import Torus32, modSwitchToTorus32

# *#*****************************************
# zones on the torus -> to see
# *#*****************************************

# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped NAND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_NAND_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,1/8) - ca - cb
    NandConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, NandConst)
    lweSubTo(temp_result, ca)
    lweSubTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped OR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_OR_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,1/8) + ca + cb
    OrConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, OrConst)
    lweAddTo(temp_result, ca)
    lweAddTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped AND gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_AND_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) + ca + cb
    AndConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndConst)
    lweAddTo(temp_result, ca)
    lweAddTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped XOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_XOR_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,1/4) + 2*(ca + cb)
    XorConst = modSwitchToTorus32(1, 4)
    lweNoiselessTrivial(temp_result, XorConst)
    lweAddMulTo(temp_result, numpy.int32(2), ca)
    lweAddMulTo(temp_result, numpy.int32(2), cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped XNOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_XNOR_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,-1/4) + 2*(-ca-cb)
    XnorConst = modSwitchToTorus32(-1, 4)
    lweNoiselessTrivial(temp_result, XnorConst)
    lweSubMulTo(temp_result, numpy.int32(2), ca)
    lweSubMulTo(temp_result, numpy.int32(2), cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped NOT gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_NOT_(result: LweSampleArray, ca: LweSampleArray) -> None:
    lweNegate(result, ca)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped COPY gate (doesn't need to be bootstrapped)
 * Takes in input 1 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_COPY_(result: LweSampleArray, ca: LweSampleArray) -> None:
    lweCopy(result, ca)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic Trivial Constant gate (doesn't need to be bootstrapped)
 * Takes a boolean value)
 * Outputs a LWE sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_CONSTANT_(
    result: LweSampleArray, vals: bool | NDArray[numpy.int32]
) -> None:
    MU = modSwitchToTorus32(1, 8)
    if isinstance(vals, numpy.ndarray):
        mus = cast(NDArray[Torus32], numpy.array([MU if x else -MU for x in vals]))
    else:
        mus = cast(
            NDArray[Torus32],
            numpy.ones(result.shape, numpy.int32) * (MU if vals else -MU),
        )
    lweNoiselessTrivial(result, mus)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped NOR gate
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_NOR_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) - ca - cb
    NorConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, NorConst)
    lweSubTo(temp_result, ca)
    lweSubTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped AndNY Gate: not(a) and b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_ANDNY_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) - ca + cb
    AndNYConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndNYConst)
    lweSubTo(temp_result, ca)
    lweAddTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped AndYN Gate: a and not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_ANDYN_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,-1/8) + ca - cb
    AndYNConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndYNConst)
    lweAddTo(temp_result, ca)
    lweSubTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped OrNY Gate: not(a) or b
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_ORNY_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,1/8) - ca + cb
    OrNYConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, OrNYConst)
    lweSubTo(temp_result, ca)
    lweAddTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped OrYN Gate: a or not(b)
 * Takes in input 2 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_ORYN_(
    bk: TFHECloudKey, result: LweSampleArray, ca: LweSampleArray, cb: LweSampleArray
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params

    temp_result = LweSampleArray(in_out_params, result.shape)

    # compute: (0,1/8) + ca - cb
    OrYNConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result, OrYNConst)
    lweAddTo(temp_result, ca)
    lweSubTo(temp_result, cb)

    # if the phase is positive, the result is 1/8
    # if the phase is positive, else the result is -1/8
    tfhe_bootstrap_FFT(result, bk.bkFFT, MU, temp_result)


# pylint: disable=pointless-string-statement
"""
 * Homomorphic bootstrapped Mux(a,b,c) = a?b:c = a*b + not(a)*c
 * Takes in input 3 LWE samples (with message space [-1/8,1/8], noise<1/16)
 * Outputs a LWE bootstrapped sample (with message space [-1/8,1/8], noise<1/16)
"""
# pylint: enable=pointless-string-statement


def tfhe_gate_MUX_(
    bk: TFHECloudKey,
    result: LweSampleArray,
    a: LweSampleArray,
    b: LweSampleArray,
    c: LweSampleArray,
) -> None:
    MU = modSwitchToTorus32(1, 8)
    in_out_params = bk.params.in_out_params
    extracted_params = bk.params.tgsw_params.tlwe_params.extracted_lweparams

    temp_result = LweSampleArray(in_out_params, result.shape)
    temp_result1 = LweSampleArray(extracted_params, result.shape)
    u1 = LweSampleArray(extracted_params, result.shape)
    u2 = LweSampleArray(extracted_params, result.shape)

    # compute "AND(a,b)": (0,-1/8) + a + b
    AndConst = modSwitchToTorus32(-1, 8)
    lweNoiselessTrivial(temp_result, AndConst)
    lweAddTo(temp_result, a)
    lweAddTo(temp_result, b)
    # Bootstrap without KeySwitch
    tfhe_bootstrap_woKS_FFT(u1, bk.bkFFT, MU, temp_result)

    # compute "AND(not(a),c)": (0,-1/8) - a + c
    lweNoiselessTrivial(temp_result, AndConst)
    lweSubTo(temp_result, a)
    lweAddTo(temp_result, c)
    # Bootstrap without KeySwitch
    tfhe_bootstrap_woKS_FFT(u2, bk.bkFFT, MU, temp_result)

    # Add u1=u1+u2
    MuxConst = modSwitchToTorus32(1, 8)
    lweNoiselessTrivial(temp_result1, MuxConst)
    lweAddTo(temp_result1, u1)
    lweAddTo(temp_result1, u2)

    # Key switching
    lweKeySwitch(result, bk.bkFFT.ks, temp_result1)
