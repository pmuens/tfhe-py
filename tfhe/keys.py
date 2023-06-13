from typing import Tuple, cast

import numpy
from numpy.typing import NDArray

from .lwe import LWEKey, LWEParams, LWESampleArray, lwe_phase, lwe_sym_encrypt
from .lwe_bootstrapping import LWEBootstrappingKeyFFT
from .numeric_functions import mod_switch_to_torus32
from .tgsw import TGSWKey, TGSWParams
from .tlwe import TLWEParams


class TFHEParameters:
    def __init__(self) -> None:
        # In the reference implementation there was a parameter `minimum_lambda` here,
        # which was unused.

        # the parameters are only implemented for about 128bit of security!

        def mul_by_sqrt_two_over_pi(x: float) -> float:
            return cast(float, x * (2 / numpy.pi) ** 0.5)

        N = 1024
        k = 1
        n = 500
        bk_l = 2
        bk_bg_bit = 10
        ks_base_bit = 2
        ks_length = 8
        ks_stdev = mul_by_sqrt_two_over_pi(1 / 2**15)  # standard deviation
        bk_stdev = mul_by_sqrt_two_over_pi(9e-9)  # standard deviation
        max_stdev = mul_by_sqrt_two_over_pi(
            1 / 2**4 / 4
        )  # max standard deviation for a 1/4 msg space

        params_in = LWEParams(n, ks_stdev, max_stdev)
        params_accum = TLWEParams(N, k, bk_stdev, max_stdev)
        params_bk = TGSWParams(bk_l, bk_bg_bit, params_accum)

        self.ks_t = ks_length
        self.ks_base_bit = ks_base_bit
        self.in_out_params = params_in
        self.tgsw_params = params_bk


class TFHESecretKey:
    def __init__(
        self, params: TFHEParameters, lwe_key: LWEKey, tgsw_key: TGSWKey
    ) -> None:
        self.params = params
        self.lwe_key = lwe_key
        self.tgsw_key = tgsw_key


class TFHECloudKey:
    def __init__(self, params: TFHEParameters, bk_fft: LWEBootstrappingKeyFFT) -> None:
        self.params = params
        self.bk_fft = bk_fft


def tfhe_parameters(
    key: TFHECloudKey,
) -> TFHEParameters:  # union(TFHESecretKey, TFHECloudKey)
    return key.params


def tfhe_key_pair(rng: numpy.random.RandomState) -> Tuple[TFHESecretKey, TFHECloudKey]:
    params = TFHEParameters()

    lwe_key = LWEKey.from_rng(rng, params.in_out_params)
    tgsw_key = TGSWKey(rng, params.tgsw_params)
    secret_key = TFHESecretKey(params, lwe_key, tgsw_key)

    bk_fft = LWEBootstrappingKeyFFT(
        rng, params.ks_t, params.ks_base_bit, lwe_key, tgsw_key
    )
    cloud_key = TFHECloudKey(params, bk_fft)

    return secret_key, cloud_key


def tfhe_encrypt(
    rng: numpy.random.RandomState,
    key: TFHESecretKey,
    message: NDArray[numpy.bool_],
) -> LWESampleArray:
    result = empty_ciphertext(key.params, message.shape)
    _1s8 = mod_switch_to_torus32(1, 8)
    mus = numpy.array([_1s8 if bit else -_1s8 for bit in message])
    alpha = (
        key.params.in_out_params.alpha_min
    )  # TODO: specify noise # pylint: disable=fixme
    lwe_sym_encrypt(rng, result, mus, alpha, key.lwe_key)
    return result


def tfhe_decrypt(
    key: TFHESecretKey, ciphertext: LWESampleArray
) -> NDArray[numpy.bool_]:
    mus = lwe_phase(ciphertext, key.lwe_key)
    return numpy.array([(mu > 0) for mu in mus])


def empty_ciphertext(params: TFHEParameters, shape: Tuple[int, ...]) -> LWESampleArray:
    return LWESampleArray(params.in_out_params, shape)
