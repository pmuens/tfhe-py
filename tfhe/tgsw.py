from typing import Tuple, cast

import numpy
from numpy.typing import NDArray

from .numeric_functions import int64_to_int32
from .polynomials import (
    IntPolynomialArray,
    LagrangeHalfCPolynomialArray,
    TorusPolynomialArray,
    int_polynomial_ifft,
)
from .tlwe import (
    TLWEKey,
    TLWEParams,
    TLWESampleArray,
    TLWESampleFFTArray,
    tlwe_fft_clear,
    tlwe_from_fft_convert,
    tlwe_sym_encrypt_zero,
    tlwe_to_fft_convert,
)


class TGSWParams:
    def __init__(
        self, l: int, bg_bit: int, tlwe_params: TLWEParams  # noqa: E741
    ) -> None:
        bg = 1 << bg_bit
        half_bg = bg // 2

        h = numpy.int32(1) << (
            32 - numpy.arange(1, l + 1) * bg_bit
        )  # 1/(bg^(i+1)) as a Torus32

        # offset = bg/2 * (2^(32-bg_bit) + 2^(32-2*bg_bit) + ... + 2^(32-l*bg_bit))
        offset = int64_to_int32(
            cast(
                numpy.int64, sum(1 << (32 - numpy.arange(1, l + 1) * bg_bit)) * half_bg
            )
        )

        self.l = l  # decomp length # noqa: E741
        self.bg_bit = bg_bit  # log_2(bg)
        self.bg = bg  # decomposition base (must be a power of 2)
        self.half_bg = half_bg  # bg/2
        self.mask_mod = bg - 1  # bg-1
        self.tlwe_params = tlwe_params  # Params of each row
        self.kpl = (tlwe_params.k + 1) * l  # number of rows = (k+1)*l
        self.h = h  # powers of bg_bit
        self.offset = offset  # offset = bg/2 * (2^(32-bg_bit) + 2^(32-2*bg_bit) + ... + 2^(32-l*bg_bit)) # pylint: disable=line-too-long # noqa: E501


class TGSWKey:
    def __init__(self, rng: numpy.random.RandomState, params: TGSWParams) -> None:
        tlwe_key = TLWEKey(rng, params.tlwe_params)
        self.params = params  # the parameters
        self.tlwe_params = params.tlwe_params  # the tlwe params of each rows
        self.tlwe_key = tlwe_key


class TGSWSampleArray:
    def __init__(self, params: TGSWParams, shape: Tuple[int, ...]) -> None:
        self.k = params.tlwe_params.k
        self.l = params.l  # noqa: E741
        self.samples = TLWESampleArray(params.tlwe_params, shape + (self.k + 1, self.l))


class TGSWSampleFFTArray:
    def __init__(self, params: TGSWParams, shape: Tuple[int, ...]) -> None:
        self.k = params.tlwe_params.k
        self.l = params.l  # noqa: E741
        self.samples = TLWESampleFFTArray(
            params.tlwe_params, shape + (self.k + 1, self.l)
        )


# Result += mu*H, mu integer
def tgsw_add_mu_int_h(
    result: TGSWSampleArray, messages: NDArray[numpy.int32], params: TGSWParams
) -> None:
    # TYPING: messages::Array{Int32, 1}

    k = params.tlwe_params.k
    l = params.l  # noqa: E741
    h = params.h

    # compute result += H

    # returns an underlying coefs_t of TorusPolynomialArray, with the total size
    # (N, k + 1 [from TLweSample], l, k + 1 [from TGswSample], n)
    # messages: (n,)
    # h: (l,)
    # TODO: use an appropriate method # pylint: disable=fixme
    # TODO: not sure if it's possible to fully vectorize it # pylint: disable=fixme
    for bloc in range(k + 1):
        result.samples.a.coefs_t[:, bloc, :, bloc, 0] = result.samples.a.coefs_t[
            :, bloc, :, bloc, 0
        ] + messages.reshape(messages.size, 1) * h.reshape(1, l)


# Result = tGsw(0)
def tgsw_encrypt_zero(
    rng: numpy.random.RandomState, result: TGSWSampleArray, alpha: float, key: TGSWKey
) -> None:
    rlkey = key.tlwe_key
    tlwe_sym_encrypt_zero(rng, result.samples, alpha, rlkey)


# encrypts a constant message
def tgsw_sym_encrypt_int(
    rng: numpy.random.RandomState,
    result: TGSWSampleArray,
    messages: NDArray[numpy.int32],
    alpha: float,
    key: TGSWKey,
) -> None:
    # TYPING: messages::Array{Int32, 1}
    tgsw_encrypt_zero(rng, result, alpha, key)
    tgsw_add_mu_int_h(result, messages, key.params)


def tgsw_torus32_polynomial_decomp_h(
    result: IntPolynomialArray, sample: TorusPolynomialArray, params: TGSWParams
) -> None:
    # GPU: array operations or (more probably) a custom kernel

    N = params.tlwe_params.N
    l = params.l  # noqa: E741
    bg_bit = params.bg_bit

    mask_mod = params.mask_mod
    half_bg = params.half_bg
    offset = params.offset

    def decal(p: NDArray[numpy.int32]) -> NDArray[numpy.int32]:
        return 32 - p * bg_bit

    ps = numpy.arange(1, l + 1).reshape(1, 1, l, 1)
    sample_coefs = sample.coefs_t.reshape(sample.shape + (1, N))

    # do the decomposition
    result.coefs[:, :, :, :] = (
        ((sample_coefs + offset) >> decal(ps)) & mask_mod
    ) - half_bg


# For all the kpl TLWE samples composing the TGSW sample
# It computes the inverse FFT of the coefficients of the TLWE sample
def tgsw_to_fft_convert(result: TGSWSampleFFTArray, source: TGSWSampleArray) -> None:
    tlwe_to_fft_convert(result.samples, source.samples)


def tLwe_fft_add_mul_t_to(
    res: NDArray[numpy.complex128],
    a: NDArray[numpy.complex128],
    b: NDArray[numpy.complex128],
    bk_idx: int,
) -> None:
    # GPU: array operations or (more probably) a custom kernel

    ml, k_plus1, n_div2 = res.shape
    l = a.shape[-2]  # noqa: E741

    d = a.reshape(ml, k_plus1, l, 1, n_div2)
    for i in range(k_plus1):
        for j in range(l):
            res += d[:, i, j, :, :] * b[bk_idx, i, j, :, :]


# External product (*): accum = gsw (*) accum
def tgsw_fft_extern_mul_to_tlwe(
    accum: TLWESampleArray,
    gsw: TGSWSampleFFTArray,
    bk_idx: int,
    params: TGSWParams,
    tmp_a: TLWESampleFFTArray,
    deca: IntPolynomialArray,
    deca_fft: LagrangeHalfCPolynomialArray,
) -> None:
    tgsw_torus32_polynomial_decomp_h(deca, accum.a, params)

    int_polynomial_ifft(deca_fft, deca)

    tlwe_fft_clear(tmp_a)

    res = tmp_a.a.coefs_c
    a = deca_fft.coefs_c
    b = gsw.samples.a.coefs_c

    tLwe_fft_add_mul_t_to(res, a, b, bk_idx)

    tlwe_from_fft_convert(accum, tmp_a)
