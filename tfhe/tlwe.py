from typing import Tuple, cast

import numpy
from numpy.typing import NDArray

from .lwe import LWEParams, LWESampleArray
from .numeric_functions import (
    Torus32,
    rand_gaussian_torus32,
    rand_uniform_int32,
    rand_uniform_torus32,
)
from .polynomials import (
    IntPolynomialArray,
    LagrangeHalfCPolynomialArray,
    TorusPolynomialArray,
    int_polynomial_ifft,
    lagrange_polynomial_clear,
    lagrange_polynomial_mul,
    torus_polynomial_add_to,
    torus_polynomial_clear,
    torus_polynomial_fft,
    torus_polynomial_ifft,
    torus_polynomial_mul_by_xai_minus_one,
)


class TLWEParams:
    def __init__(self, N: int, k: int, alpha_min: float, alpha_max: float) -> None:
        self.N = N  # a power of 2: degree of the polynomials
        self.k = k  # number of polynomials in the mask
        self.alpha_min = alpha_min  # minimal noise s.t. the sample is secure
        self.alpha_max = alpha_max  # maximal noise s.t. we can decrypt
        self.extracted_lwe_params = LWEParams(
            N * k, alpha_min, alpha_max
        )  # lwe params if one extracts


class TLWEKey:
    def __init__(self, rng: numpy.random.RandomState, params: TLWEParams) -> None:
        N = params.N
        k = params.k
        # GPU: array operation or RNG on device
        key = IntPolynomialArray(N, (k,))
        key.coefs[:, :] = rand_uniform_int32(rng, (k, N))

        self.params = params  # the parameters of the key
        self.key = key  # the key (i.e k binary polynomials)


class TLWESampleArray:
    def __init__(self, params: TLWEParams, shape: Tuple[int, ...]) -> None:
        self.k = params.k

        # array of length k+1: mask + right term
        self.a = TorusPolynomialArray(params.N, shape + (self.k + 1,))

        # avg variance of the sample
        self.current_variances = numpy.zeros(shape, numpy.float64)

        self.shape = shape


class TLWESampleFFTArray:
    def __init__(self, params: TLWEParams, shape: Tuple[int, ...]) -> None:
        self.k = params.k

        # array of length k+1: mask + right term
        self.a = LagrangeHalfCPolynomialArray(params.N, shape + (self.k + 1,))

        # avg variance of the sample
        self.current_variances = numpy.zeros(shape, numpy.float64)

        self.shape = shape


def tlwe_extract_lwe_sample_index(
    result: LWESampleArray,
    x: TLWESampleArray,
    index: int,
    params: LWEParams,
    r_params: TLWEParams,
) -> None:
    N = r_params.N
    k = r_params.k
    assert params.n == k * N

    # TODO: use an appropriate method to get coefs_t # pylint: disable=fixme
    a_view = result.a.reshape(result.shape + (k, N))
    a_view[:, :, : (index + 1)] = x.a.coefs_t[:, :k, index::-1]
    a_view[:, :, (index + 1) :] = -x.a.coefs_t[:, :k, :index:-1]

    numpy.copyto(result.b, x.a.coefs_t[:, k, index])


def tlwe_extract_lwe_sample(
    result: LWESampleArray, x: TLWESampleArray, params: LWEParams, r_params: TLWEParams
) -> None:
    tlwe_extract_lwe_sample_index(result, x, 0, params, r_params)


# create an homogeneous tlwe sample
def tlwe_sym_encrypt_zero(
    rng: numpy.random.RandomState, result: TLWESampleArray, alpha: float, key: TLWEKey
) -> None:
    N = key.params.N
    k = key.params.k

    # TODO: use an appropriate method # pylint: disable=fixme

    result.a.coefs_t[:, :, :, k, :] = rand_gaussian_torus32(
        rng, cast(Torus32, 0), alpha, result.shape + (N,)
    )

    result.a.coefs_t[:, :, :, :k, :] = rand_uniform_torus32(rng, result.shape + (k, N))

    tmp1 = LagrangeHalfCPolynomialArray(N, key.key.shape)
    tmp2 = LagrangeHalfCPolynomialArray(N, result.shape + (k,))
    tmp3 = LagrangeHalfCPolynomialArray(N, result.shape + (k,))
    tmp_r = TorusPolynomialArray(N, result.shape + (k,))

    int_polynomial_ifft(tmp1, key.key)
    torus_polynomial_ifft(
        tmp2, TorusPolynomialArray.from_arr(result.a.coefs_t[:, :, :, :k, :])
    )
    lagrange_polynomial_mul(tmp3, tmp1, tmp2)
    torus_polynomial_fft(tmp_r, tmp3)

    for i in range(k):
        result.a.coefs_t[:, :, :, k, :] = (
            result.a.coefs_t[:, :, :, k, :] + tmp_r.coefs_t[:, :, :, i, :]
        )

    result.current_variances.fill(alpha**2)


# Arithmetic operations on TLwe samples


# result = sample
def tlwe_copy(result: TLWESampleArray, sample: TLWESampleArray) -> None:
    # GPU: array operations or a custom kernel
    numpy.copyto(
        result.a.coefs_t, sample.a.coefs_t
    )  # TODO: use an appropriate method? # pylint: disable=fixme
    numpy.copyto(result.current_variances, sample.current_variances)


# result = (0,mu)
def tlwe_noiseless_trivial(result: TLWESampleArray, mu: TorusPolynomialArray) -> None:
    # GPU: array operations or a custom kernel
    torus_polynomial_clear(result.a)
    result.a.coefs_t[
        :, result.k, :
    ] = mu.coefs_t  # TODO: wrap in a function? # pylint: disable=fixme
    result.current_variances.fill(0.0)


# result = result + sample
def tlwe_add_to(result: TLWESampleArray, sample: TLWESampleArray) -> None:
    # GPU: array operations or a custom kernel
    torus_polynomial_add_to(result.a, sample.a)
    result.current_variances += sample.current_variances


# mult externe de X^ai-1 par bki
def tlwe_mul_by_xai_minus_one(
    result: TLWESampleArray, ai: NDArray[numpy.int32], bk: TLWESampleArray
) -> None:
    # TYPING: ai::Array{Int32}
    torus_polynomial_mul_by_xai_minus_one(result.a, ai, bk.a)


# Computes the inverse FFT of the coefficients of the TLWE sample
def tlwe_to_fft_convert(result: TLWESampleFFTArray, source: TLWESampleArray) -> None:
    torus_polynomial_ifft(result.a, source.a)
    numpy.copyto(result.current_variances, source.current_variances)


# Computes the FFT of the coefficients of the TLWEfft sample
def tlwe_from_fft_convert(result: TLWESampleArray, source: TLWESampleFFTArray) -> None:
    torus_polynomial_fft(result.a, source.a)
    numpy.copyto(result.current_variances, source.current_variances)


# Arithmetic operations on TLwe samples


# result = (0,0)
def tlwe_fft_clear(result: TLWESampleFFTArray) -> None:
    lagrange_polynomial_clear(result.a)
    result.current_variances.fill(0.0)
