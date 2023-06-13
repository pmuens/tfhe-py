from typing import Tuple, cast

import numpy
from numpy.typing import NDArray

from .numeric_functions import Torus32, float_to_int32


# This structure represents an integer polynomial modulo X^N+1
class IntPolynomialArray:
    def __init__(self, N: int, shape: Tuple[int, ...]) -> None:
        self.coefs = numpy.empty(shape + (N,), numpy.int32)
        self.polynomial_size = N
        self.shape = shape


# This structure represents an torus polynomial modulo X^N+1
class TorusPolynomialArray:
    def __init__(self, N: int, shape: Tuple[int, ...]) -> None:
        self.coefs_t = numpy.empty(shape + (N,), cast(Torus32, numpy.int32))
        self.polynomial_size = N
        self.shape = shape

    @classmethod
    def from_arr(cls, arr: NDArray[Torus32]) -> "TorusPolynomialArray":
        obj = cls(arr.shape[-1], arr.shape[:-1])
        obj.coefs_t = arr
        return obj


# This structure is used for FFT operations, and is a representation
# over C of a polynomial in R[X]/X^N+1
class LagrangeHalfCPolynomialArray:
    def __init__(self, N: int, shape: Tuple[int, ...]) -> None:
        assert N % 2 == 0
        self.coefs_c = numpy.empty(shape + (N // 2,), numpy.complex128)
        self.polynomial_size = N
        self.shape = shape


def coefs(
    p: TorusPolynomialArray | IntPolynomialArray | LagrangeHalfCPolynomialArray,
) -> NDArray[Torus32 | numpy.int32 | numpy.complex128]:
    # TODO: different field names help with debugging, remove later # pylint: disable=fixme # noqa: E501
    if isinstance(p, IntPolynomialArray):
        return p.coefs
    if isinstance(p, TorusPolynomialArray):
        return p.coefs_t
    return p.coefs_c


def flat_coefs(
    p: TorusPolynomialArray | IntPolynomialArray | LagrangeHalfCPolynomialArray,
) -> NDArray[Torus32 | numpy.int32 | numpy.complex128]:
    cp = coefs(p)
    return cp.reshape(numpy.prod(p.shape), cp.shape[-1])


def polynomial_size(
    p: TorusPolynomialArray | IntPolynomialArray | LagrangeHalfCPolynomialArray,
) -> int:
    return p.polynomial_size


def prepare_ifft_input(
    rev_in: NDArray[numpy.float64],
    a: NDArray[Torus32 | numpy.int32 | numpy.complex128],
    coeff: float,
    N: int,
) -> None:
    rev_in[:, :N] = a * coeff
    rev_in[:, N:] = -rev_in[:, :N]


def prepare_ifft_output(
    res: NDArray[Torus32 | numpy.int32 | numpy.complex128],
    rev_out: NDArray[numpy.complex128],
    N: int,
) -> None:
    # FIXME: when Julia is smart enough, can be replaced by: # pylint: disable=fixme
    res[:, : N // 2] = rev_out[:, 1 : N + 1 : 2]


def int_polynomial_ifft(
    result: LagrangeHalfCPolynomialArray, p: IntPolynomialArray
) -> None:
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], 2 * N), numpy.float64)
    prepare_ifft_input(in_arr, a, 1 / 2, N)
    out_arr = numpy.fft.rfft(in_arr)
    prepare_ifft_output(res, out_arr, N)


def torus_polynomial_ifft(
    result: LagrangeHalfCPolynomialArray, p: TorusPolynomialArray
) -> None:
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], 2 * N), numpy.float64)
    prepare_ifft_input(in_arr, a, 1 / 2**33, N)
    out_arr = numpy.fft.rfft(in_arr)
    prepare_ifft_output(res, out_arr, N)


def prepare_fft_input(
    fw_in: NDArray[numpy.complex128],
    a: NDArray[Torus32 | numpy.int32 | numpy.complex128],
    N: int,
) -> None:
    fw_in[:, 0 : N + 1 : 2] = 0
    fw_in[:, 1 : N + 1 : 2] = a


def prepare_fft_output(
    res: NDArray[Torus32 | numpy.int32 | numpy.complex128],
    fw_out: NDArray[numpy.float64],
    coef: float,
    N: int,
) -> None:
    res[:, :] = float_to_int32(fw_out[:, :N] * coef)


def torus_polynomial_fft(
    result: TorusPolynomialArray, p: LagrangeHalfCPolynomialArray
) -> None:
    res = flat_coefs(result)
    a = flat_coefs(p)
    N = polynomial_size(p)

    in_arr = numpy.empty((res.shape[0], N + 1), numpy.complex128)
    prepare_fft_input(in_arr, a, N)
    out_arr = numpy.fft.irfft(in_arr)

    # the first part is from the original libtfhe;
    # the second part is from a different FFT scaling in Julia
    coeff: float = (2**32 / N) * (2 * N)
    prepare_fft_output(res, out_arr, coeff, N)


def torus_polynomial_add_mul(
    result: TorusPolynomialArray, poly1: IntPolynomialArray, poly2: TorusPolynomialArray
) -> None:
    N = polynomial_size(result)
    tmp1 = LagrangeHalfCPolynomialArray(N, poly1.shape)
    tmp2 = LagrangeHalfCPolynomialArray(N, poly2.shape)
    tmp3 = LagrangeHalfCPolynomialArray(N, result.shape)
    tmpr = TorusPolynomialArray(N, result.shape)
    int_polynomial_ifft(tmp1, poly1)
    torus_polynomial_ifft(tmp2, poly2)
    lagrange_polynomial_mul(tmp3, tmp1, tmp2)
    torus_polynomial_fft(tmpr, tmp3)
    torus_polynomial_add_to(result, tmpr)


# sets to zero
def lagrange_polynomial_clear(reps: LagrangeHalfCPolynomialArray) -> None:
    reps.coefs_c.fill(0)


# termwise multiplication in Lagrange space */
def lagrange_polynomial_mul(
    result: LagrangeHalfCPolynomialArray,
    a: LagrangeHalfCPolynomialArray,
    b: LagrangeHalfCPolynomialArray,
) -> None:
    numpy.copyto(result.coefs_c, a.coefs_c * b.coefs_c)


# TorusPolynomial = 0
def torus_polynomial_clear(result: TorusPolynomialArray) -> None:
    result.coefs_t.fill(0)


# TorusPolynomial += TorusPolynomial
def torus_polynomial_add_to(
    result: TorusPolynomialArray, poly2: TorusPolynomialArray
) -> None:
    result.coefs_t = cast(NDArray[Torus32], result.coefs_t + poly2.coefs_t)


# result = (X^ai-1) * source
def torus_polynomial_mul_by_xai_minus_one(
    out: TorusPolynomialArray, ais: NDArray[numpy.int32], in_: TorusPolynomialArray
) -> None:
    out_c = out.coefs_t
    in_c = in_.coefs_t

    N = out_c.shape[-1]
    for i in range(out.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i, :, :ai] = (
                -in_c[i, :, (N - ai) : N] - in_c[i, :, :ai]
            )  # sur que i-a<0
            out_c[i, :, ai:N] = (
                in_c[i, :, : (N - ai)] - in_c[i, :, ai:N]
            )  # sur que N>i-a>=0
        else:
            aa = ai - N
            out_c[i, :, :aa] = (
                in_c[i, :, (N - aa) : N] - in_c[i, :, :aa]
            )  # sur que i-a<0
            out_c[i, :, aa:N] = (
                -in_c[i, :, : (N - aa)] - in_c[i, :, aa:N]
            )  # sur que N>i-a>=0


# result= X^{a}*source
def torus_polynomial_mul_by_xai(
    out: TorusPolynomialArray, ais: NDArray[numpy.int32], in_: TorusPolynomialArray
) -> None:
    out_c = out.coefs_t
    in_c = in_.coefs_t

    N = out_c.shape[-1]
    for i in range(out.shape[0]):
        ai = ais[i]
        if ai < N:
            out_c[i, :ai] = -in_c[i, (N - ai) : N]  # sur que i-a<0
            out_c[i, ai:N] = in_c[i, : (N - ai)]  # sur que N>i-a>=0
        else:
            aa = ai - N
            out_c[i, :aa] = in_c[i, (N - aa) : N]  # sur que i-a<0
            out_c[i, aa:N] = -in_c[i, : (N - aa)]  # sur que N>i-a>=0
