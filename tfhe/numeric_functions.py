from typing import NewType, Tuple, cast

import numpy
from numpy.typing import NDArray

Torus32 = NewType("Torus32", numpy.int32)


def rand_uniform_int32(
    rng: numpy.random.RandomState, shape: Tuple[int, ...]
) -> NDArray[numpy.int32]:
    return rng.randint(0, 2, size=shape, dtype=numpy.int32)


def rand_uniform_torus32(
    rng: numpy.random.RandomState, shape: Tuple[int, ...]
) -> Torus32 | NDArray[Torus32]:
    # TODO: if dims == () (it happens), the return value # pylint: disable=fixme
    #   is not an array -> type instability also, there's probably
    #   instability for arrays of different dims too. Naturally,
    #   it applies for all other rand_ functions.
    return cast(
        NDArray[Torus32],
        rng.randint(-(2**31), 2**31, size=shape, dtype=numpy.int32),
    )


def rand_gaussian_float(
    rng: numpy.random.RandomState, sigma: float, shape: Tuple[int, ...]
) -> NDArray[numpy.float64]:
    return rng.normal(size=shape, scale=sigma)


# Gaussian sample centered in message, with standard deviation sigma
def rand_gaussian_torus32(
    rng: numpy.random.RandomState,
    message: Torus32,
    sigma: float,
    shape: Tuple[int, ...],
) -> Torus32:
    # Attention: all the implementation will use the stdev instead of the
    #   gaussian fourier param
    return cast(
        Torus32, message + double_to_torus32(rng.normal(size=shape, scale=sigma))
    )


# Used to approximate the phase to the nearest message possible in the message space
# The constant m_size will indicate on which message space we are working
#   (how many messages possible)
#
# "work on 63 bits instead of 64, because in our practical cases, it's more precise"
def mod_switch_from_torus32(phase: Torus32, m_size: int) -> NDArray[numpy.int32]:
    # TODO: check if it can be simplified (wrt type conversions) # pylint: disable=fixme
    interval = (1 << 63) // m_size * 2  # width of each intervall
    half_interval = interval // 2  # begin of the first intervall
    phase64 = (phase.astype(numpy.uint32).astype(numpy.uint64) << 32) + half_interval
    # floor to the nearest multiples of interval
    return cast(
        NDArray[numpy.int32],
        (phase64 // interval).astype(numpy.int64).astype(numpy.int32),
    )


# Used to approximate the phase to the nearest message possible in the message space
# The constant m_size will indicate on which message space we are working
#   (how many messages possible)
#
# "work on 63 bits instead of 64, because in our practical cases, it's more precise"
def mod_switch_to_torus32(mu: int, m_size: int) -> Torus32:
    interval = ((1 << 63) // m_size) * 2  # width of each intervall
    phase64 = mu * interval
    # floor to the nearest multiples of interval
    return cast(Torus32, numpy.int32(phase64 >> 32))


# from double to Torus32
def double_to_torus32(d: NDArray[numpy.float64]) -> Torus32:
    return cast(Torus32, ((d - numpy.trunc(d)) * 2**32).astype(numpy.int32))


def int64_to_int32(x: numpy.int64) -> numpy.int32:
    return x.astype(numpy.int32)


def float_to_int32(x: NDArray[numpy.float64]) -> numpy.int32:
    return cast(numpy.int32, numpy.round(x).astype(numpy.int64).astype(numpy.int32))
