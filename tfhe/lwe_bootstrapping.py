from typing import Tuple, cast

import numpy
from numpy.typing import NDArray

from .lwe import LWEKey, LWEKeySwitchKey, LWESampleArray, lwe_key_switch
from .numeric_functions import Torus32, mod_switch_from_torus32
from .polynomials import (
    IntPolynomialArray,
    LagrangeHalfCPolynomialArray,
    TorusPolynomialArray,
    torus_polynomial_mul_by_xai,
)
from .tgsw import (
    TGSWKey,
    TGSWParams,
    TGSWSampleArray,
    TGSWSampleFFTArray,
    tgsw_fft_extern_mul_to_tlwe,
    tgsw_sym_encrypt_int,
    tgsw_to_fft_convert,
)
from .tlwe import (
    TLWESampleArray,
    TLWESampleFFTArray,
    tlwe_add_to,
    tlwe_copy,
    tlwe_extract_lwe_sample,
    tlwe_mul_by_xai_minus_one,
    tlwe_noiseless_trivial,
)


def lwe_bootstrapping_key(
    rng: numpy.random.RandomState,
    ks_t: int,
    ks_base_bit: int,
    key_in: LWEKey,
    rgsw_key: TGSWKey,
) -> Tuple[TGSWSampleArray, LWEKeySwitchKey]:
    bk_params = rgsw_key.params
    in_out_params = key_in.params
    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lwe_params

    n = in_out_params.n
    N = extract_params.n

    accum_key = rgsw_key.tlwe_key
    extracted_key = LWEKey.from_key(extract_params, accum_key)

    ks = LWEKeySwitchKey(rng, N, ks_t, ks_base_bit, extracted_key, key_in)

    bk = TGSWSampleArray(bk_params, (n,))
    kin = key_in.key
    alpha = accum_params.alpha_min

    tgsw_sym_encrypt_int(rng, bk, kin, alpha, rgsw_key)

    return bk, ks


class LWEBootstrappingKeyFFT:
    def __init__(
        self,
        rng: numpy.random.RandomState,
        ks_t: int,
        ks_base_bit: int,
        lwe_key: LWEKey,
        tgsw_key: TGSWKey,
    ) -> None:
        in_out_params = lwe_key.params
        bk_params = tgsw_key.params
        accum_params = bk_params.tlwe_params
        extract_params = accum_params.extracted_lwe_params

        bk, ks = lwe_bootstrapping_key(rng, ks_t, ks_base_bit, lwe_key, tgsw_key)

        n = in_out_params.n

        # Bootstrapping Key FFT
        bk_fft = TGSWSampleFFTArray(bk_params, (n,))
        tgsw_to_fft_convert(bk_fft, bk)

        self.in_out_params = (
            in_out_params  # paramètre de l'input et de l'output. key: s
        )
        self.bk_params = bk_params  # params of the Gsw elems in bk. key: s"
        self.accum_params = accum_params  # params of the accum variable key: s"
        self.extract_params = extract_params  # params after extraction: key: s'
        self.bk_fft = bk_fft  # the bootstrapping key (s->s")
        self.ks = ks  # the keyswitch key (s'->s)


def tfhe_mux_rotate_fft(
    result: TLWESampleArray,
    accum: TLWESampleArray,
    bki: TGSWSampleFFTArray,
    bk_idx: int,
    barai: NDArray[numpy.int32],
    bk_params: TGSWParams,
    tmpa: TLWESampleFFTArray,
    deca: IntPolynomialArray,
    deca_fft: LagrangeHalfCPolynomialArray,
) -> None:
    # TYPING: barai::Array{Int32}
    # ACC = BKi*[(X^barai-1)*ACC]+ACC
    # temp = (X^barai-1)*ACC
    tlwe_mul_by_xai_minus_one(result, barai, accum)

    # temp *= BKi
    tgsw_fft_extern_mul_to_tlwe(result, bki, bk_idx, bk_params, tmpa, deca, deca_fft)

    # ACC += temp
    tlwe_add_to(result, accum)


def tfhe_blind_rotate_fft(
    accum: TLWESampleArray,
    bk_fft: TGSWSampleFFTArray,
    bara: NDArray[numpy.int32],
    n: int,
    bk_params: TGSWParams,
) -> None:
    """
    * multiply the accumulator by X^sum(bara_i.s_i)
    * @param accum the TLWE sample to multiply
    * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
    * @param bara An array of n coefficients between 0 and 2N-1
    * @param bk_params The parameters of bk
    """
    # TYPING: bara::Array{Int32}

    temp = TLWESampleArray(bk_params.tlwe_params, accum.shape)
    temp2 = temp
    temp3 = accum

    accum_in_temp3 = True

    # For use in tgsw_fft_extern_mul_to_tlwe(), so that we don't have to
    #   allocate them `n` times
    tmpa = TLWESampleFFTArray(bk_params.tlwe_params, accum.shape)
    deca = IntPolynomialArray(bk_params.tlwe_params.N, accum.a.shape + (bk_params.l,))
    deca_fft = LagrangeHalfCPolynomialArray(
        bk_params.tlwe_params.N,
        accum.shape + (bk_params.tlwe_params.k + 1, bk_params.l),
    )

    for i in range(n):
        # GPU: will have to be passed as a pair `bara`, `i`
        barai = bara[:, i]  # !!! assuming the ciphertext is 1D

        # FIXME: We could pass the view bk_fft[i] here, but on the current # pylint: disable=fixme # noqa: E501
        #   Julia it's too slow
        tfhe_mux_rotate_fft(
            temp2, temp3, bk_fft, i, barai, bk_params, tmpa, deca, deca_fft
        )

        temp2, temp3 = temp3, temp2
        accum_in_temp3 = not accum_in_temp3

    if not accum_in_temp3:  # temp3 != accum
        tlwe_copy(accum, temp3)


def tfhe_blind_rotate_and_extract_fft(
    result: LWESampleArray,
    v: TorusPolynomialArray,
    bk: TGSWSampleFFTArray,
    barb: NDArray[numpy.int32],
    bara: NDArray[numpy.int32],
    n: int,
    bk_params: TGSWParams,
) -> None:
    """
    * result = LWE(v_p) where p=barb-sum(bara_i.s_i) mod 2N
    * @param result the output LWE sample
    * @param v a 2N-elt anticyclic function (represented by a TorusPolynomial)
    * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
    * @param barb A coefficients between 0 and 2N-1
    * @param bara An array of n coefficients between 0 and 2N-1
    * @param bk_params The parameters of bk
    """
    # TYPING: barb::Array{Int32},
    # TYPING: bara::Array{Int32}

    accum_params = bk_params.tlwe_params
    extract_params = accum_params.extracted_lwe_params
    N = accum_params.N

    # Test polynomial
    test_vect_bis = TorusPolynomialArray(N, result.shape)
    # Accumulator
    acc = TLWESampleArray(accum_params, result.shape)

    # testvector = X^{2N-barb}*v
    # GPU: array operations or a custom kernel
    torus_polynomial_mul_by_xai(test_vect_bis, 2 * N - barb, v)

    tlwe_noiseless_trivial(acc, test_vect_bis)

    # Blind rotation
    tfhe_blind_rotate_fft(acc, bk, bara, n, bk_params)

    # Extraction
    tlwe_extract_lwe_sample(result, acc, extract_params, accum_params)


def tfhe_bootstrap_wo_ks_fft(
    result: LWESampleArray, bk: LWEBootstrappingKeyFFT, mu: Torus32, x: LWESampleArray
) -> None:
    """
    * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
    * @param result The resulting LWESample
    * @param bk The bootstrapping + keyswitch key
    * @param mu The output message (if phase(x)>0)
    * @param x The input sample
    """

    bk_params = bk.bk_params
    accum_params = bk.accum_params
    in_params = bk.in_out_params
    N = accum_params.N
    n = in_params.n

    test_vec = TorusPolynomialArray(N, result.shape)

    # Modulus switching
    # GPU: array operations or a custom kernel
    barb = mod_switch_from_torus32(cast(Torus32, x.b), 2 * N)
    bara = mod_switch_from_torus32(cast(Torus32, x.a), 2 * N)

    # the initial test_vec = [mu,mu,mu,...,mu]
    # TODO: use an appropriate method # pylint: disable=fixme
    # GPU: array operations or a custom kernel
    test_vec.coefs_t.fill(mu)

    # Bootstrapping rotation and extraction
    tfhe_blind_rotate_and_extract_fft(
        result, test_vec, bk.bk_fft, barb, bara, n, bk_params
    )


def tfhe_bootstrap_fft(
    result: LWESampleArray, bk: LWEBootstrappingKeyFFT, mu: Torus32, x: LWESampleArray
) -> None:
    """
    * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
    * @param result The resulting LweSample
    * @param bk The bootstrapping + keyswitch key
    * @param mu The output message (if phase(x)>0)
    * @param x The input sample
    """

    u = LWESampleArray(bk.accum_params.extracted_lwe_params, result.shape)

    tfhe_bootstrap_wo_ks_fft(u, bk, mu, x)

    # Key switching
    lwe_key_switch(result, bk.ks, u)
