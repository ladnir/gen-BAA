#pragma once
#include "SmallAlphabetBaaCode.h"

namespace osuCrypto::chosen_baa_detail
{
    template<typename F, typename Field = ChosenBaaField<F>>
    struct TwoCoefficientArithmetic : SmallBaaArithmetic<F, Field> {};

    template<>
    struct TwoCoefficientArithmetic<block, ChosenBaaField<block>>
    {
        // The public coefficient is 1 or x. Multiplication by x shifts the
        // complete 128-bit polynomial, including across the 64-bit boundary,
        // and reduces its top bit by x^7+x^2+x+1. Never treat lanes as fields.
        static OC_FORCEINLINE block madd(block state, u32 coefficient, block input)
        {
#ifdef OC_ENABLE_SSE2
            const auto shift = _mm_cvtsi32_si128(coefficient - 1);
            const auto reverseShift = _mm_cvtsi32_si128(65 - coefficient);
            const block carry = _mm_srl_epi64(state, reverseShift);
            const block low = _mm_sll_epi64(state, shift);
            const auto reduction = ZeroBlock.sub_epi64(carry.srli_si128<8>()) & block(0, 0x87);
            return input ^ low ^ carry.slli_si128<8>() ^ reduction;
#else
            // Fixed shifts also work in the scalar block implementation;
            // unlike SSE2, shifting a C++ uint64_t by 64 is undefined.
            const auto carry = state.srli_epi64(63);
            const auto doubled = state.slli_epi64(1) ^ carry.slli_si128<8>() ^
                (ZeroBlock.sub_epi64(carry.srli_si128<8>()) & block(0, 0x87));
            const u64 mask = u64{0} - (coefficient - 1);
            return input ^ state ^ ((state ^ doubled) & block(mask, mask));
#endif
        }
    };

    // Experimental nonbinary low-distance candidate, not a VOLE selector.
    // Same MDS [16,8,9] constituent and independent weighted scans as the
    // high tier, with two distinct nonzero coefficient choices. In the
    // canonical GF128 basis these are 1 and x (integer encodings 1 and 2).
    // This does not use the binary RM constituent or plain XOR scans.
    template<typename F = block, typename Field = ChosenBaaField<F>>
    using TwoCoefficientBaaCode = SmallAlphabetBaaCode<F, Field,
        TwoCoefficientArithmetic<F, Field>, 2>;

    // Opt-in experiment; existing profiles and the unfused candidate keep
    // their current choices. The fused path implements the identical map.
    template<typename F = block, typename Field = ChosenBaaField<F>>
    using FusedTwoCoefficientBaaCode = SmallAlphabetBaaCode<F, Field,
        TwoCoefficientArithmetic<F, Field>, 2, true>;
}
