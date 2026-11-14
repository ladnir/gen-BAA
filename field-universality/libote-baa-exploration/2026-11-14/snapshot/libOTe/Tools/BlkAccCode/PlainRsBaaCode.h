#pragma once
#include "SmallAlphabetBaaCode.h"

namespace osuCrypto::chosen_baa_detail
{
    template<typename F, typename Field = ChosenBaaField<F>>
    struct PlainBaaArithmetic
    {
        static OC_FORCEINLINE F madd(F state, u32, F input)
        { return Field::add(state, input); }
    };

    // Experimental MDS RS[16,8,9] followed by two permuted plain scans.
    // Field::point(0) must be one, as in the default field policies.
    // The local RS transform supplies nonbinary coefficients. These scans
    // use only addition; coefficient extraction is dead code in the kernel.
    // Fused changes the transpose schedule, not the encoded linear map.
    template<typename F = block, typename Field = ChosenBaaField<F>, bool Fused = false>
    using PlainRsBaaCode = SmallAlphabetBaaCode<F, Field, PlainBaaArithmetic<F, Field>, 1, Fused>;
}
