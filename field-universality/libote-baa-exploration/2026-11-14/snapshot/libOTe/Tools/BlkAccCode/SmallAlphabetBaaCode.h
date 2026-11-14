#pragma once
#include "ChosenBaaCode.h"
#include <algorithm>
#include "AdditiveRs8.h"
#include "SmallBaaGfni.h"

namespace osuCrypto
{
    namespace chosen_baa_detail { struct BaaStageAccess; }
    template<typename F, typename Field = ChosenBaaField<F>>
    struct SmallBaaArithmetic
    {
        static F madd(F state, u32 coefficient, F input)
        { return Field::add(input, Field::mul(state, Field::point(coefficient - 1))); }
    };
    template<>
    struct SmallBaaArithmetic<block, ChosenBaaField<block>>
    {
        // Polynomial coefficient 1..256 in the canonical GF(2^128) basis.
        // Two independent 64x9 carryless products; the overflow has at most
        // eight bits, so multiplication by x^7+x^2+x+1 cannot overflow again.
        static OC_FORCEINLINE block madd(block state, u32 coefficient, block input)
        {
#if defined(OC_ENABLE_SSE2) && defined(__GFNI__)
            return chosen_baa_detail::smallGfniMadd(state, coefficient, input);
#else
            const block c(0, coefficient);
            const auto lo = state.clmulepi64_si128<0x00>(c);
            const auto hi = state.clmulepi64_si128<0x01>(c);
            const auto overflow = hi.srli_si128<8>();
            return input ^ lo ^ hi.slli_si128<8>() ^ overflow ^
                overflow.slli_epi64(1) ^ overflow.slli_epi64(2) ^ overflow.slli_epi64(7);
#endif
        }
    };

    // [16,8,9] RS followed by two permuted recursive scans. Independent
    // coefficients use AlphabetSize distinct nonzero field elements. This is a
    // different map from ChosenBaaCode's full-field weighted accumulators.
    // Packed metadata supports n <= 2^24, with eight coefficient bits.
    // The default alphabet and seed schedule preserve the q=256 profile.
    template<typename F = block, typename Field = ChosenBaaField<F>,
             typename Arithmetic = SmallBaaArithmetic<F, Field>, unsigned AlphabetSize = 256,
             bool FusedTranspose = false>
    class SmallAlphabetBaaCode
    {
        friend struct chosen_baa_detail::BaaStageAccess;
        static_assert(AlphabetSize >= 1 && AlphabetSize <= 256 &&
            (AlphabetSize & (AlphabetSize - 1)) == 0);
        static constexpr u32 indexMask = (1u << 24) - 1;
        u64 mN = 0;
        std::array<AlignedUnVector<u32>, 2> mMeta;
        AlignedUnVector<u32> mInverseFirst;
        std::array<F, 64> mMatrix, mForward;
        static constexpr bool additiveLocal = std::is_same_v<Field, ChosenBaaField<block>>;
        chosen_baa_detail::AdditiveRs8 mAdditive;
        static u32 below(PRNG& random, u32 bound)
        {
            u64 p = u64(random.get<u32>()) * bound;
            if (u32(p) < bound)
            {
                const auto threshold = (u32(0) - bound) % bound;
                while (u32(p) < threshold) p = u64(random.get<u32>()) * bound;
            }
            return u32(p >> 32);
        }
        template<bool Transpose, unsigned J = 0>
        static OC_FORCEINLINE void prefetch(const F* in, F* out, const u32* p)
        {
            if constexpr (J < 16)
            {
#ifdef OC_ENABLE_SSE2
                const F* address = Transpose ? out + (p[J] & indexMask) : in + (p[J] & indexMask);
                _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T2);
#endif
                prefetch<Transpose, J + 1>(in, out, p);
            }
        }
        template<unsigned J = 0>
        static OC_FORCEINLINE void gather(const F* in, const u32* p, F* values)
        {
            if constexpr (J < 16) { values[J] = in[p[J] & indexMask]; gather<J + 1>(in, p, values); }
        }
        template<unsigned J = 0>
        static OC_FORCEINLINE void forwardBatch(const F* values, F* out, const u32* p, F& sum)
        {
            if constexpr (J < 16)
            {
                out[J] = sum = Arithmetic::madd(sum, (p[J] >> 24) + 1, values[J]);
                forwardBatch<J + 1>(values, out, p, sum);
            }
        }
        template<unsigned J = 0>
        static OC_FORCEINLINE void transposeBatch(const F* in, F* out, const u32* p, F& sum, u32& coefficient)
        {
            if constexpr (J < 16)
            {
                constexpr auto pos = 15 - J;
                sum = Arithmetic::madd(sum, coefficient, in[pos]);
                coefficient = (p[pos] >> 24) + 1;
                out[p[pos] & indexMask] = sum;
                transposeBatch<J + 1>(in, out, p, sum, coefficient);
            }
        }
        template<bool Transpose>
        void stage(const F* in, F* out, unsigned which) const
        {
            const auto* p = mMeta[which].data();
            F sum = Field::zero();
            u32 coefficient = 1;
            for (u64 j = 0; j < mN; j += 16)
            {
                const u64 i = Transpose ? mN - j - 16 : j;
                constexpr u64 lead = 256;
                if (j + lead + 16 <= mN)
                    prefetch<Transpose>(in, out, p + (Transpose ? i - lead : i + lead));
                if constexpr (Transpose) transposeBatch(in + i, out, p + i, sum, coefficient);
                else
                {
                    std::array<F, 16> values;
                    gather(in, p + i, values.data());
                    forwardBatch(values.data(), out + i, p + i, sum);
                }
            }
        }
        void localForward(const F* in, F* out) const
        {
            for (u64 i = 0; i < mN / 16; ++i)
            {
                const auto* src = in + 8 * i; auto* dst = out + 16 * i;
                std::copy_n(src, 8, dst);
                if constexpr (additiveLocal) mAdditive.apply(src, dst + 8);
                else Field::template localForward<8>(src, dst, mForward.data());
            }
        }
        OC_FORCEINLINE void localTransposeBlock(const F* in, F* out) const
        {
            if constexpr (additiveLocal)
            {
                std::array<block, 8> values;
                mAdditive.apply(in + 8, values.data());
                for (u64 j = 0; j < 8; ++j) out[j] = in[j] ^ values[j];
            }
            else Field::template local<8>(in, out, mMatrix.data());
        }
        void localTranspose(F* inOut) const
        {
            for (u64 i = 0; i < mN / 16; ++i)
                localTransposeBlock(inOut + 16 * i, inOut + 8 * i);
        }
        template<unsigned J = 0>
        static OC_FORCEINLINE void suffixBatch(F* in, const u32* p, F& sum, u32& coefficient)
        {
            if constexpr (J < 16)
            {
                constexpr unsigned pos = 15 - J;
                in[pos] = sum = Arithmetic::madd(sum, coefficient, in[pos]);
                coefficient = (p[pos] >> 24) + 1;
                suffixBatch<J + 1>(in, p, sum, coefficient);
            }
        }
        void transposeSuffix(F* in) const
        {
            F sum = Field::zero();
            u32 coefficient = 1;
            for (u64 end = mN; end; end -= 16)
                suffixBatch(in + end - 16, mMeta[0].data() + end - 16, sum, coefficient);
        }
        template<u64 Lead = 256>
        void gatherLocalTranspose(const F* in, F* out) const
        {
            const auto* p = mInverseFirst.data();
            for (u64 i = 0; i < mN; i += 16)
            {
                if constexpr (Lead)
                    if (i + Lead + 16 <= mN) prefetch<false>(in, out, p + i + Lead);
                std::array<F, 16> values;
                gather(in, p + i, values.data());
                localTransposeBlock(values.data(), out + i / 2);
            }
        }
    public:
        void config(u64 n, block seed)
        {
            if (!n || n % 16 || n > (1u << 24)) throw std::invalid_argument("Small-alphabet BAA length.");
            // The proof requires distinct nonzero alphabet elements, not
            // integer scalar multiples of one in positive characteristic.
            for (u64 i = 0; i < AlphabetSize; ++i)
            {
                const F c = Field::point(i);
                if (c == Field::zero()) throw std::invalid_argument("BAA coefficient alphabet contains zero.");
                for (u64 j = 0; j < i; ++j)
                    if (c == Field::point(j)) throw std::invalid_argument("BAA coefficient alphabet repeats.");
            }
            using LocalField = std::conditional_t<additiveLocal, chosen_baa_detail::AdditivePoints, Field>;
            ChosenBaaCode<8, F, LocalField> local;
            local.config(16, seed);
            for (u64 row = 0; row < 8; ++row) for (u64 col = 0; col < 8; ++col)
                mMatrix[col * 8 + row] = mForward[row * 8 + col] = local.matrix(row, col);
            if constexpr (additiveLocal)
            {
                for (u64 i = 0; i < 8; ++i) for (u64 j = 0; j < 8; ++j)
                    if (mMatrix[i * 8 + j] != mMatrix[i ^ j]) throw std::logic_error("Additive RS symmetry.");
                mAdditive.init(mMatrix.data());
            }
            PRNG master(seed);
            const std::array<block, 2> permutationSeeds{master.get<block>(), master.get<block>()};
            const std::array<block, 2> coefficientSeeds{master.get<block>(), master.get<block>()};
            for (unsigned s = 0; s < 2; ++s)
            {
                auto& p = mMeta[s]; p.resize(n);
                for (u64 i = 0; i < n; ++i) p[i] = u32(i);
                PRNG random(permutationSeeds[s]), coefficients(coefficientSeeds[s]);
                for (u64 i = n; i > 1; --i) std::swap(p[i - 1], p[below(random, u32(i))]);
                for (u64 i = 0; i < n; ++i)
                    p[i] |= (u32(coefficients.get<u8>()) & (AlphabetSize - 1)) << 24;
            }
            if constexpr (FusedTranspose)
            {
                mInverseFirst.resize(n);
                for (u64 i = 0; i < n; ++i) mInverseFirst[mMeta[0][i] & indexMask] = u32(i);
            }
            mN = n;
        }
        u64 messageSize() const { return mN / 2; }
        u64 codewordSize() const { return mN; }
        u32 metadata(unsigned stage, u64 i) const
        {
            if (i >= mN) throw std::out_of_range("Small-alphabet BAA metadata index.");
            return mMeta.at(stage)[i];
        }
        F matrix(u64 row, u64 col) const
        {
            if (row >= 8 || col >= 8 || !mN) throw std::out_of_range("Small-alphabet BAA matrix index.");
            return mMatrix[col * 8 + row];
        }
        // Caller-owned disjoint buffers. No allocation inside either encoder.
        void encode(span<const F> input, span<F> output, span<F> scratch) const
        {
            if (!mN || input.size() != mN / 2 || output.size() != mN || scratch.size() != mN)
                throw std::invalid_argument("Small-alphabet BAA forward buffer size.");
            localForward(input.data(), output.data());
            stage<false>(output.data(), scratch.data(), 0);
            stage<false>(scratch.data(), output.data(), 1);
        }
        void dualEncode(span<F> inOut, span<F> scratch) const
        {
            if (!mN || inOut.size() != mN || scratch.size() != mN)
                throw std::invalid_argument("Small-alphabet BAA transpose buffer size.");
            stage<true>(inOut.data(), scratch.data(), 1);
            if constexpr (FusedTranspose)
            {
                // Preserve the weighted transpose exactly. First form the
                // suffix in permutation order, then gather the inverse
                // permutation directly into each local RS compression.
                // This replaces the last random scatter with a gather and
                // costs one extra u32 inverse index per codeword element.
                transposeSuffix(scratch.data());
                gatherLocalTranspose(scratch.data(), inOut.data());
            }
            else
            {
                stage<true>(scratch.data(), inOut.data(), 0);
                localTranspose(inOut.data());
            }
        }
    };
}
