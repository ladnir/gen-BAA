#include "libOTe/Tools/BlkAccCode/TwoCoefficientBaaCode.h"
#include "libOTe/Tools/BlkAccCode/PlainRsBaaCode.h"
#include "libOTe/Tools/BlkAccCode/ReedMullerBaaCode.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace osuCrypto;
using Clock = std::chrono::steady_clock;
using Times = std::array<Clock::time_point, 4>;

// Benchmark-only access to the actual kernels: no timers or runtime hooks
// are compiled into the encoders. Buffers and setup are outside all timings.
namespace osuCrypto::chosen_baa_detail
{
    struct BaaStageAccess
    {
        template<typename Code>
        static Times forward(const Code& code, const block* msg, block* out, block* scratch)
        {
            Times t; t[0] = Clock::now();
            code.localForward(msg, out); t[1] = Clock::now();
            code.template stage<false>(out, scratch, 0); t[2] = Clock::now();
            code.template stage<false>(scratch, out, 1); t[3] = Clock::now();
            return t;
        }
        template<bool Fused, u64 Lead = 256, typename Code>
        static Times transpose(const Code& code, block* out, block* scratch)
        {
            Times t; t[0] = Clock::now();
            code.template stage<true>(out, scratch, 1); t[1] = Clock::now();
            if constexpr (Fused)
            {
                code.transposeSuffix(scratch); t[2] = Clock::now();
                code.template gatherLocalTranspose<Lead>(scratch, out); t[3] = Clock::now();
            }
            else
            {
                code.template stage<true>(scratch, out, 0); t[2] = Clock::now();
                code.localTranspose(out); t[3] = Clock::now();
            }
            return t;
        }
        static Times forward(const ReedMullerBaaCode<>& code, const block* msg, block* out, block* scratch)
        {
            Times t; t[0] = Clock::now();
            for (u64 i = 0; i < code.mN; i += 32) code.localEncode(msg + i / 2, out + i);
            t[1] = Clock::now();
            code.stage<false>(out, scratch, 0); t[2] = Clock::now();
            code.stage<false>(scratch, out, 1); t[3] = Clock::now();
            return t;
        }
        template<bool Fused, u64 Lead = 256>
        static Times transpose(const ReedMullerBaaCode<>& code, block* out, block* scratch)
        {
            static_assert(Fused);
            Times t; t[0] = Clock::now();
            code.stage<true>(out, scratch, 1); t[1] = Clock::now();
            if constexpr (ReedMullerBaaCode<>::fusedTranspose) code.finishTranspose(scratch, out);
            else
            {
                code.stage<true>(scratch, out, 0);
                for (u64 i = 0; i < code.mN; i += 32) code.localTranspose(out + i, out + i / 2);
            }
            t[2] = Clock::now();
            t[3] = t[2];
            return t;
        }
    };
}

template<typename Code, bool Fused, bool Rm = false, u64 Lead = 256>
void profile(const char* name, u64 n, u64 seed, unsigned iterations)
{
    Code code; code.config(n, block(125, seed));
    PRNG random(block(126, seed));
    AlignedUnVector<block> msg(n / 2), original(n), out(n), scratch(n), expected(n), expectedT(n);
    random.get(msg.data(), msg.size()); random.get(original.data(), n);
    code.encode(msg, expected, scratch);
    expectedT = original; code.dualEncode(expectedT, scratch);
    using Access = chosen_baa_detail::BaaStageAccess;
    for (bool trans : {false, true})
    {
        std::array<std::vector<double>, 4> samples;
        block checksum = ZeroBlock;
        for (unsigned j = 0; j < iterations + 2; ++j)
        {
            if (trans) std::copy(original.begin(), original.end(), out.begin());
            const auto times = trans ? Access::transpose<Fused, Lead>(code, out.data(), scratch.data()) :
                Access::forward(code, msg.data(), out.data(), scratch.data());
            if (j >= 2)
            {
                for (unsigned i = 0; i < 3; ++i)
                    samples[i].push_back(std::chrono::duration<double, std::milli>(times[i + 1] - times[i]).count());
                samples[3].push_back(std::chrono::duration<double, std::milli>(times[3] - times[0]).count());
            }
            checksum ^= out[j % (trans ? n / 2 : n)];
        }
        const auto& reference = trans ? expectedT : expected;
        if (!std::equal(out.begin(), out.begin() + (trans ? n / 2 : n), reference.begin()))
            throw std::runtime_error("Stage profile changed encoder map");
        const char* phases[4] = {"local", "scan0", "scan1", "total"};
        if (trans)
        {
            phases[0] = "scan1";
            phases[1] = Rm ? "final_scan_local" : Fused ? "suffix0" : "scan0";
            phases[2] = Rm ? "unused" : Fused ? "gather_local" : "local";
        }
        for (unsigned i = 0; i < 4; ++i)
        {
            if (Rm && trans && i == 2) continue;
            auto& times = samples[i]; std::sort(times.begin(), times.end());
            std::cout << name << ',' << (trans ? "transpose" : "forward") << ',' << n << ',' << seed
                << ',' << phases[i] << ',' << times[iterations / 2] << ',' << checksum << '\n';
        }
        std::cout.flush();
    }
}

int main(int argc, char** argv)
{
    const u64 n = argc > 1 ? std::stoull(argv[1]) : (1u << 21);
    const unsigned iterations = argc > 2 ? std::stoul(argv[2]) : 31;
    const u64 seed = argc > 3 ? std::stoull(argv[3]) : 1;
    if (!iterations || !n || n % 32 || n > (1u << 24)) throw std::invalid_argument("benchmark arguments");
    std::cout << "code,direction,n,seed,phase,median_ms,checksum\n";
    if (argc > 4 && std::string(argv[4]) == "plain")
    {
        using Plain = chosen_baa_detail::PlainRsBaaCode<>;
        using Fused = chosen_baa_detail::PlainRsBaaCode<block, ChosenBaaField<block>, true>;
        profile<Plain, false>("rs_plain", n, seed, iterations);
        profile<Fused, true>("rs_plain_fused", n, seed, iterations);
        return 0;
    }
    if (argc > 4 && std::string(argv[4]) == "plain-sweep")
    {
        using Code = chosen_baa_detail::PlainRsBaaCode<block, ChosenBaaField<block>, true>;
        profile<Code, true, false, 64>("plain_fused_pf64", n, seed, iterations);
        profile<Code, true, false, 128>("plain_fused_pf128", n, seed, iterations);
        profile<Code, true, false, 256>("plain_fused_pf256", n, seed, iterations);
        profile<Code, true, false, 512>("plain_fused_pf512", n, seed, iterations);
        profile<Code, true, false, 1024>("plain_fused_pf1024", n, seed, iterations);
        return 0;
    }
    if (argc > 4 && std::string(argv[4]) == "sweep")
    {
        using Code = SmallAlphabetBaaCode<block, ChosenBaaField<block>, SmallBaaArithmetic<block>, 2, true>;
        profile<Code, true, false, 0>("q2_fused_pf0", n, seed, iterations);
        profile<Code, true, false, 32>("q2_fused_pf32", n, seed, iterations);
        profile<Code, true, false, 64>("q2_fused_pf64", n, seed, iterations);
        profile<Code, true, false, 128>("q2_fused_pf128", n, seed, iterations);
        profile<Code, true, false, 256>("q2_fused_pf256", n, seed, iterations);
        profile<Code, true, false, 512>("q2_fused_pf512", n, seed, iterations);
        profile<Code, true, false, 1024>("q2_fused_pf1024", n, seed, iterations);
        return 0;
    }
    profile<ReedMullerBaaCode<>, true, true>("binary_rm", n, seed, iterations);
    profile<chosen_baa_detail::TwoCoefficientBaaCode<>, false>("q2_shift", n, seed, iterations);
    profile<chosen_baa_detail::FusedTwoCoefficientBaaCode<>, true>("q2_shift_fused", n, seed, iterations);
    profile<SmallAlphabetBaaCode<block, ChosenBaaField<block>, SmallBaaArithmetic<block>, 2>, false>(
        "q2_general", n, seed, iterations);
    profile<SmallAlphabetBaaCode<block, ChosenBaaField<block>, SmallBaaArithmetic<block>, 2, true>, true>(
        "q2_general_fused", n, seed, iterations);
    profile<SmallAlphabetBaaCode<>, false>("q256", n, seed, iterations);
    profile<SmallAlphabetBaaCode<block, ChosenBaaField<block>, SmallBaaArithmetic<block>, 256, true>, true>(
        "q256_fused", n, seed, iterations);
}
