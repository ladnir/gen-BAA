#include "libOTe/Tools/BlkAccCode/PlainRsBaaCode.h"
#include "libOTe/Tools/BlkAccCode/ReedMullerBaaCode.h"
#include <chrono>
#include <iostream>
#include <vector>
using namespace osuCrypto;

template<typename Code>
void run(const char* name, u64 n, u64 seed, unsigned iterations)
{
    Code code;
    auto setupStart = std::chrono::steady_clock::now();
    code.config(n, block(125, seed));
    const double setup = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setupStart).count();
    AlignedUnVector<block> message(n / 2), original(n), input(n), scratch(n), forward(n), reference(n);
    PRNG random(block(126, seed));
    random.get(message.data(), message.size()); random.get(original.data(), n);
    code.encode(message, forward, scratch);
    reference = original; code.dualEncode(reference, scratch);
    std::vector<double> samples; samples.reserve(iterations);
    block checksum = ZeroBlock;
    for (unsigned j = 0; j < iterations + 2; ++j)
    {
        std::copy(original.begin(), original.end(), input.begin());
        const auto start = std::chrono::steady_clock::now();
        code.dualEncode(input, scratch);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (j >= 2) samples.push_back(elapsed);
        checksum ^= input[j % (n / 2)];
    }
    if (!std::equal(input.begin(), input.begin() + n / 2, reference.begin()))
        throw std::runtime_error("Transpose changed between iterations");
    block lhs = ZeroBlock, rhs = ZeroBlock;
    for (u64 i = 0; i < n; ++i) lhs ^= original[i].gf128Mul(forward[i]);
    for (u64 i = 0; i < n / 2; ++i) rhs ^= message[i].gf128Mul(input[i]);
    if (lhs != rhs) throw std::runtime_error("Transpose identity failed");
    std::sort(samples.begin(), samples.end());
    std::cout << name << ',' << n / 2 << ',' << n << ',' << seed << ',' << setup << ','
        << samples[iterations / 2] << ',' << checksum << std::endl;
}

int main(int argc, char** argv)
{
    const u64 n = argc > 1 ? std::stoull(argv[1]) : (1u << 21);
    const unsigned iterations = argc > 2 ? std::stoul(argv[2]) : 31;
    const u64 seed = argc > 3 ? std::stoull(argv[3]) : 1;
    if (!iterations || !n || n % 32 || n > (1u << 24)) throw std::invalid_argument("benchmark arguments");
    std::cout << "code,k,n,seed,setup_ms,transpose_ms,checksum\n";
    // Rotate case order between seeds, while keeping every run sequential.
    for (unsigned j = 0; j < 5; ++j)
        switch ((j + seed - 1) % 5)
        {
        case 0: run<ReedMullerBaaCode<>>("binary_rm", n, seed, iterations); break;
        case 1: run<chosen_baa_detail::PlainRsBaaCode<>>("rs_plain", n, seed, iterations); break;
        case 2: run<chosen_baa_detail::PlainRsBaaCode<block, ChosenBaaField<block>, true>>(
            "rs_plain_fused", n, seed, iterations); break;
        case 3: run<SmallAlphabetBaaCode<block, ChosenBaaField<block>, SmallBaaArithmetic<block>, 2>>(
            "rs_q2", n, seed, iterations); break;
        case 4: run<SmallAlphabetBaaCode<>>("rs_q256", n, seed, iterations); break;
        }
}
