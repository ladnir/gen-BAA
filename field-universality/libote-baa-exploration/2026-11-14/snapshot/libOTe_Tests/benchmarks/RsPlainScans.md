# GF128 RS blocks followed by plain scans

Consider the existing MDS RS[16,8,9] block encoder over GF(2^128), followed
by two independently and uniformly permuted plain prefix sums. Fix
k=2^20 and n=2^21. Only the permutations are random; all scan coefficients
are one. Distance counts nonzero field symbols in the complete output.

The direct q=1 certificate in gen-BAA proves minimum distance at least
138413 (above 6.6%) except with probability below 2^-53.96. This statement
holds simultaneously for all nonzero messages, including messages chosen
after setup. It concerns ideal uniform permutations, not a statistical
guarantee for AES-expanded seeds or arbitrary padded/truncated VOLE sizes.

The numerical ledger was originally evaluated for a prime-field Newton
MDS constituent. Its basic bound transfers to the GF128 RS constituent:

* The shortened dimension of an MDS [16,8,9] block on a support of size
  d>=9 is d-8. The dimension-weighted source polynomial is therefore
  `1 + sum(d=9..16, binom(16,d) z^d tau^(d-8))` in either field.
* The small-source bound counts necessary support patterns for runs of a
  plain accumulator. A return requires a nonzero input and nonzero prior
  state. The bound allows every cancellation with that support pattern;
  it needs neither random coefficients nor odd characteristic.
* The dense bound tracks the rank of imposed zero equations. Independent
  input and feedback forms charge one rank for a zero output. Dependent
  forms may cancel deterministically at q=1. The two-state bound explicitly
  permits that worst case, including characteristic-two cancellation.
* The same rank is tracked across both scans. Uniform permutation averaging
  and the final nonnegative support sums do not depend on characteristic.

The replay covers the small source supports and all 137 dense leaves with
the original rational verifier. It contains no projective-pair or
complete-run refinement. `certificates/verify_q1_field_transfer.py` checks
the exact input ledger hash and reports distance and failure probability,
omitting prime-field implementation cost figures that do not transfer.
As with the other transfers, numerical replay verifies the ledger; the
argument above supplies the change of field.

The stronger Fp result above 11.7% uses refinements with odd-characteristic
cancellation estimates. In particular, the projective-pair proof uses
`P(opposite) <= P(proportional)/2`, which is unavailable in characteristic
two: identical nonzero forms cancel. That stronger result is not claimed
here. Improving the GF128 lower bound to 10% remains a separate task; the
6.6% bound is not an upper bound on the construction's actual distance.

The RS blocks supply nonbinary coefficients. Applying invertible binary
prefix sums and permutations afterward does not turn that code into a
binary scalar extension. This is distinct from the binary RM constituent
with plain scans. It does not by itself settle the VOLE hardness analysis.

Plain scans remove the weighted-scan arithmetic while retaining the local
RS transform. The experimental `chosen_baa_detail::PlainRsBaaCode` uses a
compile-time addition-only arithmetic policy. Its optional fused transpose
replaces the final scatter with a contiguous suffix sum followed by inverse
gathers into the local RS transpose. It adds an n-entry u32 inverse
permutation (8 MiB at this size). Existing selectors and defaults are unchanged.

## Transpose benchmark, 2026-11-14

Peach, Ryzen 7950X, one thread pinned to CPU 2, boost disabled; active
frequency snapshots were approximately 4.49 GHz. Every benchmark ran
sequentially, including the two builds. Each seed used two warmups followed
by 31 measured transposes. Case order rotated between seeds. Setup, input
restoration, forward encoding, and correctness checks were outside the timer.
These are medians of seed medians, with their ranges in parentheses.

| Implementation | AVX-512/GFNI, 5 seeds, ms | AVX2/PCLMUL, 3 seeds, ms |
| --- | ---: | ---: |
| Binary RM comparison | 9.594 (9.523–9.633) | 10.330 (10.208–10.357) |
| RS + plain scans | 15.958 (15.904–16.203) | 19.130 (19.102–19.195) |
| RS + plain scans, fused | **13.284 (13.169–13.482)** | **17.173 (16.995–17.240)** |
| RS + q=2 weighted scans, general multiplier | 16.961 (16.930–17.049) | 23.677 (23.672–23.682) |
| RS + q=256 weighted scans | 16.962 (16.853–17.013) | 23.523 (23.492–23.600) |

Fusion reduces plain-scan transpose time by 16.8% on the AVX-512 build.
The fused plain version takes 21.7% less time than q=256, but remains
2.58 ms above the 10.7 ms target. This is a performance result for the
6.6%-certified construction; it does not establish a 10% distance bound.
The q=2 row uses the general multiplier, not the shift-specialized candidate.

GF128 and Goldilocks passed scalar forward references, full transpose
comparisons against the general arithmetic implementation, and adjoint
checks under native and portable ASan/UBSan. Every timed case also checked
its adjoint identity, and fused/unfused checksums agreed for every seed.

`ChosenBaaTranspose.cpp` supplies the benchmark (`LIBOTE_BAA_BENCHMARK=ON`,
target `chosen_baa_transpose`; arguments: output length, iterations, seed).
`results/peach-rs-plain-20261114.tar.gz` archives the measured sources, build
commands and hashes, runner, per-seed CSVs, and host snapshots.
`results/rs-plain-summary.json` contains the aggregate results.

The next useful step is profiling this plain fused path to distinguish the
RS arithmetic cost from inverse-gather cost before selecting an optimization.

## AVX2/GFNI and the 10% target

The current cryptoTools CMake AVX2 option adds `-mavx2`; it does not add
`-mgfni`. `SmallBaaArithmetic` uses GFNI only under `__GFNI__`. The local
`AdditiveRs8` transform uses `ChosenBaaField<block>::mul8`, which selects
VPCLMUL independently. GFNI therefore accelerates the weighted scans but
does not accelerate the current addition-only variant's local RS transform.

Sequential Peach runs with three seeds and 31 samples per seed gave:

| Flags beyond AES/SSE/PCLMUL | Plain fused transpose, ms | q=256 transpose, ms |
| --- | ---: | ---: |
| AVX2 + GFNI | 17.356 | 20.782 |
| AVX2 + GFNI + VPCLMUL | 14.454 | 19.408 |

The AVX-512 plain fused stage profile gave median phase times of 5.103 ms
for the first reverse scan, 0.693 ms for the suffix, and 7.957 ms for the
gathered RS transform; the median total was 13.758 ms. Instrumented timings
are kept separate from the earlier uninstrumented 13.284 ms measurement.
The first profiling seed's contiguous RS phase was anomalously slower
(5.967 ms versus 4.454 ms on the following two seeds); no cause is claimed.
An exploratory one-seed prefetch sweep favored lead 128 at 13.623 ms over
lead 256 at 13.853 ms. That small saving does not close the gap, and the
default remains 256. All profiled outputs matched the normal encoder.

At forbidden output weight 209715, the characteristic-independent sparse
support bound for source weights through 31 evaluates to less than
2^-47.919090471278. Thus that part already supports the 10% target. The
unresolved part is the dense-support bound. Numerical optimization of the
basic two-state transfer at source/intermediate fractions (0.2,0.12), for
example, returned a rationally evaluated log2 upper bound of approximately
+139549, which is useless. Other sampled profiles also failed. These are
diagnostics, not an impossibility proof for this transfer or a low-weight
codeword. Increasing the local MDS block size in the sampled numerical
proposals did not resolve the issue either.

The odd-characteristic projective-pair improvement cannot be imported:
in characteristic two, equal nonzero forms cancel, so proportional forms
can all be cancelling forms. An RS-specific argument could instead charge
for the zero equations needed to make many coordinate forms equal in the
surviving quotient. It must retain that accounting across both scans; a
claim about distinct original RS columns alone is insufficient after
conditioning. This is a proposed proof direction, not an established bound.

On performance, a concrete same-map experiment is to accumulate the local
RS products before modular reduction, sharing a reduction among terms
contributing to the same output. The current kernel reduces products
individually. This may reduce carryless arithmetic, but its register and
gather scheduling costs must be measured. Neither a 10% certificate nor
a 10.7 ms transpose has been achieved for this variant.

Sources, runner, compile flags, phase CSVs, and host snapshots are archived
in `results/peach-rs-plain-investigate-20261114.tar.gz`. Numerical proposal
records, the exact small-support result, and timing aggregates are in
`results/rs-plain-investigate-summary.json`.
