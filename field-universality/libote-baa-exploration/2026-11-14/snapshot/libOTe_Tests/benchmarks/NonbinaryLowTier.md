# Nonbinary low-distance candidate

The current GF128 `ChosenBaa10` uses a binary RM code with plain XOR scans.
It remains unchanged while this candidate is evaluated. Its binary-lift
distance certificate and timings do not describe this candidate.

`chosen_baa_detail::TwoCoefficientBaaCode<F, Field>` uses an MDS [16,8,9]
constituent followed by two independently permuted, weighted scans. Each
scan computes `y[i] = input[perm[i]] + a[i] * y[i-1]`, starting at zero.
Each coefficient is sampled independently from two distinct nonzero field
elements. For GF128 these are `1` and `x`, represented by integers 1 and 2
in the canonical polynomial basis. Multiplication by `x` shifts across the
whole 128-bit element and reduces modulo `x^128 + x^7 + x^2 + x + 1`.
The constituent is the same nonbinary additive RS map as ChosenBaa30.

This is an experimental standalone encoder. No VOLE selector, default,
existing ChosenBaa10 alias, or PSI dependency pin is changed by the candidate.

## Distance argument

Fix a field admitting the local MDS code and two distinct nonzero scan
coefficients. Sample both permutations uniformly and independently, and
sample every scan coefficient uniformly and independently of the others and
of the permutations. At `n=2^21`, `k=2^20`, the direct quotient-rank ledger
bounds the probability of a nonzero codeword of weight at most 209715 by

```
135807966222723081714348558404059809 /
1461501637330902918203684832716283019655932542976
```

This is below `2^-43.2909`. Thus the corresponding minimum-distance bound
is 209716, slightly above 10%, except with that setup probability. The
statement quantifies over all nonzero messages, including those chosen
after setup.

The original ledger was evaluated for a prime-field Newton constituent.
Its transfer to this GF128 constituent uses the same facts as the direct
q=256 certificate: an MDS [16,8,9] code has shortened dimension `d-8`
on a support of size `d>=9`; imposing an independent zero equation raises
rank; and, for dependent nonzero input and feedback forms, at most one of
the two fresh coefficients cancels them. These statements hold over the
fixed field without a characteristic restriction. The source polynomial
is `1 + sum(d=9..16, binom(16,d) z^d tau^(d-8))`. The direct width-one
transfer uses only this polynomial, the coefficient atom bound 1/2, and
permutation averaging. It does not invoke the all-one, fresh-pair, or
large-characteristic arguments.

`certificates/verify_q2_field_transfer.py` replays all 167 sparse intervals
and 45 dense leaves with the original rational verifier. Replaying the
arithmetic verifies the ledger, not the mathematical transfer argument.
The implementation expands AES seeds, whereas the certificate uses ideal
independent randomness. Neither the certificate nor a check for bit mixing
is a complete VOLE security argument. Other lengths, padding, and output
truncation require their own treatment.

## Implementation checks

The candidate preserves the fixed unrolled gathers, weighted scans, and
additive RS kernel. The two-choice GF128 multiplier uses shifts, reduction,
and XORs without input-dependent branches or tables. A separate benchmark
row uses the existing general small-coefficient multiplier as a control.
The q=256 specialization retains its previous coefficient stream and map.

Tests compare the forward map with scalar field arithmetic, check the
field-valued transpose identity, and verify both multiplier choices on
all 128 basis vectors, including the 64-bit boundary and reduction at the
top bit. A Frobenius check verifies that `x` is outside GF(2^64), which
contains every proper subfield of GF(2^128). This is an algebraic check,
not an attack-cost estimate.

The native AVX2/VPCLMUL build passed all 239 enabled libOTe tests (39 were
disabled by configuration). Standalone Clang ASan/UBSan checks passed for
both that backend and the all-ISA-disabled portable backend, including
GF128 and Goldilocks scalar forward references and transpose identities.

## Initial performance result

On Peach, at `n=2^21`, the first four sequential seeds (31 timed iterations
per direction after two warmups) had the following medians of per-seed
medians. Encoding excludes setup and the copy preceding each transpose.

| Encoder | Forward ms | Transpose ms |
| --- | ---: | ---: |
| Existing binary RM low tier | 10.66 | 9.56 |
| Nonbinary q=2, shift multiplier | 16.91 | 17.53 |
| Nonbinary q=2, general GFNI multiplier | 16.95 | 16.83 |
| Existing nonbinary q=256 tier | 17.29 | 16.88 |

Six seeds were run in total. The host slowed down late in seed 5 and all
timings roughly doubled in seed 6, including the unchanged controls. The
cause was not established. All six seeds and their unfiltered summary are
archived in `results/peach-nonbinary-q2-20261114.tar.gz`; the table describes
the initial stable portion rather than presenting the whole run as stable.
Both q=2 implementations produced identical checksums for every seed. The
q=256 and binary controls also match the prior archived seed-1 checksums.

This candidate removes the binary-map limitation but does not meet the
10.7 ms transpose target or provide a meaningful speed advantage over the
30% tier. No promotion or default change follows from these measurements.
The next performance experiment should reduce the nonbinary constituent
cost or batch the weighted scan, with a matching distance argument.

## Stage profile and fused-transpose experiment

The follow-up measured each actual encoder stage and implemented an opt-in
fused transpose, without changing any existing selector or default.
`SmallAlphabetBaaCode`'s final template argument enables this path; the
default is false. `FusedTwoCoefficientBaaCode` names the shift-multiplier
experiment. The original scatter path remains available in every comparison.

The fused path first computes the weighted suffix in place, in permutation
order. An inverse permutation then gathers each local block directly into
the RS transpose. The coefficient carried between suffix batches is the
next forward coefficient, exactly as in the original reverse scan. Setup
stores one additional u32 inverse index per element: 8 MiB at n=2^21.
Encoding still uses caller-owned buffers and performs no allocations.
This rearranges memory accesses; it does not reduce the local arithmetic
or the total number of full-length reads and writes in these two stages.

The profiler is a separate `chosen_baa_stages` executable under
`LIBOTE_BAA_BENCHMARK`. Its friend access calls the actual kernels; there
are no timing hooks in ordinary encoding. Phase timings have clock calls
between stages, so separate uninstrumented `chosen_baa_directions` runs
determine end-to-end performance.

On Peach, CPU 2, n=2^21, the AVX-512/GFNI results below are medians of three
seed medians, each with 31 samples and two warmups. All runs, including
the AVX2 check and prefetch sweep, were sequential. Setup and the copy
before each transpose are excluded from encoding time.

| Forward phase | Binary RM ms | Nonbinary q=2, GFNI ms |
| --- | ---: | ---: |
| Local transform | 2.019 | 5.418 |
| First scan | 4.222 | 5.734 |
| Second scan | 4.411 | 5.636 |

The local transform accounts for about 3.40 ms of the forward gap; the
weighted scans account for another 2.74 ms. The transpose profile is:

| Transpose phase | Original q=2 GFNI ms | Fused q=2 GFNI ms |
| --- | ---: | ---: |
| First reverse scan | 6.014 | 5.893 |
| Final scan: scatter / contiguous suffix | 6.387 | 4.396 |
| Local transform: contiguous / inverse gather | 4.457 | 7.946 |
| Profiled total | 16.882 | 18.229 |

Fusion saves about 1.99 ms in the final scan but adds 3.49 ms to the local
stage. Thus the random gather into the RS transform costs more than the
contiguous suffix saves. This establishes where the regression appears;
it does not separate cache stalls, instruction scheduling, and spills.

The uninstrumented comparisons confirm the regression:

| Transpose implementation | Original ms | Fused ms |
| --- | ---: | ---: |
| q=2, GFNI (three seeds) | 16.735 | 18.079 |
| q=2, shift multiplier (three seeds) | 17.492 | 18.115 |
| q=256, GFNI (three seeds) | 16.835 | 18.205 |
| q=2, shift multiplier, AVX2 (one seed) | 21.155 | 22.138 |
| q=2, CLMUL multiplier, AVX2 (one seed) | 23.259 | 25.157 |

A separate q=2 GFNI sweep tried inverse-gather prefetch leads of 0, 32,
64, 128, 256, 512, and 1024 elements. At its single seed and 15 samples,
128 was best at 18.162 ms total; disabling prefetch took 31.263 ms. The
sweep did not close the gap to the original path. The experimental default
lead remains 256; no single-seed tuning result was promoted.

The full-output regression compares the fused and unfused transpose
prefixes element by element, in addition to scalar forward references and
adjoint identities. GF128 and Goldilocks are covered, including partial
batch counts and small valid sizes. Native and portable sanitizer checks
exercise the same cases. No native Apple timing was run.

Raw CSVs, host snapshots, source hashes, compile/run recipes, and archived
source are in `results/peach-fused-stages-20261114.tar.gz`; the extracted
summary is `results/fused-stage-summary.json`. The archive's RM phase label
`suffix_gather_local` is called `final_scan_local` in the current profiler
so the same label also describes its existing ARM path. That label change
does not change the measurements.

The measured fused path remains opt-in. The next useful experiment is a
smaller nonbinary local constituent with a matching distance argument, or
batching the weighted recurrence. The binary RM timing remains a comparison
point rather than the performance claim for a nonbinary low tier.
