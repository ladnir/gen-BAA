# Status: large-field inheritance for BAA codes

## Bottom line

The original equality hypothesis is false. Reusing a binary BAA matrix over
`F_p` can reduce the distance for a particular encoder. Even `p>N` does not
prevent this reduction.

The weaker cryptographic claim remains plausible:

```text
For delta below the binary ensemble threshold,
Pr[d_min(C_Fp) <= delta N] is negligible
when log_2(p) is the security parameter.
```

Large characteristic removes every known local obstruction. The unresolved
problem is uniformity: a minimum-distance witness chooses its support after it
sees both interleavers.

Outside the BAA ensemble, existence is settled by a dense baseline. A random
binary generator of sufficiently small constant rate has relative distance
close to one half over `F_2` and over any prescribed finite collection of
prime fields. The proof uses uniform convergence for Boolean-cube hyperplane
sections and is independent of the field sizes. Direct encoding takes
quadratically many additions. The open problem is therefore efficient,
structured field universality rather than field-universal existence. See
[`DENSE_BASELINE.md`](DENSE_BASELINE.md).

## Model and target

For `N=3K`, the encoder is

```text
F^K --B_3--> F^N --P_1--> F^N --A--> F^N --P_2--> F^N --A--> F^N,
```

where `B_3` repeats each message coordinate three times, `A` is the prefix
accumulator, and `P_1,P_2` are independent uniform coordinate permutations.
The same integer circuit defines a code over every prime field.

The main target is one-sided threshold inheritance, not equality of realized
distances. For every `delta` below a proved binary threshold, the odd-field
code should have distance greater than `delta N` with overwhelming
probability over `(P_1,P_2)`.

## What is known

| Claim | Status | Meaning |
|---|---|---|
| Dense field-universal binary code | Proved from a standard VC bound | Random binary generators have positive rate and relative distance near `1/2` over any prescribed finite field family, but direct encoding is quadratic. |
| Samplewise equality `d_p=d_2` | False | Small encoders give strict inequalities in both directions. |
| The condition `p>N` prevents loss | False | At `N=6`, one encoder has `d_2=4` and `d_7=3`. |
| Ternary inheritance at `0.1323` | Proved | The stronger ternary threshold `0.196` is certified for repetition three. |
| Sparse dual formulation | Proved | A word supported in `D` exists exactly when the dual columns `H_D` are dependent. |
| Large-prime absolute floor | Proved | `d_p >= min(d_2,ceil(log_(sqrt(20))(p)))`; a 128-bit prime gives at least 59. |
| Local torsion barrier | Proved | A characteristic-`p` residual component needs cycle rank at least `ceil(log_2(p))`. |
| Complete length-six prime census | Certified computation | All `720^2` interleaver pairs have been checked. `PAPA` has bad odd primes `3,5,7,11`; `APAP` has `3,5,7`; neither schedule has a characteristic-zero reduction. |
| Sampled all-prime scans at `N=9,12` | Certified per sampled encoder | No characteristic-zero reduction was found. `PAPA` terminates at `p=11`; one `APAP` encoder at `N=12` has `d_13=4<5=d_2`. |
| Selected-prime scans at `N=15,18` | Certified per sampled encoder | No loss occurred in 2,300 exact comparisons for ten primes from 17 through `2^61-1`; 2,270 comparisons improved and 30 tied. |
| Adaptive multi-round peeling | Proved, but insufficient | Deferred decisions remain valid for any fixed number of reservoirs; the present recurrence cannot approach a complete peel. |
| Fixed random deletion at the target scale | Experimental | Ten interleaver samples peeled completely over every field. |
| Uniform rank for every low-weight deletion | Open | This is the missing minimum-distance theorem. |

For the concrete experiments, `N=1048575` and

```text
p = 2^128-159.
```

No counterexample was found at this characteristic. This observation cannot
exclude events with probability near `2^-128`.

## Current proof route

Let `D subseteq [N]` be a candidate output support. The sparse dual matrix has
the form

```text
H_D = H_0 P_1 T P_2 T E_D,
```

where `T` is the inverse transpose of an accumulator. The proof now classifies
`D` by its number of interval runs.

If `D` has size `d`, has `r` runs, and contains the final coordinate exactly
when `tau=1`, then

```text
T E_D(Z^D)
```

is a direct sum of zero-sum lattices and has `d+r-tau` active rows. The number
of deletion sets with this profile is exactly

```text
binom(d-1,r-1) binom(N-d+1,r).
```

This identity creates two regimes.

- **Many runs.** The second interleaver expands the active rows. The first
  interleaver then creates many degree-one dual checks. At the target
  parameters, the proved one-round bound removes at least 23,594 columns with
  failure probability below `0.002954`. A staged two-round proof removes at
  least 17,441 columns with valid conditioning between rounds. The general
  multi-reservoir theorem closes the adaptive-conditioning issue for this
  construction. Four equal reservoirs certify 14,682 pivots with failure
  probability below `0.003952`.

- **Few runs.** Peeling is weaker, but there are far fewer deletion sets.
  At deletion fraction `0.1323`, the exceptional profiles cost at most about
  `2^(0.1314N)` rather than `2^(0.5638N)`. They also retain an exact random
  zero-sum-space description.

The experiments have much more margin than the theorem. For the fixed random
deletion at the target scale, the first round removed between 120,466 and
120,731 of 138,726 columns in ten samples. Every sample then peeled completely.

## What remains

The project does not yet prove large-field threshold inheritance. Two results
would complete the present route.

1. **Many-run theorem.** Prove an exponential bound for complete peeling, or
   prove that every residual core has linear row surplus. The proof must remain
   valid through adaptively selected peeling rounds.

2. **Few-run theorem.** Prove rank anti-concentration for the randomly embedded
   zero-sum spaces. The bound must survive the exact run-profile count and the
   union over all deletion supports.

The main danger is a deletion profile that creates a nearly square core with
at least 128 independent cycles. No current theorem excludes that profile.

## Recommended decision

Stop adding reservoirs to the present recurrence. For every reservoir
partition, its certified count is at most 26,216.04 columns, or 18.90% of the
target deletion, even before concentration slacks. Its zero-slack fine-mesh
limit is smaller: `0.0201382N`, or 15.22% of the deletion. These are limits of
the lower-bound method, not upper bounds on the actual decoder; the experiments
still peel completely. The loss comes from charging each pivot for as many as
four destroyed singleton rows.

The next bounded task should be a direct anti-concentration lemma for the
few-run zero-sum spaces. A conditional occupancy theorem for the many-run core
is also viable, but it must improve the singleton-loss charge; another
reservoir partition cannot do so.

Detailed proofs and data are in
[`HIGH_CHARACTERISTIC.md`](HIGH_CHARACTERISTIC.md),
[`RUN_PROFILE_THEORY.md`](RUN_PROFILE_THEORY.md), and
[`results/run-profile-bound.json`](results/run-profile-bound.json).
