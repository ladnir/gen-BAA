# Field Inheritance for Addition-Only BAA Codes

This directory contains an independent research project on addition-only linear
encoders over finite fields. It does not form part of the BAA paper or the
`lean/BAA` formalization. Results may be transferred to another project only by
an explicit later decision.

## Question

This project studies one concrete ensemble. Fix a repetition factor `r` and
set `N=rK`. The map `B_r` repeats each message symbol `r` times. Two
independent uniform permutations separate `B_r` from two prefix accumulators:

```text
F^K --B_r--> F^N --P_1--> F^N --A--> F^N --P_2--> F^N --A--> F^N.
```

We call this the repetition-based BAA ensemble. Its encoder is

```text
E_{P_1,P_2} := A P_2 A P_1 B_r.
```

The same sampled permutations and the same integer circuit define a code over
every prime field. No random diagonal multipliers are used.

The central question is whether an odd-prime implementation inherits the
binary ensemble's relative-distance threshold. Exact equality is no longer
the leading hypothesis: the current evidence instead suggests that odd fields
have a better bulk threshold and a characteristic-specific sparse failure
tail.

For an odd prime `p`, define

```text
Delta_{p,N} := d_min(C_{F_p,N}) / N.
```

The binary code uses the same sampled permutations and defines `Delta_{2,N}`.

## Research notes

[`INITIAL_ANALYSIS.md`](INITIAL_ANALYSIS.md) records the counterexample search,
the algebraic rank formulation, and the evidence that led away from literal
field universality. [`CLEAN_THEORY.md`](CLEAN_THEORY.md) develops the current
sparse--bulk theory, including an exact vector-composition enumerator and a
finite-dimensional bulk exponent.

## Current theorem

For repetition three over `F_3`, the project now proves

```text
delta_* := 49/250,
Pr[Delta_{3,N} <= delta_*] -> 0.
```

The proof uses a directed interval certificate for message supports of at
least `0.0001 K`. An exact random-spacing argument covers every smaller
support. The sparse estimate sums over all ternary messages and does not
assume random diagonal multipliers.

Two implementations certify the bulk bound. The primary certificate is
[`results/ternary-bulk-0.196-eta-0.0001.json`](results/ternary-bulk-0.196-eta-0.0001.json).
The independent certificate is
[`results/ternary-bulk-0.196-eta-0.0001-independent.json`](results/ternary-bulk-0.196-eta-0.0001-independent.json).

Kliewer et al. report `0.1323` to four decimal places as the corresponding
binary lower bound. The ternary theorem is therefore stronger than the
published binary guarantee. It proves one-sided threshold inheritance, not
equality of the binary and ternary distances. See [Kliewer et al., *Coding
Theorems for Repeat Multiple Accumulate
Codes*](https://arxiv.org/abs/0810.3422), Table I.

## Conjecture hierarchy

Let `delta_bin(r)` be a proved binary relative-distance threshold for this
ensemble. The first target is threshold inheritance. For every fixed odd prime
`p` and every `delta < delta_bin(r)`,

```text
Pr[Delta_{p,N} <= delta] -> 0.
```

This statement is sufficient for the intended application: an encoder over a
prime field would inherit the certified binary distance guarantee while using
only additions, permutations, and the fixed repetition map.

If the binary relative distance also converges to `delta_bin(r)`, threshold
inheritance implies the coupled one-sided statement

```text
Pr[Delta_{p,N} < Delta_{2,N} - epsilon] -> 0
```

for every `epsilon>0`. The convergence assumption is necessary for this
deduction; two unrelated lower bounds do not compare the realized distances.

The stronger coupled-universality conjecture was the original motivation:

```text
Delta_{p,N} - Delta_{2,N} -> 0 in probability.
```

The experiments and bulk optimization do not currently support this stronger
statement. It remains useful as a falsifiable comparison, not as the main
proof target. Generalization beyond the repetition-based, two-accumulator
circuit should wait until the sparse and bulk bounds are proved in this model.

## Rank formulation

For a full-row-rank generator matrix `G` over `F`,

```text
d_min(rowspan_F(G))
  = min { |T| : rank_F(G restricted to the complement of T) < dimension(G) }.
```

Equivalently, minimum distance is the cogirth of the column matroid represented
by `G`. The conjecture therefore asks for universality of an extremal deletion
rank, not a correspondence between individual codewords.

For a backward constraint matrix `Q_F`, define the worst-support rank profile

```text
r_{F,N}(t) := min_{T subseteq [N], |T| = t} rank_F((Q_F)_T).
```

The distance is the first `t` for which `r_{F,N}(t) < t`. A field-independent
limiting rank profile would imply coupled distance universality.

## Necessary qualifications

Sample-by-sample equality is false. With the length-three repetition input
`(1,1,1)`, identity permutations, and two prefix accumulators, the output has
weight two over `F_2` and weight one over `F_3`.

The conjectures must therefore quantify over the random interleavers. Initial
work will also fix `p` before taking `N -> infinity`. Uniformity over primes,
adversarial characteristics, deterministic interleavers, and unknown binary
base encoders are outside the first scope.

A later extension may replace `B_r` with a fixed blockwise base code. Such an
extension must state checkable hypotheses on its finite symbol-composition
enumerator. Binary minimum distance alone is not a sufficient hypothesis.

## Proof program

1. Build a full-composition bulk certificate for `F_5`; its sparse bound is
   already proved through `0.26`.
2. Certify a positive ternary bulk type at `0.197` and locate the first-moment
   transition above `0.196`.
3. Establish the rank/cogirth formulation as an independent algebraic view of
   characteristic-specific failures.
4. Investigate the matching upper bound required for coupled universality.
5. Extract checkable enumerator hypotheses for blockwise BAA base codes only
   after the repetition-based theorem is complete.

The first-moment proof resolves the uniform quantifier for the ternary
`0.196` theorem. Rank and residual-core diagnostics remain useful for finding
counterexamples, but the theorem does not depend on a uniform rank comparison.

The same spacing argument now proves the sparse half of a quinary theorem
through output fraction `0.26`, using message cutoff `0.00001`. The remaining
quinary task is a rigorous bulk certificate. The ternary sign reduction does
not generalize: an explicit stationary type over `F_5` makes sign averaging
decrease the full exponent.

## Experimental program

Experiments must use the same permutations for every field. Random-message
searches are insufficient because minimum distance is an adversarial property.
The first implementation should compute or certify:

1. exact minimum distance for small instances over `F_2`, several odd prime
   fields, and the rationals;
2. the gap `(d_p - d_2) / N` for coupled samples;
3. minimum rank-deficient deletion sets;
4. Smith `p`-nullities and determinantal-divisor certificates for candidate
   integer submatrices, which identify the characteristics that create
   additional dependencies; and
5. the effect of no diagonal multipliers, sign-only diagonals, and unrestricted
   field multipliers.

The rational comparison is important. If the rational distance remains above
the binary distance by a linear amount, sufficiently large characteristics may
improve the distance rather than match it.

## Project boundary

All notes, experiments, proofs, and future formalization for this conjecture
belong under `field-universality/`. In particular:

- do not modify `main.tex` for exploratory results;
- do not import project files into `lean/BAA`;
- do not describe numerical evidence as a theorem; and
- do not transfer a claim to the BAA paper before its assumptions and proof
  status are explicit.

## Exact experiment harness

The Rust program in this directory constructs the encoder once over the
integers. It reduces that same matrix over `F_2`, each requested odd prime
field, and the rationals. Every field therefore receives exactly the same local
map and permutation sample.

For each field, the program exhaustively evaluates every coordinate-deletion
set and reports the worst-support rank profile and exact minimum distance. This
is exponential in the global length and is intended for small, certified
instances. A work-budget guard prevents accidental large runs.

The schedule is explicit. For example, `APAP` denotes

```text
B -> accumulator -> permutation 1 -> accumulator -> permutation 2,
```

whereas `PAPA` denotes

```text
B -> permutation 1 -> accumulator -> permutation 2 -> accumulator.
```

Build and test the harness with

```text
cd field-universality
cargo test --release
```

Run the default coupled experiment with

```text
cargo run --release -- \
  --local examples/repetition3.matrix \
  --blocks 2 \
  --schedule APAP \
  --samples 10 \
  --primes 3,5 \
  --output results/repetition3-apap.csv
```

For larger lengths and small prime fields, enumerate all field-valued messages
instead of all coordinate-deletion sets:

```text
cargo run --release -- \
  --local examples/repetition3.matrix \
  --blocks 12 \
  --schedule PAPA \
  --samples 100 \
  --primes 3 \
  --distance-only \
  --no-rational \
  --classify-witnesses \
  --support-spectrum \
  --max-minors 100000 \
  --max-n 60 \
  --output results/rep3-papa-b12-s100-classified.csv
```

Each CSV row records the field, exact distance, normalized distance, binary
baseline, signed distance gap, normalized gap, complete deletion-rank profile,
minimum message, minimum codeword, sample seed, and every permutation. The
profile column is empty in distance-only mode.

With `--classify-witnesses`, every odd-field minimum is assigned to one of four
exact algebraic classes: distance-beating `p`-torsion, nonbeating `p`-torsion,
a dependency shared with characteristic two, or a dependency already present
over the rationals. The output records binary and rational ranks, Smith
2- and `p`-nullities, an explicit maximal-minor certificate when one exists,
and the exact top determinantal divisor when its maximal-minor count fits the
`--max-minors` budget.

The classifier also records every intermediate circuit word, its weight,
number of nonzero runs, value composition, and a normalized trace signature.
It computes a colored Weisfeiler--Leman fingerprint of the peeled sparse
factor core. This value is a reproducible graph fingerprint, not a proof of
graph isomorphism or a canonical labeling. The retained-matrix core is a dense
diagnostic; the factor core preserves the local-map, permutation, and
accumulator equations needed for a peeling argument. A zero minor budget
disables only the exhaustive determinantal-divisor gcd.

In distance-only mode, `--support-spectrum` records the minimum codeword
weight at every exact message support. It is optional because retaining the
default early exit matters for large exhaustive message enumerations. The old
`--classify-losses` spelling remains as a deprecated alias.

The modular and rational rank routines reuse flat row-major workspaces
throughout the exhaustive subset loop. The rational rank routine uses checked
`i128` fraction-free elimination. Torsion certificates use arbitrary-precision
integers, so determinant overflow cannot silently invalidate a classified
loss.

## Bulk exponent scripts

The numerical optimizers require Python 3, NumPy, and SciPy. The primary
certificate uses Python's standard `decimal` module. The independent checker
uses exact rational boxes and the interval module from mpmath. These scripts
analyze the Markov-type exponent derived in `CLEAN_THEORY.md`:

```text
python scripts/optimize_bulk_exponent.py \
  --prime 3 --repetition 3 --delta 0.196 --eta 0.0001 --restarts 64

python scripts/optimize_symmetric_ternary.py \
  --repetition 3 --delta 0.195 --eta 0.05

python scripts/certify_symmetric_ternary.py \
  --delta 0.196 --eta 0.0001 --precision 80

python scripts/verify_symmetric_ternary.py \
  --delta 0.196 --eta 0.0001 --precision 80

python scripts/sparse_spacing_bound.py \
  --delta 0.196 --eta 0.0001

python scripts/sparse_spacing_bound.py \
  --prime 5 --repetition 3 --delta 0.26 --eta 0.00001
```

The first command searches the full vector-composition problem. The second
uses the exact sign-symmetric ternary reduction and separately checks the
message-cutoff and output-boundary faces.

A positive value identifies a candidate bad type. A negative numerical value
is not a proof that the global maximum is negative.

For `F_3`, the theory note proves that sign symmetrization is without loss.
The certificate script covers the reduced three-dimensional outer domain with
deterministic entropy bounds. Its default 80-digit arithmetic adds a directed
interval around every arithmetic operation and logarithm. The independent
checker constructs another cover using exact rational boxes and mpmath
intervals. Both implementations certify the bulk half of the `0.196` theorem
at cutoff `0.0001`.

The two sparse commands report the constants in the exact ternary and quinary
proofs. The proofs use integer inequalities; the displayed floating-point
values are diagnostic.
