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

For every fixed prime `p`, the project now gives an explicit integer `m` such
that

```text
288(p-1)^2 m^4 < (m-2)^6(m+2)^3,
Pr[Delta_{p,N} <= 1/m^2] -> 0.
```

This first ladder theorem uses only the random-spacing estimate and the
deterministic fact that two accumulators can reduce Hamming weight by at most
a factor of four. The least certified roots begin with `m=7,8,10,10` for
`p=5,7,11,13`, respectively, and give distances of order `p^(-4/5)`. In
particular, the field-uniform criterion alone gives the quinary bound `1/49`.
A separately verified sign-orbit bulk certificate raises the proved quinary
distance to `1/20`. The same method, closed under field-multiplication
automorphisms, raises the septic bound from `1/64` to `9/500`. These constants
do not locate the true prime-field thresholds.

At the binary benchmark `1323/10000`, exact support-indicator certificates now
control initial bulk slabs over `F_5`, `F_7`, `F_11`, `F_13`, and `F_17`.
The bound follows by retaining only the zero/nonzero indicators in the two
mutual-information terms. Its small-weight leading term is
`(1/r-1/2)A log(1/A)`, which is negative for every repetition factor greater
than two and every fixed field. Exact sparse-spacing specializations meet the
lower endpoint of every certified slab. This is a proved field-uniform tangent
mechanism, not yet a proof for the remaining bulk domains.

The support analysis also gives one field-uniform theorem. For every fixed odd
prime `p`, the expected number of low-output codewords whose message support is
at most

```text
K / (60 (p-1)^2)
```

tends to zero at output fraction `1323/10000`. The proof combines the
support-indicator bound above `K/(100(p-1)^2)` with the exact random-spacing
estimate below it. `scripts/verify_uniform_support_slab.py` checks all
constants using rational arithmetic.

At `p>=5`, the binary-endpoint proof also covers a field-uniform full-symbol
face. If both the message and first accumulator state use zero and one
opposite symbol pair, the first transition pays at least `(A/2)log(2)` beyond
its support mutual information. After charging the message sign entropy, this
leaves a net `(A/6)log(2)` toll. Together with the exact support bound, this
proves negativity for every such type with `A>=1/500`. This result identifies
a concrete mechanism behind the `p=5,7,11` optimizers, but it does not yet
control compositions spread across several sign orbits.

For globally sign-symmetric transitions, the remaining sign loss also has an
exact orbit-level formula. It is the sum, over active increment--state orbit
pairs, of `w(log(2)-h(q))`, where `q` chooses between the two possible folded
target-orbit branches. Pinsker's inequality turns a small loss into an
explicit near-balance condition. This reduces the next proof step to a
quantitative statement about a nearly unbiased walk on the folded odd cycle,
together with the separately measured magnitude mutual information.

A stronger reduction removes sign symmetry altogether. For each accumulator,
the internal-information excess is a KL divergence from the coupling that
keeps the observed support cells but makes the nonzero symbol values
conditionally independent. Data processing first to cancellation and then to
the nonzero target symbol gives an explicit two-level lower bound. Applied to
both stages, it produces a rigorous composition-only upper bound with
`3p+2` variables. Numerical maximization is negative at the binary endpoint
for `p=5,7,11`.

For `F_5`, canonicalizing the largest nonzero message mass and applying the
entropy conjugate removes the remaining conditional compositions. The middle
and output laws reduce to sixteen symbol branches, each of which may use its
own valid KL cut. This leaves only the five support variables
`(A,B,C,x_1,x_2)`. Rational branch certificates give the exact upper bound
`-0.00138189` at the numerical maximizer, close to the reduced value
`-0.00146130`. The global five-variable cover, rather than the original
14-variable cover, is now the main bulk obligation.

The exact checker proposes fresh cuts on off-curve support boxes and verifies
their rationalized potentials independently. It bounds the complete convex
surrogate on vertices of the joint five-dimensional support polytope, so the
support and conditional penalties cannot be maximized at incompatible
points. The current 100-box exact upper bound is `0.09312368`; the cover is
improving but is not yet a theorem at `0.1323`.

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

1. Strengthen the proved `F_5` bound from `1/20` toward the published binary
   threshold using a full-composition bulk certificate; its sparse bound is
   already proved through `0.26`.
2. Certify a positive ternary bulk type at `0.197` and locate the first-moment
   transition above `0.196`.
3. Raise the proved `F_7` bound from `9/500` and determine how certificate size
   scales with the number of sign orbits.
4. Establish the rank/cogirth formulation as an independent algebraic view of
   characteristic-specific failures.
5. Investigate the matching upper bound required for coupled universality.
6. Extract checkable enumerator hypotheses for blockwise BAA base codes only
   after the repetition-based theorem is complete.

The first-moment proof resolves the uniform quantifier for the ternary
`0.196` theorem. Rank and residual-core diagnostics remain useful for finding
counterexamples, but the theorem does not depend on a uniform rank comparison.

The spacing argument and deterministic accumulator floor prove quinary
relative distance `1/49` without bulk optimization. A rigorous sign-orbit
bulk certificate raises the bound to `1/20`. The spacing argument also covers
the sparse half of a stronger quinary theorem through output fraction `0.26`,
using message cutoff `0.00001`. Reaching that stronger range still requires a
full-symbol bulk certificate. The ternary sign reduction does not generalize:
an explicit stationary type over `F_5` makes sign averaging decrease the full
exponent.

The general certificate under development uses entropy duality. Arbitrary
symbol potentials give affine upper bounds on each transition entropy.
Secants upper-bound the negative composition entropies on a box. Each box then
reduces to a linear program that retains all field-symbol constraints.

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

python scripts/sparse_spacing_bound.py \
  --prime 5 --repetition 3 --delta 0.05 --eta 0.016666666666666666

python scripts/verify_sign_orbit_bulk.py \
  --delta 1/20 --eta 1/60 --max-boxes 8000 \
  --maximum-cut-pairs 16 --precision 80

python scripts/sparse_spacing_bound.py \
  --prime 7 --repetition 3 --delta 0.018 --eta 0.0625

python scripts/verify_sign_orbit_bulk.py \
  --prime 7 --delta 9/500 --eta 1/16 --max-boxes 500 \
  --maximum-cut-pairs 96 --automorphism-cuts --precision 80

python scripts/optimize_sign_orbit_bulk.py \
  --prime 5 --repetition 3 --delta 0.05 --eta 0.016666666666666666

python scripts/search_sign_orbit_cover.py \
  --prime 5 --repetition 3 --delta 0.05 --eta 0.016666666666666666 \
  --maximum-cut-pairs 16 --max-boxes 8000

python scripts/verify_prime_ladder.py --prime 5

python scripts/verify_prime_ladder.py --prime 7

python scripts/diagnose_entropy_dual.py \
  --prime 5 --repetition 3 --delta 0.196 --eta 0.0001

python scripts/search_entropy_dual_cover.py \
  --prime 5 --repetition 3 --delta 0.196 --eta 0.0001 --max-boxes 100

python scripts/optimize_lifted_orbit_bulk.py \
  --prime 5 --repetition 3 --delta 0.1323 --eta 0.0005

python scripts/search_full_symbol_cover.py \
  --prime 5 --repetition 3 --delta 0.1323 --eta 0.0005 \
  --maximum-cut-pairs 32 --max-boxes 1000

python scripts/verify_support_indicator_bulk.py \
  --prime 5 --repetition 3 --delta 1323/10000 \
  --eta 1/2000 --cutoff 1/500 --precision 80

python scripts/verify_uniform_support_slab.py

python scripts/verify_support_indicator_bulk.py \
  --prime 5 --delta 1323/10000 --eta 1/500 \
  --cutoff 1323/2500 --internal-values-paid \
  --target-upper-bound 1/10000 --precision 80

python scripts/search_internal_information_gap.py \
  --prime 5 --delta 0.1323 --eta 0.002 --mode lagrangian \
  --gap-multiplier 1.5 --restarts 32

python scripts/verify_one_orbit_toll.py \
  --eta 1/500 --support-upper-bound 1/10000

python scripts/optimize_sign_symmetric_bulk.py \
  --prime 7 --delta 0.1323 --eta 0.0008 --objective exponent \
  --restarts 16

python scripts/optimize_conditional_product_bound.py \
  --prime 5 --delta 0.1323 --eta 0.002 --projection two-level \
  --restarts 32

python scripts/verify_reduced_composition_bulk.py \
  --prime 5 --delta 1323/10000 --eta 1/500 \
  --max-boxes 1000 --precision 80

python scripts/verify_eliminated_composition_bulk.py \
  --prime 5 --delta 1323/10000 --eta 1/500 \
  --max-boxes 1000 --precision 80

python scripts/verify_sign_orbit_bulk.py \
  --prime 5 --delta 1323/10000 --eta 1/500 \
  --sign-symmetric --automorphism-cuts --maximum-cut-pairs 64
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

The sparse commands report the constants in the exact ternary and quinary
proofs. The sign-orbit verifier proves the bulk half of the quinary `1/20`
and septic `9/500` theorems with directed logarithm intervals and exact
rational LP duals. The prime-ladder verifier independently checks the all-integer
criterion and its least passing denominator. The displayed floating-point
values from the sparse script are diagnostic.

The sign-orbit optimizer and floating cover are certificate-design tools.
Their output is not a proof; only the exact verifier may discharge the bulk
condition.

The entropy-dual commands develop the full-composition certificate. They use
floating-point optimization to design entropy-dual cuts and a box cover.
Their negative output would still require the independent interval checker
described in `CLEAN_THEORY.md`.

The lifted-orbit and full-symbol commands diagnose the remaining quinary gap.
They are numerical design tools. Their output cannot certify a theorem.

The support-indicator command is an exact checker for the small-message slab.
The final sign-symmetric command invokes the exact full-symbol quotient under
global sign symmetry. Its current box search is incomplete and does not
justify restricting the unrestricted quinary problem to that quotient.
