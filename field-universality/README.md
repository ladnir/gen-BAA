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

[`expander-lift/README.md`](expander-lift/README.md) begins a separate study of
addition-only Expand--Accumulate encoders. It gives a fixed-message binary
tail comparison, a fixed-prime asymptotic distance argument, and an exact
large-prime transfer of the regular ensemble's two-coordinate failure event.
Its continuation transfers binary witnesses through support 58 at degree 64
for every 128-bit prime, and classifies three- and four-coordinate cancellations.
The subsequent spacing bound and shared opposite-pair constraints cover all
messages through support 50,000 at the reference profile with failure below
`2^-20`, for every prime above 127. A collision-exact transfer and one common
coloring across regions give the latest extension beyond support 4,000.
The original rate-one-half all-support bound remains open. The EA-only
continuation now gives complete joint binary/large-prime guarantees at
rates 8.345% and 3.242%, with respective degrees 32 and 64 and distances
about 6.10% and 12.10%; see
[`COLLISION_PROJECTION.md`](expander-lift/COLLISION_PROJECTION.md). These match
the two preceding certified binary reference tuples, not an optimal frontier.

[`STATUS.md`](STATUS.md) gives the one-page research state: the surviving
conjecture, disproved stronger claims, proved results, evidence, and the two
remaining proof obligations.

[`paper/main.tex`](paper/main.tex) is the publication-oriented manuscript.
The Markdown files in this directory remain the proof audit, experimental
record, and source for claims that are not ready for the paper.

[`DENSE_BASELINE.md`](DENSE_BASELINE.md) gives a positive baseline outside the
BAA ensemble. A random dense binary generator has constant rate and relative
distance close to one half over the binary field and any prescribed finite
collection of prime fields. Its encoding uses only additions but costs
quadratically many of them. This result isolates efficient, structured field
universality as the actual open problem.

[`INITIAL_ANALYSIS.md`](INITIAL_ANALYSIS.md) records the counterexample search,
the algebraic rank formulation, and the evidence that led away from literal
field universality. [`CLEAN_THEORY.md`](CLEAN_THEORY.md) develops the current
sparse--bulk theory, including an exact vector-composition enumerator and a
finite-dimensional bulk exponent. [`F5_BOUND_OBSTRUCTION.md`](F5_BOUND_OBSTRUCTION.md)
explains why the present quinary certificate stalls, separates verified facts
from numerical evidence, and ranks the most promising routes to a stronger
bound. [`SYMMETRY_REDUCTION.md`](SYMMETRY_REDUCTION.md) derives the exact
stationary and two-stage symmetry groups, records the failure of unconditional
sign averaging, and develops symmetry-adapted certificate and analytic routes.
[`HIGH_CHARACTERISTIC.md`](HIGH_CHARACTERISTIC.md)
studies the separate cryptographic regime `N=2^20` with a 128-bit prime. It
gives deterministic dual-minor and residual-core bounds, an exact
counterexample to the condition `p>N`, and a revised random-saturation target.
It also records fixed-deletion experiments at the intended million-coordinate
scale. The sparse harness builds only the requested columns of the dual matrix,
peels field-independent pivots, and ranks the residual core. Run it with
`--fixed-deletion-only --deletion-sizes D,... --no-rational`; the `prefix`,
`even`, and `random` deletion patterns are available through
`--deletion-pattern`.

[`RUN_PROFILE_THEORY.md`](RUN_PROFILE_THEORY.md) proves the exact integer
decomposition by deletion runs, counts every run profile, and gives a
run-sensitive lower bound for the first peeling round. It also proves a
multi-reservoir deferred-decision theorem whose later stages remain valid
after conditioning on earlier pivots. At the intended parameters, the
one-round theorem certifies 23,594 pivots. A four-round instance certifies
14,682 pivots with failure probability below `0.003952`. The zero-slack
fluid limit reaches 15.22% of the deletion. More generally, the recurrence
cannot certify more than 18.90% for any reservoir partition, so reservoir
splitting alone cannot prove complete peeling.

[`WHT_RESTRICTED.md`](WHT_RESTRICTED.md) studies a separate
Walsh-Hadamard quasi-abelian route. It proves that sign-only and every other
fixed random spectral alphabet have vanishing relative distance. It also
identifies a finite-parameter variant with a growing alphabet of small integer
multipliers, for which the pointwise products can be implemented by short
addition chains. The global distance theorem for that variant remains open.

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

A complementary projection retains the complete source and target symbols.
After the support-preserving product law is pushed through addition, the true
transition must lie among couplings whose row and column marginals are both
the state composition. Its KL projection is a matrix-scaling problem with a
low-dimensional dual over target potentials. Taking the larger of this
stationary-coupling payment and the cell-target payment blocks different
composition escape routes. At output fraction `0.22`, hybrid searches give
`-0.00118836` over `F_5` and `-0.00004311` over `F_7`; these are numerical
diagnostics, not global certificates.

The hybrid relaxation is not sharp over `F_5`: its numerical transition lies
between `0.225` and `0.230`, whereas the exact full-composition transition is
near `0.260`. A new joint constrained projection explains the gap. It imposes
the full increment marginal, both stationary state marginals, the support
table, and the sum constraint on one comparison law. After minimizing over
the support crossing, this projection equals the exact stationary information
payment. At the hybrid false maximizers for output fractions `0.24` and
`0.26`, it changes the respective values from `+0.00848367` and `+0.02260611`
to `-0.01705459` and `-0.00677242`. These are numerical point diagnostics,
not global certificates. The implementation is
`scripts/diagnose_joint_information_projection.py`.

The joint projection now has a 17-variable outer optimizer backed by the
normalized KL dual. After removing an unintended distinguished-symbol
constraint, it reproduces the full-composition values throughout the quinary
transition. The largest observed gap is `1.87e-7`; the gaps at `0.26`,
`0.260600`, `0.260612`, and `0.261` are below `1.11e-13`. The exponent is
`-6.77e-6` at `0.260600` and `+2.15e-7` at `0.260612`. This is a numerical
sign bracket, not a global certificate. The optimizer is
`scripts/optimize_joint_projection_bound.py`, and the recorded sweep is
`results/f5-joint-projection-sweep.json`.

Two further diagnostics now separate the lower-bound and typical-distance
questions. For the lower bound, the entropy differences of the two
accumulators are optimized jointly; this lowers the root box upper at
`delta=0.22` from `0.5364793` to `0.2772976`. The next cover will combine the
existing broad projected-KL relaxation with conditional residual cuts from
the exact joint projection. See `F5_BOUND_OBSTRUCTION.md`.

For the typical-distance converse, stationary information tensorizes exactly
at product pair compositions. At `delta=0.261`, the numerical pair exponent
agrees with twice the single exponent to `3.2e-13`. The full 48-dimensional
correlation Hessian at that product is negative definite, with largest
eigenvalue `-2.50e-5`. This is local numerical evidence for a second-moment
argument, not a global pair certificate. See `F5_TYPICAL_DISTANCE.md` and the
three `diagnose_pair_*` scripts.

The stationary payment also has a direct fixed-dual representation on two
adjacent composition laws. A cut has the form
`c+<u,(alpha,beta)>-log(alpha^T K beta)` and is valid globally. Keeping a bank
of these cuts active at one shared middle law removes the quantifier loss of
the earlier eliminated checker. A 72-cut bank finds no positive box on the
saved `delta=0.22` frontier. A separate 93-cut bank gives a complete 101-box
numerical scan at `delta=0.24`, with largest upper `-0.0005925298`.
`scripts/scan_fixed_stationary_bank.py` records the latter scan with bank and
frontier hashes. These are floating local-optimization audits, not distance
theorems; the proved quinary bound remains `0.05` until the box maxima and
transcendental constants receive exact replay.

One local exact replay is now complete at `delta=0.22`. Two rational
stationary cuts, directed exponential and logarithm intervals, and a rational
LP dual certify a radius-`1/2000` composition box with upper
`-0.00000872850168`. The independent verifier checks 65 variables and 306
constraints, the exact `eta=1/500`, `delta=11/50` domain, and a rational
in-box composition witness. This validates the certificate machinery but not the full
frontier: a 300-split floating cover of the surrounding broad box still has
upper `+0.00489876` from the McCormick rank relaxation. See
`results/f5-stationary-box-certificate-0.22.json` and
`F5_BOUND_OBSTRUCTION.md`.

A second exact result closes the normalized bulk tube at `delta=0.22` and
message support at least `1/100000`. A scalar cancellation reduction uses the
mass-scaled radii `w=(A-x)z` together with a coupled message
entropy--cancellation payment. A 22,471-split rational certificate checks
22,414 terminal boxes and has exact negative worst upper (approximately
`-1.92e-9`). This endpoint meets the exact sparse-spacing theorem. The result
does not yet prove the `0.22` distance theorem: the orientation-sensitive
complement of the tube remains open. See `SYMMETRY_REDUCTION.md` and
`results/f5-scalar-tube-cover-0.22-eta-1e-5.json`.

The primary global route now certifies the projected KL relaxation directly,
which would avoid a separate tube-complement theorem. The active floating
checkpoint has 11,414 splits, 10,948 leaves, and upper `+0.00772705`; it is not a
certificate. Five enrichment epochs increased each stage bank from 124 to 380
cuts. At the current worst LP composition the saved bank gives `-0.00641496`,
so the remaining positive value is a lifted relaxation gap.
Exhaustive split probes remain useful diagnostics, but mature campaign tests
found that they did not improve global progress per unit time; the production
run therefore uses the ordinary rank-gap split. Exact lifted replay is now
implemented: on a radius-`0.01` box around the current worst composition, the
independent verifier proves upper `-0.0185485` from 1,105 reconstructed
constraints and a rational LP dual. The global partition also has an exact
replay chain through split 11,414. The verifier reconstructs a frozen
250,553-node cover through split 10,100, then checks 1,314 recorded splits and
their 10,948 active cells. This result closes the topology audit, but not the
analytic bound on every active cell. See `F5_BOUND_OBSTRUCTION.md`.

An explicit vertex-product LP now shows that a generic convex strengthening
is unlikely to close the gap. On the worst leaf, its optimum mixes seven
vertex triples, although every triple has a negative exact bank value. The
leading experiment therefore preserves the disjunction between competing cut
regimes instead of adding more undirected moment inequalities.

One local disjunction is now complete in floating arithmetic. Sixteen midpoint
splits replace frontier cell 18053 by 17 negative leaves; a 20-round refresh
has worst upper `-0.0000689819`. The exact split-log verifier replays the added
topology. A second cell remains positive after 98 targeted splits, so the next
analytic target is a perspective or chamber inequality for the cut switch.
The second tree's 98 verified splits are also part of the global checkpoint;
its refreshed local upper is `+0.000961979`.

Split thresholds are chosen at the worst LP coordinate, clipped to the middle
half of its interval. From the same 8,000-split checkpoint, 500 such splits
reached `+0.0101227`, while midpoint splitting reached `+0.0101736`; the
clipped rule gave about 9% more bound reduction at comparable cost.

An exact stationary-dual split-tree fallback was also implemented and tested.
It certifies a radius-`0.01` neighborhood of the current worst composition
with exact upper about `-0.0200`, but a global 500-split pilot closed no
feasible leaves. The projected lifted cover therefore remains the primary
global route; the stationary tree is retained as local exact replay machinery.

For `F_5`, canonicalizing the largest nonzero message mass and applying the
entropy conjugate removes the remaining conditional compositions. The middle
and output laws reduce to sixteen symbol branches. A complete Fenchel
certificate must be fixed before maximizing over those branches. This leaves
only the five support variables `(A,B,C,x_1,x_2)`.

An audit found that the first checker selected cuts after fixing each branch,
which reversed the required minimax order. The branch-wise diagnostics are
withdrawn. The corrected checker uses one common rational cut mixture. Its
20-box upper bound at `delta=0.1323` is `0.140051`, so the current common dual
is too loose to prove the binary benchmark. `AUDIT.md` gives the counterexample
and records the scope of the correction.

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

The all-prime scanner has a specialized dimension-two mode. At length six it
enumerates every ordered interleaver pair and derives every bad odd
characteristic from the projective column determinants:

```text
python scripts/scan_bad_primes.py --blocks 2 --schedule PAPA \
  --exhaustive-interleavers
python scripts/scan_bad_primes.py --blocks 2 --schedule APAP \
  --exhaustive-interleavers
```

The certified counts are recorded in
[`results/exhaustive-n6-primes.json`](results/exhaustive-n6-primes.json).
Sampled all-prime scans at lengths nine and twelve are recorded in
[`results/bad-prime-growth.json`](results/bad-prime-growth.json).
Exact selected-prime scans at lengths fifteen and eighteen are recorded in
[`results/selected-prime-ladder.json`](results/selected-prime-ladder.json).

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

python scripts/optimize_character_surrogate.py \
  --prime 5 --delta 0.1323 --eta 0.002 --restarts 12 --seed 17

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

python scripts/explore_perron_bound.py \
  --delta 0.1323 --eta 0.002 --restarts 8 \
  --skip-box --search-splits 60 --search-cuts 8

python scripts/generate_perron_point_certificate.py

python scripts/verify_perron_point_certificate.py \
  results/f5-perron-point-certificate.json --precision 100

python scripts/generate_perron_box_certificate.py

python scripts/verify_perron_box_certificate.py \
  results/f5-perron-box-certificate.json --precision 100
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

The Perron command tests a spectral upper bound for the exact accumulator
enumerator. It contracts the second accumulator over the low-output event and
fixes every cut mixture before the transition LP is maximized. Its floating
search closes the previously worst retained-composition box, but not the full
bulk domain. `F5_BOUND_OBSTRUCTION.md` gives the results and proof obligations.

The Perron generator uses floating optimization only to propose rational
fugacities and Collatz vectors. The independent checker verifies one stored
point certificate with rational matrix inequalities and directed logarithm
intervals. That point certificate validates the replay mechanism; it is not a
global distance certificate.

The box generator and checker extend the stored cut pair to one nonempty
rational neighborhood. The checker reconstructs the 65-variable transition
LP, verifies rational entropy secants, and checks a sparse exact dual. The
certified local upper is about `-0.00177103`. This is still a local certificate,
not a cover of the bulk domain.

The support-indicator command is an exact checker for the small-message slab.
The final sign-symmetric command invokes the exact full-symbol quotient under
global sign symmetry. Its current box search is incomplete and does not
justify restricting the unrestricted quinary problem to that quotient.

The projection-surrogate command applies data processing to the exact
internal-information KL identity and retains `(U,V,D+S)`. Independent
floating-point searches currently find a negative maximum at the binary
benchmark. This is the leading certificate-design route, but the global
optimization has not yet been replayed with exact boxes and directed
logarithm bounds.

The current local strengthening uses affine cut gates. A gate is a
composition-dependent partition of unity over valid cut pairs. Its weighted
sum of exponentiated cut bounds is therefore a pointwise upper bound. Eighteen
rational splits reduce the worst projected cell to 19 chambers. Multistart
continuous searches place every chamber below `-0.0001`; the smallest located
margin is `-0.0001110275`.

The split topology now has an exact rational manifest. An independent checker
replays all 19 chambers. The affine weights also have rational coefficients,
sum identically to one, and are exactly nonnegative on each product box.
Numerical optimization of the rational gates retains every margin. These
checks do not prove the distance bound. The analytic mixtures still require
directed global maximization, and the remaining frontier cells require
adaptive gates.

The saved gates do not transfer unchanged. Only the source cell passes the
necessary vertex test among the 50 worst frontier cells. For the next cell,
one transferred chamber reopened in the interior. Splitting that chamber at
coordinate 14 equal to `0.0825` and refitting both children closed them at
`-0.000389849` and `-0.000291670`. This experiment supports adaptive local
refinement, not a universal gate. See `F5_BOUND_OBSTRUCTION.md` for the
certificate data and remaining proof obligations.

The projected-KL dual eliminates the retained tables and leaves 15
composition variables. Run its fixed-bank global relaxation with

```text
python scripts/explore_projected_kl_box_cover.py \
  --bank-iterations 18 --bank-restarts 12 --max-boxes 5000 --seed 11 \
  --outer-iterations 12 --target-upper -0.0001
```

The bank and best-bound queue are cached in `results/`. The first global
floating zero crossing occurred after 3799 splits. After 3854 splits, every
box had upper at most `-0.0001036133`. See
`results/f5-projected-kl-box-cover.json`. This closes the floating
relaxation, but it is not yet a distance proof. The next step is directed or
exact replay of the cut constants and LP certificates.
