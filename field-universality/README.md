# Field Universality for Addition-Only Encoders

This directory contains an independent research project on addition-only linear
encoders over finite fields. It does not form part of the BAA paper or the
`lean/BAA` formalization. Results may be transferred to another project only by
an explicit later decision.

## Question

Fix a family of binary local maps `B_N`, a constant number of additive linear
transforms, and independent uniform coordinate permutations. Interpret the same
integer circuit over a field `F`, without random diagonal multipliers, and let
`C_{F,N}` be its row-space code.

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

## Conjecture hierarchy

The first target is one-sided distance inheritance. For every fixed odd prime
`p` and every `epsilon > 0`,

```text
Pr[Delta_{p,N} < Delta_{2,N} - epsilon] -> 0.
```

This statement is sufficient for the intended application: an encoder over a
prime field would inherit the binary relative-distance guarantee while using
only additions, permutations, and fixed local binary maps.

The stronger coupled-universality conjecture was the original motivation:

```text
Delta_{p,N} - Delta_{2,N} -> 0 in probability.
```

The experiments and bulk optimization do not currently support this stronger
statement. It remains useful as a falsifiable comparison, not as the main
proof target. Generalization beyond the two-accumulator circuit should wait
until the sparse and bulk bounds are proved in this model.

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
adversarial characteristics, deterministic interleavers, and arbitrary binary
local maps are outside the first scope.

The local maps must preserve dimension and minimum distance over the fields
being compared. This property is an assumption until proved for a particular
family.

## Proof program

1. Prove exact distance preservation for scalar extension from `F_2` to
   `F_{2^r}`. This gives a baseline theorem and isolates the odd-characteristic
   obstruction.
2. Establish the rank/cogirth formulation for the complete encoder circuit.
3. Prove field-independent rank for supports whose backward constraint graph
   can be eliminated by degree-one peeling.
4. Bound the probability of a non-peelable low-weight residual core.
5. Prove one-sided distance inheritance uniformly over all supports below a
   target relative weight.
6. Investigate the matching upper bound required for coupled universality.
7. Generalize only after the two-layer model is understood.

The uniform quantifier over exponentially many supports is the main obstacle.
A fixed-support rank theorem is useful but does not by itself imply a distance
theorem.

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

The numerical optimizers require Python 3, NumPy, and SciPy. The certificate
uses only Python's standard `decimal` module. These scripts analyze the
Markov-type exponent derived in `CLEAN_THEORY.md`:

```text
python scripts/optimize_bulk_exponent.py \
  --prime 3 --repetition 3 --delta 0.13 --eta 0.01 --restarts 64

python scripts/optimize_symmetric_ternary.py \
  --repetition 3 --delta 0.195 --eta 0.05

python scripts/certify_symmetric_ternary.py \
  --delta 0.19 --eta 0.01 --precision 80
```

The first command searches the full vector-composition problem. The second
uses the exact sign-symmetric ternary reduction and separately checks the
message-cutoff and output-boundary faces.

A positive value identifies a candidate bad type. A negative numerical value
is not a proof that the global maximum is negative.

For `F_3`, the theory note proves that sign symmetrization is without loss.
The certificate script then covers the reduced three-dimensional outer domain
with deterministic entropy bounds. Its default 80-digit arithmetic adds a
directed interval around every arithmetic operation and logarithm. The command
above certifies a negative bulk exponent through output fraction `0.19` for
message supports of at least one percent.
