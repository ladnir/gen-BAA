# Lean formalization plan for Generalized BAA Codes

## 1. Purpose and proof policy

This directory will contain a Lean 4 formalization of the finite algebraic and
combinatorial claims in `../main.tex`. The formalization is also a proof audit:
Lean statements must make every index convention, feasibility condition,
sample space, and endpoint case explicit.

The governing rules are:

- Do not use `sorry`, `admit`, project-local axioms, or opaque declarations that
  merely restate a paper claim.
- Do not silently weaken or alter a manuscript claim. First add an entry to the
  audit log in Section 9, then state and prove the corrected theorem.
- Use exact finite counts and rational probabilities for the finite ensemble.
  Do not introduce measure theory merely to express a uniform finite sample.
- Keep theorem statements abstract over finite fields where the mathematics is
  field-independent. Use concrete computation only for fixed local codes.
- Keep asymptotic entropy arguments, floating-point optimization, and open
  concentration claims outside the initial formalization scope.
- Prefer definitions that reduce predictably. Concrete exhaustive checks must
  use compact `BitVec` data and shared verified evaluators rather than lists of
  heap-allocated vectors.

Completion means that every item marked **Formalize** in Section 7 builds, has
no unfinished proof, and has an audited `#print axioms` result.

## 2. Reproducible project setup

The first implementation milestone will add, beside this document:

```text
lean/
  lakefile.toml
  lean-toolchain
  Main.lean
  BAA/
    Basic.lean
    ...
```

Pin both Lean and mathlib to `v4.32.1`:

```text
lean-toolchain: leanprover/lean4:v4.32.1
mathlib revision: v4.32.1
```

Use a `lakefile.toml` package named `baa` and a library named `BAA`. Fetch
precompiled dependencies with `lake exe cache get`. Commit `lake-manifest.json`
after the first successful dependency resolution. Do not track `.lake/` or
build output.

The standard verification command is:

```powershell
cd lean
lake build
```

Run exhaustive code-spectrum checks serially. Never run two performance or
exhaustive-enumeration benchmarks at the same time.

## 3. Core mathematical interfaces

All project declarations live in namespace `BAA`.

### 3.1 Words, supports, and codes

Use the following representation throughout the abstract development:

```lean
abbrev Word (F : Type*) (n : Nat) := Fin n -> F
abbrev Code (F : Type*) (n : Nat) [Semiring F] :=
  Submodule F (Word F n)
```

Use mathlib's `hammingNorm` for weight. Define `support x` as the finite set of
coordinates at which `x` is nonzero, and prove once that
`(support x).card = hammingNorm x`.

Define the coordinate subspace supported inside `S : Finset (Fin n)` and then

```lean
shortening (C : Code F n) (S : Finset (Fin n)) : Code F n
```

as its infimum with `C`. Define `shorteningDim C S` as its finite dimension.
Keep "support equals `S`" separate from "support is contained in `S`"; this
distinction is essential in the proof of (3).

Define minimum distance as the minimum `hammingNorm` among nonzero codewords,
with the zero-code edge case stated explicitly. The main BAA theorems assume a
nontrivial local code.

### 3.2 Ordered linear maps

Coordinates are `Fin n`, ordered from `0` through `n - 1`. Define:

- `accumLinearEquiv`: prefix sum, with output coordinate `i` equal to the sum
  over `j <= i`.
- `diffLinearEquiv`: the inverse, with coordinate zero unchanged and later
  coordinates equal to the current value minus the previous value.
- `permLinearEquiv`: coordinate reindexing by `Equiv.Perm (Fin n)`.
- `diagLinearEquiv`: coordinatewise multiplication by `Fin n -> Fˣ`.
- `monomialLinearEquiv`: the declared composition of the permutation and
  diagonal maps.

Prove both inverse identities pointwise. State one row/column convention in
`Basic.lean` and use it everywhere; matrix formulas must be derived from the
linear maps rather than used as their definitions.

### 3.3 Block direct sums

Let output length be definitionally `B * m`, avoiding divisibility hypotheses
and natural-number division. Define a named equivalence

```lean
blockCoordEquiv (B m : Nat) : Fin B x Fin m ~= Fin (B * m)
```

using mathlib's standard finite-product equivalence. Define `outerCode C0 B` as
the vectors whose restriction to every block belongs to `C0`. Prove:

- membership is equivalent to blockwise membership;
- `finrank (outerCode C0 B) = B * finrank C0`;
- its weight enumerator is the `B`-th power of the local enumerator (B1).

The BAA code is the image of the outer code under two independent monomial
linear equivalences followed by accumulators. Its dimension theorem follows
from invariance of finite dimension under linear equivalence.

### 3.4 Exact finite probability

For a finite sample type `Omega`, define uniform probability and expectation by
cardinality in `Rat`:

```lean
uniformProb (E : Finset Omega) : Rat
uniformExpectation (X : Omega -> Nat) : Rat
```

The binary ensemble sample type is a pair of permutations. The nonbinary sample
type additionally contains the two functions of nonzero diagonal entries. Prove
independence facts as factorization identities for finite sums, not as
measure-theoretic lemmas.

All divisions in displayed probability formulas are interpreted in `Rat`.
Separate natural-valued enumerators from rational-valued probabilities so that
binomial identities remain in `Nat` as long as possible.

### 3.5 Polynomials and generic rank

Use `Polynomial Nat` for ordinary weight enumerators and a two-variable
`MvPolynomial` for support/dimension and block-rank enumerators. Prove
coefficient statements before mapping coefficients into `Rat` or `Real`.

For the two independent diagonal layers, use variables indexed by
`Sum (Fin n) (Fin n)`. Define generic rank as matrix rank after mapping an
`MvPolynomial` matrix into its fraction field. This prevents specialization of
diagonal variables from being confused with generic rank.

## 4. Concrete local-code certification

The abstract code API remains function-based. Fixed binary examples use a
computational `BitVec n` representation plus proved conversions to and from
`Word (ZMod 2) n`. A spectrum certificate consists of a theorem equating the
abstract weight enumerator to a concrete polynomial. Use `native_decide` for
fast regression checks, but not as the final publication-facing certificate:
it introduces a generated native-code axiom. The accepted theorem must use
kernel reduction (`decide`) or a symbolic proof over the same verified `BitVec`
evaluator.

### 4.1 Extended binary Golay code

Construct the binary cyclic `[23,12,7]` code using

```text
g(X) = X^11 + X^9 + X^7 + X^6 + X^5 + X + 1
```

and extend every word by its parity coordinate. Prove injectivity of the
12-bit message encoder, then certify dimension 12, minimum distance 8, and

```text
1 + 759 z^8 + 2576 z^12 + 759 z^16 + z^24.
```

### 4.2 Reed--Muller `RM(2,5)`

Represent a message by the coefficients of the 16 square-free monomials in
five variables of degree at most two. Evaluate it on all 32 Boolean inputs in
lexicographic bit order. Prove injectivity and certify dimension 16, minimum
distance 8, and the spectrum printed in the manuscript.

### 4.3 The explicit `[32,8,12]` BCH subcode

Decode each appendix hexadecimal row least-significant-bit first. Prove the
eight rows linearly independent and certify the 256-word spectrum and minimum
distance.

For the provenance claim, construct `GF(32)` from the irreducible primitive
polynomial `X^5 + X^2 + 1`. Define the narrow-sense length-31 primitive BCH code
by the root checks at powers `alpha^1` through `alpha^10`, and add the parity
extension. Produce an explicit 11-row encoder for this kernel, prove it has
dimension 11 and minimum distance 12, and prove that each of the eight appendix
rows belongs to it. The exhaustive distance computation ranges over the 2,048
messages of the proved-injective encoder, never over all 32-bit words.

### 4.4 Reed--Solomon/MDS codes

Define an evaluation code at `m` pairwise distinct field points using
polynomials of degree less than `k`. Prove injectivity, dimension `k`, and
minimum distance `m - k + 1` using the root bound. Define `IsMDS` through the
equivalent property that every restriction to `k` coordinates is an
isomorphism. Use that interface to prove the shortening identity (5) and the
support/dimension polynomials L1 and R3.

## 5. Formalization milestones

Each milestone starts only after its prerequisites build without unfinished
proofs.

| Milestone | Prerequisites | Deliverables | Acceptance |
|---|---|---|---|
| M0: Bootstrap | None | Pinned Lake project, imports, CI/build command | Clean `lake build` from the `lean/` directory |
| M1: Finite-code core | M0 | Words, weight/support, codes, shortenings, ordered linear equivalences, block direct sums | Inverse and dimension theorems build; small `Fin` examples evaluate correctly |
| M2: Binary accumulator | M1 | Run decomposition, B2 and B3, uniform-permutation slice lemma, B1, B4, B5 | Exact brute-force comparison for every `n <= 8`, plus symbolic theorems |
| M3: Binary local codes | M1 | Golay, `RM(2,5)`, and appendix BCH-subcode constructions and spectra | All stated dimensions, distances, spectra, and BCH membership are theorems |
| M4: MDS interface | M1 | RS evaluation code, MDS restriction theorem, (4), (5), L1, L2, R3, R4 | Shortening dimensions agree with brute force on small fields/codes |
| M5: Large-field trace bound | M1, M4 | Zero-return traces, formula (1), projective normalization/count, (2), audited master bound (3) | Exhaustive comparison on small fields and `n <= 7`; abstract bound builds |
| M6: Backward rank | M1, M4 | G1, G2, neighborhood and block-rank definitions, G5 | Matrix and linear-map conventions agree on small examples; G5 is field-generic |
| M7: Generic paths and containers | M6 | Generic-rank/path-packing equivalence, G6--G8, boundary-corrected G9 | Exhaustive set/permutation check for `N <= 8`; exact expectation theorem builds |
| M8: Audit closure | M2--M7 | Completed claim matrix, discrepancy resolutions, axiom report | No forbidden placeholders; all **Formalize** rows are proved |

### Current implementation status

| Milestone | Status | Current result |
|---|---|---|
| M0 | Complete | Lean/mathlib `v4.32.1` is pinned and `lake build` succeeds. |
| M1 | Complete | Core words, support/weight, total minimum distance, shortenings, ordered equivalences, block sums, construction dimension, and B1 are proved. |
| M2 | Complete | B1, the symbolic B2 run-composition equivalence and exact formula, B3's finite conditional probability, the uniform-permutation slice theorem, B4's exact two-stage expectation, and B5's first-moment bound are proved. `binaryAccumulatorFormula_small` is a kernel-reduced exhaustive B2 certificate covering every positive input/output weight through `N = 8`. |
| M3 | Complete | The appendix range code has proved dimension `8`, exact spectrum, minimum distance `12`, and inclusion in the concrete extended primitive BCH code. The systematic parent encoder is proved equal to the ten-root-plus-parity kernel and has certified dimension `11`, exact spectrum, and minimum distance `12`. The cyclic Golay construction has proved dimension `12`, exact spectrum, and minimum distance `8`, with cached literal rows certified against the parity-extended cyclic generator and a symbolic bridge to the abstract code. The degree-two `RM(2,5)` evaluation code has proved dimension `16`, exact abstract weight enumerator, and minimum distance `8`; its 65,536-message kernel certificate is split into sixteen serially cacheable blocks. |
| M4 | Complete | Reed--Solomon evaluation is proved injective with dimension `k`; the root bound and an explicit degree-`k-1` witness prove exact minimum distance `m-k+1`. A reusable MDS restriction interface proves (5). Global block shortenings are proved equivalent to products of local shortenings, their dimensions and support sizes decompose as exact sums, and corrected (4) is proved for admissible supports. The exact MDS support/dimension polynomial and balance identity prove L1 and L2, including the kernel-checked `[16,4,13]` polynomial and balance specializations R3 and R4. |
| M5 | Complete | Canonical projective normalization, exact projective counts, the cancellation factor (2), zero-return semantics, equation (1), the concrete one- and two-stage bounds, and the full concrete master union bound (3) are proved. The exact-support/projective distinction and the independent support-only kernel branch are explicit. A kernel-reduced exhaustive equation-(1) regression covers `N <= 7`; direct cancellation counts over `ZMod 3` and `ZMod 5` and a symbolic `GF(4)` instance cover the planned small-field checks. |
| M6 | Complete | G1--G5, support propagation, coefficient-independent block obstruction, exact MDS selected-column ranks, and the standard-basis parity-check matrix bridge are proved. Rectangular Cauchy--Binet identifies actual generic minors with tagged selection expansions, and generic rank is exactly the maximum algebraic layered packing size. Difference-matrix minors are equivalent to unique vertex-disjoint self/successor linkages, while repeated MDS parity checks realize the exact `h = m-k` local block capacities. Matrix/linear-map agreement is a theorem, and the path-minor convention is exhaustively checked through `N = 8`. |
| M7 | Complete | G6--G8, the exact endpoint-refined support partition, corrected neighborhood sizes, G8 coefficient/count equality, and the exact four-index corrected G9 expectation are proved. Every counted structural container forces an actual nonzero BAA codeword. The concrete repeated-MDS generic rank is proved equal to the maximum two-layer vertex-disjoint path packing with local block capacities, with exhaustive path/minor regression through `N = 8`. |
| M8 | Complete | The claim matrix separates formalized, definition-only, corrected, deferred, and open items. `lake build` and `lake build BAA.AxiomAudit` pass; the placeholder scan contains no declarations or proof terms using `sorry`, `admit`, or project axioms. Publication-facing theorems audit to standard Lean/mathlib foundations only. Native-decider axioms occur only in the explicitly labeled regression module and are duplicated by symbolic all-size theorems, as required by I-001. |

The status table is descriptive only: the claim matrix and milestone acceptance
conditions remain the authority for completion.

## 6. Proof strategy and edge cases

### 6.1 Binary accumulator count

Prove B2 through a bijection, not factorial algebra alone. A nonzero binary
input toggles the accumulator state, so its output support is a union of runs.
Map each input support of weight `w` and output weight `h` to the positive gap
lengths counted by the two binomial factors. State `w > 0`; allow impossible
parameter choices to evaluate to zero through `Nat.choose` only after proving
the correspondence with the feasibility inequalities. This is implemented as
an explicit equivalence between fixed-length run compositions and the
output-side accumulator event, using `w > 0`, `h > 0`, and `h <= N`.

The uniform-slice lemma says a fixed word of weight `w`, acted on by a uniform
permutation, induces the uniform distribution on all weight-`w` words. The
implemented proof uses an explicit finite-fiber count. B4 then follows by
finite sum reindexing and B5 by the exact finite first-moment bound; both are
now machine-checked.

### 6.2 Large-field zero returns and projective words

Represent an active support by its increasing enumeration. Define the state
after each active symbol and the zero-return trace from that state. Prove that
the first active index cannot be a zero return and that zero returns are
nonadjacent.

Prove (1) with an explicit bijection among:

- a nonadjacent trace;
- zero-state gap lengths;
- positive nonzero-state gap lengths.

State all side conditions on `s`, `h`, `a`, and `N`; prove the probability is
zero outside the feasible region.

For projective counting, normalize each nonzero word by the inverse of its
first nonzero coordinate. Prove existence and uniqueness of the representative
and derive `(q^r - 1) / (q - 1)` without relying on an opaque projective-space
cardinality theorem. For a fixed normalized word and feasible trace, expose the
fresh diagonal variable forced by each zero return. Product counting yields the
`(q - 1)^(-c)` factor; a finite union bound and truncation at one yields (2).

For (3), partition candidate words by **exact** support `S`, while using the
dimension of the shortening supported **inside** `S` only as an upper bound on
the number of projective candidates. This logical distinction must appear in
the theorem statement and proof.

### 6.3 Backward rank and containers

Derive G1 by transporting a parity-check kernel through the inverse linear
equivalences. Prove G2 by identifying words supported in `T` with the kernel of
the column-restricted matrix.

Prove G5 first by a direct rank upper bound: every relevant column terminates
inside the two-step neighborhood, and each outer MDS block contributes at most
`h = m - k`. The full generic-rank/path-packing equality is a separate theorem:
expand minors by Cauchy--Binet over the polynomial diagonal variables, prove
that distinct layer selections yield distinct monomials, and connect nonzero
difference-matrix minors to vertex-disjoint paths. This equality is not needed
as an unproved shortcut in G5.

### 6.4 Exact endpoint correction for G9

For `1 <= t <= N`, refine the run count by an endpoint flag `e` indicating
whether `N - 1` belongs to the subset:

```text
R^0_N(t,r) = choose(t-1,r-1) * choose(N-t,r)
R^1_N(t,r) = choose(t-1,r-1) * choose(N-t,r-1).
```

Their sum is G6. If `e` is interpreted as zero or one, the one-step
neighborhood has exact size `t + r - e`.

The corrected G9 theorem sums over `r1`, `r2`, `e1`, and `e2`. Set

```text
u1 = t  + r1 - e1
u2 = u1 + r2 - e2.
```

The corresponding summand is

```text
R^e1_N(t,r1)
  * R^e2_N(u1,r2) / choose(N,u1)
  * A_N,m,h(u2,t-1) / choose(N,u2).
```

This is the exact finite expectation. G9 as printed is retained only as an
asymptotic shorthand after separately bounding the endpoint correction.

## 7. Manuscript claim matrix

Statuses mean:

- **Formalize**: required in M8.
- **Definition**: introduce the object, but no independent proof obligation.
- **Deferred**: asymptotic, analytic, complexity, or numerical work outside the
  finite formalization.
- **Open**: the manuscript itself does not claim a proof.
- **Correct**: replace the printed statement with the exact audited version.

| Claim | Status | Lean obligation or reason |
|---|---|---|
| Construction and deterministic dimension | **Formalize** | Define the BAA image and prove dimension `B * k` from linear equivalences |
| Encoding `O(N)` | **Deferred** | Requires a separate cost model |
| One-accumulator insufficiency | **Deferred** | Asymptotic second-moment argument |
| B1 | **Formalize** | Direct-sum weight enumerator product and coefficient identity |
| B2 | **Formalize** | Exact binary accumulator input/output count |
| B3 | **Formalize** | Uniform-slice probability derived from B2 |
| B4 | **Formalize** | Exact two-interleaver expected low-weight count |
| B5 | **Formalize** | Finite union/first-moment bound |
| B6 | **Deferred** | Fixed-weight two-accumulator asymptotic |
| B7 | **Deferred** | One-active-block asymptotic contribution |
| B8 | **Deferred** | Fixed-active-block-family big-O bound |
| B9 | **Definition** | Coefficient upper exponent, recorded when the analytic phase begins |
| B10 | **Definition** | Binary accumulator exponent, recorded when the analytic phase begins |
| B11 | **Definition** | Binary BAA exponent, recorded when the analytic phase begins |
| B12 | **Deferred** | Asymptotic entropy criterion |
| Golay and `RM(2,5)` parameters/spectra | **Formalize** | Certified from explicit encoders |
| Appendix `[32,8,12]` parameters/spectrum and BCH inclusion | **Formalize** | Certified from rows and the concrete BCH definition |
| (1) | **Formalize** | Exact nonbinary zero-return support kernel with feasibility conditions |
| (2) | **Formalize** | Projective count and anti-cancellation union bound |
| (3) | **Formalize** | Audited finite master bound with exact-support partition |
| (4) | **Correct** | Add nontrivial/admissible local-shortening hypothesis; then prove the global support-excess bound |
| (5) | **Formalize** | Exact MDS shortening dimension |
| (6) | **Deferred** | Fixed-active-block large-field asymptotic |
| (7) | **Deferred** | Sparse-event big-Theta bound |
| L1 | **Formalize** | MDS support/dimension enumerator |
| L2 | **Formalize** | Finite block-distribution identity |
| L3 | **Definition** | Entropy object deferred with the analytic phase |
| L4 | **Deferred** | First-stage Stirling/entropy limit |
| L5 | **Deferred** | Second-stage Stirling/entropy limit |
| L6 | **Deferred** | Real-analysis maximization theorem |
| L7 | **Deferred** | Asymptotic first-moment exponent |
| R1 | **Deferred** | Numerical rate-`1/2` transition |
| R2 | **Deferred** | Numerical GV root |
| R3 | **Formalize** | Concrete specialization of L1 for `[16,4,13]` |
| R4 | **Formalize** | Concrete specialization of L2 at rate `1/4` |
| R5 | **Deferred** | Numerical rate-`1/4` transition |
| R6 | **Deferred** | Asymptotic marked one-block term |
| R7 | **Deferred** | Numerically supported distance target |
| G1 | **Formalize** | Backward parity-check formula under the fixed convention |
| G2 | **Formalize** | Support/kernel/rank equivalence |
| Generic-rank/path-packing prose claim | **Formalize** | Polynomial-minor theorem and layered vertex-disjoint paths |
| G3 | **Definition** | Two-step neighborhood |
| G4 | **Definition** | Capped local block-rank function |
| G5 | **Formalize** | Coefficient-independent obstruction |
| G6 | **Formalize** | Run count, refined internally by endpoint flag |
| G7 | **Definition** | Local block-rank polynomial |
| G8 | **Definition** | Global coefficient sum |
| G9 | **Correct** | Prove the four-index endpoint-refined formula in Section 6.4 |
| G10 | **Deferred** | Numerical structural transitions |
| G11 | **Deferred** | Numerical rate-`1/2` structural saddle |
| G12 | **Deferred** | Numerical rate-`1/4` structural saddle |
| C1 | **Deferred** | Numerical summary |
| Container-count concentration | **Open** | Record only; do not introduce as an assumption |

## 8. Testing and acceptance criteria

Every module must pass `lake build`. Additionally:

1. Search the project for forbidden placeholders:

   ```powershell
   rg -n "\bsorry\b|\badmit\b|^\s*axiom\b" BAA Main.lean
   ```

   The result must be empty, excluding explanatory comments that are clearly
   marked as such.

2. Add exhaustive small-case tests for B2, B3, (1), run counts, neighborhood
   sizes, and the corrected G9 formula. Cover empty/full supports, `n = 0`,
   `w = 0`, endpoint membership, impossible trace counts, and maximal runs.

3. For each fixed local code, test encoder injectivity, word count, every
   enumerator coefficient, coefficient sum, symmetry where claimed, and minimum
   nonzero weight.

4. For large-field statements, instantiate abstract theorems over at least
   `ZMod 3`, `ZMod 5`, and a small extension field. Compare theorem formulas to
   direct enumeration for small lengths.

5. For rank statements, compare linear-map, matrix, and brute-force kernel
   computations on small random-looking fixed permutations and diagonal values.

6. Maintain an `AxiomAudit.lean` module containing `#print axioms` commands for
   all publication-facing theorems. Document any standard Lean/mathlib
   foundations reported there; no project-local axiom is permitted.

7. Keep `native_decide` checks as fast regression tests only. Publication-facing
   certificates must use `decide`/kernel reduction or symbolic proofs over the
   same proved encoder and conversion lemmas. Do not accept an external
   script's output as a theorem.

## 9. Proof-audit log

Add new entries before changing a paper claim. Each entry records the source
claim, issue, exact Lean replacement, proof status, and required manuscript
edit.

### A-001: G9 omits endpoint corrections

- **Source:** G9 and the two preceding sentences.
- **Issue:** The displayed equality uses `t + r1` and `t + r1 + r2`, while a
  prefix-difference neighborhood has size `t + r - 1` when the final coordinate
  lies in the set. The prose acknowledges only an `O(1)` correction, so G9 is
  not an exact finite equality as printed.
- **Lean replacement:** The endpoint-flag formula in Section 6.4.
- **Status:** Proved. Both endpoint-refined run counts and their sum to G6 are
  proved. The exact one-step neighborhood cardinality `t + r - e` is proved,
  the endpoint slices form an exact support partition, G8's polynomial
  coefficient sum is proved to equal the direct block-pattern count, and the
  concrete two-permutation container expectation is proved equal to the exact
  four-index corrected expression.
- **Manuscript action:** Replace G9 by the exact four-index formula or label the
  existing display explicitly as an asymptotic expression.

### A-002: B2 needs an output-weight side condition

- **Source:** B2 and the sentence introducing it.
- **Issue:** The text assumes `w >= 1` but does not state `h >= 1`. For a
  nonzero input, invertibility makes `h = 0` impossible, while the displayed
  term `choose(h-1, r_w-1)` is not an ordinary natural binomial at `h = 0`.
  Interpreting subtraction as truncated natural subtraction would incorrectly
  make the right side positive in small cases.
- **Lean replacement:** State B2 for `1 <= w <= N` and `1 <= h <= N`; prove
  separately that the count is zero when `h = 0` and `w > 0`.
- **Status:** The general positive-weight theorem and the infeasible `h = 0`
  theorem are proved in M2.
- **Manuscript action:** Add `1 <= h <= N` or state an explicit extended-binomial
  convention that makes the infeasible case zero.

### A-003: (4) needs nontrivial local shortenings

- **Source:** The paragraph deriving (4), beginning “If `C_0` has minimum
  distance `d`”.
- **Issue:** The claimed local inequality `s_T - r_0(T) >= d - 1` is false
  for an arbitrary coordinate set `T` whose shortening is zero. For example,
  a `[3,1,3]` repetition code shortened inside a singleton has `s_T = 1` and
  `r_0(T) = 0`, hence `1 < d - 1 = 2`. The Singleton argument requires the
  shortened code to be nontrivial. In the intended application this follows
  when `T` is the local part of an actually realized exact codeword support,
  but “touching a block” alone does not state that condition.
- **Lean replacement:** Require every block counted as active to have a
  nontrivial local shortening (for MDS codes, equivalently require
  `m - k + 1 <= |T|`). Under this admissibility condition Lean proves the
  stronger equality `s = r + t * (m - k)`.
- **Status:** Corrected theorem proved in M4.
- **Manuscript action:** Add the nontrivial-shortening/exact-realized-support
  hypothesis before (4), or restrict “active support” explicitly to the local
  support of a nonzero codeword.

### Audit watch items

These are not yet recorded as errors; the relevant milestone must resolve them:

- In (3), verify that the sum is a partition by exact support while `r_S`
  belongs to the shortening with support contained in `S`.
- In G1, verify all inverses and transposes after fixing one matrix convention.
- In the generic-rank argument, prove that distinct layer selections really do
  yield distinct diagonal monomials, including repeated coordinates and all
  endpoint cases.
- For the BCH provenance claim, verify the paper's least-significant-bit
  coordinate convention against the chosen primitive element and parity
  coordinate.

### I-001: Native decision tactics are outside the kernel axiom boundary

- **Source:** Initial formalization plan, concrete-code acceptance policy.
- **Issue:** `#print axioms` reports generated native-decider axioms for
  theorems proved by `native_decide` and, in Lean `v4.32.1`, `bv_decide`.
  This conflicts with the requirement that publication-facing finite
  certificates be kernel-checkable.
- **Resolution:** Keep native checks for performance, but duplicate accepted
  certificates with `decide` or symbolic proofs. B2's symbolic theorem and
  small checks, the appendix and Golay certificates, and the split `RM(2,5)`
  certificate now feed publication-facing theorems whose audit contains no
  generated native-decider axiom. The `N <= 8` difference-linkage regression
  is explicitly audited as native-only; its publication-facing counterpart is
  the symbolic all-`N` minor/linkage equivalence and exact generic-rank theorem,
  neither of which depends on a generated native-decider axiom.
- **Manuscript action:** None; this is a proof-engineering audit item.

## 10. Definition of done

The finite formalization is complete when M0 through M8 pass, every
**Formalize** or **Correct** row has a theorem, all fixed-code certificates are
reproducible from their explicit encoders, and the audit log has no unresolved
item affecting a proved theorem. Deferred and open claims must remain visibly
separate; they are not prerequisites for declaring the finite development
complete.
