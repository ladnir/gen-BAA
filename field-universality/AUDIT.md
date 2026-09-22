# Formalization and certificate audit

## 2026-09-05: investigation-paper editorial pass

The paper now treats the specified encoder factorization as part of the
lifting model; lifting a collapsed binary generator is a separate operation.
The fixed-prime baseline theorems are retained in an appendix rather than
advertised as the main contribution. The quinary `0.2606` value remains a
numerical prediction, with the cutoff and global-optimization gaps stated
in the main text. The stopped `0.22` campaign is condensed to its methodological
limitation; its detailed history remains in the research notes.

The Walsh--Hadamard fixed-alphabet probability statement is scoped to
independent identically distributed projective types and fixed block count.
It does not assert failure for every deterministic assignment from a finite
alphabet. Product tensorization alone does not settle finite-length second
moments; the prose now separates that identity from the numerical pair tests
and the remaining concentration problem.

## 2026-09-05: cutoff coverage and pair-diagnostic scope

The sign sweep near `0.2606` uses message fraction `A>=0.002`. The sparse
specialization previously stated in `CLEAN_THEORY.md`, Proposition 14, uses
`A<=10^-5` and output fraction `delta<=0.26`. These sources do not cover
`10^-5<A<0.002`. The exact normalized-tube certificate concerns only a
restricted set of compositions at `delta=0.22`; it does not close that gap
near the proposed transition. References to a completed sparse-to-bulk bridge
must be restricted to the tube, with its complement still open.

There is a small exact improvement to the sparse specialization. Its proof
uses only `4 delta(1-delta)<(22/25)^2`, which still holds at `delta=131/500`.
The new margin is `61/62500`; the remaining sufficient inequalities are
unchanged. The specialization can therefore extend through `0.262` at the
same message cutoff. This does not establish a global distance bound.

Product-pair tensorization and the numerical 48-dimensional Hessian concern
fixed one-replica marginals. Neither searches all pair compositions. Agreement
between the full and projected single-replica optimizers also uses a shared
initialization and utilities; it is not independent global optimization.

`F5_DISTANCE_EVIDENCE.md` records the coverage ledger, new bounded searches,
and a covariance-based check of the local pair Hessian. Unsuccessful or
insufficiently feasible optimizer outputs are retained but excluded from
the accepted search evidence. Floating curvature checks are not certificates.

## 2026-08-25: branch-dependent Fenchel cuts

The first eliminated-composition checker optimized its Fenchel cut after
fixing a pure nonzero-symbol branch. Its computation had the order

```text
max_branch min_cut bound(branch,cut).
```

A valid upper bound requires

```text
min_cut max_branch bound(branch,cut).
```

Equivalently, the checker must fix a complete cut certificate before it
maximizes over the conditional symbol compositions. The same requirement
applies to both accumulator stages.

### Counterexample

At `p=5`, `delta=0.21`, and `eta=0.002`, consider the support box

```text
A  in [2351/128000, 277/12800],
B  in [1321/64000, 87/3200],
C  in [1261/8000, 21/100],
x1 in [451/64000, 277/25600],
x2 in [87/25600, 87/12800].
```

Branch-specific cuts gave the purported exact upper bound `-0.108823`.
The full transition optimizer produced a feasible type in this box with
exponent `-0.008383997`. Hence the branch-specific value was not an upper
bound.

The regression test
`test_global_cut_quantifiers_bound_feasible_full_type` stores this feasible
type and checks it against the corrected envelope.

### Correction and scope

The checker now uses one common second-stage cut mixture across all sixteen
nonzero-symbol branches. A first-stage certificate must also use one common
cut mixture. Floating-point optimization proposes the mixture; exact rational
arithmetic and directed intervals check the resulting bound.

The change withdraws the eliminated-checker diagnostics `-0.00138189`,
`0.09312368`, `0.00846858`, `0.00271039`, and the later regional-bank values.
No theorem in the paper used these incomplete-cover diagnostics. The proved
sign-orbit bounds `0.05` over `F_5` and `0.018` over `F_7` are unaffected. The
exact sparse-spacing certificates are also unaffected.

## 2026-08-26: septenary local maximum

The first cell-target search at `p=7`, `delta=0.22`, and `eta=0.0008`
reported `-0.00036564`. A later full-composition search found a dense,
sign-symmetric feasible point with cell-target value `-0.00004311`. The first
run had therefore missed a higher local maximum.

The value `-0.00036564` is withdrawn as an estimate of the global maximum. No
proved result used it. The replacement value is also only the largest value
found by floating-point multistart optimization; its negative sign is not a
certificate. The optimizer now reports the cell-target and
stationary-coupling payments separately and supports an exact
sign-symmetric parameterization for auditing this branch.

## 2026-08-26: stale retained-composition box value

`F5_BOUND_OBSTRUCTION.md` initially recorded `0.902508` as the worst upper
bound after 200 splits of the retained-composition checker. The current
command

```text
python scripts/verify_conditional_product_bulk.py \
  --prime 5 --delta 1323/10000 --eta 1/500 --max-boxes 200
```

returns `1.4638637550165112`. Its worst box has both KL-payment lower bounds
equal to zero. Changes to the checker and its split order preceded this audit,
and the earlier run was not stored with a complete certificate trace.
Therefore, the earlier number cannot currently be reproduced.

Both values describe incomplete covers. Neither value is a theorem or a
counterexample. The discrepancy does not affect the proved quinary bound
`0.05`. Future cover diagnostics must store the command, worst box, cut bank,
and split trace together.

## 2026-08-26: the internal-information quantity can be negative

The decomposition `Psi=Phi-G` does not imply `Psi<=Phi`. A constrained
quinary search at message support `A=0.1` found

```text
G = -0.0462098120.
```

This sign is consistent with the definition

```text
G = [I(D_1;S_1)-I(U_1;V_1)]
  + [I(D_2;S_2)-I(U_2;V_2)]
  - H(D_1|U_1)/3.
```

The two bracketed terms are nonnegative, but the last term can be larger.
Consequently, a support box with `Phi<0` cannot be discarded without a lower
bound on `G`. The scripts that extract and probe the nonnegative support
frontier are now labeled heuristic-only. No theorem or stored certificate
used the invalid pruning rule.

The corrected leading reduction bounds each bracket directly. For one stage,
define

```text
Q(d,s)=P_UV(u,v)P_(D|U)(d|u)P_(S|V)(s|v).
```

Then `D(P_DS||Q)=I(D;S)-I(U;V)`. Data processing under
`(d,s)|->(u,v,d+s)` supplies the support-cell projection bound recorded in
`CLEAN_THEORY.md`.

## 2026-08-26: projected-KL cut bank missed a dense composition

The local constraint-generation search for the projected-KL dual bank
reported envelope `-0.0007291058` after 18 rounds. A subsequent global box
cover found a feasible composition with bank value

```text
+0.0000455302.
```

Fitting the information projection at that composition gave exponent upper
`-0.1488975`. Thus the projected bound did not fail; the local nonlinear
search had failed to generate a necessary dense-regime cut. The bank was
extended with the fitted cuts and their multiplicative `F_5^*` orbits.
Additional frontier refinements produced a final floating bank of 120 cuts
per stage.

The value `-0.0007291058` remains a diagnostic from the initial local search.
It is withdrawn as a bound on the complete finite-bank envelope. The global
floating box cover with the enriched bank reached upper
`-0.0001036133` after 3854 splits. Floating arithmetic prevents this value
from being a theorem until directed replay is complete.

## 2026-08-28: a near-zero affine-gate closure was not stable

The first continuous search on `cell20766/leaf1` reported a near-zero negative
maximum. A resumed search found interior basins as large as approximately
`+0.00239`. The earlier result had too few effective restart basins and did not
satisfy the later strict-margin acceptance rule.

The near-zero result is withdrawn. The chamber was split into seven leaves,
each of which now has a located maximum below `-0.0001`. The replacement cover
uses 19 leaves in total. Every accepted result records `closed=true`,
`complete=true`, and termination by the strict target margin.

This discrepancy did not affect a theorem. It changed the search protocol:
child runs inherit counterexample points but restart from the child's local
expert bank. Inheriting every parent expert caused large LPs and poorer
conditioning without improving closure.

## 2026-08-28: source-cell affine gates do not transfer directly

The exact split tree for frontier cell 20766 was evaluated on the 50 worst
frontier cells. The audit reused the saved affine coefficients and expert
pairs. Only the source cell passed the necessary vertex-margin,
nonnegativity, and partition tests. Some transferred weights became smaller
than `-1`, so a negative transferred mixture alone would not be a valid upper
bound.

This result rejects direct gate reuse. It does not reject adaptive affine
gates. On frontier cell 1, one broad chamber remained positive after eight
exchange rounds. Splitting coordinate 14 at `0.0825` and refitting each child
closed both children numerically.

## 2026-08-28: strict rational positivity can damage a gate

The first rationalization forced every affine weight above `10^-8`. Eighteen
chambers retained their numerical margins, but `leaf4a` reopened at
`+0.00158659`. A formerly zero expert received enough weight to dominate the
mixture. The positive value was a rationalization artifact, not a new maximum
of the saved floating gate.

The replacement rationalization uses denominator `10^14` and positivity
buffer `10^-9`. Its largest coefficient adjustment is `1.49121e-7`. All 19
rationalized gates again have located maxima below `-0.0001`. An independent
checker verifies exact partition identities and nonnegativity. Directed global
maximization of the analytic mixtures remains open.

## 2026-08-28: the first Bernstein remainder used a signed weight bound

The first Arb/Bernstein artifact for `leaf4b2a1` treated every affine gate
weight as nonnegative on the rectangular Bernstein domain. The exact gate
certificate proves nonnegativity on the chamber. The rectangle is larger
because it ignores some bounds on dependent simplex coordinates. One affine
weight reaches

```text
-2951351877696512146548371136527
 / 25000000000000000000000000000000
```

on that extension. The first Taylor remainder therefore used an invalid
one-sided weight bound.

The corrected remainder uses the maximum absolute weight on the rectangular
extension. All 32 dyadic boxes still close. The generator also parses source
decimals directly as rationals. A separate 256-bit verifier reconstructs the
face, its dyadic cover, and every Bernstein/Taylor bound. Its worst upper bound
is strictly negative. The superseded artifact is not evidence for any claim;
only the corrected `exact-source` artifact and its replay are retained as
certificate evidence.

This correction closes one named candidate face. It does not prove that the
candidate-face list is exhaustive, and it does not prove the `0.22` distance
bound.

## 2026-08-28: coordinate monotonicity does not select one simplex face

The first log-Hessian reduction treated each fixed derivative sign as if it
selected one coordinate endpoint. That inference is valid on a rectangle.
It is incomplete on a bounded simplex.

For example, let (q_{12}) be the dependent coordinate in the last symbol
simplex. If the derivative along (q_{14}-q_{12}) is negative, a maximizer
can reach either of two boundaries:

```text
q14 = q14_min    or    q12 = q12_max.
```

The active boundary depends on the other coordinates. The same issue occurs
for the middle simplex when a positive derivative can terminate at either an
explicit upper bound or the lower bound on the dependent coordinate.

The directed sign certificates remain valid. In particular, the saved
certificates prove their stated Hessian signs over their complete source
faces. The invalid step was the inference from one sign certificate to one
successor face. The two-dimensional certificate for the selected log
derivative is therefore evidence for one endpoint branch only. It does not
yet prove monotonicity on the complete high-`q3` chamber.

The replacement proof must construct a face union. For every monotone
coordinate, the union includes each explicit-coordinate boundary and each
dependent-coordinate boundary that can become active. A verifier must check
that this union covers all possible terminal points before the proof uses a
bound from any residual face.

## 2026-08-28: a message derivative omitted the normalized-coordinate width

An exploratory derivative bound parameterized each free coordinate as

```text
q_j = q_j^min + t_j (q_j^max-q_j^min).
```

The polynomial derivatives were taken with respect to `t_j`. The first
message calculation inserted the entropy derivative with respect to `q_j`
without multiplying by `q_j^max-q_j^min`. It consequently reported large
positive bounds for message coordinates 2, 3, and 4. At the located maximizer,
the direct tangent derivatives for coordinates 2 and 3 are negative. This
comparison exposed the missing factor.

The implementation now includes the coordinate width. All message-derivative
claims produced before this correction are withdrawn. No theorem or exact
certificate used them. The output-coordinate derivative certificates are
unaffected because the entropy term does not depend on the output block.

## 2026-08-28: rectangular derivative corners were outside the reduced simplex

The first calculation for the final `q_4` interval found negative derivative
values at Boolean corners of the normalized parameter box. Those corners did
not satisfy the inherited bounds on the dependent coordinates. For example,
some corners forced `q_10<0.78` or placed `q_11` above its chamber upper bound.
They belong to the rectangular polynomial extension, not to the feasible
polytope.

The corrected certificate subtracts nonnegative facet slacks before applying
the Bernstein bound. Its entropy term also retains the dependence between
`q_0` and `q_4`. A constant lower bound for the entropy slope loses this
dependence and does not close the final interval. The retained lower envelope
uses a secant for `log(q_0)` and a tangent for `-log(q_4)`.

The first exact anisotropic transform used only each polynomial's power
degree. Its bound was weaker than the floating proposal, which had elevated
every coordinate to Bernstein degree three. Selective degree elevation showed
that only normalized coordinate `q_9` requires degree three. This elevation is
an exact change of Bernstein representation; it does not change the
polynomial.

The final certificate partitions the `q_4` interval into eleven boxes and
checks their exact union on a 1,024-cell dyadic audit grid. An independent
320-bit implementation reconstructs every box and source hash. The smallest
replayed derivative lower bound is approximately `2.00435e-7`. Thus, after
the separately certified reduction `q_10=q_10^min`, every maximizer in this
chamber satisfies `q_4=q_4^max`.

## 2026-08-28: coordinate-2 monotonicity fails on the reduced face

Both observed maximizer signatures put message coordinates `q_2` and `q_3`
at their lower endpoints. This observation does not extend to a global
monotonicity statement. On the reduced face `q_10=q_10^min` and
`q_4=q_4^max`, feasible product vertices give both signs for the derivative in
the `q_2-q_0` direction. The sampled range is approximately
`[-0.00370,0.00305]` for the mixture derivative.

No theorem claim depended on coordinate-2 monotonicity. A valid exhaustive
reduction must combine direct value bounds with first-order conditions, or
must use a complete active-set cover. The two observed faces cannot be declared
exhaustive from their numerical frequency.

## 2026-08-28: the first coordinate-2 split used the raw box, not the feasible simplex

The first child split used the midpoint of the stored coordinate interval for
`q_2`. That interval ignores the message-simplex identity. After fixing
`q_4=q_4^max`, the inherited lower bounds on `q_0`, `q_1`, and `q_3` reduce
the feasible upper endpoint of `q_2` to approximately `0.01812307`. The first
threshold, approximately `0.01867474`, was larger than this endpoint. Its
upper child was empty, and its lower child still contained the complete
feasible chamber.

The retained split uses the feasible midpoint

```text
q_2 = 0.0136425285272678.
```

The two children share this boundary and cover the reduced chamber. Fresh
constraint generation, rather than inherited parent weights, closes both
children numerically. Their rational mixtures have 15 and 13 experts. Each
partition has denominator `10^14`, sums identically to one, and has a strictly
positive exact minimum weight.

This split exposed a second provenance error. With `--result-override`, the
rationalizer and the face-certificate tools loaded the parent gate from the
tree manifest. They therefore ignored the child restriction. The tools now
load the gate named by the overridden result and preserve that gate in the
rational partition. Regression tests cover this rule. All child partitions
and face certificates were regenerated after the correction.

The six active-face signatures found by multistart optimization now have
exact analytic certificates. The low child uses four covers with 2, 32, 2,
and 8 boxes. The high child uses covers with 4 and 1 boxes. The two faces with
varying message coordinates use an affine entropy tangent on each message
subbox. Every cover uses exact rational source data and Arb balls. A separate
320-bit implementation reconstructs and closes all 49 boxes.

These certificates exhaust the observed signatures only. They do not prove
that every child maximizer lies on a listed face. Sampled derivatives suggest
two next reductions on the high child: the `q_3-q_0` derivative is uniformly
negative in 100,000 samples, and the `q_14-q_10` second derivative is positive
in 5,000 samples. Neither sampled statement is a theorem.
