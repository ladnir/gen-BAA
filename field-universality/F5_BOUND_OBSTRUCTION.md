# Improving the quinary distance bound

## Bottom line

The proved relative-distance bound over `F_5` remains `0.05`. The normalized
bulk tube at output fraction `0.22` is now certified exactly, but the
orientation-sensitive complement remains open. The tube starts at the exact
sparse cutoff, so its message-weight range meets the sparse theorem. This
does not cover the orientation-sensitive complement at those weights.
Fixed-dual envelope
scans support the stronger output fractions `0.22`
and `0.24`, but neither scan is a proof: both use floating duals and local
nonlinear optimization.

The shared-middle-law obstruction has been removed at the level of the
numerical objective. A stationary entropy dual gives one globally valid
function of the two adjacent composition laws. Keeping a bank of these
functions active at the same middle law produces a pointwise envelope. This
envelope is negative on the saved global frontiers for `0.22` and `0.24`.

For the current investigation of a threshold near `0.2606`, see
`F5_DISTANCE_EVIDENCE.md`. The tube certificate is restricted to `0.22`;
it does not bridge the message cutoffs near `0.26`.

The remaining obstruction in this stopped `0.22` campaign is certification. We need directed constants and
a globally valid upper bound for each composition box. An SLSQP maximum,
even when reproduced from several starts, cannot supply that theorem.

## Target and current evidence

Let `N=3K`. The encoder over `F_5` is

```text
F_5^K --B_3--> F_5^N --P_1--> F_5^N --A--> F_5^N
      --P_2--> F_5^N --A--> F_5^N,
```

where `P_1` and `P_2` are independent uniform coordinate permutations. The
target theorem is

```text
Pr[d_min(im(A P_2 A P_1 B_3)) <= 0.22 N] -> 0
```

as `K -> infinity`. The probability is over `(P_1,P_2)`.

The current evidence has four different statuses.

| statement | value | status |
|---|---:|---|
| Published quinary lower bound | `0.05` | proved |
| Full-composition transition | about `0.260` | numerical |
| Fixed-dual scan at `0.22` | no positive box found | complete saved-frontier numerical scan; not replayed as one audit artifact |
| Fixed-dual scan at `0.24` | `-0.00059253` | complete 101-box numerical scan; not a certificate |
| Hybrid composition bound at `0.22`, `A>=0.002` | `-0.00118836` | older numerical local maximum search |
| Hybrid bound at its apparent maximizer | `-0.00115832` | exact point certificate |
| Normalized tube at `0.22`, `A>=10^-5` | `< -1.91e-9` | exact 22,471-split certificate; complement not included |

The exact point certificate proves an inequality at one composition. It does
not bound all compositions.

The sparse-spacing argument is already strong enough at `0.22`. It controls
message supports up to `10^-5 K`. The original specialization used
`delta<=13/50`; the 2026-09-05 audit extends it to `delta<=131/500`.

The older numerical hybrid search used the larger cutoff `A>=0.002`. The
strengthened exact tube certificate now covers the interval

```text
10^-5 <= A < 0.002
```

within the normalized tube. Together with the spacing theorem below
`10^-5`, it removes the cutoff gap only for this restricted family.
The unproved complement certificate must use the same matched cutoff.

## Reduced bulk objective

Write

```text
A := 1-mu_0,
B := 1-beta_0,
C := 1-zeta_0.
```

The variables `A`, `B`, and `C` are the nonzero fractions of the message,
middle, and output words. Let `x_1` and `x_2` be the two stationary crossing
masses. Let `mu+`, `beta+`, and `zeta+` be the corresponding conditional laws
on `F_5^*`.

For each accumulator stage, the internal information beyond the support
projection equals

```text
D(P || Q),
```

where `P` is the increment--source law. The reference `Q` preserves the four
zero/nonzero cells and makes the symbols conditionally independent inside
each cell.

The repetition-three exponent has the form

```text
Phi(A,B,C,x_1,x_2)
  + A H(mu+)/3
  - D_1(mu+,beta+)
  - D_2(beta+,zeta+).
```

The support term `Phi` is explicit. The difficulty lies in the final three
terms. Both payments depend on `beta+`.

## Available KL payments

The exact checkers use lower bounds on `D(P || Q)`.

The cell-target payment retains the support cell and the target symbol. The
stationary payment maps `(D,S)` to `(S,S+D)` and retains both state
marginals. If `K` is the image of `Q`, then

```text
S(alpha,beta,x)
  := inf { D(R || K) : row(R)=beta, column(R)=beta }.
```

Both payments are valid. Neither dominates the other. Their maximum gives the
hybrid numerical bound.

A stationary cut consists of rational source potentials `u`, rational target
potentials `v`, and directed bounds on

```text
exp(u_s+v_t)-1.
```

For every stationary coupling `R`, the cut proves

```text
D(R || K)
  >= <u,beta> + <v,beta>
     - sum_(s,t) K_st (exp(u_s+v_t)-1).
```

Floating point proposes `(u,v)`. Exact rational arithmetic and directed
intervals replay the displayed inequality.

## Obstruction in the five-variable checker

The five-variable checker removes `mu+`, `beta+`, and `zeta+`. For a fixed
complete cut certificate, the remaining dependence on `beta+` and `zeta+` is
multilinear. The maximum therefore occurs at pure conditional laws.

The cut must be fixed before this maximization. Choosing a cut after replacing
the conditional laws by a pure branch reverses the required order. The first
implementation made that error; `AUDIT.md` records a concrete counterexample.

The corrected common-cut order is sound but weak. An orbit-complete stationary
bank gives no improvement at the quinary numerical support point:

```text
exact eliminated upper = +0.00118844.
```

After 20 support-box splits at `delta=0.22`, the worst exact upper is
`0.336818`. The cell-target and hybrid banks give the same value on that box.

The observed loss has a structural explanation. The stationary payment is
strong at an interior conditional law. A common cut must also control every
pure branch created by the elimination. The current orbit-complete bank loses
its symbol-sensitive advantage under that requirement.

This computation does not prove that every possible common stationary cut is
useless. It shows that better cut placement alone has not repaired the
elimination.

## Obstruction in the retained-composition checker

The retained checker keeps nine conditional coordinates: three independent
coordinates for each of `mu+`, `beta+`, and `zeta+`. A fixed rational cut is
evaluated at every vertex of each conditional box. This order is sound and
preserves the stationary payment.

At the apparent quinary maximizer, a degenerate rational box gives

```text
exact retained-composition upper = -0.00115832.
```

The global cover remains loose. After 200 splits, its worst exact upper is
`0.902508`. On that broad box, both KL payments have lower bound zero.

This run does not yet isolate the cause. The split heuristic gives support
coordinates priority `4`, crossings priority `2`, and conditional coordinates
priority `1`. It may therefore leave the symbol simplexes broad while refining
variables that do not activate either payment.

The large number is not a numerical counterexample. It reflects the current
box and split choices. The checker also performs three incompatible
optimizations:

1. it maximizes the message entropy over its conditional box;
2. it minimizes the first payment over the first two boxes;
3. it minimizes the second payment over the last two boxes.

These operations may use three different values of `beta+`. Their combination
forgets that one middle law must serve both accumulator stages.

## Best repair of the current checker

Let `ell_i^1` and `ell_j^2` be finite banks of valid cuts for the two stages.
The strongest bound supplied by these banks is

```text
A H(mu+)/3
  - max(0,max_i ell_i^1(mu+,beta+))
  - max(0,max_j ell_j^2(beta+,zeta+)).
```

A single fixed cut pair gives a sound upper bound. It still computes the weak
order `min_cuts max_compositions`. The desired pointwise envelope retains
`max_compositions min_cuts`.

Introduce payment variables `t_1,t_2` and impose

```text
t_1 >= 0,              t_2 >= 0,
t_1 >= ell_i^1  for every i,
t_2 >= ell_j^2  for every j.
```

Then maximize `A H(mu+)/3-t_1-t_2`. These constraints keep every cut active at
the actual shared value of `beta+`.

Each cut is affine in the products

```text
Y_(d,s) := mu+_d beta+_s,
W_(s,t) := beta+_s zeta+_t.
```

On a conditional box, McCormick inequalities give a rational outer
approximation of the two rank-one product tables. A cross-entropy tangent
upper-bounds `H(mu+)`. The resulting node problem is a linear program.

Every true conditional triple has a feasible lift. The linear-program optimum
is therefore a valid upper bound. Subdivision makes the rank-one relaxation
tighter. A floating-point solver may propose an active basis; a rational dual
witness must replay the accepted bound exactly.

The split rule should use the lifted rank-one gaps, not coordinate widths.
For example, split the marginal that contributes most to
`|Y_(d,s)-mu+_d beta+_s|` or `|W_(s,t)-beta+_s zeta+_t|` at the LP witness.

Before global integration, evaluate this epigraph on the broad box responsible
for `0.902508`. Compare it with a fixed-pair vertex bound. If the epigraph does
not materially reduce that box, the current finite cut bank is insufficient.

## Preferred new route: Perron coefficient cuts

The exact accumulator enumerator has transfer matrix

```text
M(X,Y)_(s,t) := X_(t-s) Y_t.
```

Fix positive rational vectors `x,y,r`. Suppose a rational `Lambda` satisfies

```text
M(x,y) r <= Lambda r
```

componentwise. Positivity and coefficient extraction give the finite-length
bound

```text
T_N(q,h)
  <= (r_0/min_s r_s) Lambda^N x^(-q) y^(-h).
```

The prefactor is subexponential. Hence

```text
T_5(alpha,beta)
  <= log Lambda
     - <alpha,log x>
     - <beta,log y>.
```

Optimized Perron cuts are dual-tight on interior feasible types. Unlike the
conditional-product payments, they upper-bound the complete accumulator path
exponent.

The second accumulator and the output-weight sum can be contracted together.
For `0<q<=1`, define

```text
M_2(x,q)_(s,t) := x_(t-s) q^[t!=0].
```

If `M_2(x,q)r<=Lambda r`, then

```text
sum_(z: wt(z)<=d) T_N(h,z)
  <= (r_0/min_s r_s) Lambda^N x^(-h) q^(-d).
```

After one first-stage cut and one contracted second-stage cut, only the
message and middle compositions remain. The resulting upper bound is

```text
U(mu,beta)
  = log Lambda_1 + log Lambda_2 - delta log q
    - (2/3) H(mu) - H(beta)
    - <mu,log x_1>
    - <beta,log y_1+log x_2>.
```

For fixed cuts, `U` is convex. Its maximum on a rational composition polytope
occurs at a vertex. A common convex mixture of cuts can be optimized before
the vertex maximum without changing the quantifier order.

Exact replay is small. Store rational `x,y,r,q`, compute a rational row bound
`Lambda`, verify the matrix inequality, and enclose only logarithms with
directed intervals. No exponential interval is needed for the Perron step.

This route removes the specific obstruction in the current checker. It does
not split one transition entropy into independently optimized support and
symbol payments. It also reduces the quinary bulk problem from 14 variables
to eight independent composition coordinates after output contraction.

The first prototype should target `delta=0.1323` with `A>=0.002`. The existing
small-message theorem already matches that cutoff. Test point fidelity,
broad-box strength, exact rational replay, and closing trend under a modest
box budget. A successful prototype would move the proved quinary bound to the
certified binary benchmark before returning to `0.22`.

## Perron prototype results

The floating prototype is `scripts/explore_perron_bound.py`. It performs four
separate computations:

1. It fits one Perron cut to each accumulator at a numerical maximizer of the
   exact full-composition exponent.
2. It contracts the second accumulator over every output composition of
   weight at most `delta`.
3. It fixes a common mixture of cut pairs before maximizing over a box.
4. It retains the exact two-transition polytope in the inner linear program.

The quantifier order in the third step is

```text
for each box B:
  choose one common mixture w_B;
  maximize the resulting affine upper bound over every feasible type in B.
```

The prototype never chooses a cut after observing the maximizing type. A
child may also inherit its parent's mixture. This inheritance makes the
reported cover bound monotone under subdivision.

At `delta=0.1323` and `eta=0.002`, the optimized Perron value matches the
exact full-composition objective at the numerical maximizer. The best run
observed the values

```text
full-composition exponent  = -0.00194309884237,
Perron upper at that point = -0.00194309883103,
gap                        =  1.13e-11.
```

The two stage gaps were `7.83e-12` and `3.50e-12`. This agreement confirms
that output contraction does not introduce a visible loss at the relevant
optimizer. It is numerical evidence, not a proof of dual attainment.

The current conditional-product checker gives `1.4638637550` after 200 splits
on its worst box. An earlier run recorded `0.902508`; the current tree does not
reproduce that number. `AUDIT.md` records the discrepancy. Both values are
incomplete-cover diagnostics and have no theorem status.

On the current worst box, a common mixture of 24 Perron cut pairs gives
`0.13860` when maximized over the product of the message and middle
composition projections. Retaining the complete transition polytope and
using entropy secants gives `0.15657`. These bounds use different outer
relaxations and should not be compared term by term. Both replace the value
above one from the conditional-product checker by an ordinary subdivision
problem.

Two independent adaptive runs close this particular box numerically:

| seed | cut pairs proposed per box | splits | final upper bound |
|:---:|---:|---:|---:|
| `5` | `8` | `29` | `-0.00338527` |
| `11` | `8` | `38` | `-0.00184501` |

The search splits the coordinate whose entropy secant is loosest at the
current LP maximizer. Each child retains the smaller of its local bound and
the inherited parent bound. The closing trend therefore does not rely on
discarding a valid but inconvenient certificate.

The same implementation does not yet close the full bulk domain. With the
canonical constraint that `mu_1` is a largest nonzero message mass, one run
gave

| split budget | current numerical upper bound |
|---:|---:|
| `0` | `0.88517` |
| `100` | about `0.21` during the same search scale |
| `200` | `0.14865` |

The exact intermediate value depends on the proposed cut bank and seed. The
200-split endpoint is the reproducible conclusion of that run. The decrease
shows that the spectral architecture has not reached a visible structural
floor, but the present splitter is too slow for a global certificate.

All prototype calculations use floating eigenvectors, logarithms, and LP
solutions. None of the negative values is kernel-checkable. The experiment
supports the Perron route because it establishes point fidelity and removes
the recorded broad-box obstruction. It does not improve the proved quinary
distance bound.

The next certificate-design tasks are now concrete:

1. retain parent cut banks, rather than only parent scalar bounds;
2. replace entropy-gap splitting with exact one-step lookahead on the worst
   few boxes; and
3. identify whether the remaining full-domain boxes are loose because of
   entropy secants, sparse-boundary behavior, or missing spectral cuts.

### Exact replay of one point certificate

The first task above is complete for one exact rational type. The generator
`scripts/generate_perron_point_certificate.py` fits floating spectral cuts and
rationalizes the proposed data. Its output is stored in
`results/f5-perron-point-certificate.json`.

The independent script `scripts/verify_perron_point_certificate.py` ignores
the floating metadata. It checks the following statements:

1. The two stored transition matrices have the claimed row, column, and
   increment marginals over `F_5`.
2. The output composition has nonzero mass exactly `1323/10000`.
3. Every fugacity and Collatz vector entry is a positive rational number.
4. Each stored `Lambda` is the exact maximum rational row ratio.
5. Both componentwise Collatz inequalities hold over the rationals.
6. Directed logarithm intervals make the Perron point upper strictly
   negative.

The exact message and middle compositions are

```text
(499/500, 1/1000, 0, 0, 1/1000).
```

The exact output composition is

```text
(8677/10000, 3/50, 123/20000, 123/20000, 3/50).
```

At 100 decimal digits, the checker returns

```text
exact type exponent upper = -0.00374666771745934...,
Perron point upper        = -0.00194326971127067....
```

The finite-length coefficient bound also carries the checked prefactor

```text
1267650600228229401496703205376
----------------------------------------------.
53491235639023502748608145
```

This prefactor is independent of block length and therefore disappears from
the asymptotic exponent. The certificate verifies the spectral lemma and its
directed-log implementation at one exact point. It does not bound a
neighborhood, close a cover, or improve the proved distance.

### Exact replay of one box certificate

The second certificate extends the same cut pair to a rational neighborhood.
For both the message and middle compositions, it uses the coordinate bounds

| coordinate | lower | upper |
|:---:|---:|---:|
| `0` | `99799/100000` | `99801/100000` |
| `1` | `99/100000` | `101/100000` |
| `2` | `0` | `1/1000000` |
| `3` | `0` | `1/1000000` |
| `4` | `99/100000` | `101/100000` |

The exact point above lies in this box, so the certified region is nonempty.
The checker bounds each `x log x` term by a rational secant. Directed
logarithm intervals supply rational upper coefficients for the spectral
terms.

The resulting linear program has 65 nonnegative variables. These variables
represent three compositions and two `5`-by-`5` transition types. Its 156
constraints enforce normalization, both stationary transition polytopes, the
message cutoff, the output threshold, the canonical message sector, the box,
and global coordinate bounds.

The stored exact dual has ten nonzero multipliers. It proves

```text
box upper =
  -1177052369346563038820700893128997870861
  ------------------------------------------------
   664613997892457936451903530140172288000000

           = -0.00177103156580975... .
```

The generator is `scripts/generate_perron_box_certificate.py`. The independent
checker is `scripts/verify_perron_box_certificate.py`, and the stored data is
`results/f5-perron-box-certificate.json`.

This first dual uses only box-face constraints. The checker nevertheless
reconstructs and validates the complete transition LP. A broad-box
certificate must use transition constraints or a substantially richer cut
mixture. Thus this result validates rational entropy secants and exact LP
replay, but it does not yet validate the global cover architecture.

## Structural zero-payment set

The shared middle law does not guarantee that one payment is positive. A
Fourier calculation characterizes the zero set of the target mismatch.

Let `u` and `b` be the conditional nonzero increment and state laws. For every
nontrivial additive character `k`,

```text
beta_full^(k)-K^(k)
  = (1-u^(k)) (x+(A-x)b^(k)).
```

Because `u` has no mass at zero, the first factor cannot vanish over `F_5`.
The target mismatch vanishes exactly when

```text
x=A/5  and  b is uniform on F_5^*.
```

Both stage payments can therefore vanish when

```text
x_1=A/5, beta+ uniform, x_2=B/5, zeta+ uniform.
```

The message law remains arbitrary. Let `Psi_0(A,B,C)` denote the resulting
support exponent. Its output derivative is

```text
d Psi_0/dC
  = log( C(1-C-B/5) / ((1-C)(C-4B/5)) ).
```

The ratio exceeds one exactly when `B(4/5-C)>0`. Hence `Psi_0` is increasing
in `C` throughout the target domain, and the maximum has `C=0.22`. The zero
set therefore reduces exactly to a two-variable triangle.

Numerical maximization on that triangle gives `-0.0053834522` at
`A=0.002`, `B=0.0029876046`. A rational certificate covers the triangle with
22 leaves. The independent verifier checks exact polygon coverage,
binary-entropy secants, cross-entropy tangents, and directed logarithm bounds.
It proves

```text
Psi_0 <= -1/200
```

throughout the zero set. The largest exact replay upper is approximately
`-0.00507563718`. Generate and verify the certificate with

```text
python scripts/generate_zero_payment_certificate.py
python scripts/verify_zero_payment_certificate.py \
  results/f5-zero-payment-certificate.json --precision 100
```

The zero-set result is complete. A stronger exact result now covers a full
normalized neighborhood of that set.

For one stage, define `r_0:=x` and `r_a:=(A-x)b_a` for `a!=0`. The Fourier
identity and Pinsker's inequality prove

```text
D(beta || K)
  >= (1-cos(2 pi/5))^2 ||r-(A/5)1||_2^2.
```

This field-specific inequality has the exact zero set above. It is not strong
enough globally. Floating optimization of the resulting two-stage upper gives
`+0.283595`. Even the exact target KL projection gives `+0.180729`.

The support-normalized cancellation payment is more useful locally. Write
`q=||b-u_4||_2^2`, `z=sqrt(3q/4)`, and `w=(A-x)z`. Every coordinate of `b`
lies in `[1/4-z,1/4+z]`, so data processing gives a binary cancellation floor.
In these variables the scalar tube is diagonal:

```text
(5x_1-A)^2/20+4w_1^2/3
  +(5x_2-B)^2/20+4w_2^2/3 <= 10^-6.
```

The message entropy and first cancellation payment require one further
coupling. If `m` is the largest nonzero message probability, the cancellation
probability is at most `m`. Two applications of Pinsker and minimization over
`m` prove

```text
message entropy deficit + first cancellation KL
  >= A(5x-A)_+^2/[8(A-x)(4A-3x)].
```

With this payment, an exact box certificate closes the tube at
`delta=11/50` and `A>=1/100000`. It uses 22,471 rational splits and has exact
negative worst upper, approximately `-1.919e-9`. The independent replay
checks 22,414 leaves. This endpoint meets the exact sparse-spacing theorem,
so there is no remaining sparse-to-bulk gap.

The complement cannot be handled by a stronger scalar constant alone.
Floating searches remain positive for the proved residual--cancellation and
Fourier--cancellation bounds, and even for the conjectural coefficient
`(4/5)R/max(A,B)`. The positive witnesses retain substantial symbol
orientation, which the scalar reductions discard. The complement must use the
stronger cell-target or stationary cuts.

## Next complement interface

Let the ten raw mismatch coordinates be the five coordinates of each stage's
vector `r-(A/5)1`. Outside the certified tube, at least one coordinate has
absolute value greater than `sqrt(10^-7)`. Thus twenty signed slabs cover the
complement. Each slab is linear if the raw transition coordinates are kept.
The present projected LP has eliminated those coordinates, so its interface
must be extended before this split can be imposed. Necessary lifted moment
conditions for the sixteen nonzero-coordinate slabs were also tested. They
preserve the negative bound on frontier box 0 but leave hard box 166 near
`+0.033`. Raising the covariance loop from 12 to 64 iterations and using 64
affine bilinear cuts still leaves `+0.03254`. This failure is caused by the
moment rank relaxation: the lifted point need not arise from one transition
satisfying the slab.

This is preferable to blind retained-composition subdivision. A bounded
1,000-split complement run was already too slow, and earlier 200--300 split
runs retained positive McCormick artifacts. The slab split restores exactly
the information needed to distinguish the complement from the tube.

The numerical bulk cutoff now matches the analytic small-message endpoint at
`A=1/100000`. No separate cutoff bridge remains.

## Other routes to evaluate

### Verified local models

Cover neighborhoods of all numerical stationary points with interval Hessian
or Taylor certificates. Use a coarse global inequality to exclude the
remaining domain. This route requires a verified list of candidate stationary
regions; numerical restarts alone cannot provide completeness.

### Sound and unsound uses of symmetry

The exact transition exponent has an order-16 action. Common multiplication
by `F_5^*` acts on every composition. Transposition and simultaneous negation
also give independent sign changes of the message, middle, and output laws.

This action supports canonical sectors and orbit caching. It does not support
averaging. Exact high-output examples and a low-output numerical example show
that sign averaging can decrease the exponent. Full multiplicative averaging
also fails.

A weaker theorem could still help: every point with nonnegative exponent might
symmetrize without loss. Another viable split would certify neighborhoods of
sign-symmetric maximizers by negative curvature in sign-odd directions, then
cover the asymmetric complement separately. Neither statement is proved.

Unconditionally sound uses include canonical box representatives,
orbit-complete cut banks, invariant common certificates, and separate
sign-even and sign-odd coordinates. These changes reduce cost but not
dimension.

### Direct certification of the full transition problem

The full exponent has transition matrices with linear marginal constraints.
An entropy dual gives affine upper cuts for each transition entropy. Retain an
epigraph variable for each accumulator and impose every dual upper cut. This
keeps the pointwise minimum of the entropy-cut bank and preserves the shared
middle composition linearly.

Composition entropies enter with negative signs. Secants of `x log x` give
rational upper bounds on those terms. Each node then reduces to a linear
program with directed bounds on the dual log-partition constants.

This route certifies the full-composition exponent instead of the weaker
hybrid relaxation. Its numerical transition near `0.260` offers more margin.
Boundary faces and large dual potentials are the main expected difficulties.

### A joint three-variable information projection

For fixed compositions `mu,beta,zeta`, the two exact stationary information
payments equal one three-variable projection:

```text
I_stat(mu,beta)+I_stat(beta,zeta)
  = min_R D(R || mu tensor beta tensor zeta),
```

where `R` satisfies `X+B~beta` and `B+S~zeta`. The identity follows by gluing
the two optimal pair laws conditionally through `B`. The reverse inequality
is the nonnegativity of `I(X;S|B)`.

This formulation makes the shared-middle obstruction explicit. Its dual has
125 exponential terms over `F_5`, so it is better suited as a theory lemma or
tensor-scaling experiment than as the first exact checker.

### A lower target

A theorem at `0.10`, `0.1323`, `0.15`, or `0.18` would improve the proved
`0.05`. A target sweep should use the sound checker and report certificate
cost, not only numerical maxima. Lowering `delta` does not by itself fix the
loss on broad conditional boxes.

### Relation to weighted nonbinary RAA results

Classical nonbinary RAA theorems often insert independent random nonzero field
multipliers. Those multipliers reduce the analysis to a scalar weight
enumerator. The addition-only ensemble has no value randomization, and its
maximizing symbol laws need not be uniform. Weighted results are therefore a
benchmark, not a domination theorem for the present ensemble.

## Support-cell information projection

The most promising current reduction avoids a spectral cover. For one
accumulator, let `P` be the joint law of the increment `D` and source state
`S`, and let `U` and `V` be their zero/nonzero indicators. Define

```text
Q(d,s) = P_UV(u,v) P_(D|U)(d|u) P_(S|V)(s|v).
```

The comparison law preserves the exact support table and removes only the
dependence between field values inside its four cells. Expansion of relative
entropy gives the exact identity

```text
D(P_DS || Q) = I(D;S)-I(U;V).
```

Push both laws through `(d,s) |-> (u,v,d+s)`, and denote the image of `Q` by
`R`. Data processing then gives

```text
I(D;S) >= I(U;V) + D(P_UVT || R).
```

This inequality has the correct rare-event scale. Ordinary Pinsker applied
before conditioning produces an `O(A^2)` payment and is much too weak.
Retaining the support cell before observing the sum produces an `O(A)` KL
term.

At the exact-exponent optimizer for `delta=0.1323`, the projection term
captures `0.0006916531` of the first stage's `0.0006916926` internal excess.
It captures `0.0004196419` of `0.0004294042` in the second stage. Replacing
both exact information payments by the support information plus these
projection terms gives a rigorous pointwise upper bound on the exponent.

The script `scripts/optimize_character_surrogate.py` numerically maximizes
this upper bound. Two independent 12-restart runs report

| seed | numerical surrogate maximum |
|---:|---:|
| 17 | `-0.0018144965` |
| 31 | `-0.0018210461` |

Both candidates have message support `A=0.002` and output support
`C=0.1323`. The KL inequality is rigorous, and a randomized unit test checks
it against the exact information excess. The displayed global maxima remain
floating-point search results. They do not yet prove the distance theorem.

The `20`-cell tables can be removed by dualizing their information
projections. For compositions `a,b`, define `J(a,b)` as the minimum KL
divergence from the independent `(U,V,D+S)` law, subject to the required
support and sum marginals. Its dual has four target potentials and two
support potentials. The resulting composition-only bound is

```text
H(mu)/3 - J(mu,beta) - J(beta,zeta).
```

The complete projected-table relaxation has numerical maximum
`-0.0015187126`. A local constraint-generation search gave numerical
envelope `-0.0013026013`, but a global cover later found a positive escape of
`0.0000455302` in its finite cut bank. The true projected bound at that point
was `-0.1488975`. Adding the fitted cuts and their multiplicative orbits
removed the escape.

The current global checker fixes such a bank and covers all 15 composition
coordinates. Its box LP uses degree-two moments, simplex moment identities,
covariance cuts, first-level RLT constraints, and generated tangents for
entropy and logarithms. The final bank contains 120 cuts per stage. The first
global zero crossing occurred after 3799 splits, at upper
`-6.3289919e-7`. After 3854 splits, the global floating upper was
`-0.0001036133`.

This computation closes the floating relaxation over the complete
composition domain. It does not yet prove the distance theorem. A final
checker must replace the floating potentials, exponentials, logarithms, and
LP solves by directed or exact bounds.

## Matching the first-moment transition

The projected relaxation is strong enough to move beyond the binary
benchmark, but it does not reach the full-composition transition. At
`eta=0.002`, numerical optimization gives

| output fraction | projected upper found |
|---:|---:|
| `0.220` | `-0.00274964` |
| `0.225` | `-0.00028082` |
| `0.230` | `+0.00243353` |
| `0.240` | `+0.00848367` |
| `0.260` | `+0.02260611` |

Thus this projection has a numerical transition between `0.225` and `0.230`.
No amount of exact replay can push that relaxation to the conjectured
first-moment transition near `0.260`.

A joint constrained projection recovers the missing information. Fix
increment composition `alpha`, state composition `beta`, and crossing mass
`x`. Let `J_x` be the resulting `2`-by-`2` support table. Define

```text
Q_x(d,s)
  := J_x([d!=0],[s!=0])
     alpha(d | [d!=0]) beta(s | [s!=0]).
```

Let `R(alpha,beta,x)` contain the laws `R` on `(D,S)` that satisfy

```text
law(D)=alpha,
law(S)=law(D+S)=beta,
law([D!=0],[S!=0])=J_x.
```

For every `R` in this set,

```text
I_R(D;S) = I_{J_x}(U;V) + D(R || Q_x).
```

Consequently, the exact stationary information payment is

```text
I_stat(alpha,beta)
  = min_x [ I_{J_x}(U;V)
            + min_{R in R(alpha,beta,x)} D(R || Q_x) ].
```

The earlier cell-target and stationary projections enforce different subsets
of the constraints defining `R(alpha,beta,x)`. Taking their maximum need not
enforce both subsets on one law. The joint projection does so and is exact
after minimizing over `x`.

The first diagnostic evaluates the joint projection at the false maximizers
of the weaker hybrid relaxation:

| output fraction | hybrid value | joint value at the same compositions |
|---:|---:|---:|
| `0.240` | `+0.00848367` | `-0.01705459` |
| `0.260` | `+0.02260611` | `-0.00677242` |

The joint projection therefore removes the known false escape even at the
numerical full-composition endpoint. The calculation is implemented in
`scripts/diagnose_joint_information_projection.py`. Its unit test checks that
the payment vanishes for an independent stationary law and never exceeds the
mutual information of a supplied stationary transition.

A second prototype keeps both exact transition polytopes and imposes the
entropy-dual cuts through separate stage epigraphs. This repairs the common
mixture quantifier problem: at each fitted composition, the entropy dual
matches the exact local value to numerical precision. Its broad-box loss now
comes from secants for the negative composition entropies. At `delta=0.22`,
the global upper falls from `2.19` to `1.16` in 30 splits, while the worst
local exact value is `-0.117`. This confirms sound point fidelity but is not
yet a competitive cover.

The joint constrained projection has now been dualized directly. The
normalized KL dual uses thirteen independent potentials per stage at interior
quinary points. Its outer problem has three five-symbol compositions and two
crossings, for seventeen variables in total. The implementation is
`scripts/optimize_joint_projection_bound.py`.

The reduced optimizer reproduces the full transition search at the numerical
maximizers:

| output fraction | full value | joint value | joint minus full |
|---:|---:|---:|---:|
| `0.220000` | `-0.0013925661` | `-0.0013925636` | `+2.51e-9` |
| `0.240000` | `-0.0012805762` | `-0.0012803888` | `+1.87e-7` |
| `0.250000` | `-0.0012252644` | `-0.0012252223` | `+4.21e-8` |
| `0.260000` | `-0.0003530571` | `-0.0003530571` | `+8.12e-14` |
| `0.260600` | `-0.0000067697` | `-0.0000067697` | `+3.71e-14` |
| `0.260612` | `+0.0000002151` | `+0.0000002151` | `+1.10e-13` |
| `0.261000` | `+0.0002273217` | `+0.0002273217` | `+1.11e-13` |

The larger discrepancies at `0.24` and `0.25` occur on sparse symbol faces
and track the tolerance of the inner dual. They do not indicate a relaxation:
the information-projection identity equates the two objectives before
optimization. The sweep refines the numerical sign transition to
`[0.260600,0.260612]`. It remains heuristic because neither outer search is a
global certificate.

There is a useful global entropy inequality that does not require a dual cut.
For one stationary accumulator,

```text
I_stat(alpha,beta) >= H(alpha)-H(beta).
```

Indeed, `(D,S)` and `(S,D+S)` are in bijection, and both state marginals are
`beta`. The inequalities for two accumulators telescope. Thus

```text
H(mu)/3 - I_stat(mu,beta)-I_stat(beta,zeta)
  <= H(zeta)-2H(mu)/3.
```

Keeping the two payments separate is stronger on broad boxes:

```text
I_stat(mu,beta)   >= max(0,H(mu)-H(beta)),
I_stat(beta,zeta) >= max(0,H(beta)-H(zeta)).
```

The checker now maximizes

```text
H(mu)/3
  - max(0,H(mu)-H(beta))
  - max(0,H(beta)-H(zeta))
```

over rigorous entropy ranges for the three composition boxes. This is a
five-variable linear program after introducing the two payment epigraphs.
At `delta=0.22`, it lowers the root upper from `0.5364793` to `0.2772976`.
The earlier telescoped inequality alone gave `0.3908506` after 100 splits.

The new floating checker `scripts/explore_joint_projection_box_cover.py`
combines this inequality with inherited joint-dual cuts, explicit support
coordinates, simplex-tight entropy bounds, and a common-composition LP. It
can now seed its cut bank at the numerical saddle. At `delta=0.22`, that
saddle has value `-0.00139258` and message support `0.002`, exactly the dense
cutoff. The broad-box cut lower bounds still collapse to zero, so blind
subdivision remains ineffective. The immediate loss is cut placement and
box geometry rather than point fidelity.

The projected KL relaxation supplies the natural global pruning layer. Let
`T=D+S`. Under the independent reference law, `T` has distribution
`alpha*beta`; every feasible stationary law has `T~beta`. The KL chain rule
therefore gives the exact decomposition

```text
D(R || alpha x beta)
  = D(beta || alpha*beta)
    + E_{t~beta} D(R_{D,S|T=t} || Q_{D,S|T=t}),
```

where `Q` is the independent reference conditioned on `T`. The first term is
the projected relaxation whose floating cover already closes at
`delta=0.1323`. Its numerical maximum is also negative at `delta=0.22`, but
the stored `0.22` box covers remain positive and incomplete. The second term
is nonnegative and is precisely the information discarded by that
relaxation. The next combined checker should use projected cuts to eliminate
broad boxes and invoke joint residual cuts only on the surviving
false-maximizer region. Starting the exact joint cover from the full root
throws away the strongest available global pruning layer.

## Fixed stationary-dual envelope

The exact stationary payment can be dualized without the crossing variable.
For increment law `alpha` and stationary state law `beta`, write

```text
I_stat(alpha,beta)
  := inf D(R || alpha tensor beta),
```

where the infimum ranges over nonnegative laws `R(d,s)` satisfying

```text
D~alpha,  S~beta,  D+S~beta.
```

Fixing feasible entropy-dual potentials gives a globally valid lower bound
of the form

```text
L(alpha,beta)
  = c + <u,(alpha,beta)> - log(alpha^T K beta),
```

where `K` is a positive `5`-by-`5` kernel. The coefficients and kernel are
fixed before `alpha` and `beta` are optimized. Consequently,

```text
I_stat(alpha,beta) >= max_i L_i(alpha,beta)
```

for every bank of valid cuts. For three adjacent composition laws
`mu,beta,zeta`, the resulting exponent upper is

```text
H(mu)/3
  - max_i L_i(mu,beta)
  - max_j L_j(beta,zeta).
```

Both maxima use the same `beta`. This is the quantifier order that the earlier
eliminated and separately minimized checkers lost.

The implementation is `scripts/exact_stationary_dual.py`. The diagnostic
`scripts/diagnose_fixed_stationary_box.py` maximizes a fixed bank on one saved
composition box using an epigraph formulation. It rejects dual fits with a
large primal residual and approaches singular stationary faces through
interior reference laws. Raising an underflowed kernel entry only weakens the
dual lower bound, so boundary evaluation remains safe at the mathematical
level. Floating cut coefficients and SLSQP output remain uncertified.

At `delta=0.22`, a shared bank of 72 cuts per accumulator was grown from the
980-box projected frontier saved after 1,025 splits. The projected LP still
gave upper `+0.0424768` on its worst box because its lifted moment table could
mix incompatible rank-one products. Direct fixed-bank maximization made that
box negative. A sequential scan found no positive local maximum among the 810
boxes whose projected upper was positive; the remaining 170 boxes were
already closed by the projected LP. The weakest box was enriched when its
margin reached `-0.0000638`. This scan is strong numerical evidence, but it
predates the resumable audit script and is not a certificate.

At `delta=0.24`, the same bank left positive boxes, so a separate bank was
grown to 93 cuts per accumulator. The saved projected frontier contains 101
boxes after 100 splits. The resumable scanner
`scripts/scan_fixed_stationary_bank.py` records a fixed-bank maximum on every
box, together with the bank and frontier hashes. Its complete replay stores

```text
boxes requested              = 101,
boxes with positive result   = 0,
boxes with no successful run = 0,
largest upper found          = -0.0005925298.
```

The result is `results/f5-fixed-stationary-scan-0.24.json`. Ninety-four boxes
use two successful SLSQP starts, and seven boxes inherit an already negative
projected-LP upper. These facts make the experiment reproducible; they do not
establish global optimality of the nonlinear searches.

The first exact stationary box certificate is complete. It targets a
coordinatewise radius `1/2000` neighborhood of the sparse maximizer in
frontier box 166 at `delta=0.22`. The certificate stores two sets of twelve
rational stationary potentials. Directed intervals upper-bound the 25
exponentials in each cut and every logarithm in the objective. A rational LP
dual then checks the entropy tangent, the support constraints, and both
McCormick product tables. The independent verifier checks 65 variables and
306 constraints. It also checks that the stored box lies inside the stated
`eta=1/500`, `delta=11/50` domain and contains a rational composition witness.
Its exact rational upper is strictly negative; numerically,

```text
certified box upper = -0.00000872850168.
```

The generator is `scripts/generate_stationary_box_certificate.py`. The
independent replay is `scripts/verify_stationary_box_certificate.py`, and the
stored certificate is
`results/f5-stationary-box-certificate-0.22.json`. Tests reject both an
understated exponential bound and a missing LP dual, and they reject changes
to the target domain or the stored witness.

This certificate validates the complete arithmetic path, but it does not
close frontier box 166. The original projected upper on that box is
`+0.0362898`. A fixed-bank McCormick cover lowers it to `+0.00489876` after
300 splits, with 103 children closed and 170 still active. The worst LP
witness has a strongly negative value under the true fixed-bank envelope;
the remaining positive value is a rank-one relaxation gap. A radius
`3/5000` box around the same maximizer no longer closes with the selected cut
pair, so naive uniform tiling is not attractive.

The next checker should retain the exact replay above but strengthen the
global product relaxation or use locally fitted cuts on the active leaves.
Once frontier box 166 closes, replay its weakest leaves over the rationals and
then apply the same procedure to the remaining `0.22` frontier.

## Direct projected-cover campaign at `0.22`

The projected KL relaxation is itself sufficient for a `0.22` theorem if its
global maximum is certified negative. This route does not need the tube
decomposition: the projected payment is a lower bound for the exact
stationary payment at every composition. Numerical optimization places its
maximum below zero at `delta=0.22`, but the box relaxation must still recover
the rank-one products shared by the two stages.

The resumable floating cover now uses three ingredients:

1. a bank of locally fitted projected duals, closed under nonzero scaling;
2. the full first-level RLT system and separated covariance-PSD cuts for the
   lifted product matrix; and
3. rank-gap subdivision, with optional strong-branch probes for diagnostics.

Dual enrichment is essential. At one former worst LP composition, the saved
bank gave `+0.0142723`, whereas a newly fitted dual gave `-0.0305560`; the two
dual residuals were below `6e-10`. Four enrichment rounds increased each
stage bank from 124 to 348 cuts. At 10,000 splits the worst saved-bank value
rose to `+0.0002052`, triggering eight diverse frontier refinements. The bank
now has 380 cuts per stage. The active checkpoint contains 11,414 splits and
10,948 leaves; its floating upper is `+0.00772705`, and the current worst
saved-bank value is `-0.00641496`. Thus stale local cut
coverage is not the active obstruction: the remaining positive value comes
from the lifted product-mixture and tangent relaxations.

An exhaustive split audit on five leading leaves found that the best
coordinates were `11,14,14,6,12`. The original local score selected the best
coordinate only once. A two-tier rule evaluates the eight nonzero middle
and output candidates with a zero-round probe and applies the full separation
loop only to the winner. In its first 25-split test, the global bound fell
from `+0.0271169` to `+0.0267606` in 35.5 seconds. This is a floating
performance diagnostic, not evidence for a negative global bound. Longer
campaign comparisons found no wall-clock advantage over ordinary rank-gap
splitting, so the production campaign disables the probes.

Threshold placement was tested separately from coordinate selection. Starting
from the identical 8,000-split checkpoint, 500 midpoint splits reached
`+0.0101736`; placing the threshold at the worst LP coordinate, clipped to the
middle half of the interval, reached `+0.0101227` at comparable cost. The
production campaign now uses this clipped rule. The gain is modest but
repeatable on the mature frontier.

A four-point stationary bank, expanded to 32 universal symmetry cuts per
stage, was also tested on a copy of the 9,000-split frontier. Refreshing 50
leaves and processing 100 splits reached `+0.0093734`. The reduction per split
improved, but the larger LPs lost the gain in wall-clock time. These cuts are
therefore used for exact leaf replay, where strength matters more than
throughput, and not in the production floating campaign.

Two alternative formulations were tested on the same frontier. Retaining the
crossing masses and imposing a large signed residual slab did not improve the
inherited projected cap after 20 splits; broad log-partition intervals made
the joint payment vanish. Retaining both complete transition tables made the
positive transition entropies accurate to `1.3e-6`, but separating the
negative composition entropies introduced a secant gap of about `0.544`.
These tests reject the present implementations of those routes. They do not
reject the underlying exact formulations.

The projected cover is therefore the primary certificate route. Its next
gate is a negative floating frontier with a reproducible cut bank and split
tree. Only then should each leaf be rationalized and replayed with directed
logarithm bounds. The exact scalar tube remains an independent audit and a
fallback decomposition if the projected partition becomes too large.

The lifted leaf replay has now passed its first end-to-end test. The generator
`generate_projected_lifted_box_certificate.py` fits projected and stationary
duals, rationalizes their exponential kernels with directed intervals,
separates rational covariance-square cuts, and exports a rational LP dual.
The independent script `verify_projected_lifted_box_certificate.py`
reconstructs the full 138-variable relaxation rather than trusting stored
matrix rows. On a radius-`0.01` box around the current worst frontier
composition, it checks 1,105 constraints and proves exact upper approximately
`-0.0185485`. Removing a dual multiplier or lowering a stored exponential
kernel causes replay to fail. Thus individual exact leaves are no longer the
unknown part of the certificate; exact full-domain coverage and sufficient
subdivision remain open.

The saved checkpoint did not contain its historical split parents. An exact
coverage-map verifier and reconstructor now recover that missing topology.
The verifier accepts a terminal only after rational domain propagation shows
that its exact cell is contained in a saved frontier box. It also checks every
split and infeasibility witness. Strict routing at half-open split boundaries
reconstructs the 10,100-split checkpoint with 250,553 nodes, including 125,276
splits, 110,483 frontier cells, and 14,794 infeasible cells.

Subsequent splits are recorded directly. An independent verifier starts from
the reconstructed frontier and replays 1,314 records through split 11,414. It
checks each rational split, each propagated child, and 23 discarded children
as infeasible. The replay ends with exactly the 10,948 cells in the floating
checkpoint. The split topology is therefore auditable through 11,414. The
remaining task is analytic: subdivide until every active cell
admits a negative rational lifted proof.

One thousand additional coordinate splits reduced the floating upper from
`+0.00851859` to `+0.00785152`. Re-evaluating the leading 100 leaves with 20
outer-separation rounds changed the global upper only to `+0.00785070`; an
unrefreshed leaf became worst. The obstruction is therefore spread across a
broad frontier rather than concentrated in one stale tangent.

A joint-concavity diagnostic also tested whether one fixed pair of stationary
cuts could bypass the moment relaxation. The active pair has four positive
Hessian directions on the 12-dimensional simplex tangent space. Among all
2,397 bank pairs that remain negative at the reported composition, none has a
nonpositive Hessian there. Oblique splits along the four positive modes were
then compared with all coordinate splits on the same box. The best oblique
split gave child upper `+0.00672220`, while coordinate 2 gave
`+0.00524272`. The production rank-gap rule already selected coordinate 2.
These tests reject fixed-pair concavity and Hessian-mode splitting as primary
improvements. They leave higher-order rank-one cuts and a more structured
conditional decomposition as the leading analytic options.

An entropy-charged coupling tests the latter option. For fixed marginals
`alpha,beta`, kernel `K`, and `c>0`, define

```text
U_c(alpha,beta;K)
  = max_P { log(<K,P>) - c D(P || alpha tensor beta) },
```

where `P` ranges over couplings of `alpha` and `beta`. The independent
coupling is feasible, so `U_c` is always at least
`log(alpha^T K beta)`. The divergence term charges exactly the dependence
that the lifted relaxation currently receives for free. At the current worst
composition, the resulting fixed-pair exponent is `+0.00440637` for `c=32`
and `-0.00403531` for `c=64`; it approaches the exact fixed-pair value
`-0.0119436` as `c` grows. These are floating inner optimizations, not box
bounds.

The whole-box test does not preserve this margin. A conditional-entropy
formulation gives upper `+6.56094` after 512 separation rounds at `c=64`.
Its remaining gap comes almost entirely from secants for the varying marginal
entropy. The equivalent exponential-kernel inequality gives pointwise upper
`-0.00389947`, but gives `+0.0123772` on the box. Adding its tangents to the
existing relaxation leaves the reported upper unchanged to numerical
precision. Entropic transport is therefore not a useful broad-box bound in
this form.

An explicit vertex-product diagnostic identifies a sharper obstruction. The
three boxed simplexes at the current worst leaf have 16, 19, and 5 vertices.
Optimizing over all 1,520 vertex triples gives upper `+0.00774258`, essentially
the same as the degree-two moment relaxation. The optimum mixes seven vertex
triples. Every one of those triples has a negative exact bank value, ranging
from `-0.0192` to `-0.234`. Thus generic RLT, covariance, or convex-hull cuts
cannot remove the main gap. The relaxation earns a positive value by mixing
points whose minima are witnessed by incompatible cuts.

The cut identifiers confirm that interpretation. The seven triples use first-
stage cuts `371,5,143,366,348,352,364` and second-stage cuts
`263,210,263,363,363,363,363`. The mixed hull instead rests near first-stage
cut `214` and second-stage cut `2`. At the averaged rank-one composition, the
minimizers are cuts `212` and `209`. The three selections disagree in both
stages. A cut-regime formulation must therefore preserve the selection before
averaging; choosing a better cut only after forming the mixed moments repeats
the same relaxation.

A first disjunctive diagnostic is positive. Grouping the 1,520 vertex triples
by their first-stage minimizing cut gives 35 regimes. Thirty-two regimes meet
the average domain constraints, and every feasible restricted hull is
negative; the weakest upper is `-0.0124175`. Grouping by the second-stage cut
gives 32 regimes, of which 28 are feasible. Those hulls are also all negative,
with weakest upper `-0.00575033`. The calculation uses the existing tangent
bank without additional separation.

This result proves neither disjunctive cover. A composition inside the box need
not be a convex combination of box vertices that share its minimizing cut.
The calculation instead answers the immediate design question: cross-regime
mixing is sufficient to explain the positive hull optimum, and removing it
leaves a useful margin. The next step is to cover the continuous box by sound
regions, each assigned a fixed cut. Candidate regions should be defined by
rational linear inequalities so the existing LP and exact replay machinery can
check them.

The first continuous test also succeeds, but only locally. Exhaustive
coordinate lookahead closes frontier cell 18053 with 16 midpoint splits and
17 negative leaves. Re-evaluating every leaf with 20 separation rounds gives
worst upper `-0.0000689819`. The importer replaces the source cell in the
global checkpoint, and the exact topology verifier replays all 16 new splits.
This is a complete floating disjunction for one cell. It is not a rational
analytic certificate for those leaves.

The next cell exposes the scaling limit of coordinate trees. After 98 targeted
splits, its local upper falls from `+0.00772738` to `+0.000972549`, but 66 of
99 leaves remain positive. A 20-round refresh lowers its upper to
`+0.000961979`. The partial tree is now part of the global checkpoint. The
explicit vertex-product hull for this cell has
upper `+0.00731925`. All 38 feasible first-stage regime hulls and all 20
feasible second-stage regime hulls are negative, with weakest values
`-0.00723695` and `-0.00695333`. Thus the cut-regime explanation generalizes,
whereas a small coordinate tree does not.

Two further simplifications were tested. On all 2,565 vertex triples, one
second-stage cut is negative everywhere. Two first-stage cuts cover all
vertices. Applying only those three cuts to the continuous hull nevertheless
gives upper `+0.0186728`; interpolation between vertices remains essential.
A convex mixture of cuts can make the Hessian nearly negative semidefinite at
the worst composition, but the resulting pointwise upper is `+2.35818`.
Conversely, a mixture constrained below `-0.001` has minimum observed largest
Hessian eigenvalue `+13.2705`. Global concavity therefore destroys the useful
cut value. The remaining target is a perspective or chamber inequality that
preserves the cut switch without paying for an axis-aligned tessellation.

### Piecewise affine cut gates

An affine partition of unity gives a concrete chamber inequality. Let
`f_e(x)` be the exponent upper bound supplied by cut pair `e`, and choose
affine weights `lambda_e(x)` that are nonnegative on the chamber and sum to
one identically. Pointwise validity of every cut gives

```text
exp(exact exponent at x)
    <= sum_e lambda_e(x) exp(f_e(x)).
```

Nonnegativity of an affine weight on a product of boxed simplexes is checked
at the product vertices. The remaining problem is to bound the displayed
analytic function throughout the chamber; checking it only at vertices is
not sound.

This formulation substantially reduces the current worst cell, identifier
20766. Four coordinate cuts first produce five chambers. Affine gates are
negative on every product vertex of all five chambers, but continuous
maximization finds interior violations. An exchange procedure therefore adds
each violating maximizer to the gate LP. It closes three chambers with
floating continuous margins `-0.0000642497`, `-0.0000301136`, and
`-0.0000326693`. Splitting the remaining 21-expert chamber at middle
coordinate 6 equal to `0.04125`, and then its lower child at output coordinate
11 equal to `0.055`, produces three subchambers. Two close with margins
`-0.000328608` and `-0.00362903`.

Further column generation and targeted coordinate splits close the complete
source cell numerically. The final tree has 18 rational splits and 19 leaves.
Every leaf has a located continuous maximum below `-0.0001`. The weakest leaf,
`leaf4b2a1`, has value `-0.0001110275` before rationalization.

The file `results/f5-cell20766-affine-cover-0.22.json` records the exact split
tree. `verify_cell_affine_cover.py` checks its topology, artifact references,
and strict floating margins. The rational partition file replaces every
affine coefficient by a rational with denominator `10^14`. The independent
checker proves that the weights sum identically to one and are nonnegative on
the full product box. The largest coefficient adjustment is
`1.49121e-7`. Multistart optimization of these rational gates keeps all 19
located maxima below `-0.0001`.

These results certify the affine algebra, but not the analytic maximum. The
nonlinear searches use multistart SLSQP and cannot exclude an unlocated
maximum. The cut constants, exponentials, and logarithms also await directed
replay. The source-cell data are therefore certificate-design evidence, not a
distance theorem.

The observed maximizers provide a useful reduction target. The affine
mixtures are not concave on their full 12-dimensional domains. However, every
located maximizer lies on a face of dimension between two and five. The
Hessian restricted to that face is negative definite in all 19 chambers. The
smallest observed curvature gap is `0.280109`. A proof can exploit this
structure only after it excludes higher-dimensional interior maxima and all
unvisited faces.

A separate 64-restart search on the weakest chamber returned 75 numerical
local maxima. All 75 lie on five-dimensional faces, and only two active-face
signatures occur. This concentration suggests a finite face analysis. It does
not prove that the optimizer found every face that can contain a maximum.

One of these faces now has a complete analytic certificate. On the face named
`message-lower-output-11`, the message distribution is fixed. Each gated
expert is therefore a polynomial factor times the exponential of an affine
form in five variables. A degree-eight Taylor expansion and degree-twelve
tensor Bernstein bound close a uniform 32-box dyadic cover. The corrected
worst exact upper bound on the mixture minus one is approximately
`-2.4467e-5`. The remainder uses an absolute affine-weight bound on the
rectangular extension. A separate 256-bit implementation reconstructs and
replays all 32 boxes from the source artifacts.

The weakest chamber also has an exact partial reduction before face analysis.
On the complete chamber, the mixture increases when output mass moves from
coordinate 10 to coordinate 12. If coordinate 12 is at its upper bound, the
mixture increases when output mass moves from coordinate 10 to coordinate 14.
Both statements use rational affine-weight ranges, Arb bounds on the stage
factors, and an exact bounded-simplex linear program. Independent 256-bit
replay verifies both derivative certificates. The chamber bounds satisfy

```text
q_11^min + q_12^max + q_14^max = 0.22.
```

Hence these two derivative statements force `q_10=q_10^min` at every chamber
maximizer.

The message reduction now closes exactly. After fixing `q_10=q_10^min`, eleven
rational boxes cover the complete normalized `q_4` interval. Four boxes cover
the interval below `15/16`. Seven boxes partition the final interval by
normalized coordinates `q_7` and `q_8`. Each box has a positive exact lower
bound for the derivative in the `q_4-q_0` direction. The independent replay
checks the eleven source hashes and their exact union on a 1,024-cell dyadic
grid. Its weakest derivative lower bound is approximately `2.00435e-7`.
Therefore every maximizer in this chamber satisfies `q_4=q_4^max`.

Both observed terminal faces also have exact analytic certificates. The face
`message-lower-output-11` and the face
`message-upper-output-13-14` each close a 32-box Arb/Bernstein cover under an
independent replay. These certificates prove strict negativity on the named
faces. They do not prove that every maximizer belongs to one of them.

The remaining local problem is face exhaustiveness on the reduced domain

```text
q_10 = q_10^min,   q_4 = q_4^max.
```

A global sign argument for `q_2` is unavailable. Feasible points on this
reduced face have both signs for the `q_2-q_0` derivative, even though every
located maximizer puts `q_2` at its lower endpoint. Generic ten-dimensional
interval bounds are also too loose at the root: the best current Taylor upper
is about `1.2774` for a target below `1`.

A feasible coordinate split now gives a sharper local decomposition. The
message-simplex identity reduces the feasible `q_2` interval to approximately

```text
[0.0091619872, 0.0181230699].
```

The exact-decimal threshold `0.0136425285272678` partitions this interval.
Fresh affine-gate searches close both children numerically. After exact
rationalization, their located maxima are approximately `-0.000201584` and
`-0.000203383`. The exact partitions contain 15 and 13 experts and remain
strictly positive on their complete child domains.

Multistart optimization finds four face signatures in the lower child and
two in the upper child. All six now have exact analytic certificates. Four
faces have fixed message distributions. Two lower-child faces retain one or
two message coordinates; their certificates use concavity of entropy to
replace entropy by an affine tangent on each subbox. Selective subdivision
produces 49 exact boxes in total. An independent 320-bit verifier reconstructs
the child gate, rational partition, face, subdivision schedule, and every
Bernstein bound. All 49 replayed upper bounds are strictly negative.

This result closes every observed child face, not the child chambers. The
face list still comes from floating multistart optimization. An exhaustive
proof must exclude unobserved stationary faces or cover them directly. The
best current structural lead concerns the upper child. In 100,000 feasible
samples, the `q_3-q_0` derivative lies below `-0.104`. In 5,000 samples, the
second derivative in the `q_14-q_10` direction lies between approximately
`0.920` and `1.519`. Exact versions would force `q_3` to its lower face and
reduce `q_14` to its two endpoints. These sampled signs guide the next
certificate; they are not proof claims.

The next certificate should combine direct value bounds with blockwise
first-order conditions. The child refits materially change the curvature.
For the lower child, the sampled largest within-block eigenvalues are about
`-8.40`, `-0.84`, and `+0.024`. For the upper child, they are about `-10.84`,
`-0.20`, and `+8.24`. Thus one shared concavity argument will not cover both
children. The upper child instead offers strong one-coordinate signs for
`q_3` and `q_14`. These eigenvalue and derivative samples guide certificate
design; they are not proof claims.

The exact fallback has now been tested rather than left hypothetical.
`verify_stationary_cover_certificate.py` replays an exact rational split tree
whose leaves use stationary duals, directed exponential and logarithm
intervals, McCormick products, and rational LP duals. At the current worst
projected composition, a radius-`0.01` box has exact upper approximately
`-0.0200`. A global pilot nevertheless produced no feasible certified leaf
after 500 splits; it had 477 pending boxes and 24 exactly infeasible leaves.
This route reaches local certainty but partitions the broad 15-dimensional
domain too slowly to replace the lifted projected cover.

## Synthesis of the five independent reviews

The reviews produced one new leading route and several lower-cost diagnostics.

| rank | route | role | first decisive test |
|---:|---|---|---|
| 1 | Support-cell information projection | Reduced KL certificate at the binary benchmark | Certify the negative projection-surrogate maximum at `delta=0.1323` |
| 2 | Perron coefficient cuts | Exact-exponent fallback | Reproduce stored full-composition points and bound a broad box at `delta=0.1323` |
| 3 | Matched cutoff and conditional lookahead | Audit the evidence and splitter | Prove the `1/8000` bridge; test all nine conditional splits on the `0.902508` box |
| 4 | Pointwise cut epigraph with McCormick products | Repair of the earlier KL checker | Compare its broad-box value with independent payment minima |
| 5 | Zero-set stability and local sign curvature | Auxiliary theory and dimension reduction | Certify the zero set and sign-odd curvature near numerical maximizers |

The reviews rejected three shortcuts:

- a global sign-symmetrization theorem is false;
- one fixed cut pair per broad box retains a weak minimax order;
- weighted nonbinary RAA theorems do not transfer to the addition-only
  ensemble.

## Decision criteria

A new idea should pass three tests before a large computation begins.

1. **Point fidelity.** It must reproduce a negative exact bound near the
   numerical maximizer.
2. **Broad-box strength.** It must materially reduce the active global box
   before extensive subdivision.
3. **Exact replay.** Every accepted box must use rational data and directed
   transcendental bounds.

The present retained checker passes the first test. Its 200-box run does not
yet decide the second because the split heuristic may starve conditional
coordinates. The eliminated checker fails the first test at the same support
point.

## Current recommendation

Treat frontier cell 20766 as a completed floating prototype. Its topology and
affine partitions now replay exactly. The next local proof task is a directed
face certificate for each rational mixture. Such a certificate must bound all
faces, not only the 19 faces selected by floating optimization.

Do not copy the source-cell gates across the frontier. Among the 50 worst
frontier cells, 49 fail the necessary transferred-gate vertex test. Most
failures make an affine weight negative. The next frontier cell nevertheless
responds to adaptive refinement. One difficult chamber closes after a split at
coordinate 14 equal to `0.0825`; its children have located maxima
`-0.000389849` and `-0.000291670`. A scalable campaign should refit gates per
cell, retain counterexample points, and split only persistent violations.

Ordinary rank-gap subdivision remains the production fallback. A controlled
100-split comparison from the same checkpoint gave exactly `+0.00772761` for
both ordinary branching and top-three strong branching. Strong branching took
longer, so it is disabled. The checkpoint must continue to store the complete
cut bank, every rational box, and enough information to reconstruct inherited
bounds.

The numerical discovery phase is complete. Local duals remain negative at
the current worst compositions, so further unconstrained optimization is
lower priority than removing the lifted product-mixture gap. The earlier
fixed-stationary scans remain useful audits: the `0.22` scan found no positive
box on its then-current 980-box frontier, and the independent `0.24` scan has
maximum `-0.0005925298`. Neither scan is a global theorem.

The closed projected cover at `delta=0.1323` remains a fallback theorem. Its
floating margin is `0.0001036133`, and its exact replay would move the proved
quinary bound from `0.05` to the binary benchmark. The new joint route is more
ambitious because its underlying objective retains the numerical transition
near `0.260`.

The sparse and bulk cutoffs must still meet in the exact theorem. Before a
full box replay, prove or reject the proposed `A=1/8000` bridge at
`delta=0.22`; otherwise retain the existing sparse theorem and certify the
uncovered intermediate slab separately.

Keep the Perron prototype as the exact-exponent fallback. Its point fidelity
is excellent, but its full-domain cover remains too expensive. The projection
checker should be rejected only if a sound broad-box test loses most of the
observed margin.

The next concrete milestone is a negative floating projected frontier at
`0.22`, followed by a rational certificate generator and an independent
global replay. Only after that replay closes should the same machinery be
applied to `0.24`. A later climb to `0.25` or `0.26` should not begin before
the quantifier and rounding audit is complete at `0.22`.
