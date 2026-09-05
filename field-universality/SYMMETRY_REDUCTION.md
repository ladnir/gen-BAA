# Symmetry reduction for the quinary exponent

## Bottom line

The quinary optimization has more exact symmetry than the current certificate
uses. One stationary accumulator transition has a group of order 40. The full
two-accumulator exponent retains a weight-preserving subgroup of order 16.
After the message composition has been put in canonical order, the remaining
group has order four and permits the exact chamber restrictions

```text
beta_1 >= beta_4,    zeta_1 >= zeta_4.
```

These symmetries reduce duplicate search and strengthen a finite cut bank.
They do not justify replacing a composition by its sign average. The exponent
is invariant under sign changes but is not concave on a sign orbit. There are
explicit feasible points over `F_5` where sign averaging lowers the exponent.

The most useful computational consequence is narrower than full quotienting:
close every stationary cut under the eight scaling-and-reversal symmetries.
On the current broad box at output fraction `0.22`, this reduces the floating
LP upper bound from `0.03287` to `0.01468`. Adding the five state translations
per symmetry has no visible effect. The remaining positive value is caused by
the rank-one moment relaxation, not by a missing label symmetry.

The symmetry therefore changes the next proof attempt. The certificate should
use the eight-image cut closure and pass to sign-even and sign-odd coordinates.
The current solver already separates the full covariance PSD constraint.
Further PSD block diagonalization can reduce cost, but cannot strengthen that
relaxation. The leading route is now analytic: certify the zero-payment set,
then charge departures from it through KL divergence or Fourier modes.

## The stationary transition

Let `D` be the increment and `S` the state before an accumulator step. For
compositions `alpha` and `beta`, define

```text
I_stat(alpha,beta)
  := min I(D;S)
```

over laws satisfying

```text
law(D)=alpha,    law(S)=law(S+D)=beta.
```

The addition is in `F_5`. The repetition-three first-moment exponent is

```text
Psi(mu,beta,zeta)
  = H(mu)/3 - I_stat(mu,beta) - I_stat(beta,zeta).
```

Here `mu`, `beta`, and `zeta` are the message, middle, and output symbol
compositions. Their nonzero masses are constrained by the sparse--bulk split
and the target output weight.

This formulation exposes the relevant symmetry. It also fixes the observer:
the transition payment is the mutual information between the increment and
the incoming state, minimized over stationary edge laws.

## Exact symmetry of one accumulator

Choose `a` in `F_5^*`, `b` in `F_5`, and a sign `epsilon` in `{+1,-1}`. Then

```text
I_stat(alpha,beta)
  = I_stat((epsilon a)_# alpha, (x |-> a x+b)_# beta).
```

For `epsilon=+1`, apply the bijection

```text
(D,S) |-> (aD,aS+b).
```

For `epsilon=-1`, reverse the stationary edge:

```text
(D,S) |-> (-aD,a(S+D)+b).
```

In the second map, the new outgoing state is `aS+b`. It has the same law as
the new incoming state because `S` and `S+D` have the same law. Both maps are
bijections on the two observed variables, so they preserve mutual information.

The parameters give `4 * 5 * 2 = 40` transformations. The group consists of
four nonzero scalings, five state translations, and edge reversal. A numerical
audit of all 40 images of one transition found a maximum value discrepancy of
`1.91e-14`; the largest stationarity residual was `4.65e-9`. These figures are
implementation checks, not part of the proof above.

## Symmetry of the two-stage exponent

Hamming weight distinguishes zero, so state translations do not survive in
the full distance problem. Let the three symbol compositions be relabeled by
nonzero multipliers `a_mu`, `a_beta`, and `a_zeta`. The first transition is
preserved when `a_beta/a_mu` is a sign, and the second is preserved when
`a_zeta/a_beta` is a sign. Thus the global action can be written as

```text
(a_mu,a_beta,a_zeta)
  = (a, epsilon_1 a, epsilon_1 epsilon_2 a),
```

where `a` is in `F_5^*` and both signs are independent. This group has order
16. Entropy and Hamming weight are invariant under every element, so `Psi` is
invariant as well.

The current canonicalization uses the common multiplier to place a largest
nonzero coordinate of `mu` at symbol one. The stabilizer of that choice has
order four: it changes
the signs of `beta` and `zeta` independently while leaving `mu` fixed. Every
remaining orbit therefore meets the chamber

```text
beta_1 >= beta_4,    zeta_1 >= zeta_4.
```

Boundary points may have more than one representative, which is harmless.
This is an exact reduction of the search domain. It is not an assumption about
the shape of an optimizer.

## Orbit closure of stationary cuts

A fixed stationary dual cut has the form

```text
L(alpha,beta)
  = c + <u,alpha> + <v,beta>
      - log(alpha^T K beta).
```

Suppose `f` and `g` relabel the increment and state symbols. Pulling the cut
back through the symmetry gives another globally valid cut with

```text
u'_i    = u_{f(i)},
v'_j    = v_{g(j)},
K'_{ij} = K_{f(i),g(j)}.
```

No new nonlinear optimization is needed. An implementation may store one base
descriptor and generate its orbit when evaluating the envelope. The useful
closure has eight images per cut: four scalings and the two edge orientations.
The five state translations enlarge this to 40 images, but the present
relaxation gains essentially nothing from them.

The following table reports the floating audit on broad frontier box 166 at
`eta=0.002` and `delta=0.22`. The last column adds the two residual chamber
inequalities. All entries are diagnostics from a floating LP relaxation; none
is a distance certificate.

| cut closure | cuts per stage | unrestricted upper | chamber upper |
|---|---:|---:|---:|
| original | 72 | `0.03286585` | `0.02304338` |
| four scalings | 288 | `0.03261935` | `0.02285180` |
| scaling and reversal | 576 | `0.01467547` | `0.01465772` |
| scaling and translation | 1,440 | `0.03276207` | `0.02265613` |
| full order-40 orbit | 2,880 | `0.01467511` | `0.01465684` |

Edge reversal supplies nearly all of the improvement. Once the cut bank is
closed under reversal and scaling, its envelope is invariant to numerical
precision and the chamber changes the root upper by only `1.8e-5`. The
chamber remains useful for removing duplicate boxes and cached subproblems.

At the LP maximizer, evaluating the fixed envelope on the actual composition
still gives a negative value. The positive LP upper is therefore a
pseudo-moment artifact. Further label symmetries cannot remove that gap by
themselves.

## Why sign averaging is not a valid reduction

Let `sigma(x)=-x`, and write

```text
Sym(q) := (q + sigma_# q)/2.
```

Invariance under `sigma` does not imply that `Psi(Sym(q))` is at least
`Psi(q)`. Such a conclusion would require a concavity property that the
stationary optimization does not have.

The failure is exact, not merely numerical. A feasible stationary law over
`F_5` is

```text
tau = (1/28) *
  [[6,0,0,5,0],
   [5,6,0,0,0],
   [0,0,0,0,0],
   [0,5,0,1,0],
   [0,0,0,0,0]].
```

Its increment and state marginals are

```text
alpha = (13,0,0,10,5)/28,
beta  = (11,11,0,6,0)/28.
```

For `Asym(q):=H(Sym(q))-H(q)`, this law satisfies

```text
Asym(tau)-Asym(alpha)-Asym(beta)
  = -(5/14) log(2).
```

Thus even a single stationary transition can become worse under global sign
averaging. A two-stage construction with a uniform-state second transition
gives a full-exponent change of `-(5/28)log(2)`.

Direct optimization finds similar counterexamples in the low-output domain.
They lie deep in the negative region: one search found a sign-averaging loss
of about `0.229` at exponent about `-0.638`. The counterexamples defeat an
unconditional averaging lemma, but they do not resemble the extremal witness
near exponent zero.

## The symmetry-defect identity

The lifted exponent uses stationary edge laws `tau_1` and `tau_2`:

```text
Psi
  = H(tau_1)+H(tau_2)
      -(2/3)H(mu)-2H(beta)-H(zeta).
```

Simultaneously average every law with its sign reversal. The exponent change
is exactly

```text
Delta
  = Asym(tau_1)+Asym(tau_2)
      -(2/3)Asym(mu)-2Asym(beta)-Asym(zeta).
```

Equivalently, define the two stage defects

```text
g_1 = Asym(tau_1)-Asym(mu)-Asym(beta),
g_2 = Asym(tau_2)-Asym(beta)-Asym(zeta).
```

Then

```text
Delta = g_1 + g_2 + Asym(mu)/3.
```

This identity isolates the obstruction. If averaging is harmful, at least one
stationary edge contains more sign bias than its two marginals can explain.
That hidden orientation must be maintained through the addition constraint.
The missing theorem should charge this maintenance to mutual information.

A useful target is the conditional statement

```text
Psi >= 0  ==>  Delta >= 0.
```

It would be enough to optimize on the sign-symmetric face. A quantitative
version would be stronger:

```text
Delta <= -gamma  ==>  Psi <= -m(gamma)
```

for an explicit positive function `m`. Floating constrained searches at
`delta=0.22` support this separation. The largest exponent found under
`Delta<=0` was `-0.00139255`; under `Delta<=-10^-5` it was `-0.00173111`;
under `Delta<=-10^-3` it was at most `-0.00477027` among the recorded feasible
witnesses. These are multistart lower bounds on constrained maxima, not global
upper bounds. One threshold was nonmonotone before taking the envelope over
all stricter witnesses, which also shows that the nonlinear search is not a
certificate method.

## Even and odd coordinates

For a law `q` on `F_5`, use the sign-even coordinates

```text
q_0,    q_1+q_4,    q_2+q_3
```

and the sign-odd coordinates

```text
q_1-q_4,    q_2-q_3.
```

A sign change fixes the even coordinates and negates the odd pair. For the
three compositions, the residual sign group acts independently on the odd
pairs of `mu`, `beta`, and `zeta` before message canonicalization, and on the
last two pairs afterwards.

At a fully sign-symmetric stationary point, this action block-diagonalizes the
Hessian. The even variables form one block. Each composition contributes a
separate two-dimensional odd block, and all cross terms between different
sign characters vanish. This gives a small local test for whether an extremum
can leave the symmetric face: check the three odd curvature blocks rather than
the full Hessian.

The common multiplication action gives a second decomposition. Order the
nonzero symbols as `(1,2,4,3)` and take the four-point Fourier transform. The
trivial and quadratic modes are sign-even; the conjugate odd modes measure
orientation. This is the natural basis for a Fourier or positive-semidefinite
bound.

## A quotient that retains orientation

Discarding the sign inside each orbit is too coarse. It loses exactly the
conditional orientation responsible for cancellation. A more faithful
parametrization separates orbit mass from orientation.

The sign orbits of symbols are

```text
{0},    {+1,-1},    {+2,-2}.
```

A stationary edge is indexed by `(D,S,S+D)`. Simultaneous sign reversal fixes
the all-zero edge and pairs every other edge with one opposite orientation.
The 25 valid edges therefore form 13 coarse orbits. Give a nonzero coarse cell
mass `gamma_c` and a signed orientation `rho_c`, with

```text
|rho_c| <= gamma_c.
```

The two oriented probabilities are `(gamma_c+rho_c)/2` and
`(gamma_c-rho_c)/2`. Its entropy splits exactly as

```text
H(tau)
  = H(gamma)
      + sum_{c != 0} gamma_c
          h_2((1+rho_c/gamma_c)/2).
```

Composition entropies have the same decomposition over their two nonzero sign
orbits. All even marginal constraints involve only `gamma`; all odd marginal
constraints are linear in `rho`. This quotient does not immediately reduce
the raw count—there are 13 masses and 12 orientations—but it separates the
convex magnitude problem from a collection of binary orientation entropies.
It is a better starting point for eliminating orientation variables than the
old sign-orbit relaxation, which deleted them.

## Symmetry-adapted moment relaxation

Pointwise averaging can fail, but averaging a relaxation is still valid. If
the domain, objective, and cut bank are invariant, any feasible
pseudo-distribution can be averaged over the group without changing its
objective. Its moment matrix then commutes with the group action and splits
into character blocks.

This observation applies directly to the present obstruction. The broad LP
uses independent product variables and McCormick inequalities, so it admits
pseudo-moments that do not come from one composition. A small
positive-semidefinite constraint can restore part of the missing rank-one
structure. In even/odd coordinates, the residual `C_2 x C_2` action separates
the trivial block from the `beta`-odd and `zeta`-odd blocks. Before message
canonicalization, the larger sign group gives one block for each composition's
odd character.

The relaxation must be built on an invariant root domain or on the union of a
box with its orbit. It must also use the eight-image cut closure. Under those
conditions, block diagonalization is exact and can reduce both matrix size and
the number of cross moments.

The implementation audit found that the existing box solver already uses the
stronger full `15`-by-`15` covariance constraint. It enforces the constraint
through eigenvector separation. Increasing the separation budget from 12 to
100 rounds changed the unrestricted edge-orbit upper only from `0.01467547`
to `0.01466986`. The chamber upper changed from `0.01465772` to `0.01465378`.
Thus incomplete PSD separation does not explain the remaining gap.

Symmetry blocks may still accelerate a future semidefinite solver. They cannot
close the present box unless that solver also adds constraints beyond the
current degree-two moment cone.

## A second analytic route: the zero set

There is an independent reduction that may lead to a shorter proof. The
support-projected stationary mismatch vanishes only when the support crossing
equals its independent value and the conditional nonzero state law is
uniform. For two accumulator stages, imposing both zero-payment conditions
reduces the outer problem to three scalar variables.

Let `h` denote binary entropy. For support fractions `(A,B,C)`, define

```text
J(a,b) := (1-b-a/5, b-4a/5, a/5, 4a/5),
I_0(a,b) := h(a)+h(b)-H(J(a,b)).
```

Maximizing the message entropy at fixed `A` makes its nonzero law uniform.
The reduced support exponent is therefore

```text
Psi_0(A,B,C)
  := (h(A)+A log(4))/3 - I_0(A,B)-I_0(B,C).
```

At `eta=0.002` and `delta=0.22`, the support exponent is strictly increasing
in the output support `C`. Indeed, on the zero set,

```text
d Psi_0/dC
  = log( C(1-C-B/5) / ((1-C)(C-4B/5)) ).
```

The ratio exceeds one because the numerator minus the denominator is
`B(4/5-C)`. Since `C<=0.22<4/5`, the maximum has `C=0.22`. This exact
monotonicity reduces the zero set from three variables `(A,B,C)` to a
two-dimensional triangle in `(A,B)`. Its domain is

```text
eta <= A <= 25 delta/16,
4A/5 <= B <= 5 delta/4.
```

Numerical maximization on the triangle gives `-0.0053834522` at
`A=0.002`, `B=0.0029876046`. An exact certificate covers the triangle with 22
rational leaves. It uses binary-entropy secants for the negative terms and
cross-entropy tangents for the positive table entropies. An independent
verifier checks the rational polygon cover and directed logarithm intervals.
It proves

```text
Psi_0(A,B,C) <= -1/200
```

throughout the structural zero set at `eta=1/500` and `C<=11/50`. The largest
replayed leaf upper is approximately `-0.00507563718`.

The second part should use the same even/odd decomposition. Magnitude modes
measure departure from uniformity on the two sign orbits, while the odd modes
measure orientation and cancellation bias. Unlike unconditional sign
averaging, this argument would charge every departure from the zero set rather
than assert that symmetrization improves it.

## Coercivity away from the zero set

The Fourier identity gives one exact coercivity bound. Fix one accumulator
stage. Let `A` be the nonzero increment mass and `x` its crossing mass. Let
`u` and `b` be the conditional nonzero increment and state laws. Define a
five-symbol vector `r` by

```text
r_0 := x,
r_a := (A-x)b_a  for a != 0.
```

Let `K` be the target law under the support-preserving conditional product.
For `g:=1-cos(2 pi/5)=(5-sqrt(5))/4`,

```text
D(beta || K) >= g^2 ||r-(A/5)1||_2^2.              [Fourier coercivity]
```

The squared defect has the equivalent form

```text
||r-(A/5)1||_2^2
  = (5/4)(x-A/5)^2 + (A-x)^2 ||b-u_4||_2^2,
```

where `u_4` is uniform on `F_5^*`.

To prove the bound, fix a nontrivial additive character `k`. The Fourier
mismatch identity gives

```text
beta^(k)-K^(k) = (1-u^(k)) r^(k).
```

The value `u^(k)` lies in the convex hull of the four nontrivial fifth roots
of unity. That convex hull has distance `g` from one. Parseval therefore gives

```text
||beta-K||_2^2 >= g^2 ||r-(A/5)1||_2^2.
```

The vector `beta-K` has coordinate sum zero. Hence
`||beta-K||_1^2>=2||beta-K||_2^2`. Pinsker's inequality completes the proof.

This theorem characterizes the correct zero set but is too weak globally.
The following values are floating multistart diagnostics at `delta=0.22`.
Positive values are feasible witnesses for the stated relaxation. Negative
values are not global certificates.

| retained payment | unrestricted value found |
|---|---:|
| Fourier coercivity | `+0.283595` |
| exact target `l_2` payment | `+0.226076` |
| exact target KL projection | `+0.180729` |
| two-level cancellation and conditional payment | `+0.00174497` |

The unweighted Fourier bound does remain negative in a very small tube. The
largest values found under squared defect at most `10^-8`, `10^-7`, and
`10^-6` are `-0.00464180`, `-0.00309334`, and `+0.000316918`. The entropy
gradient near `A=0.002` explains this rapid loss: it scales inversely with the
small support, while the Euclidean payment does not.

The two-level payment supplies the missing normalization. Under squared defect
at most `10^-6`, its largest observed value is `-0.00130966`. Its binary
cancellation KL charges deviations on the scale of the active support. The
conditional nonzero KL charges orientation. This is the useful local theorem
to certify. Globally, the remaining positive two-level witness lies on a
sign-symmetric two-orbit face. The cell-target and stationary payments are
both stronger there.

### Exact tube coordinates

For two accumulator stages, let `r^(1)` and `r^(2)` denote the vectors above.
The normalized tube is

```text
R := ||r^(1)-(A/5)1||_2^2
   + ||r^(2)-(B/5)1||_2^2 <= 10^-6.
```

This description gives exact rational pruning rules. For example, each stage
satisfies

```text
|x-A/5| <= sqrt(4 R/5),
|(A-x)(b_a-1/4)| <= sqrt(R).
```

The reduced-composition verifier now applies these rules before it evaluates
a box. It also computes a rational lower bound on `R` and discards a box when
that lower bound exceeds the tube radius.

The message composition can be removed from the first-stage box. Fix one
valid two-level dual cut. For fixed support data and middle law, write the cut
as

```text
L(u) = c-sum_(d!=0) u_d q_d,
```

where `u` is the conditional nonzero message law. The entropy conjugate gives

```text
max_u { A H(u)/3-L(u) }
  = -c + A/3 log sum_(d!=0) exp(3q_d/A).
```

The verifier evaluates the right-hand side with directed intervals. This
step is sound because the verifier fixes one cut before it maximizes over
`u`. It does not exchange the cut minimum with the composition maximum.

The resulting checker is exact but has not closed the tube. With support-first
subdivision, its exact upper after `100` splits is `0.103398`. After `1,000`
splits, the upper is `0.0238299`. An `r`-weighted split rule performs worse:
its upper is `0.222967` after `1,000` splits. The independent conditional-law
boxes still make the second-stage KL payment vanish on the worst leaves.

These results rule out further tuning of the same box heuristic as the main
route. A successful tube certificate should either cover coupled `r` cells or
prove a support-normalized coercivity inequality for the two-level payment.

### Support-normalized coercivity

The increment support alone is not the correct normalization. Put

```text
s:=B/A,  t:=x/A,
```

and fix a positive nonuniform law `v` on `F_5^*`. Take `t=1/5` and let the
nonzero increment law be uniform. Then

```text
L/A = s D(v || (1-1/s)v+u_4/s),
R/A^2 = (16/25)||v-u_4||_2^2.
```

Consequently, `A L/R` tends to zero as `s` tends to infinity. This family
disproves any uniform inequality of the form `L>=kappa R/A` with positive
`kappa`.

Numerical minimization supports the corrected normalization

```text
L >= (4/5) R/max(A,B).                              [KL conjecture]
```

The smallest value of `max(A,B)L/R` found over `s` in `[0.5001,220]` is
`0.8809162730`. It occurs at `s=1`. An audit of every inequivalent support
stratum for the increment law finds the same minimizer on an opposite pair.
The other two-symbol orbit and the singleton stratum remain above `1.8` near
`s=1`. These computations do not prove the displayed inequality.

A weaker conjecture has a simpler proof target. Let `u` and `v` be the
conditional nonzero increment and state laws. Define

```text
c := sum_(d!=0) u_d v_(-d),
e_0 := x-(A-x)c,
e^+ := (A-x)v-xu-(A-2x)rho^+,
```

where `rho^+` is the nonzero part of the convolution of `u` and `v`,
conditioned on noncancellation. Pinsker's inequality gives the proved bound

```text
L >= [2e_0^2+||e^+||_2^2]/max(A,B).                 [residual bound]
```

It remains to prove the algebraic inequality

```text
2e_0^2+||e^+||_2^2 >= (2/5)R.                      [residual conjecture]
```

The residual conjecture contains no entropy and no support ratio. Numerical
minimization over all inequivalent support strata gives `0.4105458279` for
the ratio of its left-hand side to `R`. The candidate minimizer is

```text
u=v=(delta_a+delta_(-a))/2,
t=(sqrt(129)-3)/20.
```

On this family, the defect and residual are

```text
R/A^2 = (3/2)t^2-t+3/10,
(2e_0^2+||e^+||_2^2)/A^2 = (17/2)t^2-7t+3/2.
```

The ratio is strictly greater than `2/5` for every `t`: after clearing
denominators, the difference is `395t^2-330t+69`, whose discriminant is
`-120`. A global proof must still exclude the remaining increment and state
laws.

There is also a proved nonlinear payment. Since `c` is a convex combination
of the coordinates of `v`, it lies in `[min_a v_a,max_a v_a]`. Let `c_*` be
the projection of `x/(A-x)` onto this interval. Data processing gives

```text
L >= (A-x)d(x/(A-x) || c_*).                        [cancellation floor]
```

The cancellation floor and the conjectured residual bound are complementary.
The former is strong when the crossing ratio leaves the range permitted by
`v`; the latter controls orientation inside that range. The residual
conjecture is no longer needed for the normalized-tube theorem below.

This combination is strong enough numerically. On the two-stage tube
`R_1+R_2<=10^-6`, the corrected constrained optimizer finds the following
upper bounds at `delta=0.22` and `eta=0.002`:

| retained payment | upper found |
|---|---:|
| `(4/5)R/max(A,B)` alone | `+0.00016146` |
| `max(cancellation floor,(4/5)R/max(A,B))` | `-0.00050769` |
| `max(cancellation floor,(2/5)R/max(A,B))` | `-0.00050642` |

The last two rows use the optimizer's nonlinear feasibility filter. Their
reported defects differ from `10^-6` by less than `2*10^-13`.

### Exact scalar tube certificate

The cancellation floor alone admits a smaller exact relaxation. Put

```text
q:=||v-u_4||_2^2,  z:=sqrt(3q/4),  w:=(A-x)z.
```

For every coordinate `a`, Cauchy--Schwarz gives

```text
|v_a-1/4| <= z.
```

Hence the cancellation probability lies in
`[1/4-z,1/4+z]`. The stage defect is at least

```text
(5x-A)^2/20 + 4w^2/3.
```

The mass-scaled variable `w` is essential: it removes the singular
conditional radius when `A` is small. It also makes zero scalar payment
equivalent to `|5x-A|<=4w`.

Let `J(A,B,x)` be the four-cell support law

```text
(1-B-x, B-A+x, x, A-x).
```

After maximizing the output support at `C=delta`, the scalar upper is

```text
H(J(A,B,x))+H(J(B,delta,y))
  -(2/3)h(A)-2h(B)-h(delta)+(A/3)log 4
  -Pay(A,x,w_1)-Pay(B,y,w_2),
```

where `Pay` is the binary cancellation floor with reference interval
`[1/4-w/(A-x),1/4+w/(A-x)]`. The maximization at `C=delta` is exact: the
derivative of the second support term has the sign of `B(1-C)-y`, which is
positive because `y<=B/2` and `C<=0.22`.

At small support, the message entropy and first cancellation payment cannot
be optimized independently. If `m=max_a u_a` and `q=x/(A-x)`, then the
cancellation probability is at most `m`. Pinsker's inequality gives

```text
log 4-H(u) >= 2(m-1/4)^2,
d(q || c) >= 2(q-m)_+^2.
```

Minimizing the weighted sum over `m` proves the coupled payment

```text
T(A,x) = A(5x-A)_+^2/[8(A-x)(4A-3x)].
```

The exact checker also uses a stronger directed grid envelope for the same
one-dimensional minimization, based on
`d(m||1/4)/3+(1-t)d(t/(1-t)||m)` with `t=x/A`. This step prevents the
relaxation from earning uniform message entropy while demanding deterministic
cancellation.

An exact rational box cover proves that this upper is negative on

```text
(5x-A)^2/20 + 4w_1^2/3
  +(5y-B)^2/20 + 4w_2^2/3 <= 10^-6
```

for `A>=1/100000` and `delta=11/50`. The strengthened certificate contains
22,471 splits and 22,414 terminal boxes. Its exact worst upper is negative;
numerically it is `-1.91896927570e-9`. The independent replay reconstructs
every rational box, checks the entropy secants and cross-entropy tangents with
directed logarithm intervals, and verifies both cancellation bounds. Thus the
normalized tube is closed without the residual conjecture, and its lower
endpoint meets the exact sparse-spacing theorem.

Global scalar coercivity still fails. Floating searches find positive uppers
`+0.2611` for the proved residual--cancellation bound, `+0.2514` for the
Fourier--cancellation bound, and `+0.1653` even for the conjectural
`(4/5)R/max(A,B)` bound combined with cancellation. These witnesses lie far
outside the tube. They show that the complement must retain nonzero-symbol
orientation.

## Decision

The normalized tube and the sparse-to-bulk junction are complete. A global
certificate outside that tube remains open. Two routes now have different
roles.

1. **Primary route: certify the projected KL relaxation directly.** A
   negative global projected bound implies the exact `0.22` distance theorem
   without partitioning by residual defect. The current floating checkpoint
   has 11,414 splits, 10,948 leaves, and upper `+0.00772705`. A fifth enrichment
   epoch grew the bank to 380 cuts per stage after its worst point became
   slightly positive at 10,000 splits. The saved bank is again negative at the
   current worst composition (`-0.00641496`); the positive
   value is therefore a lifted relaxation gap. The production campaign uses
   ordinary rank-gap subdivision. Strong-branch probes remain diagnostic but
   did not improve mature global progress per unit time. Split thresholds use
   the worst LP coordinate clipped to the middle half of its interval; a
   controlled 500-split comparison improved bound reduction by about 9% over
   midpoint thresholds.
   The leading local strengthening now preserves cut selection with an affine
   partition of unity. A seven-chamber refinement of the current worst cell
   has five chambers with negative multistart continuous upper bounds. Two
   chambers remain open. On one of them, interior column generation retains a
   sampled margin `-0.000395992` after 20 rounds while reducing the largest
   located violation to `+0.000263612`. This replaces broad coordinate
   subdivision as the next experiment, but it is still floating evidence.
2. **Fallback route: cover the tube complement by signed residual slabs.** If
   the ten squared residual coordinates sum to more than `10^-6`, one signed
   coordinate has magnitude greater than `sqrt(10^-7)`. Necessary lifted
   moment inequalities close some orientations but leave others essentially
   unchanged. Retaining the crossings removes that particular elimination,
   yet the current broad-box log-partition bounds still collapse to zero.

The slab experiments explain why a hybrid certificate may still help later.
At the current worst projected composition, the minimum residual defect over
both crossings is about `0.0117`; some residual coordinate must therefore
have magnitude at least `0.034`. Several signed nonzero-coordinate slabs at
threshold `0.03` are already negative. The surviving negative orientations
remain loose because the lifted inequality replaces the crossing by only
`x<=A/2`. This is useful localization evidence, but it is not yet a global
cover.

Full state-translation closure remains low priority: it multiplies the cut
bank by five without a measurable improvement over scaling and reversal.
Exact lifted leaf replay is now implemented. On a radius-`0.01` box around the
current worst composition, an independent 138-variable verifier reconstructs
1,105 semantic constraints and proves upper approximately `-0.0185485` from a
rational LP dual. The partition topology is also exact through split 11,414.
The verifier reconstructs the frozen 10,100-split frontier from 250,553 nodes,
then replays 1,314 directly recorded splits to obtain the 10,948 current cells.
The next proof-engineering gate is analytic coverage: subdivision must continue
until every active cell admits a negative rational lifted proof. The tube
certificate remains an independent check on the most delicate sparse boundary.

## Reproducibility

The exact cut transformations and their tests are in
`scripts/exact_stationary_dual.py` and
`scripts/test_exact_stationary_dual.py`. The group, chamber, cut-family, and
averaging audits are in `scripts/analyze_stationary_symmetry.py`, with output
in `results/f5-stationary-symmetry-audit.json`. The constrained
symmetry-defect search is in `scripts/analyze_symmetry_split.py`, with output
in `results/f5-symmetry-split-audit.json`. The converged PSD audit is in
`results/f5-stationary-symmetry-psd-audit.json`. The zero-set reduction and
floating cover are in `scripts/analyze_zero_payment_surface.py` and
`results/f5-zero-payment-surface.json`. The exact certificate is generated by
`scripts/generate_zero_payment_certificate.py` and replayed by
`scripts/verify_zero_payment_certificate.py`. The stored artifact is
`results/f5-zero-payment-certificate.json`. The coercivity diagnostics are in
`scripts/analyze_fourier_coercivity.py` and
`results/f5-fourier-coercivity.json`. The exact tube pruning and message
conjugate are in `scripts/verify_reduced_composition_bulk.py`; their regression
tests are in `scripts/test_reduced_composition_verifier.py`. The normalized
KL scan is in `scripts/analyze_normalized_two_level.py`. The support-stratum
audit is in `scripts/analyze_normalized_boundaries.py`, and the residual audit
is in `scripts/analyze_residual_coercivity.py`. Their stored reports are
`results/f5-normalized-two-level.json`,
`results/f5-normalized-boundaries.json`, and
`results/f5-residual-coercivity.json`. The two-stage comparison is stored in
`results/f5-normalized-tube-diagnostic.json`. The scalar tube search is in
`scripts/verify_scalar_tube_bound.py`. Its split tree is
`results/f5-scalar-tube-cover-0.22-eta-1e-5.json` and is independently
replayed by `scripts/verify_scalar_tube_certificate.py`. The lifted-slab
frontier diagnostic is `scripts/diagnose_projected_residual_slabs.py`. The
crossing-retaining slab diagnostic is
`scripts/diagnose_joint_residual_slab_cover.py`. The full-transition entropy
diagnostic is `scripts/diagnose_transition_residual_slab.py`, and the
all-coordinate branch audit is
`scripts/diagnose_projected_split_candidates.py`. The active projected
campaign driver is `scripts/run_projected_kl_campaign.py`. The active projected
checkpoint and cut bank are
`results/f5-projected-kl-box-cover-0.22.checkpoint.json` and
`results/f5-projected-kl-dual-bank-0.22.json`.
The exact stationary fallback generator and independent tree verifier are
`scripts/generate_stationary_cover_certificate.py` and
`scripts/verify_stationary_cover_certificate.py`; its bounded work checkpoint
is `results/f5-stationary-cover-0.22.checkpoint.json`.
The exact lifted leaf generator and verifier are
`scripts/generate_projected_lifted_box_certificate.py` and
`scripts/verify_projected_lifted_box_certificate.py`. Their first replayed
artifact is `results/f5-projected-lifted-worst-local-0.22.json`.

Every numerical value in this note is evidence unless an exact argument is
given in the surrounding paragraph.
