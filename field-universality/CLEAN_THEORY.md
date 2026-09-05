# A sparse-core theory for addition-only BAA codes

## 1. Current thesis

The experiments no longer suggest equality of coupled binary and prime-field
distances. They suggest a cleaner theorem for one explicit BAA ensemble.

Sparse messages create characteristic-specific traps with polynomially small
probability. Bulk messages obey a vector-composition first moment. A proof that
this ensemble inherits a certified binary distance threshold should treat
these regimes separately.

The central obstruction is not the existence of odd-characteristic torsion.
Most successful samples contain such torsion above the binary minimum. The
desired theorem must instead exclude torsion below a certified binary linear
threshold.

This note develops eight ingredients:

1. a vector-composition enumerator for the complete addition-only ensemble;
2. an exact embedding count for the dominant repetition-three trap;
3. an algebraic classification of every sampled odd-field minimum witness;
4. an exact sign-symmetrization theorem for the ternary bulk exponent;
5. a rigorous sign-orbit relaxation for the quinary bulk exponent;
6. a uniform random-spacing bound for sparse messages;
7. a deterministic weight floor for two accumulators; and
8. independently checkable interval and rational certificates.

For ternary repetition-three codes, the sparse and bulk bounds prove relative
distance `0.196`. A sign-orbit certificate proves quinary relative distance
`0.05` and septic relative distance `0.018`. The spacing bound and
deterministic floor also prove a positive relative distance over every fixed
prime. Stronger bounds beyond these rungs still require full-composition
analysis.

## 2. Ensemble and probability space

Fix a prime `p`, a repetition factor `r`, and an integer `K>=1`. Set `N=rK`.
All words in this note lie in `F_p^N` unless another space is specified.

The repetition map `B_r:F_p^K -> F_p^N` repeats each message coordinate `r`
times. The accumulator `A:F_p^N -> F_p^N` is

```text
(Au)_j := sum_{i=0}^j u_i.
```

Let `P_1` and `P_2` be independent uniform coordinate permutations. The
addition-only encoder is

```text
E_{P_1,P_2} := A P_2 A P_1 B_r.
```

All probabilities and expectations below are over `(P_1,P_2)`. A coupled
binary comparison uses the same realized permutations in characteristic two.

Let `delta_bin(r)` be a proved binary relative-distance threshold for this
ensemble. The target theorem is

```text
for every fixed odd prime p and every delta<delta_bin(r),
Pr[d_min(im E_{P_1,P_2}) <= delta N] -> 0 as K -> infinity.
```

The image and minimum distance in this statement are taken over `F_p`. The
statement transfers a certified threshold. It does not assert samplewise
distance monotonicity or equality of limiting distances.

The repetition map is part of the theorem, not a placeholder for an unknown
binary encoder. A later blockwise extension must control the base code through
its finite symbol-composition enumerator. Binary minimum distance alone cannot
provide that control.

For a word `u`, define its composition by

```text
comp(u) := (n_a(u))_{a in F_p},
n_a(u) := |{j : u_j=a}|.
```

For a nonnegative vector `q=(q_a)_{a in F_p}` with total `N`, define

```text
Mult(q) := N! / product_a q_a!.
```

Thus `Mult(q)` is the size of the type class with composition `q`.

## 3. Exact vector-composition enumerator

Random multipliers make a scalar weight enumerator sufficient in weighted
nonbinary RAA. The addition-only ensemble has no such value randomization.
The correct exact object records every field-symbol multiplicity.

### 3.1 One accumulator

For compositions `q` and `h` of length `N`, define

```text
T_N(q,h)
  := |{u in F_p^N : comp(u)=q and comp(Au)=h}|.
```

The following transfer matrix gives all values of `T_N` simultaneously. Let
`X=(X_a)_{a in F_p}` and `Y=(Y_a)_{a in F_p}` be commuting variables. Define
the `p x p` matrix `M(X,Y)` by

```text
M(X,Y)_{s,t} := X_{t-s} Y_t.
```

All subscripts and differences in this definition lie in `F_p`.

**Lemma 1 (accumulator transfer matrix).** For every pair `(q,h)`,

```text
T_N(q,h)
  = [X^q Y^h] e_0^T M(X,Y)^N 1.
```

Here `e_0` selects state zero, and `1` sums over the final state.

**Proof.** Set `y_{-1}=0`. Each accumulator path has states
`y_0,...,y_{N-1}` and increments `u_j=y_j-y_{j-1}`. A transition from `s` to
`t` contributes `X_{t-s}Y_t`. Hence its monomial records the increment and
state compositions. Matrix multiplication concatenates transitions. The
coefficient extraction selects the required compositions. ∎

**Lemma 2 (permutation of a type).** Fix `u` with `comp(u)=q`. If `P` is a
uniform coordinate permutation, then `Pu` is uniform over the type class of
`q`.

**Proof.** Every word in the type class has exactly `product_a q_a!`
permutation preimages. This number does not depend on the output word. ∎

Lemmas 1 and 2 imply the exact transition probability

```text
Pr[comp(A P u)=h | comp(u)=q] = T_N(q,h) / Mult(q).
```

### 3.2 Two accumulators

For a composition `z`, let `A_z(P_1,P_2)` be the number of nonzero messages
whose encoded word has composition `z`.

**Theorem 3 (exact first moment).** Let `m` range over compositions of length
`K`, except the all-zero composition. Then

```text
E[A_z]
  = sum_m Mult(m)
      sum_h
        T_N(r m,h) T_N(h,z)
        / (Mult(r m) Mult(h)).
```

The product `r m` is coordinatewise.

**Proof.** There are `Mult(m)` messages of composition `m`. Repetition changes
their composition to `r m`. Lemma 2 makes the first permuted word uniform over
that type class. Lemma 1 gives the first accumulator transition to `h`.
Conditioned on `h`, the second permutation is fresh and uniform over its type
class. A second use of Lemma 1 gives the transition from `h` to `z`. Summing
over `h` and `m` proves the formula. ∎

For `d>=1`, let `Z_p(d)` be the number of nonzero encoded words of Hamming
weight at most `d`. Theorem 3 gives the exact union bound

```text
Pr[d_p <= d]
  <= E[Z_p(d)]
  = sum_{z : N-z_0 <= d} E[A_z].
```

No torsion or rank comparison appears in this formula. A direct prime-field
distance proof can therefore avoid a uniform comparison of exponentially many
coupled deletion sets.

### 3.3 Bulk exponent

Theorem 3 has a finite-dimensional exponential form. Let `Delta_p` be the
probability simplex on `F_p`. For `alpha,beta in Delta_p`, define
`Gamma(alpha,beta)` as the set of arrays
`tau=(tau_{s,t})_{s,t in F_p}` satisfying

```text
sum_t tau_{s,t} = beta_s,
sum_s tau_{s,t} = beta_t,
sum_s tau_{s,s+a} = alpha_a.
```

The first two constraints make `beta` the source and destination marginal.
The last constraint makes `alpha` the increment marginal. Define

```text
T_p(alpha,beta)
  := max_{tau in Gamma(alpha,beta)} H(tau)-H(beta),
```

and set the value to minus infinity when `Gamma(alpha,beta)` is empty. All
entropies in this subsection use natural logarithms.

The quantity `T_p(alpha,beta)` is the conditional-entropy rate of an
accumulator path. For a finite path, the source and destination marginals
differ at the two endpoints by at most `1/N`. This discrepancy disappears in
the exponent.

For normalized message, intermediate, and output compositions
`mu,beta,zeta in Delta_p`, define

```text
Psi_{p,r}(mu,beta,zeta)
  := H(mu)/r
     + T_p(mu,beta)-H(mu)
     + T_p(beta,zeta)-H(beta).
```

The four terms have direct meanings. The first counts messages. The next two
terms give the probability exponent of the first accumulator transition. The
last two terms give the second transition.

Fix `eta>0`. Restrict attention to messages with at least `eta K` nonzero
coordinates. The method of Markov types applied to Theorem 3 gives

```text
limsup_N (1/N) log E[Z_p^bulk(delta N)]
  <= max Psi_{p,r}(mu,beta,zeta),
```

where the maximum satisfies

```text
1-mu_0 >= eta,
1-zeta_0 <= delta.
```

To justify the bound, fix a transition-count array. The number of compatible
paths is at most a polynomial factor times

```text
exp(N[H(tau)-H(row(tau))]).
```

There are polynomially many composition and transition types because `p` is
fixed. Maximizing their exponents gives the displayed bound. Endpoint
imbalances change each normalized type by `O(1/N)` and do not change the
limit superior.

Therefore, a negative maximum proves an exponential bulk-distance bound. The
optimization has `p^2+O(p)` variables and linear constraints. It is the first
place to search for a linear-support counterexample.

The script `scripts/optimize_bulk_exponent.py` performs a constrained
multistart search. Its output is numerical evidence, not a certificate of the
global maximum. The following sign checks use repetition factor three.

| field | bulk cutoff | output fraction | exponent found |
|:---|---:|---:|---:|
| `F_2` | 0.01 | 0.132 | -0.000150 |
| `F_2` | 0.01 | 0.133 |  0.000381 |
| `F_3` | 0.05 | 0.195 | -0.001034 |
| `F_3` | 0.05 | 0.197 |  0.000260 |
| `F_5` | 0.05 | 0.260 | -0.000353 |
| `F_5` | 0.05 | 0.261 |  0.000227 |

The binary sign change reproduces the known numerical RAA threshold near
`0.132`. This agreement checks the formulation and implementation. The odd
sign changes occur substantially later. They support strict odd-field
improvement rather than equality with the binary growth rate.

At output fraction `0.13` and cutoff `eta=0.01`, the exponents found are
`-0.00117`, `-0.00608`, and `-0.00720` for `p=2,3,5`. For `p=3`, reducing the
cutoff to `0.001` raises the value to `-0.000998`. The maximizing sequence
approaches zero message support, as expected from the polynomial sparse tail.

For `p=5`, the low-output optimizer concentrates the message on one opposite
pair of nonzero values. It does not use a uniform nonzero composition. This
behavior confirms that the unweighted ensemble needs the vector enumerator.

The full-composition problem still has a field-uniform family of affine upper
bounds. Fix vectors `u,v,w in R^p` and define

```text
Z(u,v,w)
  := sum_{s,t in F_p} exp(-u_s-v_t-w_{t-s}).
```

For every `tau in Gamma(alpha,beta)`,

```text
H(tau)
  <= log Z(u,v,w) + <u+v,beta> + <w,alpha>.
```

Indeed, define

```text
q_{s,t} := exp(-u_s-v_t-w_{t-s})/Z(u,v,w).
```

The inequality is `D(tau || q)>=0`. The three linear terms follow from the
row, column, and increment marginals of `tau`.

This dual bound converts a box certificate into linear programming. On an
interval `[a,b]`, the secant through `x log x` at `a` and `b` is an upper bound
because `x log x` is convex. Apply these secants to the three negative entropy
terms in `Psi_{p,r}`. Apply one or more entropy-dual bounds to each transition
entropy. The remaining maximization is a linear program over the original
composition and transition constraints.

Consequently, a finite certificate for fixed `p` can contain:

1. a box cover of the three composition simplices;
2. dual potentials for both transition entropies on every box; and
3. a feasible dual witness for each resulting linear program.

The checker needs only interval evaluation of logarithms and exact validation
of linear inequalities. Numerical optimization may design the cover, but it
does not belong to the trusted proof.

The ternary optimization admits a useful information-theoretic form. For
compositions `alpha` and `beta` on `F_3`, define

```text
I_stat(alpha,beta)
  := min I(D;S),
```

where the minimum ranges over pairs `(D,S)` such that

```text
law(D)=alpha,
law(S)=law(S+D)=beta.
```

The identity `H(S+D|S)=H(D|S)` gives

```text
T_3(alpha,beta)-H(alpha) = -I_stat(alpha,beta).
```

Consequently,

```text
Psi_{3,r}(mu,beta,zeta)
  = H(mu)/r-I_stat(mu,beta)-I_stat(beta,zeta).
```

The exponent therefore measures the information required to keep two
nonuniform state distributions stationary under their prescribed increments.

### 3.3.1 A field-uniform support-indicator bound

The zero/nonzero indicators retain enough information to control the
small-message boundary. Let a stationary accumulator transition have
nonzero-increment fraction `a` and nonzero-state fraction `b`. Write `x` for
the probability of a transition from zero to a nonzero state. Stationarity
makes the reverse crossing probability equal to `x`. Both crossings use a
nonzero increment, so `2x<=a`. The joint law of the nonzero-increment and
nonzero-source indicators is

```text
           source 0       source nonzero
inc 0        1-b-x             b-a+x
inc nz          x                 a-x.
```

For `a<=2b` and `b<1/2`, let `J(a,b)` be the mutual information of this table
at `x=a/2`.

**Support-indicator lemma.** Suppose `C<1/4`, and let `A`, `B`, and `C` be the
nonzero fractions of `mu`, `beta`, and `zeta`. Every feasible full-symbol
point satisfies

```text
Psi_{p,r}
 <= (h(A)+A log(p-1))/r - J(A,B)-J(B,C),
```

where `h` is binary entropy.

**Proof.** Applying the zero/nonzero map to the increment and source at each
stage cannot increase mutual information. With fixed Bernoulli marginals,
mutual information is convex in the crossing parameter and is minimized at
independence, where `x=a(1-b)`. Since `b<1/2`, this point lies at or to the
right of `a/2`. Feasibility restricts `x<=a/2`, so the minimum over feasible
crossings is attained at `x=a/2`.

The accumulator support inequalities give `A<=2B` and `B<=2C`. Thus
`B<1/2` and `C<1/2`, and the preceding argument applies at both stages.
Finally,

```text
H(mu) <= h(A)+A log(p-1).
```

Substitution in the information form of `Psi_{p,r}` proves the claim. ∎

This lemma explains the vanishing bulk margin. On the branch `B=Theta(A)`,
the first support mutual information has leading term
`(A/2)log(1/A)`, whereas `H(mu)/r` contributes
`(A/r)log(1/A)`. Consequently,

```text
Psi_{p,r}
 <= (1/r-1/2) A log(1/A) + O_p(A).
```

For every fixed field and repetition factor `r>2`, the leading coefficient is
negative. At repetition three it is `-1/6`. This is a field-uniform tangent
mechanism; the field size enters only the linear term `A log(p-1)/r`.

The tangent mechanism gives a uniform theorem, not only an asymptotic
expansion.

**Theorem (all-primes small-message slab).** Let `p` be an odd prime and set
`q:=p-1`. In the repetition-three addition-only BAA ensemble, the expected
number of codewords with

```text
0 < message support <= K/(60 q^2),
output support <= (1323/10000) N
```

tends to zero.

**Proof.** Set `delta_0:=1323/10000` and

```text
Lambda := log(1/(4 delta_0(1-delta_0))).
```

The conditional distribution of the state indicator given a nonzero
increment is uniform when the crossing parameter is `a/2`. Therefore,

```text
J(a,b)
 >= a D(Bernoulli(1/2) || Bernoulli(b))
  = (a/2) log(1/(4b(1-b))).
```

For `C<=delta_0`, the support-indicator lemma and `1-B<=1` give

```text
Psi_{p,3}
 <= (h(A)+A log q)/3
    -(A/2)log(1/(4B))-(B/2)Lambda.
```

Minimizing the last two terms over positive `B`, then using
`h(A)<=A log(e/A)`, yields

```text
Psi_{p,3}/A
 <= (1/6)log(q^2 A)-1/6-(1/2)log(Lambda/4).
```

The integer checks recorded below imply `Lambda>3/4`. If
`q^2 A<=1/60`, the right-hand side is strictly less than

```text
-(1/6) log(e * 60 * (3/16)^3) < 0.
```

For the last inequality, the exponential series gives `e>8/3`, and

```text
(8/3) * 60 * (3/16)^3 = 135/128 > 1.
```

This proves a negative bulk exponent on every closed slab bounded away from
`A=0`.

It remains to cover the sparse endpoint. Take `eta:=1/(100q^2)`. The spacing
bound of Section 4 has squared geometric base at most

```text
309760000000 / 435691903941 < 1.
```

Here we used `q>=2`, `e<11/4`, and
`exp(D(1/2 || delta_0))-1>19/40`. Hence the sparse expected count below
`eta K` tends to zero. The bulk exponent handles

```text
eta <= A <= 1/(60q^2).
```

The union bound proves the theorem. ∎

All transcendental constants in this proof reduce to integer comparisons.
The bound `e<11/4` and

```text
(11/4)^3 * (4 delta_0(1-delta_0))^4 < 1
```

imply `Lambda>3/4`. The script
`scripts/verify_uniform_support_slab.py` checks these comparisons and the two
rational margins. Its output is recorded in
`results/uniform-support-slab.json`.

At `r=3` and `C<=1323/10000`, the checker
`scripts/verify_support_indicator_bulk.py` gives the following exact ladder.

| field | lower endpoint | upper endpoint | boxes split | exact upper bound |
|---:|---:|---:|---:|---:|
| `F_5` | `1/2000` | `1/500` | 3,165 | `-1.7042e-7` |
| `F_7` | `1/5000` | `1/1250` | 7,112 | `-4.3382e-8` |
| `F_11` | `1/10000` | `1/4000` | 9,286 | `-1.6792e-8` |
| `F_13` | `1/15000` | `1/5000` | 18,981 | `-2.3037e-9` |
| `F_17` | `1/25000` | `1/10000` | 18,441 | `-7.6116e-11` |

Directed intervals enclose every logarithm, and rational arithmetic checks
every box bound. The exact numerators and denominators are recorded in
`results/support-indicator-binary-benchmark-ladder.json`.

The sparse-spacing estimate also holds below each listed lower endpoint. A
single rational specialization proves this fact. At
`delta=1323/10000`,

```text
sqrt(4 delta (1-delta)) < 40/59,
exp(D(1/2 || delta))-1 > 19/40.
```

Together with `e<11/4`, the remaining check is a rational inequality in `p`
and the lower endpoint; `scripts/sparse_spacing_bound.py` verifies it exactly.
Thus the sparse and support-indicator arguments meet at every row. These
results certify only the displayed small-message slabs. They do not certify
the remaining full-symbol domains above the upper endpoints.

### 3.3.2 Paying for internal symbol entropy

The support projection separates the remaining obstruction. For stage `i`,
let `I_i` be the full mutual information and let `I_i^supp` be the mutual
information after mapping both symbols to their zero/nonzero indicators. Set

```text
H_int(mu) := H(mu)-h(A),

Phi := h(A)/3-I_1^supp-I_2^supp,

G := (I_1-I_1^supp)+(I_2-I_2^supp)-H_int(mu)/3.
```

Then the full repetition-three exponent satisfies the exact identity

```text
Psi_{p,3} = Phi-G.
```

The quantity `G` measures whether the two accumulator transitions pay for the
message entropy hidden inside the nonzero symbols. A global inequality `G>=0`
would reduce the problem to the binary support exponent. It is false. Over
`F_5`, numerical minimization finds `G=-0.10395`, but the same type has full
exponent `-0.328996`. Thus violations found so far occur far below the
maximizing region.

There is nevertheless a field-uniform lower bound on each internal payment.
It removes the transition type from the problem.

Let `P` be the increment--source joint law of one stationary transition. Its
marginals are the increment composition `alpha` and the state composition
`beta`. Let

```text
U:=1[D!=0],  V:=1[S!=0],  T:=D+S,
```

and let `J` be the joint law of `(U,V)`. Define the support-preserving product
law

```text
Q(d,s)
  := J(u,v) Pr[D=d | U=u] Pr[S=s | V=v].
```

Terms in a zero-mass support cell use the continuous extension.

**Conditional-product lemma.** The internal information beyond the support
projection satisfies

```text
I(D;S)-I(U;V) = D(P || Q).                          [product identity]
```

**Proof.** Expand the relative entropy:

```text
D(P || Q)
  = -H(D,S)+H(U,V)+H(D | U)+H(S | V)
  = I(D;S)-I(U;V).
```

The second equality uses that `U` and `V` are deterministic functions of `D`
and `S`. ∎

The product identity yields an explicit composition-only bound. Write

```text
A:=Pr[U=1],  B:=Pr[V=1],  x:=J(1,0).
```

Stationarity gives `Pr[V=1,T=0]=x`, so

```text
J = (1-B-x, B-A+x, x, A-x)
```

in the order `(0,0),(0,1),(1,0),(1,1)`. In particular,

```text
max(0,A-B) <= x <= min(A/2,1-B).
```

Let `alpha^+` and `beta^+` be the conditional nonzero-symbol laws. Under
`Q`, two nonzero symbols cancel with probability

```text
kappa := sum_(d!=0) alpha^+_d beta^+_(-d).
```

Under `P`, the cancellation probability in the `(1,1)` cell is
`q:=x/(A-x)`. Hence data processing already gives the binary-divergence
payment

```text
(A-x) d(q || kappa).
```

There is a second payment after conditioning on a nonzero target. Let `rho^+`
be the law of `D+S` conditioned on `D!=0`, `S!=0`, and `D+S!=0` under the
product law `alpha^+ tensor beta^+`. Define

```text
nu^+
  := [(B-A+x) beta^+ + x alpha^+ + (A-2x) rho^+] / B.
```

Tilting `Q` only enough to match the forced cancellation probability leaves
`nu^+` as its conditional nonzero-target law. The KL chain rule, followed by
data processing to the target symbol, therefore proves

```text
I(D;S)-I(U;V)
  >= L(alpha,beta,x),

L(alpha,beta,x)
  := (A-x)d(q || kappa)+B D(beta^+ || nu^+).        [two-level bound]
```

This argument does not assume sign symmetry. It uses only stationarity,
addition in the field, and the zero/nonzero projection.

Stationarity yields a second, incomparable payment. Push the product law `Q`
through the bijection

```text
(D,S) -> (S,T),  T:=S+D,
```

and call the resulting source--target law `K`. The true source--target law
has both marginals equal to `beta`. Consequently,

```text
I(D;S)-I(U;V)
  = D(P || Q)
  >= S(alpha,beta,x),

S(alpha,beta,x)
  := inf_(R: row(R)=column(R)=beta) D(R || K).      [stationary bound]
```

The optimization defining `S` is the information projection of `K` onto the
stationary couplings of `beta`. Because `K` already has source marginal
`beta`, eliminating the source potentials gives

```text
S(alpha,beta,x)
  = sup_v { <beta,v>
            -sum_s beta_s log(sum_t (K_st/beta_s) exp(v_t)) }.
```

The continuous extension covers zero-mass rows. This dual is globally valid:
the target potential is fixed before maximizing over the unknown transition
law. It may still distinguish target symbols, which is precisely the
information lost by support-only cuts.

Let `C(alpha,beta,x)` denote the cell-target KL projection, obtained by
retaining the support cell and target symbol. Since both `C` and `S` lower
bound the same divergence,

```text
I(D;S)-I(U;V) >= max(C(alpha,beta,x),S(alpha,beta,x)).
                                                               [hybrid bound]
```

The two terms are incomparable. Numerically, their maximum removes escape
directions left by either term alone.

Apply the two-level bound to both accumulators, with crossing masses `x_1`
and `x_2`. The full exponent then obeys

```text
Psi_{p,3}
  <= Phi + H_int(mu)/3
       -L(mu,beta,x_1)-L(beta,zeta,x_2).            [reduced master bound]
```

The right-hand side has `3p+2` variables and linear feasibility constraints;
the original transition formulation has `2p^2+3p` variables. More
importantly, the reduction is an inequality for every transition type, not a
symmetry ansatz.

The script `scripts/optimize_conditional_product_bound.py` numerically
maximizes the reduced master bound. At the binary output endpoint, its best
values found are as follows.

| field | message cutoff | reduced upper bound found |
|---:|---:|---:|
| `F_5` | `0.002` | `-0.00146130` |
| `F_7` | `0.0008` | `-0.00066335` |
| `F_11` | `0.00025` | `-0.00023818` |

These values are numerical diagnostics, not global certificates. Their signs
show that the two-level inequality is strong enough to support the desired
prime ladder. The next certification problem is now composition-only and
retains a visible margin at each of the first three rungs.

The same script also implements the cell-target, stationary, and hybrid
payments. At output fraction `0.22`, the hybrid upper bounds found are
`-0.00118836` for `F_5` with message cutoff `0.002`, and `-0.00004311` for
`F_7` with cutoff `0.0008`. These remain local numerical searches. In
particular, the septenary margin is too small to guide an exact cover without
a more thorough optimization audit.

The stationary dual has an exact cut representation with rational potentials
and directed exponential intervals. The current cut bank does not strengthen
the five-variable eliminated checker. At the quinary numerical support point, both the original
and hybrid cut banks give exact upper bound `+0.00118844`. The common cut
required by the pure-branch reduction loses the symbol-sensitive stationary
payment.

Retaining the three conditional symbol laws avoids that loss. The resulting
14-variable checker evaluates every fixed cut on the vertices of each
conditional simplex. At the quinary numerical maximizer, a degenerate box has
exact upper bound `-0.00115832`. A global interval cover is still open. After
200 splits, its worst upper bound is `0.902508` because broad symbol boxes make
both KL payments zero. The next certificate must couple the symbol variables
more tightly than independent interval subdivision.

The exact checker `scripts/verify_reduced_composition_bulk.py` works in the
variables `(A,B,C,x_1,x_2)` and the three conditional nonzero-symbol laws. It
uses the variational inequality

```text
D(R || K)
  >= <a,J>+<b,beta>
       -sum_(c,t) K_(c,t)(exp(a_c+b_t)-1)
```

for rational cell potentials `a` and target potentials `b`. The reference
law `K` is the conditional-product law pushed to the support cell and target
symbol. In conditional-composition coordinates, every entry of `K` is a
linear or quadratic polynomial; no small support appears in a denominator.
Directed intervals enclose the exponential coefficients and every entropy.

At the numerical quinary maximizer, a degenerate rational box receives the
exact upper bound `-0.0014613008557`, matching the numerical two-level value.
The regression test checks this point certificate. The global 14-variable
cover remains open: after 1,000 boxes, normalization-aware simplex-vertex
cuts give the exact upper bound `0.4393419`. Broad conditional-composition
boxes still make the KL dual cuts uninformative.

The conditional compositions can in fact be removed. First use field
multiplication to choose a canonical message law with

```text
mu^+_1 >= mu^+_d  for every d!=0.
```

Fix a middle symbol `j` and a first-stage Fenchel cut. On the branch
`beta^+=delta_j`, let `c_d(A,B,x)` be the resulting affine cut value when the
nonzero increment is `d`. For arbitrary nonnegative order multipliers
`lambda_d`, set

```text
c'_1 = c_1-sum_(d!=1) lambda_d,
c'_d = c_d+lambda_d  (d!=1).
```

The canonical inequalities and the entropy conjugate give

```text
A H(mu^+)/3-sum_d mu^+_d c_d
  <= A/3 log sum_(d!=0) exp(-3c'_d/A).              [ordered conjugate]
```

Indeed, subtracting
`sum_(d!=1) lambda_d(mu^+_d-mu^+_1)` can only increase the objective on the
canonical simplex. Unconstrained entropy maximization then proves the
displayed log-sum-exp formula.

There is no positive entropy term for `beta^+` or `zeta^+`. After a complete
Fenchel certificate is fixed, their remaining dependence is multilinear. Its
maximum over the two simplexes is therefore attained at a pair
`(beta^+,zeta^+)=(delta_j,delta_k)`.

The order of these operations is essential. The checker must fix one common
cut mixture before maximizing over `(j,k)`. Choosing the best cut separately
on each pure branch reverses a minimax order and can produce a false upper
bound. Branch-dependent order multipliers remain permissible only within one
fixed certificate, because each multiplier pays for the same canonical
message inequalities.

This gives a five-variable upper bound. For each `(j,k)`, combine an ordered
first-stage conjugate with a second-stage affine cut, then maximize over the
`(p-1)^2` branches. Only `(A,B,C,x_1,x_2)` remain. For fixed certificates,
the ordered conjugate is a convex perspective of log-sum-exp. Its maximum on
a rational support polytope occurs at a vertex. The checker therefore uses
directed point evaluations at finitely many support vertices instead of wide
exponential intervals.

The script `scripts/verify_eliminated_composition_bulk.py` implements the
corrected reduction for `F_5`. At the earlier diagnostic point, the corrected
common-mixture bound is `+0.00002672`. Thus that point is not yet certified
negative by this relaxation.

The five-variable global cover remains open. The checker uses floating point
only to propose a common mixture. It rationalizes that mixture before exact
evaluation. A 20-box corrected diagnostic at `delta=0.1323` and `eta=0.002`
has exact upper bound `0.140051`.

The checker keeps the support and conditional terms coupled. After the
two positive joint entropies are replaced by cross-entropy tangents, the
complete surrogate is convex in `(A,B,C,x_1,x_2)`: negative binary entropy is
convex, the ordered log-sum-exp term is a convex perspective, and every cut
term is affine. Its maximum on a rational support box is therefore attained
at a vertex of the joint five-dimensional feasibility polytope. Evaluating
the complete surrogate at those vertices avoids maximizing its support and
conditional pieces at incompatible points.

The proved global support estimate supplies a second envelope. On every box,
the full exponent is at most both the coupled surrogate and

```text
1/10000 + (conditional message entropy - two KL payments).
```

The checker can propagate common mixtures down the branch-and-bound tree or
select them by support region. The earlier branch-wise diagnostics
`0.09312368`, `0.00846858`, and `0.00271039` used the invalid minimax order and
are withdrawn. `AUDIT.md` records the counterexample and correction.

This observation gives a quantitative proof split. On the domain

```text
A >= 1/500,
C <= 1323/10000,
```

the exact checker `scripts/verify_support_indicator_bulk.py` proves

```text
Phi < 1/10000.
```

It uses cross-entropy tangents for the two positive joint entropies and exact
endpoint maximization of the remaining convex one-variable terms. The cover
closes after 1,629 rational boxes. Its exact upper bound is
`0.00009797319670870162`, and the full numerator and denominator are recorded
in `results/binary-support-upper-0.1323-eta-0.002.json`.

Consequently, every type in this domain with `G>=1/10000` has negative full
exponent. Only the region `G<=1/10000` remains.

The low-gap region has a useful Lagrangian formulation. For every type with
`G<=gamma` and every `lambda>=0`,

```text
Psi <= max(Psi-lambda G)+lambda gamma.
```

At `lambda=3/2`, the full-symbol optimizer finds

```text
max(Psi-(3/2)G) = -0.0025410203.
```

This would give `Psi<-0.002391` throughout `G<=1/10000`. These two values are
numerical diagnostics, not certificates. They reduce the remaining quinary
binary-benchmark proof to an unconstrained full-symbol Lagrangian bound with
a visible margin.

### 3.3.2 A finite information-projection upper bound

The Lagrangian reduction preserves the exact exponent but still leaves a
large transition-entropy cover. A smaller upper bound follows from one
application of data processing.

Let `P` be the joint law of an accumulator increment `D` and source state
`S`. Put

```text
U := 1[D!=0],   V := 1[S!=0],   T := D+S.
```

Define a comparison law `Q` on `(D,S)` by

```text
Q(d,s) := P_UV(u,v) P_(D|U)(d|u) P_(S|V)(s|v),
```

where `u=1[d!=0]` and `v=1[s!=0]`. Thus `Q` keeps the exact support coupling
but makes the two field values conditionally independent once their support
bits are fixed. Let `R` be the image of `Q` under
`(d,s) |-> (u,v,d+s)`.

**Support-cell projection lemma.** For every joint law `P` on `(D,S)`,

```text
I_P(D;S)
  >= I_P(U;V) + D(P_UVT || R).
```

**Proof.** Direct expansion gives

```text
D(P_DS || Q)
  = H(D)+H(S)-H(D,S)-H(U)-H(V)+H(U,V)
  = I_P(D;S)-I_P(U;V).
```

Relative entropy cannot increase under a deterministic map. Applying data
processing to `(d,s) |-> (u,v,d+s)` yields

```text
D(P_DS || Q) >= D(P_UVT || R),
```

which proves the claim. ∎

For stage `i`, write `L_i:=D(P_(U_i,V_i,T_i)||R_i)`. The exact two-stage
exponent therefore satisfies

```text
Psi_(5,3)
  <= H(mu)/3 - I(U_1;V_1) - L_1 - I(U_2;V_2) - L_2.
```

The right-hand side depends on each transition only through the finite table
`P_(U,V,T)`, with `20` cells over `F_5`. It discards information and is an
upper bound, but it retains the additive output constraint that the earlier
support relaxation lost.

At the numerical maximizer of the exact exponent, the first projection term
is `0.0006916531`, compared with an exact internal-information excess of
`0.0006916926`. The corresponding second-stage values are `0.0004196419` and
`0.0004294042`. Thus the projection retains nearly all of the missing
field-value payment at the relevant sparse type.

Two independent constrained searches at `delta=0.1323` and `eta=0.002`
found projection-surrogate maxima

```text
-0.0018144965   (seed 17, 12 restarts),
-0.0018210461   (seed 31, 12 restarts).
```

Both candidates lie on `A=0.002` and `C=0.1323`. These values are numerical,
not certified global maxima. They nevertheless supply the first simple
pointwise upper bound whose numerical optimization reaches the certified
binary benchmark over the complete quinary bulk domain. The remaining task
is to certify this reduced projection problem with rational boxes and
directed bounds for the KL terms. The raw data are recorded in
`results/f5-projection-surrogate-0.1323.json`.

The projected transition tables can be eliminated without weakening this
composition-only relaxation. For increment composition `a` and state
composition `b`, let `M_(a,b)` be the law of

```text
(1[D!=0],1[S!=0],D+S),   D~a, S~b independently.
```

Let `J(a,b)` be the minimum of `D(W||M_(a,b))` over laws `W` whose first
support marginal is `a(F_5^*)`, whose second support marginal is
`b(F_5^*)`, and whose sum marginal is `b`. Every projected stationary
transition is feasible for this minimization. Hence

```text
Psi_(5,3)(mu,beta,zeta)
  <= H(mu)/3 - J(mu,beta) - J(beta,zeta).
```

The information projection has the six-parameter dual

```text
J(a,b) = sup_(y,lambda,rho)
  <y,b> + lambda a(F_5^*) + rho b(F_5^*)
  - log sum_(d,s) a_d b_s
      exp(y_(d+s) + lambda 1[d!=0] + rho 1[s!=0]),
```

where the gauge `y_0=0` leaves four target potentials and two support
potentials. A numerical optimization of the complete projected-table
relaxation gives `-0.0015187126`. At its optimizer, the primal and dual
values differ by at most `1.1e-10` in each stage.

A constraint-generation search collected each dual cut together with its
four multiplicative `F_5^*` images. Its local nonlinear search became
negative after 18 rounds and reached `-0.0013026013` after 30 rounds. A later
global cover found a missed composition with bank value `0.0000455302`.
Fitting the information projection at that composition gave `-0.1488975`.
The final floating bank contains 120 cuts per stage, including multiplicative
orbits fitted at weak points from the global frontier.

The script `scripts/explore_projected_kl_box_cover.py` instead covers the
full composition domain with LP boxes for a fixed cut bank. It uses entropy
and logarithm tangents, all degree-two composition moments, McCormick
envelopes, simplex moment identities, covariance cuts, and first-level RLT
constraints. The checker saves its dual bank and best-bound queue. At
`delta=0.1323` and `eta=0.002`, representative floating bounds are

| boxes split | global LP upper |
|---:|---:|
| 1500 | `0.00361292` |
| 3000 | `0.00213081` |
| 3500 | `0.000481865` |
| 3750 | `0.0000885919` |
| 3799 | `-0.000000632899` |
| 3854 | `-0.000103613` |

At split 3799, every live box had a negative floating upper. Twelve tangent
rounds and 55 further splits increased the global margin to `0.000103613`.
This computation closes the complete composition domain for the floating
relaxation; it does not yet prove the distance theorem. The cut potentials,
exponentials, logarithms, and LP certificates still require directed or
exact replay. The compact results are in
`results/f5-projected-kl-box-cover.json`.

There is one field-uniform face on which the internal-information payment is
already exact. Let `D` be the first-stage increment, let `S` be its source
state, and put `T:=S+D`. Write `U:=1[D!=0]` and `V:=1[S!=0]`.

**One-orbit information toll.** Fix an odd prime `p>=5`. Suppose the
increment composition is sign-symmetric and supported on `{0,+a,-a}`, and
the stationary state composition is sign-symmetric and supported on
`{0,+b,-b}`. If `A:=Pr[U=1]`, then

```text
I(D;S)-I(U;V) >= (A/2) log 2.
```

**Proof.** Let `W:=1[T!=0]`. Stationarity gives `V` and `W` the same
distribution. If

```text
x:=Pr[V=0,W=1],
```

then `Pr[V=1,W=0]=x`. Both events require a nonzero increment. The first is
the support cell `(U,V)=(1,0)`, whereas the second lies in `(U,V)=(1,1)`.
Consequently,

```text
Pr[U=1,V=1] = A-x >= A/2.
```

Every nonzero support cell contains at most two compatible symbol pairs
`(D,S)`. This is immediate for `(0,1)` and `(1,0)`. For `(1,1)`, fix
`S=b`. The only possible nonzero increments that keep `T` in
`{0,+b,-b}` are `-b` and `-2b`. Both cannot lie in one sign orbit unless
`[b]=[2b]`, which would force `3b=0`. The case `S=-b` is identical. This is
the only point at which `p>=5` is used.

The message and state sign symmetries give

```text
H(D)-h(A)=A log 2,
H(S)-h(B)=B log 2,
```

where `B:=Pr[V=1]`. Conditioning `(D,S)` on `(U,V)` costs at most one bit on
each nonzero support cell. Hence

```text
H(D,S | U,V)
  <= (A+B-Pr[U=1,V=1]) log 2.
```

Subtracting the support mutual information from the full mutual information
now gives at least `Pr[U=1,V=1] log 2`, which is at least
`(A/2)log 2`. ∎

If both `mu` and `beta` have this one-orbit form, the second-stage internal
information is nonnegative by data processing, while
`H_int(mu)=A log 2`. Therefore

```text
G >= (A/6) log 2.
```

On the domain `A>=1/500`, the exact support certificate above gives
`Phi<1/10000`. Directed interval arithmetic and an exact rational comparison
then verify

```text
1/10000 < (1/500)(log 2)/6.
```

Thus every such one-orbit type has `Psi<0` at output fraction
`1323/10000`, over every odd prime `p>=5`. The checked negative margin is
`0.0001310490601`; `scripts/verify_one_orbit_toll.py` records its exact
numerator and denominator in
`results/one-orbit-toll-eta-0.002.json`.

The direct counting argument above has an exact extension for globally
sign-symmetric transitions. Write `[x]:={x,-x}` for a sign orbit and set

```text
M:=[D],  R:=[S],  L:=[T].
```

Let `J:=Pr[D!=0,S!=0]`. Then

```text
I(D;S)-I(M;R) = J log 2-H(L | M,R).       [branch identity]
```

To see this, condition a symbol pair `(D,S)` on `(M,R)`. If exactly one
symbol is nonzero, there is one global sign bit and `L` is determined. If
both are nonzero, there are two relative-sign branches, with target orbits
`[s+d]` and `[s-d]`. These orbits are distinct over an odd field. Within
each branch, global sign symmetry contributes one unbiased sign bit. It
follows that

```text
H(D,S | M,R)
  = Pr[(D,S)!=(0,0)] log 2 + H(L | M,R).
```

Combining this equality with

```text
H(D)=H(M)+A log 2,
H(S)=H(R)+B log 2
```

proves the branch identity.

The branch identity identifies the precise loss in the sign-orbit relaxation. For a
nonzero orbit pair `(m,r)`, let `w_(m,r)` be its mass and let `q_(m,r)` be
the conditional probability of one of the two target-orbit branches. Then

```text
I(D;S)-I(M;R)
  = sum_(m,r) w_(m,r) [log 2-h(q_(m,r))]
  >= 2 sum_(m,r) w_(m,r) (q_(m,r)-1/2)^2.
```

The last inequality is Pinsker's inequality in natural units. Thus a small
conditional-sign payment is not an unstructured failure: nearly every active
orbit pair must split almost evenly between its two folded-cycle neighbors.
The magnitude mutual information `I(M;R)-I(U;V)` measures the other possible
payment. A general proof can therefore split on two checkable quantities:
dependence between the increment and state orbits, and imbalance between the
two relative-sign branches.

The reduced optimizer makes the same face visible at the next primes. The
following best values found in the globally sign-symmetric subproblem are
numerical diagnostics.

| field | cutoff `A` | exponent found | `Phi` | `G` |
|---:|---:|---:|---:|---:|
| `F_5` | `0.002` | `-0.00194321` | `-0.00128306` | `0.00066015` |
| `F_7` | `0.0008` | `-0.00090459` | `-0.00062603` | `0.00027857` |
| `F_11` | `0.00025` | `-0.00033118` | `-0.00024186` | `0.00008933` |

In all three searches, the best point found has `mu` and `beta` concentrated
on one nonzero sign orbit and lies at the lower message cutoff. The first-stage
internal increment approaches the exact `(A/2)log 2` toll. This repeated structure
suggests a useful route to a field-uniform theorem: control departures from a
single sign orbit, then analyze the remaining accumulator as a lazy walk on
the odd cycle generated by that orbit. The table does not prove that this
face contains the unrestricted maximum. In particular, sign averaging is not
valid over `F_5`, as the counterexample below shows.

The numerical maximizers are invariant under the sign map `sigma(a):=-a`.
The following lemma proves that this symmetry is without loss over `F_3`.

For a distribution `q`, define its sign-asymmetry by

```text
Asym(q) := H((q+sigma q)/2)-H(q).
```

Apply `sigma` to both coordinates of a transition type.

**Lemma 4 (ternary circulation symmetrization).** Let `tau` be a transition
type on `F_3 x F_3`. Suppose its row and column marginals both equal `beta`,
and its difference marginal equals `alpha`. Then

```text
Asym(tau) >= Asym(alpha)+Asym(beta).
```

**Proof.** For nonnegative `x,y`, set

```text
j(x,y)
  := x log(2x/(x+y))+y log(2y/(x+y)).
```

Terms with zero coefficient use the continuous extension. The function `j`
is jointly convex and positively homogeneous.

The off-diagonal entries of `tau` form a circulation on three vertices. After
exchanging the two nonzero symbols if necessary, write that circulation as

```text
tau_01=c+q,  tau_10=c,
tau_12=h+q,  tau_21=h,
tau_20=b+q,  tau_02=b,
```

where `b,c,h,q>=0`. Write `tau_11=e` and `tau_22=f`. Direct substitution gives

```text
Asym(tau)
  = j(c+q,b)+j(c,b+q)+j(e,f)+j(h+q,h),

Asym(alpha) = j(b+c+h+3q,b+c+h),

Asym(beta) = j(c+e+h+q,b+f+h+q).
```

Joint convexity and homogeneity imply

```text
j(e,f)+j(c+h+q,b+h+q)
  >= j(c+e+h+q,b+f+h+q).
```

It remains to prove that `G>=0`, where

```text
G := j(c+q,b)+j(c,b+q)+j(h+q,h)
     -j(b+c+h+3q,b+c+h)
     -j(c+h+q,b+h+q).
```

Set `m:=b+c`, `x:=c-b`, and

```text
J(s,d) := j((s+d)/2,(s-d)/2).
```

Then

```text
G = J(m+q,x+q)+J(m+q,x-q)+J(2h+q,q)
    -J(2m+2h+3q,3q)-J(m+2h+2q,x).
```

For fixed `m,h,q`, this expression is even in `x`. Moreover,

```text
d^2 J(s,d)/d d^2 = s/(s^2-d^2).
```

Write `f_s(d):=s/(s^2-d^2)`. This function is convex in `d` and decreases as
`s` increases. Since `m+2h+2q>=m+q`,

```text
d^2G/dx^2
  = f_(m+q)(x+q)+f_(m+q)(x-q)-f_(m+2h+2q)(x)
  >= 2f_(m+q)(x)-f_(m+2h+2q)(x)
  >= 0.
```

The feasible interval for `x` is symmetric. Hence `G` is minimized at `x=0`.
At that point,

```text
G = 2J(m+q,q)+J(2h+q,q)-J(2m+2h+3q,3q).
```

Joint convexity and homogeneity of `j` make `J` subadditive. Applying
subadditivity to the three pairs

```text
(m+q,q), (m+q,q), (2h+q,q)
```

proves `G>=0`. ∎

**Corollary 5 (exact sign reduction).** The maximum of `Psi_{3,r}` is attained
by sign-symmetric compositions and transition types.

**Proof.** Average any feasible point with its sign image. Feasibility is
preserved. The change in the objective is

```text
Asym(tau_1)+Asym(tau_2)
  -(1-1/r)Asym(mu)-2Asym(beta)-Asym(zeta).
```

Lemma 4 lower-bounds this change by `Asym(mu)/r`, which is nonnegative. ∎

The one-stage inequality in Lemma 4 does not extend to every odd prime. Over
`F_5`, consider the stationary transition type

```text
tau := 1/28 *
  [ 6  0  0  5  0 ]
  [ 5  6  0  0  0 ]
  [ 0  0  0  0  0 ]
  [ 0  5  0  1  0 ]
  [ 0  0  0  0  0 ].
```

Its state and increment marginals are

```text
beta  = (11,11,0,6,0)/28,
alpha = (13,0,0,10,5)/28.
```

Rows and columns of `tau` both sum to `beta`. Direct entropy evaluation gives

```text
Asym(tau)-Asym(alpha)-Asym(beta) = -(5/14) log 2 < 0.
```

The obstruction also survives the two-stage objective. Set `mu:=alpha` and
choose the second transition by

```text
(tau_2)_{s,t} := beta_{t-s}/5.
```

This transition has increment marginal `beta` and uniform state marginal.
For repetition factor three, sign averaging changes the full exponent by

```text
-(5/14)log 2 + Asym(alpha)/3 = -(5/28)log 2 < 0,
```

where `Asym(alpha)=(15/28)log 2`. Thus the ternary sign reduction is
characteristic-specific. A quinary bulk certificate must retain more than one
nonzero-symbol orbit.

### 3.4 A sign-orbit relaxation for `F_5`

The failed symmetrization still identifies a useful quotient. Let `O_p` be
the set of orbits of `F_p` under `a -> -a`. For odd `p`, the zero orbit has
size one and every other orbit has size two. Write `[a]` for the orbit of
`a`.

For a distribution `nu` on `F_p`, let `bar(nu)` be its pushforward to `O_p`.
For a joint distribution, apply the orbit map to each coordinate. Define

```text
C_p := {([d],[s],[t]) : d,s,t in F_p and t=s+d}.
```

A coarse transition type is a distribution `gamma` supported on `C_p`. Its
three marginals record the increment, source-state, and target-state orbits.
Let `pi_gamma` denote its increment--source marginal.

For orbit compositions `bar(mu)`, `bar(beta)`, and `bar(zeta)`, consider two
coarse transition types. The first has marginals

```text
(bar(mu), bar(beta), bar(beta)),
```

and the second has marginals

```text
(bar(beta), bar(zeta), bar(zeta)).
```

Define the sign-orbit exponent

```text
hat(Psi)_{p,r}
  := H(pi_1)+H(pi_2)
     -(1-1/r)H(bar(mu))-2H(bar(beta))-H(bar(zeta))
     +(1-bar(mu)_0)log(2)/r.
```

**Lemma 6 (sign-orbit upper relaxation).** Every feasible full-symbol point
for `Psi_{p,r}` maps to feasible coarse types and satisfies

```text
Psi_{p,r} <= hat(Psi)_{p,r}.
```

**Proof.** Let `D_i` and `S_i` be the increment and source-state symbols at
stage `i`. The full exponent has the information form

```text
Psi_{p,r}=H(mu)/r-I(D_1;S_1)-I(D_2;S_2).
```

Data processing under the sign-orbit map gives

```text
I(D_i;S_i) >= I([D_i];[S_i]).
```

Conditioning on a nonzero orbit reveals at most one additional bit. Hence

```text
H(mu) <= H(bar(mu))+(1-bar(mu)_0)log(2).
```

Substitution gives the displayed bound. The image of each full transition is
supported on `C_p` and has the required marginals. Allowing every coarse type
with these properties can only increase the maximum. ∎

The relaxation preserves all sign orbits but forgets whether the sign choices
in different transitions are jointly consistent. A negative maximum therefore
certifies the full exponent. A positive maximum need not identify a full-field
counterexample.

**Proposition 7 (certified odd-field sign-orbit bounds).** For repetition
three, the sign-orbit exponent is negative on each domain in the following
table.

| field | message cutoff | output fraction | certified upper bound |
|---:|---:|---:|---:|
| `F_5` | `1/40` | `1/25` | `-0.0000161798496159` |
| `F_5` | `1/60` | `1/20` | `-0.0000011386223254` |
| `F_7` | `1/16` | `9/500` | `-0.0001159627871096` |

**Proof.** The verifier `scripts/verify_sign_orbit_bulk.py` covers the orbit
compositions with rational boxes. Cross-entropy inequalities upper-bound the
two positive joint-entropy terms. Rational secants upper-bound the convex
`x log x` terms on each box. The resulting problem is a linear program.

The floating solver proposes cuts and dual variables. Directed 80-digit
intervals enclose every logarithm. The verifier rounds the enclosures outward
to rational coefficients and repairs each proposed dual to exact feasibility.
It discards a structurally infeasible box only after an exact rational phase-I
dual proves positive constraint violation.

For the septic row, the verifier closes every entropy cut under multiplication
by `F_7^*/{+1,-1}`. This operation supplies all three images of each nonzero
orbit without changing the bound.

The covers close after 1,848, 5,433, and 155 splits. The verifier checks two,
ten, and ten infeasible children, respectively. The run summaries are recorded
in `results/quinary-sign-orbit-bulk-0.04-eta-0.025.json`,
`results/quinary-sign-orbit-bulk-0.05-eta-1-over-60.json`, and
`results/septic-sign-orbit-bulk-0.018-eta-0.0625.json`. ∎

The orbit relaxation does not reach the binary benchmark over `F_5`. At
`delta=0.1323` and `eta=0.0005`, its numerical maximum is `0.0465302489`.
Requiring both coarse transitions to admit one full-symbol lift leaves this
value unchanged. Thus liftability is not the missing constraint. The loss
comes from discarding conditional sign information in the two mutual
information terms.

At the same endpoint, the full-symbol optimizer returns `-0.0006005618`.
The most negative symmetrization gap found has value `-0.1374837842`, but its
full exponent is `-0.4863645818`. These are numerical diagnostics, not
certificates. They motivate a future split between a near-symmetric region and
an asymmetric region with a large negative exponent.

The support-indicator certificate sharpens this proposed decomposition. Raw
Euclidean distance from sign symmetry is poorly scaled near zero support: an
output of weight `0.1323` can be completely one-sided while having squared
asymmetry below `0.00876`. Entropy asymmetry gives the appropriate normalized
diagnostic, but its constrained maxima also remain close to the symmetric
maximum. At composition entropy-asymmetry levels `0.01`, `0.05`, and `0.09`,
the numerical full-symbol maxima are approximately `-0.000610`, `-0.000652`,
and `-0.000729`. Thus asymmetry does not provide a large-margin complement.
The more promising split isolates the field-uniform small-`A` tangent regime
before certifying the remaining full-symbol domain.

### 3.5 Certified ternary optimization

It remains to optimize the sign-symmetric family. Let `A` be the nonzero
increment fraction, let `B` be the nonzero state fraction, and let `b` be a
bridge variable. Every sign-symmetric feasible transition type has the form

```text
[ 1-B-2b             b                 b           ]
[    b       (B-A+2b)/2          A/2-2b          ]
[    b            A/2-2b       (B-A+2b)/2        ].
```

The feasible interval is

```text
max(0,(A-B)/2) <= b <= min((1-B)/2,A/4).
```

On the relative interior `0<A<1`, `0<B<1`, and `A<2B`, transition entropy is
strictly concave in `b`. Its unique maximum satisfies

```text
(1-B-2b)(A/2-2b)^2 = b^2(B-A+2b)/2.
```

The feasibility condition is `A<=2B`. For the two-stage exponent, the outer
domain therefore satisfies

```text
eta <= A <= 1,
A <= 2B,
B <= 2C,
0 <= C <= delta.
```

The script `scripts/optimize_symmetric_ternary.py` solves the cubic inner
problem numerically. It searches the full outer box and the cutoff and output
faces separately. At output fraction `0.195` and cutoff `eta=0.05`, it returns
exponent `-0.00103433377`. The full nonsymmetric optimizer returns the same
value. At output fraction `0.19` and cutoff `eta=0.01`, both searches return
`-0.00375433102`; this maximum lies on both constrained faces.

The script `scripts/certify_symmetric_ternary.py` supplies a deterministic
upper bound. It uses a supporting hyperplane of each concave transition
entropy. On an outer box, the remaining objective is a sum of convex
one-variable functions, whose maxima occur at endpoints. Branching therefore
produces a verified cover of the full outer domain. The verifier uses 80-digit
decimal intervals. Every arithmetic operation and logarithm is rounded
outwards.

The script `scripts/verify_symmetric_ternary.py` is an independent checker.
It shares no implementation code with the primary certifier. The checker uses
exact rational boxes and stationary-bridge bisection. It evaluates entropies
with the interval module from mpmath and constructs its own cover. At output
fraction `0.1323`, it reproduces the primary upper bound to the displayed
precision. The result is recorded in
`results/ternary-bulk-0.1323-eta-0.0001-independent.json`.

**Proposition 8 (certified ternary bulk bounds).** For repetition factor three,

```text
max Psi_{3,3} < 0
```

on each of the following domains:

| message cutoff | output fraction | certified upper bound |
|---:|---:|---:|
| 0.0001 | 0.13 | -0.0000033185 |
| 0.0001 | 0.1323 | -0.0000014822 |
| 0.0001 | 0.196 | -0.0000003617 |
| 0.01 | 0.13  | -0.0001378250 |
| 0.01 | 0.19  | -0.0000053629 |
| 0.05 | 0.195 | -0.0000275872 |

**Proof.** Corollary 5 reduces the full ternary problem to the sign-symmetric
domain. The supporting-hyperplane bound described above is valid on every
box. For the six rows, the primary verifier closes the covers after 312, 315,
789, 57, 219, and 233 box splits. The largest remaining box bound is the value
in the last column. The primary certificate used by Theorem 19 is recorded in
`results/ternary-bulk-0.196-eta-0.0001.json`. The independent checker closes
another 789-split cover and obtains the same upper bound to the displayed
precision. Its result is recorded in
`results/ternary-bulk-0.196-eta-0.0001-independent.json`. ∎

The first two rows certify the small-cutoff bulk through the displayed binary
benchmark. The `0.196` row reaches the new ternary theorem. Proposition 8 does
not control message supports below the stated cutoff.

## 4. Sparse path theory

The first moment has a sparse boundary that is invisible to a normalized
entropy calculation. The dominant ternary trap has an elementary exact count.

Fix one repetition block and track its `r` coordinates through `P_1`. Their
positions form a uniform `r`-subset of `[N]`. For ordered positions
`a_1<...<a_r`, define the span as `a_r-a_1`.

**Lemma 9 (span distribution).** For `t>=r-1`,

```text
Pr[span=t]
  = (N-t) binom(t-1,r-2) / binom(N,r).
```

**Proof.** Choose the first position in `N-t` ways. Choose the remaining
`r-2` interior positions from the `t-1` positions strictly between the two
endpoints. Divide by the number of `r`-subsets. ∎

For every fixed `s>=r-1`, Lemma 9 gives

```text
Pr[span <= s] = Theta(N^{1-r})
```

for one block. Since `K=N/r`, the expected number of bounded-span blocks is
`Theta(N^{2-r})`.

Now specialize to `r=p=3`. Let the active message symbol be `x!=0`. After the
first permutation, the three increments equal `x`. The first accumulator has
states `x`, `2x`, and `0` between successive increments. Its weight equals the
span of the three positions.

For span two, the three positions are consecutive. The first accumulator then
contains the adjacent values `(x,-x)` and zeros elsewhere. The second
permutation places these values at an ordered pair of distinct positions.
The final accumulator weight is their separation.

**Lemma 10 (two-stage repetition-three trap).** Fix one message coordinate and
let `W` be its final ternary weight. The event described above satisfies

```text
Pr[first span=2 and W<=d]
  = 6/[N(N-1)] * d(2N-d-1)/[N(N-1)].
```

Consequently, the expected number of message coordinates with this property
is

```text
2/(N-1) * d(2N-d-1)/[N(N-1)].
```

**Proof.** Lemma 9 gives

```text
Pr[first span=2] = (N-2)/binom(N,3) = 6/[N(N-1)].
```

There are `2(N-w)` ordered pairs at separation `w`. Therefore

```text
Pr[W<=d | first span=2]
  = 2 sum_{w=1}^d (N-w) / [N(N-1)]
  = d(2N-d-1)/[N(N-1)].
```

Multiply the two independent permutation probabilities. Multiplication by
`K=N/3` gives the expectation over message coordinates. ∎

For `d=floor(delta N)` and fixed `0<delta<1`, the last expectation is

```text
[2(2 delta-delta^2)+o(1)]/N.
```

Markov's inequality gives the same expression as an upper bound on the
probability that this particular trap occurs. This calculation explains the
`N^(2-r)` sparse-tail exponent without random multipliers.

The reset condition matters. If `r` is not zero in `F_p`, one active block
leaves a nonzero accumulator state after its last increment. A bounded span
then creates a long suffix rather than a bounded first-stage word. Further
message cancellations are necessary.

### 4.1 Random spacings

The one-block calculation does not control several active message symbols.
A uniform argument must sum over every support size below the bulk cutoff.
The following spacing identity supplies that argument.

Fix an ordered list of `t` nonzero increments. Place the increments at a
uniform `t`-subset of the `N` coordinates. Let `S_i` be the accumulator state
after increment `i`. Define

```text
a := |{i : S_i != 0}|.
```

The state `S_i` persists from increment `i` until the next increment. Let `H`
be the resulting accumulator weight.

**Lemma 11 (spacing distribution).** Conditional on the increment order,

```text
Pr[H=w]
  = binom(w-1,a-1) binom(N-w,t-a) / binom(N,t).
```

Consequently,

```text
Pr[H<=d] = Pr[X>=a],
```

where `X` is hypergeometric with population size `N`, marked-set size `d`,
and sample size `t`.

**Proof.** Let `g_0>=0` be the number of coordinates before the first
increment. For `1<=i<=t`, let `g_i>=1` be the number of coordinates on which
the state after increment `i` persists. These gaps satisfy

```text
g_0+g_1+...+g_t=N.
```

The accumulator weight is the sum of the `a` gaps whose states are nonzero.
There are `binom(w-1,a-1)` ways to choose those positive gaps with sum `w`.
The initial gap and the remaining `t-a` positive gaps can be chosen in
`binom(N-w,t-a)` ways. Division by `binom(N,t)` proves the first formula.
Summing over `w<=d` and applying Vandermonde's identity gives the
hypergeometric tail. ∎

The formula depends on the increment values only through `a`. A nonzero
increment cannot take state zero to state zero. Therefore, zero states cannot
occur consecutively in `(S_1,...,S_t)`, and

```text
a >= ceil(t/2).
```

For `0<delta<1/2`, define

```text
c_delta := D(1/2 || delta)
         = -1/2 log(4 delta(1-delta)).
```

**Lemma 12 (one-accumulator lower tail).** Let `d<=delta N`. For every fixed
nonzero increment order,

```text
Pr[H<=d] <= exp(-c_delta t).
```

**Proof.** Lemma 11 identifies the probability with a hypergeometric upper
tail. If `X` counts marked elements in a sample without replacement, then

```text
E[exp(theta X)]
  <= (1-d/N+(d/N)exp(theta))^t.
```

Maclaurin's inequality for elementary symmetric means proves this moment
bound. Chernoff optimization gives

```text
Pr[X>=a] <= exp(-t D(a/t || d/N)).
```

The binary divergence increases in its first argument above `d/N` and
decreases in its second argument below `a/t`. Since `a/t>=1/2` and
`d/N<=delta<1/2`, the last exponent is at most `-c_delta t`. ∎

Applying Lemma 12 after the second interleaver gives a factor
`exp(-c_delta H)`. The next lemma averages that factor over the first
interleaver without discarding the rarity of a small first-stage weight.

For integers `x>=a`, write `(x)_a:=x(x-1)...(x-a+1)`.

**Lemma 13 (spacing Laplace bound).** In the setting of Lemma 11, for every
`c>0`,

```text
E[exp(-cH)]
  <= (t)_a/(N)_a * (exp(c)-1)^(-a).
```

**Proof.** Lemma 11 and monotonicity of the second binomial coefficient give

```text
E[exp(-cH)]
 <= binom(N-a,t-a)/binom(N,t)
    * sum_{w=a}^infinity binom(w-1,a-1) exp(-cw).
```

The binomial ratio equals `(t)_a/(N)_a`. The negative-binomial generating
function evaluates the sum as `(exp(c)-1)^(-a)`. ∎

### 4.2 A field-uniform sparse criterion

Fix a prime `p`, a repetition factor `r>=3`, and a message support `s`. Let
`Z_{p,s}(d)` count messages of support exactly `s` whose encoded words have
weight at most `d`. Define

```text
L_delta := exp(c_delta)-1,
R_0 := r/eta,

Q_{p,r,delta}(R)
  := (p-1)eR/r * [r/((R-r)L_delta)]^(r/2).
```

**Proposition 14 (uniform sparse first moment).** Fix `0<delta<1/2` and
`0<eta<1`. Suppose

```text
R_0 > 2r/(r-2),
r/((R_0-r)L_delta) < 1,
Q_{p,r,delta}(R_0) < 1.
```

For `d:=floor(delta N)`, we then have

```text
sum_{1<=s<=eta K} E[Z_{p,s}(d)] = o(1).
```

**Proof.** Fix a nonzero message of support `s`. Its repeated word has `t=rs`
nonzero increments. Condition on their value order after `P_1`. If the first
accumulator has weight `H`, Lemma 12 bounds the conditional probability over
`P_2` by `exp(-c_delta H)`. Lemma 13 averages this bound over the first
transition positions.

Let `a` be the number of nonzero first-stage intervals. We have
`a>=ceil(t/2)`. Successive bounds from Lemma 13 have ratio

```text
(t-a)/((N-a)(exp(c_delta)-1)).
```

The ratio decreases with `a`. The second assumption bounds it by a number less
than one for `s<=eta K`. The same assumption makes the base below less than
one. Hence the largest bound occurs at `a=ceil(t/2)`, and

```text
Pr[encoded weight <= d]
 <= [t/((N-t)(exp(c_delta)-1))]^(t/2).
```

There are `(p-1)^s binom(K,s)` messages of support `s`. The estimate
`binom(K,s)<=(eK/s)^s` therefore gives

```text
E[Z_{p,s}(d)] <= Q_{p,r,delta}(N/s)^s.
```

The first assumption makes `Q_{p,r,delta}(R)` decrease for `R>=R_0`. For each
fixed `s`, the same quantity tends to zero because `r>=3`. The third assumption
therefore supplies a summable geometric dominator. Splitting the sum at a
fixed support bound and then sending that bound to infinity proves the claim.
∎

Two exact specializations start the prime ladder. For `p=r=3`, take
`eta=1/10000` and `delta<=1/5`. The function `x -> 4x(1-x)` increases on
`(0,1/2)`, so

```text
4 delta(1-delta) <= 16/25.
```

Thus `L_delta>=1/4`. Using `e<3` gives, for `R>=30000`,

```text
Q_{3,3,delta}(R) < 2R [12/(R-3)]^(3/2) < 1/2.
```

The last inequality follows from monotonicity and the integer check

```text
16 * 30000^2 * 12^3 < 29997^3.
```

The other conditions hold because `R_0=30000>6` and

```text
3/((R_0-3)L_delta) <= 12/29997 < 1.
```

For `p=5` and `r=3`, take `eta=1/100000` and `delta<=13/50`. We have

```text
4 delta(1-delta) <= 481/625,
sqrt(481) < 22,
L_delta > 3/22.
```

Using `e<3` gives, for `R>=300000`,

```text
Q_{5,3,delta}(R) < 4R [22/(R-3)]^(3/2) < 4/5.
```

After squaring, the final inequality reduces to

```text
25 * 300000^2 * 22^3 < 299997^3.
```

The other conditions hold because `R_0=300000>6` and

```text
3/((R_0-3)L_delta) < 22/299997 < 1.
```

The quinary bulk certificate uses the larger cutoff `eta=1/40` at
`delta=1/25`. Here `R_0=120`. The bounds `sqrt(6)<5/2` and `e<11/4` give

```text
L_delta > 3/2,
3/((R_0-3)L_delta) < 2/117,
Q_{5,3,delta}(R_0) < 440(2/117)^(3/2) < 1.
```

For completeness, the exponential-series estimate

```text
e < 1+1+1/2+1/6+1/18 = 49/18 < 11/4
```

uses a geometric bound on the terms from `1/4!` onward. Squaring the final
bound for `Q` leaves the integer check

```text
440^2 * 2^3 < 117^3,
```

or `1548800 < 1601613`.

The stronger endpoint `delta=1/20` uses `eta=1/60`, so `R_0=180`. The bounds
`sqrt(19)<35/8` and `e<11/4` give

```text
L_delta > 9/7,
3/((R_0-3)L_delta) < 7/531,
Q_{5,3,delta}(R_0) < 660(7/531)^(3/2) < 1.
```

After squaring, the last inequality is

```text
660^2 * 7^3 < 531^3,
```

or `149410800 < 149721291`.

The septic certificate uses `delta=9/500` and `eta=1/16`, so `R_0=48`.
Since `sqrt(4419)<133/2`,

```text
L_delta > 367/133,
3/((R_0-3)L_delta) < 133/5505.
```

Using `e<11/4` gives

```text
Q_{7,3,delta}(R_0) < 264(133/5505)^(3/2) < 1.
```

After squaring, the last inequality is

```text
264^2 * 133^3 < 5505^3,
```

or `163969388352 < 166829162625`.

Consequently, the sparse first moment vanishes through output fraction `0.2`
over `F_3` and through output fraction `0.26` over `F_5`.

Proposition 14 controls every support below the smaller bulk cutoff; it is not
only a sublinear-support statement.

### 4.3 A deterministic bulk cutoff

The sparse estimate already yields a field-uniform distance theorem once it
is paired with an elementary property of the accumulator.

**Lemma 15 (accumulator weight floor).** For every field and every word `u`,

```text
wt(Au) >= wt(u)/2.
```

Consequently, every realization of the repetition-`r` two-accumulator encoder
satisfies

```text
wt(E_{P_1,P_2}(x)) >= r wt(x)/4.
```

**Proof.** Write `y:=Au` and set `y_{-1}:=0`. The identity
`u_j=y_j-y_{j-1}` implies that `u_j` can be nonzero only if at least one of
`y_{j-1}` and `y_j` is nonzero. Counting these two possible endpoints gives
`wt(u)<=2 wt(y)`. Repetition multiplies weight by `r`, permutations preserve
weight, and applying the one-accumulator inequality twice proves the second
claim. ∎

The lemma turns output weight into a deterministic upper bound on message
support. In the repetition-three ensemble, a word of weight at most `delta N`
can come only from a message of support at most `4 delta K`.

**Theorem 16 (prime-ladder criterion).** Fix a prime `p` and an integer
`m>=4`. If

```text
288(p-1)^2 m^4 < (m-2)^6(m+2)^3,
```

then the repetition-three addition-only BAA ensemble over `F_p` satisfies

```text
Pr[d_min <= floor(N/m^2)] -> 0.
```

**Proof.** Apply Proposition 14 with `delta:=1/m^2` and `eta:=4/m^2`.
Then `R_0=3m^2/4`. The definition of `L_delta` gives

```text
L_delta > (m-2)/2.
```

Therefore

```text
3/((R_0-3)L_delta)
  < 8/((m-2)^2(m+2)).
```

This quantity is less than one for `m>=4`, and `R_0>6`. Using `e<3`, the
remaining condition in Proposition 14 follows from

```text
Q_{p,3,delta}(R_0)
  < [3(p-1)m^2/4]
    [8/((m-2)^2(m+2))]^(3/2)
  < 1.
```

Squaring the last inequality gives exactly the integer hypothesis. Proposition
14 excludes low-output words from messages of support at most `4K/m^2`.
Lemma 15 excludes every larger support deterministically. ∎

The criterion is nonempty for every fixed prime. For example, when `m>=4`,
the simpler sufficient condition

```text
m^5 > 18432(p-1)^2
```

implies its integer hypothesis. Hence this proof gives explicit relative
distance of order `p^(-4/5)` as the prime grows. The least integer roots for
the first rungs are:

| `p` | least certified `m` | certified relative distance `1/m^2` |
|---:|---:|---:|
| 5 | 7 | `1/49` |
| 7 | 8 | `1/64` |
| 11 | 10 | `1/100` |
| 13 | 10 | `1/100` |
| 17 | 11 | `1/121` |
| 19 | 12 | `1/144` |

Each row is an integer check, not a numerical entropy optimization.

**Theorem 17 (quinary BAA distance at `1/20`).** For the repetition-three
addition-only BAA ensemble over `F_5`,

```text
Pr[d_min <= floor(N/20)] -> 0.
```

**Proof.** Proposition 14 makes the expected number of low-weight words from
messages of support at most `K/60` tend to zero. Lemma 6 and Proposition 7
make the expected number from larger supports exponentially small. Markov's
inequality completes the proof. ∎

Theorem 17 is a rigorous positive-distance result, not a claim that `1/20` is
the quinary threshold. The full-composition problem remains relevant to the
much stronger binary-threshold comparison.

**Theorem 18 (septic BAA distance at `9/500`).** For the repetition-three
addition-only BAA ensemble over `F_7`,

```text
Pr[d_min <= floor(9N/500)] -> 0.
```

**Proof.** Proposition 14 makes the expected number of low-weight words from
messages of support at most `K/16` tend to zero. Lemma 6 and Proposition 7
make the expected number from larger supports exponentially small. Markov's
inequality completes the proof. ∎

Theorem 18 improves the field-uniform septic bound from `1/64` to `9/500`.
The improvement is modest, but it certifies that the bulk method transfers
beyond `F_5`.

**Theorem 19 (ternary BAA distance at 0.196).** Set `delta_*:=49/250`. For the
repetition-three BAA ensemble over `F_3`,

```text
Pr[d_min <= floor(delta_* N)] -> 0.
```

**Proof.** Proposition 14 makes the expected number of low-weight words with
message support at most `0.0001 K` tend to zero. Proposition 8 makes the
expected number from larger supports exponentially small at the same cutoff
and output fraction. Markov's inequality completes the proof. ∎

Kliewer et al. report `0.1323` to four decimal places as the binary
repetition-three lower bound. Theorem 19 therefore gives a stronger certified
lower bound for the ternary ensemble. This comparison proves one-sided
threshold inheritance. It does not prove equality of the binary and ternary
relative distances. See [Kliewer et al., *Coding Theorems for Repeat Multiple
Accumulate Codes*](https://arxiv.org/abs/0810.3422), Table I.

## 5. What successful samples contain

The exact classifier assigns each odd-field minimum witness to one of four
classes. Let `D` be its support and set `M=G_{not D}`.

- `distance_beating_p_torsion`: `rank_2(M)=rank_Q(M)=K`,
  `rank_p(M)<K`, and `d_p<d_2`.
- `nonbeating_p_torsion`: the same rank relation holds, but `d_p>=d_2`.
- `shared_modular_dependency`: `M` has full rational rank and loses rank in
  both characteristics `2` and `p`.
- `structural_dependency`: `M` already loses rank over the rationals.

For repetition three with the `PAPA` schedule, the observed class fractions
are:

| field and length | beating torsion | nonbeating torsion | shared modular | structural |
|:---|---:|---:|---:|---:|
| `F_3`, `N=6`, 10,000 samples | 0.2441 | 0.6413 | 0.0583 | 0.0563 |
| `F_3`, `N=15`, 1,000 samples | 0.1190 | 0.7840 | 0.0160 | 0.0810 |
| `F_3`, `N=24`, 1,000 samples | 0.0940 | 0.8230 | 0.0090 | 0.0740 |
| `F_3`, `N=36`, 300 samples | 0.0567 | 0.8833 | 0.0033 | 0.0567 |
| `F_5`, `N=6`, 10,000 samples | 0.0190 | 0.4915 | 0.1178 | 0.3717 |
| `F_5`, `N=15`, 1,000 samples | 0.0020 | 0.6080 | 0.0500 | 0.3400 |

Thus successful samples do not become torsion-free. Nonbeating
odd-characteristic torsion becomes the dominant class. A proof based on the
absence of all `p`-torsion cores cannot succeed.

The ternary beating fraction is close to a constant divided by `N` at the
three larger lengths. The products `N Pr[d_3<d_2]` are `1.79`, `2.26`, and
`2.04` at lengths `15`, `24`, and `36`. Lemma 10 supplies a concrete
`1/N` mechanism, but it does not prove that all beating events belong to that
family.

## 6. Sparse and bulk message supports

The harness also records the minimum codeword weight at each exact message
support. The following observations concern repetition three over `F_3`.

| `N` | samples | beating fraction | mean support of beating minimum | mean support of nonbeating torsion minimum |
|---:|---:|---:|---:|---:|
| 15 | 1,000 | 0.1190 | 1.58 | 2.48 |
| 24 | 1,000 | 0.0940 | 2.01 | 3.48 |
| 36 | 300 | 0.0567 | 2.12 | 4.55 |

At lengths 24 and 36, no sampled support of size at least six beats the binary
minimum. This is finite evidence, not a uniform bound.

The mean ternary minimum at exact support `s` has a different profile. At
length 36, the mean weights for `s=1,...,12` are

```text
16.16, 12.87, 11.48, 10.93, 10.78, 10.77,
11.05, 11.37, 11.82, 12.45, 13.27, 14.51.
```

The beating events occur at the sparse edge of this curve. The typical
nonbeating minimum moves toward larger support as `N` grows. This separation
motivated the proof split used by Theorem 19.

1. Bound every support below a fixed small linear cutoff with the random
   spacing estimate.
2. Bound every support above that cutoff with the vector-composition first
   moment from Theorem 3.

## 7. Schedule and resonance

The position of the first accumulator determines whether a local
characteristic resonance pays a permutation cost.

For repetition three over `F_3`, the schedule `APAP` applies an accumulator
before random mixing. Every active local block deterministically maps

```text
(x,x,x) -> (x,-x,0).
```

At length 12, 651 of 1,000 exact samples have `d_3<d_2`. The sparse reset is
present before the first random permutation, so Lemma 9 supplies no suppressing
factor.

The schedule `PAPA` forces the same reset to arise from a rare span event.
This difference suggests a necessary design condition:

> Every characteristic-specific bounded-support image must pay at least one
> independent random-placement cost before it reaches the final accumulator.

This condition is not yet a formal theorem hypothesis. It is a design target
for a general class of addition-only encoders.

Repetition five gives a second check. At length 30 with `PAPA`, the observed
beating fractions are `0.013` over `F_3` and `0` over `F_5` in 1,000 samples.
For `p=3`, one repetition-five block does not reset. For `p=5`, a one-block
reset has the smaller fixed-span scale `N^(2-5)=N^-3`.

## 8. General sparse--bulk reduction

The exact first moment suggests a theorem stated at a deterministic threshold,
not at the random coupled binary distance.

**Proposition 20 (sparse--bulk reduction).** Fix `r`, `p`, `delta>0`, and
`eta>0`. Let `Z_p^s(delta N)` count nonzero messages of support less than
`eta K` whose encoded word has weight at most `delta N`, and let
`Z_p^bulk(delta N)` count the remaining low-weight messages. If

```text
E[Z_p^s(delta N)] = o(1)
```

and the maximum of `Psi_{p,r}` under

```text
1-mu_0 >= eta,
1-zeta_0 <= delta
```

is negative, then

```text
Pr[d_p <= delta N] -> 0.
```

**Proof.** The Markov-type bound in Section 3.3 makes
`E[Z_p^bulk(delta N)]` exponentially small. The assumed sparse bound and the
union bound give

```text
Pr[d_p <= delta N]
  <= E[Z_p^s(delta N)] + E[Z_p^bulk(delta N)]
  = o(1).
```

∎

The reduction is elementary; proving its two hypotheses is the substantive
problem. It motivates the following ensemble-specific conjecture.

**Conjecture 21 (BAA threshold inheritance).** For the `PAPA`
repetition-`r` ensemble and every fixed odd prime `p`, there are `eta>0` and
`delta_p>0` such that Proposition 20 applies for every `delta<delta_p`.
Moreover, `delta_p` is at least the proved binary distance threshold for the
same repetition factor.

Suppose the binary repetition-`r` RAA ensemble satisfies

```text
Pr[d_2 < delta N] -> 0.
```

If Conjecture 21 covers this `delta`, Proposition 20 gives the odd-field bound.
If `d_2/N` converges in probability to `delta_2` and Conjecture 21 applies for
every `delta<delta_2`, the two bounds imply

```text
Pr[d_p < d_2-epsilon N] -> 0
```

for every `epsilon>0`.

This formulation separates two mathematical tasks.

- The sparse task is combinatorial. It counts low-complexity accumulator paths
  and their embeddings under independent permutations.
- The bulk task is analytic. It maximizes the entropy rate induced by Theorem
  3 over field-symbol and transition types.

Neither task requires samplewise field monotonicity. Neither task claims that
the binary and prime-field limiting distances are equal.

For `p=r=3`, Theorem 19 completes both tasks at output fraction `0.196` using
cutoff `eta=0.0001`. Proposition 14 already covers sparse output fractions
through `0.2`; the certified bulk exponent is now the limiting step.

For `p=5` and `r=3`, Theorem 17 completes both tasks at output fraction
`0.05` using cutoff `eta=1/60`. Raising this bound toward the binary benchmark
requires a tighter orbit relaxation or a full-symbol certificate.

For `p=7` and `r=3`, Theorem 18 completes both tasks at output fraction
`9/500` using cutoff `eta=1/16`. The certificate closes entropy cuts under the
three multiplicative automorphisms of the nonzero sign orbits.

## 9. Remaining counterexample search

The current experiments found no distance-beating witness with message support
at least six at lengths 24 or 36. They therefore found no candidate
linear-support torsion core.

This absence is not decisive. A counterexample could use a field-symbol type
whose entropy exceeds the placement cost of its residual core. Such a type
would appear only when the message support grows linearly.

The multistart searches found no positive bulk exponent below output fraction
`0.13` for `p=3` or `p=5`. Proposition 8 strengthens the ternary observation:
with cutoff `eta=0.01`, no ternary bulk counterexample exists through output
fraction `0.19`. Proposition 7 rigorously excludes quinary bulk counterexamples
through output fraction `0.05` and septic bulk counterexamples through
`9/500`. The stronger searches remain numerical.

For an uncertified prime, a positive exponent below a proved binary threshold
would identify a possible counterexample without a blind search over
permutations. A planted construction could then test whether the first moment
reflects real codewords or only correlations.

## 10. Immediate proof obligations

1. Raise the quinary bulk certificate from `0.05` toward the binary benchmark
   `0.1323` on the remaining domain above the all-primes support slab, using
   the full-symbol transition polytope.
2. Certify a positive ternary bulk type at `delta=0.197`, then locate the
   first-moment transition between `0.196` and `0.197`.
3. Evaluate or certify the bulk exponent for further primes and the repetition
   factors used by the binary RAA theorem.
4. Raise the septic certificate beyond `9/500` using automorphism-closed cuts.
5. Search the bulk optimization for a positive-exponent type below that
   threshold.

These obligations form a coherent proof program. New finite traps matter only
if they violate the sparse bound or create a positive bulk exponent.
