# A sparse-core theory for addition-only RAA codes

## 1. Current thesis

The experiments no longer suggest equality of coupled binary and prime-field
distances. They suggest a cleaner two-regime theory.

Sparse messages create characteristic-specific traps with polynomially small
probability. Bulk messages obey a vector-composition first moment. A proof of
one-sided distance inheritance should treat these regimes separately.

The central obstruction is not the existence of odd-characteristic torsion.
Most successful samples contain such torsion above the binary minimum. The
desired theorem must instead exclude torsion below a fixed linear threshold.

This note develops three exact ingredients:

1. a vector-composition enumerator for the complete addition-only ensemble;
2. an exact embedding count for the dominant repetition-three trap; and
3. an algebraic classification of every sampled odd-field minimum witness.

The remaining step is asymptotic. We must bound the sparse enumerator and the
bulk variational problem below the chosen binary threshold.

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

For `p=3`, the numerical maximizers are invariant under exchanging `1` and
`-1`. This symmetric family has an explicit reduction. Let `A` be the
nonzero increment fraction, let `B` be the nonzero state fraction, and let `b`
be a bridge variable. Every symmetric feasible transition type has the form

```text
[ 1-B-2b             b                 b           ]
[    b       (B-A+2b)/2          A/2-2b          ]
[    b            A/2-2b       (B-A+2b)/2        ].
```

The feasible interval is

```text
max(0,(A-B)/2) <= b <= min((1-B)/2,A/4).
```

For fixed `(A,B)`, transition entropy is concave in `b`. The script
`scripts/optimize_symmetric_ternary.py` solves this inner problem and performs
a global differential-evolution search over the three nonzero fractions. At
output fraction `0.195`, it returns exponent `-0.00103433377`, matching the
full optimizer to eight displayed significant digits. This agreement gives a
second implementation check. It does not prove that every global maximizer is
sign-symmetric.

## 4. Exact sparse-trap count

The first moment has a sparse boundary that is invisible to a normalized
entropy calculation. The dominant ternary trap has an elementary exact count.

Fix one repetition block and track its `r` coordinates through `P_1`. Their
positions form a uniform `r`-subset of `[N]`. For ordered positions
`a_1<...<a_r`, define the span as `a_r-a_1`.

**Lemma 4 (span distribution).** For `t>=r-1`,

```text
Pr[span=t]
  = (N-t) binom(t-1,r-2) / binom(N,r).
```

**Proof.** Choose the first position in `N-t` ways. Choose the remaining
`r-2` interior positions from the `t-1` positions strictly between the two
endpoints. Divide by the number of `r`-subsets. ∎

For every fixed `s>=r-1`, Lemma 4 gives

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

**Lemma 5 (two-stage repetition-three trap).** Fix one message coordinate and
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

**Proof.** Lemma 4 gives

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
`2.04` at lengths `15`, `24`, and `36`. Lemma 5 supplies a concrete
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
supports the following proof split.

1. Bound message supports `s=o(N)` with exact path counts and polynomial
   embedding costs.
2. Bound message supports `s=Theta(N)` with the vector-composition first
   moment from Theorem 3.
3. Choose an overlap between the two bounds for intermediate supports.

## 7. Schedule and resonance

The position of the first accumulator determines whether a local
characteristic resonance pays a permutation cost.

For repetition three over `F_3`, the schedule `APAP` applies an accumulator
before random mixing. Every active local block deterministically maps

```text
(x,x,x) -> (x,-x,0).
```

At length 12, 651 of 1,000 exact samples have `d_3<d_2`. The sparse reset is
present before the first random permutation, so Lemma 4 supplies no suppressing
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

## 8. Candidate theorem

The exact first moment suggests a theorem stated at a deterministic threshold,
not at the random coupled binary distance.

**Proposition 6 (sparse--bulk reduction).** Fix `r`, `p`, `delta>0`, and
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

**Conjecture 7 (odd-field threshold bound).** For the `PAPA`
repetition-`r` ensemble and every fixed odd prime `p`, there are `eta>0` and
`delta_p>0` such that Proposition 6 applies for every `delta<delta_p`.
Moreover, `delta_p` is at least the proved binary distance threshold for the
same repetition factor.

Suppose the binary repetition-`r` RAA ensemble satisfies

```text
Pr[d_2 < delta N] -> 0.
```

If Conjecture 7 covers this `delta`, Proposition 6 gives the odd-field bound.
If `d_2/N` converges in probability to `delta_2` and Conjecture 7 applies for
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

## 9. Remaining counterexample search

The current experiments found no distance-beating witness with message support
at least six at lengths 24 or 36. They therefore found no candidate
linear-support torsion core.

This absence is not decisive. A counterexample could use a field-symbol type
whose entropy exceeds the placement cost of its residual core. Such a type
would appear only when the message support grows linearly.

The multistart searches found no positive bulk exponent below output fraction
`0.13` for `p=3` or `p=5`. These searches are not global certificates. Their
most useful output is a candidate field-symbol type: a positive exponent below
a proved binary threshold would identify a possible counterexample without
requiring a blind search over permutations. A planted construction could then
test whether the first moment reflects real codewords or only correlations.

## 10. Immediate proof obligations

1. Replace the numerical bulk searches with rigorous upper bounds, first for
   the sign-symmetric ternary family and then for all compositions.
2. Prove a uniform sparse bound for supports `s<=eta N`, with `eta` tending to
   zero after `N` tends to infinity.
3. Evaluate or certify the bulk exponent for further primes and the repetition
   factors used by the binary RAA theorem.
4. Compare the resulting prime-field threshold with a proved binary threshold,
   not only with numerical samples.
5. Search the bulk optimization for a positive-exponent type below that
   threshold.

These obligations form a coherent proof program. New finite traps matter only
if they violate the sparse bound or create a positive bulk exponent.
