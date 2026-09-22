# Program for the Typical Quinary Distance

For the current coverage audit and reproducible local checks, see
`F5_DISTANCE_EVIDENCE.md` (2026-09-05). The lower-bound work below is a
research program, not a completed certificate.

## Target statements

Let `E_N` be the repetition-three BAA encoder over `F_5`, with independent
uniform interleavers and block length `N=3K`. The numerical first-moment
transition found by the saved local optimization, restricted to message
fraction at least `0.002`, is in

```text
[0.260600, 0.260612].
```

There are two separate targets.

1. **Certified lower bound.** Prove

   ```text
   Pr[d_min(E_N) <= 0.26 N] -> 0.
   ```

2. **Typical-distance converse.** Prove, for every fixed `epsilon>0`,

   ```text
   Pr[d_min(E_N) <= (delta_FM(5)+epsilon)N] -> 1.
   ```

The first moment can prove the first statement. It cannot prove the second.

## A new entropy bound for the lower side

Fix one accumulator. Let `D` have the increment composition `alpha`, and let
`S` and `D+S` both have the stationary composition `beta`. Every feasible
joint law satisfies

```text
I(D;S)
  = H(D)+H(S)-H(D,S)
  = H(alpha)+H(beta)-H(S,D+S)
  >= H(alpha)-H(beta).
```

The stationary information payment is the minimum of `I(D;S)` over these
laws. Therefore

```text
I_stat(alpha,beta) >= max(0,H(alpha)-H(beta)).
```

For two accumulators, with message, middle, and output compositions
`mu,beta,zeta`, the two inequalities telescope:

```text
I_stat(mu,beta)+I_stat(beta,zeta) >= H(mu)-H(zeta).
```

Hence the bad-type exponent has the global upper bound

```text
H(mu)/3 - I_stat(mu,beta)-I_stat(beta,zeta)
  <= H(zeta) - 2H(mu)/3.
```

The joint-projection box checker now combines this inequality with the exact
KL dual. In a 100-split floating run at `delta=0.22`, the global upper fell
from the trivial `log(5)/3 = 0.5364793` to `0.3908506`. The worst remaining
box is dense and is controlled by the telescoping entropy bound. This is
substantial pruning, but it is not yet close to a certificate.

## Count projective messages

For the reverse direction, scalar multiples must be removed before taking a
second moment. Let `L_K` be the set of one-dimensional subspaces of
`F_5^K`. Define

```text
Z_N(delta)
  := #{ell in L_K : wt(E_N u) <= delta N for nonzero u in ell}.
```

The definition does not depend on the chosen nonzero representative because
scalar multiplication preserves Hamming support. Moreover,

```text
{d_min(E_N) <= delta N} = {Z_N(delta) > 0}.
```

The first moment of `Z_N` differs from the nonzero-message enumerator only by
the constant factor `4`. Its exponential rate is unchanged.

## The pair enumerator

Paley--Zygmund gives

```text
Pr[Z_N(delta)>0] >= E[Z_N(delta)]^2 / E[Z_N(delta)^2].
```

The diagonal contribution to the second moment is `E[Z_N]`. When the first
moment is exponential, this contribution is negligible relative to its
square. The remaining terms correspond to two distinct projective lines.
Their representatives are linearly independent.

Encode a pair of messages coordinatewise as a word over the additive group
`F_5^2`. The same two interleavers act on both components. The exact
pair-type exponent has the same form as the single-word exponent:

```text
Psi_2(nu,beta,zeta)
  = H(nu)/3
    - I_stat^(F_5^2)(nu,beta)
    - I_stat^(F_5^2)(beta,zeta).
```

Here `nu`, `beta`, and `zeta` are distributions on `F_5^2`. Each stationary
payment minimizes mutual information over laws of `(D,S)` in
`F_5^2 x F_5^2` such that

```text
D ~ alpha,  S ~ beta,  D+S ~ beta.
```

The output constraint requires both coordinate projections of `zeta` to have
nonzero mass at most `delta`. Types supported on a one-dimensional subspace
represent proportional message pairs and belong to the diagonal analysis.

## Replica-symmetry question

Let `Psi_1(delta)` be the maximum single-word exponent. A second-moment proof
must control the correlation advantage

```text
Gamma_2(delta)
  := max Psi_2(nu,beta,zeta) - 2 Psi_1(delta).
```

Independent products of two single-word maximizers give `Gamma_2(delta)>=0`.
The desired exponential statement is `Gamma_2(delta)=0`. The difficult
possibility is a correlated pair type whose two projections look optimal but
whose shared interleavers reduce the combined information payment.

There is no such reduction at product compositions. More precisely, let
`Q_1` minimize the one-replica stationary payment for `(alpha,beta)`, and
give the two replicas the product increment and state compositions
`alpha x alpha` and `beta x beta`. Then

```text
I_stat^(F_5^2)(alpha x alpha, beta x beta)
  = 2 I_stat^(F_5)(alpha,beta).
```

For the upper bound, `Q_1 x Q_1` is feasible. For the lower bound, group the
variables of an arbitrary feasible pair law as
`X_i=(D_i,S_i)`. The KL chain rule gives

```text
D(Q_{X_1,X_2} || R_{X_1} x R_{X_2})
  = D(Q_{X_1} || R_{X_1})
    + D(Q_{X_2} || R_{X_2})
    + I_Q(X_1;X_2).
```

Each projected law `Q_{X_i}` is feasible for the one-replica problem, and
the last term is nonnegative. This proves the reverse inequality. Applying
the identity to both accumulators shows that the independent product of a
one-replica maximizer has pair exponent exactly twice the one-replica
exponent. Thus a second-moment obstruction, if one exists, must use
nonproduct message, middle, or output compositions. It cannot arise solely
from a more efficient pair transition at fixed product compositions.

The numerical implementation independently checks this identity at
`delta=0.261`. It finds the one-replica exponent
`0.00022732171499` and the optimized pair exponent
`0.00045464343029`; their difference from exact doubling is
`3.2e-13`.

The product is also a strict local maximum in the full correlation tangent
space, according to the current floating-point diagnostic. Holding both
one-replica marginals fixed leaves `16` correlation coordinates in each of
the message, middle, and output pair compositions, hence `48` coordinates
in total. At `delta=0.261`, a central-difference Hessian calculation gives

```text
smallest eigenvalue  = -1.4597e-2,
largest eigenvalue   = -2.5031e-5,
positive eigenvalues = 0.
```

The largest baseline gradient coordinate is `7.7e-9`; the maximum Hessian
asymmetry before numerical symmetrization is `1.3e-6`. Twelve independent
random-direction checks also give negative curvature, between `-0.066` and
`-0.021`. These computations rule out a local correlation instability at
the numerical level. They do not exclude a separated nonproduct maximizer,
so they are evidence for `Gamma_2(delta)=0`, not a proof of it.

Even equality of exponential rates is not by itself enough for convergence
in probability. Polynomial prefactors and neighborhoods of proportional
pairs must also be controlled. The proof program is therefore:

1. quotient exact scalar duplicates by counting projective lines;
2. prove or numerically test `Gamma_2(delta)=0`;
3. classify every equality case under `GL_2(F_5)`;
4. prove negative curvature transverse to the product manifold;
5. bound near-proportional pair types and finite-type prefactors;
6. apply Paley--Zygmund;
7. find an additional amplification or concentration argument if the second
   moment yields only a probability bounded away from zero.

## Immediate computations

A direct pair optimizer would use three `25`-symbol compositions and two
`25`-by-`25` transition laws. That formulation is too large for blind global
search. The first computations should instead:

1. decompose the computed correlation space under scalar multiplication and coordinate
   exchange;
2. turn the negative local spectrum into an interval-certified statement;
3. search for separated nonproduct maxima with a symmetry-reduced pair
   optimizer.

The reproducible diagnostics are

```text
python scripts/diagnose_pair_replica_payment.py --delta 0.261
python scripts/diagnose_pair_correlation_hessian.py --delta 0.261
python scripts/diagnose_pair_correlation_spectrum.py --delta 0.261
```

A positive correlation mode would refute the naive second-moment route. A
strictly negative transverse Hessian would justify a reduced pair search and
would be the first evidence for the converse half of the threshold
conjecture.

## Relation to binary RMA analyses

The binary repeat--multiple-accumulate literature proves high-probability
lower bounds from average enumerators and careful control of sublinear input
weights. Some papers describe the resulting threshold as a typical distance.
That terminology does not supply the reverse implication needed here. In
particular, a positive average spectral exponent can still be caused by rare
interleavers. The quinary converse should therefore cite a pair or
concentration argument explicitly.

Two references delimit the current claim.

- Ravazzi and Fagnani, *Spectra and Minimum Distances of Repeat
  Multiple--Accumulate Codes*, IEEE Transactions on Information Theory 55(11),
  2009, DOI `10.1109/TIT.2009.2030459`. The paper describes its main distance
  theorem as a probabilistic lower bound and handles the zero floor of the
  binary spectral shape with finite-length enumerator estimates.
- Kliewer, Zigangirov, Koller, and Costello, *Coding Theorems for Repeat
  Multiple Accumulate Codes*, arXiv `0810.3422`. Its uniform-interleaver
  argument proves linear lower growth for binary RAA ensembles of rate at most
  `1/3`; it does not turn a positive spectral exponent into a converse.
