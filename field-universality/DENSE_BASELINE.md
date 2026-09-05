# A dense field-universal baseline

## 1. Purpose

The characteristic-change problem is not an existence problem. A dense binary
generator can have positive rate and linear minimum distance over the binary
field and over a prescribed collection of prime fields. The same matrix uses
only additions when it encodes field-valued messages.

This construction is too slow for a competitive PCG. Its role is to isolate
the actual open problem:

> Construct a binary or signed generator with positive rate, field-uniform
> linear distance, and near-linear addition-only encoding.

The BAA and Walsh--Hadamard investigations seek the missing efficiency. They
do not seek the first existence theorem.

## 2. The full Boolean evaluation code

Fix an integer `k>=1` and a field `F`. Let `G_cube` be the `k`-by-`2^k`
binary matrix whose columns are all vectors in `{0,1}^k`. Define

```text
C_F(G_cube) := {x^T G_cube : x in F^k}.
```

Thus a message `x` produces the evaluations

```text
(x dot a)_(a in {0,1}^k).
```

The matrix is binary, regardless of `F`.

**Proposition 1 (exact field-universal distance).** For every field `F`, the
code `C_F(G_cube)` has dimension `k` and minimum distance `2^(k-1)`.

**Proof.** Fix a nonzero message `x in F^k`. Choose an index `i` with
`x_i != 0`. Pair each `a in {0,1}^k` with the vector obtained by flipping
coordinate `i`. The two inner products differ by `x_i`. At most one inner
product in each pair is zero. Hence `x^T G_cube` has at least `2^(k-1)`
nonzero coordinates.

For the reverse inequality, take `x` supported on one coordinate. Exactly
half of the Boolean columns have a one in that coordinate. The resulting
word has weight `2^(k-1)`. The positive minimum distance also proves that the
encoding map is injective. ∎

Proposition 1 holds in every characteristic. It does not compare two
unrelated distance arguments. It gives one proof for the same integer matrix
over every field.

The code has rate `k/2^k`, so Proposition 1 alone is not an asymptotically
good family. Random coordinate sampling repairs the rate.

## 3. Dense Bernoulli generator

Fix integers `k>=2` and `n>=1`. Sample independent columns

```text
A_1,...,A_n <- {0,1}^k
```

and define the binary matrix

```text
G := [A_1 | ... | A_n] in {0,1}^{k x n}.
```

For each field `F`, define

```text
C_F(G) := {x^T G : x in F^k}.
```

This ensemble is a random coordinate sample of the full Boolean evaluation
code. Sampling is with replacement because the independent-column model makes
the probability space explicit. A without-replacement variant has the same
asymptotic interpretation.

For `x in F^k \ {0}`, define the Boolean-cube hyperplane section

```text
Z_x := {a in {0,1}^k : x dot a = 0}.
```

The pairing argument from Proposition 1 gives

```text
|Z_x| / 2^k <= 1/2.                                      (1)
```

The Hamming weight of `x^T G` is

```text
wt(x^T G) = n - |{j : A_j in Z_x}|.                      (2)
```

The proof must control (2) simultaneously for every nonzero `x`. A union
bound over the `|F|^k-1` messages would introduce an unnecessary dependence
on the field size. The correct parameter is the VC dimension of the family
of sections `{Z_x}`.

**Lemma 2 (VC dimension of kernel sections).** For every field `F`, the set
system

```text
Z_F := {Z_x : x in F^k \ {0}}
```

on `{0,1}^k` has VC dimension at most `k-1`.

**Proof.** Suppose `a_1,...,a_m` are shattered. The labeling that contains
all `m` points gives a nonzero linear functional whose kernel contains every
`a_j`. Their span is therefore a proper subspace of `F^k`.

For each `j`, another labeling contains every `a_i` with `i!=j` and excludes
`a_j`. Hence `a_j` is not in the span of the other points. The vectors
`a_1,...,a_m` are linearly independent. Their span is proper, so `m<=k-1`.
∎

We use the following standard VC inequality. If a set system has VC dimension
`d>=1`, and `n` points are sampled independently from a distribution `mu`,
then

```text
Pr[sup_R |mu_n(R)-mu(R)| > epsilon]
  <= 8 (2 e n/d)^d exp(-n epsilon^2/32),                 (3)
```

whenever `2n>=d`. Here `mu_n` is the empirical distribution. The constants in
(3) are deliberately conservative. Only their independence from the field
size matters below.

**Theorem 3 (simultaneous distance over prescribed fields).** Fix fields
`F_1,...,F_t`, a real `0<epsilon<1/2`, and integers `k>=2` and `n` with
`2n>=k-1`. Sample `G` as above. Except with probability at most

```text
8t (2 e n/(k-1))^(k-1) exp(-n epsilon^2/32),             (4)
```

the following statement holds simultaneously for every `s in {1,...,t}`:

```text
dim(C_{F_s}(G)) = k,
d_min(C_{F_s}(G)) >= (1/2-epsilon)n.                     (5)
```

**Proof.** Fix `F_s`. Apply (3) to the uniform distribution on `{0,1}^k` and
the set system from Lemma 2. Equations (1) and (3) show that, outside the
failure event, every nonzero `x in F_s^k` satisfies

```text
|{j : A_j in Z_x}| / n <= 1/2+epsilon.
```

Equation (2) gives the distance bound in (5). The distance is positive, so
the encoding map has trivial kernel and dimension `k`. A union bound over the
`t` fields gives (4). ∎

For fixed `epsilon`, choose a sufficiently large constant `c_epsilon` and set
`n=c_epsilon k`. The right side of (4) is exponentially small in `k` for
every fixed number of fields. The code then has constant rate and constant
relative distance. The available rate from this conservative bound is not
intended to be competitive.

## 4. Cryptographic field families

Theorem 3 also permits the field to grow with the security parameter. Let
`lambda` be the security parameter, and let `P_lambda` be a prescribed finite
set of primes. Apply Theorem 3 to

```text
{F_2} union {F_p : p in P_lambda}.
```

The logarithm of the field count, rather than the logarithm of each field
size, enters the failure bound. In particular, the set of all primes below
`2^lambda` has size at most `2^lambda`. The union over those fields costs at
most `lambda` bits in the exponent.

Consequently, for fixed `epsilon`, a length

```text
n = c_epsilon (k+lambda)
```

with a sufficiently large constant can give simultaneous relative distance
`1/2-epsilon` over `F_2` and every prime field of at most `lambda` bits. If
`k=Omega(lambda)`, this remains a constant-rate construction. The probability
is over the one sampled binary matrix `G`.

This conclusion differs from scalar extension. No prime field contains
`F_2`. Theorem 3 controls the change of characteristic directly through the
geometry of Boolean hyperplane sections.

## 5. Encoding cost

The encoder computes

```text
(x^T G)_j = sum_{i : G_(i,j)=1} x_i.
```

It uses no general field multiplications. A dense sampled matrix contains
about `kn/2` ones, so direct encoding uses `Theta(kn)` additions. At constant
rate, the cost is `Theta(n^2)` additions. The matrix also requires `kn` bits
unless its columns are regenerated from a public seed.

Seeded regeneration reduces storage and communication. It does not reduce the
number of field additions. The dense construction is therefore a baseline,
not a competitive PCG encoder.

## 6. The sharper target

Theorem 3 separates three questions.

1. **Existence.** Dense binary matrices achieve positive rate and linear
   distance over both binary and large prime fields.
2. **Structure.** Binary distance alone does not imply prime-field distance
   for an arbitrary realized matrix.
3. **Efficiency.** The open problem is to retain the dense baseline's
   field-uniform mixing with `O(n log^c n)` or `O(n)` additions.

A useful structured theorem would sample a binary or signed matrix `G_lambda`
that admits near-linear encoding and prove

```text
Pr_G[
  min_{F in F_lambda} d_min(C_F(G)) <= delta n
] = negl(lambda)                                          (6)
```

for a constant `delta>0` and a prescribed family `F_lambda` containing
`F_2` and the deployed prime fields. The probability in (6) is over the
sampled encoder. The minimum ranges over fields only after that encoder is
fixed.

The BAA program asks whether a small number of sparse addition-only layers can
meet (6). The Walsh--Hadamard program asks whether fast transforms and a
restricted multiplier alphabet can meet it. Neither program has yet matched
the dense baseline.

## 7. Status and literature

Propositions 1 and Lemma 2 are elementary. Theorem 3 is their combination
with a standard VC uniform-convergence inequality. The proof should not be
presented as novel until a dedicated literature search has checked coding,
learning-theory, and finite-geometry formulations of the same construction.

Finite-field hyperplane classes are studied, for example, by Ruben Ascoli et
al., [*VC-Dimension of Hyperplanes over Finite
Fields*](https://arxiv.org/abs/2307.10425), 2023. That paper is useful context
for the geometric viewpoint. The exact kernel-section argument needed here is
the elementary Lemma 2 above.

The uniform-convergence step uses the classical inequality of Vladimir Vapnik
and Alexey Chervonenkis, *On the Uniform Convergence of Relative Frequencies of
Events to Their Probabilities*, Theory of Probability and Its Applications 16
(1971), 264--280.
