# Restricted-multiplier Walsh-Hadamard codes

## Bottom line

Replacing the random field multipliers in a Walsh-Hadamard quasi-abelian code
by signs does remove every pointwise multiplication. It also destroys
asymptotic linear distance.

The obstruction extends to every fixed spectral alphabet. Frequencies with
the same projective multiplier type form color classes in the Boolean cube.
A monochromatic affine flat gives an explicit low-weight codeword. For
independent types from a fixed alphabet, the largest such flat grows with the
Walsh-Hadamard dimension. The relative minimum distance therefore tends to
zero.

This negative result does not eliminate the finite large-field design. A
spectral alphabet that grows with the code length can suppress the flat
obstruction. At the intended parameters, an alphabet of about \(2^{10}\)
signed small integers makes the simplest dangerous-flat union bound smaller
than \(2^{-128}\). Each pointwise product can then be replaced by a short
addition chain. A global minimum-distance theorem for this variant remains
open.

## 1. Code model

Fix an odd-characteristic field \(F\), the group

\[
G:=\mathbb F_2^n,
\qquad N:=|G|=2^n,
\]

and the Walsh-Hadamard transform

\[
\widehat{x}(\xi)
 :=
 \sum_{u\in G}(-1)^{\langle u,\xi\rangle}x(u).
\]

The transform is invertible because \(N\ne0\) in \(F\). Fix a block count
\(c\ge2\). For each frequency \(\xi\in G\), let

\[
a(\xi)=(a_0(\xi),\ldots,a_{c-1}(\xi))\in F^c\setminus\{0\}.
\]

The corresponding rank-one quasi-abelian code maps \(x\in F^G\) to

\[
\mathsf{Enc}_a(x)_t
 :=
 \mathcal F^{-1}\bigl(a_t\widehat{x}\bigr),
 \qquad t\in\{0,\ldots,c-1\}.
\]

We normally set \(a_0=1\). This choice makes the encoder injective and gives
rate \(1/c\). Multipliers that differ by a nonzero frequency-dependent scalar
have the same projective type:

\[
\tau(\xi):=[a(\xi)]\in\mathbb P^{c-1}(F).
\]

For sign multipliers, normalization by the first block leaves at most
\(2^{c-1}\) projective types.

## 2. Affine-flat obstruction

The obstruction depends only on the projective type map \(\tau\).

**Proposition 1.** Let \(L=\xi_0+V\subseteq G\) be an affine flat of dimension
\(d\). Suppose that \(\tau\) is constant on \(L\). Then the relative minimum
distance of the length-\(cN\) code satisfies

\[
\Delta(C_a)\le 2^{-d}.
\]

**Proof.** Projective constancy gives a nonzero vector \(v\in F^c\) and
scalars \(\lambda(\xi)\in F^\times\) such that

\[
a(\xi)=\lambda(\xi)v
\qquad\text{for every }\xi\in L.
\]

Fix a character \(\chi\) of \(V\). Define a nonzero message by its transform:

\[
\widehat{x}(\xi)
 :=
 \begin{cases}
   \lambda(\xi)^{-1}\chi(\xi),&\xi\in L,\\
   0,&\xi\notin L.
 \end{cases}
\]

The transform of output block \(t\) is \(v_t\chi 1_L\). Its inverse transform
is either zero or supported on one coset of \(V^\perp\). Each nonzero block
therefore has weight \(N/|V|=N/2^d\). The complete codeword has weight at most
\(cN/2^d\). Dividing by \(cN\) proves the claim. \(\square\)

The proof constructs the message after it sees the sampled multipliers. This
adaptivity is essential because minimum distance is an extremal property.

## 3. Every fixed random alphabet fails asymptotically

Suppose that the projective types \(\{\tau(\xi)\}_{\xi\in G}\) are independent
and identically distributed on at most \(M\) types. Fix a partition of \(G\)
into \(N/2^d\) parallel affine \(d\)-flats. If the type probabilities are
\(\pi_1,\ldots,\pi_M\), then convexity gives

\[
\sum_j\pi_j^{2^d}\ge M^{1-2^d}.
\]

The events are independent within the partition. Proposition 1 therefore
gives the exact one-partition bound

\[
\Pr[\Delta(C_a)>2^{-d}]
 \le
 \left(1-M^{1-2^d}\right)^{N/2^d}
 \le
 \exp\left(-\frac{N}{2^d}M^{1-2^d}\right).
\tag{1}
\]

For fixed \(M>1\), choose

\[
d
 :=
 \left\lfloor
   \log_2\frac{n}{2\log_2 M}
 \right\rfloor.
\]

For sufficiently large \(n\), the exponent in (1) tends to infinity and

\[
2^{-d}\le\frac{4\log_2M}{n}.
\]

Consequently,

\[
\Delta(C_a)=O\left(\frac{\log M}{\log N}\right)
\]

with probability tending to one. In particular, the sign-multiplier ensemble
with fixed \(c\) is not asymptotically good.

The same conclusion applies to any fixed set of small integer multipliers.
If each of the \(c-1\) nontrivial blocks draws from an alphabet of size \(q\),
then \(M\le q^{c-1}\). In this independent-type ensemble, a positive
asymptotic distance requires \(\log q=\Omega(\log N)\). Thus the number of
available scalar multipliers must grow polynomially with \(N\).

This phenomenon is consistent with the concrete theorem of Li et al. Their
bound contains the term \((\log_pN+k)/c\). It is not an asymptotic theorem for
fixed \(p\), fixed \(c\), and \(N\to\infty\). Uniform field multipliers provide
about \(p^{c-1}\) projective types, which is enormous in their intended
large-\(p\) regime.

## 4. The concrete \(N=2^{20}\), rate-\(1/4\) regime

The asymptotic obstruction leaves room for a finite design. Set \(n=20\) and
\(c=4\). Let the first spectral multiplier be one. Draw each remaining
multiplier independently from a sign-symmetric alphabet \(A\subset F^\times\)
of size \(q\).

A three-dimensional affine flat gives relative weight \(1/8<0.1323\) when
each restricted multiplier is a scalar character. For one block, the number
of such functions on the flat is \(q2^3\), among \(q^{2^3}\) possible
functions. The expected number of dangerous three-flats is therefore

\[
\mu_3
 =
 2^{n-3}{n\brack3}_2
 \left(2^3q^{1-2^3}\right)^{c-1},
\tag{2}
\]

where \({n\brack d}_2\) is the Gaussian binomial coefficient. Markov's
inequality bounds the probability of at least one such flat by \(\mu_3\).

For \(n=20\) and \(c=4\), the base-two logarithms of (2) are:

| \(q\) | \(\log_2\mu_3\) |
|---:|---:|
| 2 | 57.608 |
| 4 | 36.608 |
| 8 | 15.608 |
| 16 | -5.392 |
| 32 | -26.392 |
| 1024 | -131.392 |

Rows with \(\log_2\mu_3>0\) are first-moment warnings, not probability lower
bounds. Rows with \(\log_2\mu_3<0\) give valid union bounds. An alphabet of
size \(1024\) makes this particular failure probability smaller than
\(2^{-128}\).

One concrete alphabet is

\[
A=\{\pm1,\pm2,\ldots,\pm512\}.
\]

Multiplication by an element of \(A\) can be implemented by an addition chain.
A binary chain uses at most 17 additions or doublings for these constants.
This replaces three full field multiplications per frequency by at most 51
additions. Better fixed addition chains reduce the average cost.

Equation (2) controls only one explicit witness family. It does not prove the
minimum distance of the restricted-multiplier code. A low-weight word need
not arise from a monochromatic or character-valued affine flat.

## 5. Exact small-instance experiment

The binary wht_flat_search enumerates monochromatic affine flats. It then
constructs the codeword from Proposition 1 and checks its weight over
\(\mathbb F_{65537}\).

For \(n=10\), \(c=4\), \(q=2\), and seed one, the enumerator found:

~~~text
dimension=1 monochromatic_flats=65249
dimension=2 monochromatic_flats=85300
dimension=3 monochromatic_flats=329
dimension=4 monochromatic_flats=0
witness_dimension=3 relative_weight=0.12500000
~~~

All ten tested seeds contained a three-flat and certified a word of relative
weight \(1/8\). With the same \(n,c,\) and seed, increasing \(q\) gave:

| \(q\) | largest enumerated flat | certified relative weight |
|---:|---:|---:|
| 2 | 3 | \(1/8\) |
| 4 | 2 | \(1/4\) |
| 16 | 1 | \(1/2\) |
| 32 | 1 | \(1/2\) |

These values certify explicit upper bounds for the sampled codes. They do not
estimate the true minimum distances when the affine-flat bound is loose.

Run the checker with:

~~~text
cargo run --release --bin wht_flat_search -- \
  --n 10 --blocks 4 --alphabet-size 2 --max-dimension 5 --seed 1
~~~

## 6. Remaining theorem

The sign-only construction is closed: it cannot have asymptotic linear
distance. The growing-alphabet construction remains viable but unproved.
Its distance analysis requires a statement of the following form.

Fix a nonzero spectrum \(z\in F^G\). For independent
\(a_1,\ldots,a_{c-1}\gets A^G\), bound

\[
\Pr\left[
  \operatorname{wt}(\mathcal F^{-1}z)
  +\sum_{t=1}^{c-1}
    \operatorname{wt}(\mathcal F^{-1}(a_tz))
  \le\delta cN
\right]
\]

uniformly enough to sum over every projective choice of \(z\). This is a
finite-field inverse Littlewood-Offord problem for a Walsh-Hadamard matrix.
Affine flats describe one structured exceptional family. A proof must show
that every other exceptional spectrum is also rare or pays sufficient
enumeration cost.

The next bounded step is to classify low-Walsh-support functions on affine
flats for \(q=2^{10}\). After that local audit, the project needs either a
global inverse theorem or a counterexample outside the flat family. Runtime
benchmarking should wait until the mathematical candidate survives that
test.

## References

- Zhe Li, Hongqing Liu, Chaoping Xing, Yizhou Yao, and Chen Yuan,
  [*Faster Pseudorandom Correlation Generators via Walsh-Hadamard Transform*](https://eprint.iacr.org/2026/196),
  2026.
- Changhao Chen and Catherine Greenhill,
  [*Threshold functions for substructures in random subsets of finite vector spaces*](https://arxiv.org/abs/1805.03778),
  2018.
- Terence Tao and Van Vu,
  [*Inverse Littlewood-Offord theorems and the condition number of random discrete matrices*](https://arxiv.org/abs/math/0511215),
  2005.
