# RM BAA with a fixed mixing constant

The question is whether adding fixed field multiplication to the two
accumulators preserves the distance of the binary RM BAA code. This note
keeps the RM(2,5)[32,16,8] constituent and both permutations fixed when
comparing the two codes. No implementation option, selector, or default
was changed during this investigation.

The full claim over GF(2^128) remains open in this investigation. There
is a pointwise proof for weights below 64, and an exact GF(4)
counterexample to the analogous unrestricted claim. The latter does not
refute the requested GF(2^128) statement.

## Construction and comparison

Let n be a positive multiple of 32 and k=n/2. Let B be the binary n-by-k
matrix applying RM(2,5) independently to each block. Let H_B be a binary
full-row-rank parity-check matrix for its image. RM(2,5) is self-dual,
so the block generator rows also give such checks.

Fix two permutation matrices P1 and P2. Let S be the n-by-n subdiagonal
shift, with S[i,i-1]=1 for 1<=i<n. Over a characteristic-two field, define

```
D(c) = I + c S,
A(c) = D(c)^(-1),
G(c) = A(c) P2 A(c) P1 B,
C(c) = image(G(c)).
```

One scan therefore computes y[i]=v[perm[i]]+c*y[i-1], starting from zero.
Since S is nilpotent, D(c) is invertible for every c. The legacy update
`state = c*(state+input)` has matrix c*A(c). For nonzero c, its two-scan
generator is c^2*G(c), which defines the same linear code C(c).

Write d1 for the minimum symbol distance of C(1). Extending the binary
code to a larger characteristic-two field preserves this distance:
each nonzero extension-field codeword has a nonzero binary component
whose support is contained in its support, and binary codewords are
still available.

The strongest desired comparison is d(C(c))>=d1 for every P1,P2 at the
chosen GF(2^128) constant. Retaining the existing probabilistic 10%
bound over independent uniform permutations would also suffice for the
distance target; it is weaker than this pointwise statement.

## A bounded-weight specialization theorem

Use the polynomial parity-check matrix

```
H(t) = H_B P1^(-1) (I+tS) P2^(-1) (I+tS).
```

For every field element c, ker H(c)=C(c). Every entry of H(t) is a
polynomial over F2 of degree at most two. This bound avoids the large
degrees in the forward generator G(t).

**Proposition.** Let c have algebraic degree m over F2. For every fixed
pair of permutations,

```
d(C(c)) >= min(d1, ceil(m/2)).
```

**Proof.** Fix a coordinate set T with w=|T|<d1 and 2w<m. No nonzero
binary codeword is supported in T. Thus the columns H(1)_T are linearly
independent. Some w-by-w minor f(t) of H(t)_T satisfies f(1)=1. Each
matrix entry has degree at most two, so deg f<=2w<m. The minimal
polynomial of c has degree m and therefore cannot divide f. It follows
that f(c)!=0 and H(c)_T has independent columns. Thus C(c) has no
nonzero codeword supported in T. This holds for every such T, proving
the proposition.

For GF(2^128) and a full-degree c, this gives

```
d(C(c)) >= min(d1,64).
```

In particular, the complete pointwise comparison holds whenever d1<=64.
For a larger binary distance, the theorem excludes new words of weight
at most 63. The number 64 counts nonzero field symbols; it is not a
security parameter. With L constant-weighted scans the same argument
gives min(d1,ceil(m/L)), since the inverse-check entries have degree L.

The exact checker verifies degree 128 for both candidate constants in
the canonical field defined by X^128+X^7+X^2+X+1:

| Constant, polynomial-basis integer | Minimal polynomial, bit encoding |
| --- | --- |
| 2, representing X | `0x100000000000000000000000000000087` |
| 4234123421, used by libOTe mulConst | `0x110bda73b1c803356ca4d4bb9046be967` |

The checker also verifies irreducibility of the canonical degree-128
field polynomial using the Frobenius criterion.

## Where a specialization can fail

For a set T independent at t=1, let Delta_T(t) be the monic greatest
common divisor of all w-by-w minors of H(t)_T. At least one minor is
nonzero, and Delta_T(1)=1. Let m_c(t) be the minimal polynomial of c.
Then

```
rank H(c)_T < w  iff  m_c(t) divides Delta_T(t).
```

Indeed, rank drops exactly when every maximal minor vanishes at c.
For a polynomial over F2, vanishing at c is equivalent to divisibility
by its minimal polynomial. Consequently the common-divisor condition
is both necessary and sufficient.

This identifies the missing step for the fixed 128-bit field: exclude
its particular degree-128 minimal polynomial from Delta_T for all
relevant supports, or bound the probability of those divisibilities
over the permutations. Nonvanishing at t=1 alone cannot do that.

At n=2^21 the existing binary certificate targets distance at least
209716. Excluding weights through 209715 using only the degree argument
would require m>419430, rather than m=128. Sampling one random field
constant and applying a root bound separately to every support also
does not yield a useful bound at this length: the support union is far
larger than the field-size denominator. This does not rule out a more
structured probability argument.

## Exact small-field obstruction

There is an example with one RM(2,5) block, n=32 and k=16, over
F4=F2[a]/(a^2+a+1). Use c=a and the two permutations stored in
`certificates/rm-constant-verification.json`. Exhausting all 65535
nonzero binary messages gives d(C(1))=5. The mixed code contains a word
of weight 4, supported at zero-based positions {5,13,15,19}.

An independent Python checker recovers that word's RM message and
re-encodes it exactly. It also computes all 1820 four-by-four minors
of the 16-by-4 restricted polynomial parity-check matrix. Their gcd is

```
Delta_T(t) = t^2+t+1.
```

At t=1 this gcd is 1, while at c=a it is zero. Hence the restricted
matrix has rank four in the binary code and rank below four in the
mixed code. The constant a has full degree over F2. Thus full-degree
mixing alone does not establish pointwise distance domination for this
RM construction across extension fields.

This particular obstruction disappears at both tested GF(2^128)
constants: their minimal polynomials do not divide t^2+t+1, and the
same four columns have rank four there. It is not a counterexample at
degree 128, nor does one fixed pair of bad permutations refute a
probabilistic bound over random permutations.

## Reproduction and remaining work

Run `python certificates/verify_rm_constant.py`. It uses exact integer
arithmetic and requires no external Python packages. The report contains
the permutations, binary minimum-distance witness, GF4 message and
word, common divisor, and the GF128 checks. The C++ discovery program
`certificates/rm_constant_probe.cpp` found the example at trial 5082
with its documented deterministic seed. Its search alone is not the
verification; the independent Python replay establishes the witness.

The generic argument over F2(t) remains valid, but its specialization
to degree 128 has not been proved at the target distance. Published
[weighted nonbinary RMA analysis](https://arxiv.org/abs/1101.5966)
uses random weighters; that result does not supply this fixed-constant,
RM-specific specialization step.

The next proof target is a bound on degree-128 factors of Delta_T using
the RM blocks and the two random permutations. A useful intermediate
result would force any such factor to involve a sufficiently large or
rare connected support pattern. If that cannot be established, a
separate distance analysis for constant-weighted RM is needed. Neither
the 10% claim nor a replacement VOLE default follows from the present
bounded-weight theorem.
