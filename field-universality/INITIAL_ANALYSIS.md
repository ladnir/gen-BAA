# Initial analysis of field inheritance

## 1. Current conclusion

Consider an integer generator matrix obtained from binary local maps,
permutations, and prefix accumulators. Reduce the same matrix over
`F_2` and `F_p`, using the same sampled permutations.

The initial evidence does not support asymptotic equality of the two relative
distances. For the standard two-interleaver schedule, odd-characteristic and
rational distances are usually larger than the binary distance. The observed
gap does not shrink over the exact lengths currently accessible.

The evidence does support a weaker and more useful target:

```text
for every fixed odd prime p and every epsilon > 0,
Pr[d_p < d_2 - epsilon N] -> 0.
```

No asymptotic counterexample to this one-sided statement has been found for
the standard two-interleaver schedule and a field-stable local code. Small
instances do contain samples with `d_p < d_2`. Their minimum words are usually
sparse-message alignment events. Their absolute deficit remains small as the
length grows.

This distinction matters for the application. One-sided inheritance would
permit an addition-only prime-field encoder to use the binary distance
guarantee. It does not require the prime-field code to have binary-like
distance. A larger prime-field distance is beneficial.

## 2. Model and schedule convention

Let `B_N` be an integer matrix with entries in `{0,1}`. Let `A_N` be the
inclusive prefix-sum matrix. Let `P_1,P_2` be independent uniform permutation
matrices. Vectors are row vectors, and operations are listed in execution
order.

The standard repeat--accumulate--accumulate schedule is

```text
PAPA: B_N -> P_1 -> A_N -> P_2 -> A_N.
```

The alternative schedule

```text
APAP: B_N -> A_N -> P_1 -> A_N -> P_2
```

has only one effective random interleaver between accumulators. The final
permutation preserves weight. This schedule can expose local cancellation
traps that the standard schedule disperses before the first accumulator.

For a field `F`, write `G_F` for the reduction of the resulting integer matrix
and define

```text
C_F := rowspan_F(G_F),
d_F := minimum Hamming weight of a nonzero word in C_F.
```

Unless stated otherwise, `G_F` has the same row rank `K` in every compared
field. A rank loss is a separate dimension-instability failure.

All coupled comparisons below use identical integer matrices and identical
permutations in every field.

## 3. Exact algebraic criterion

This section records finite statements. They do not depend on a random model.

Let `M` be a `K x s` integer matrix. Define its top determinantal divisor by

```text
Delta_K(M) := gcd of all K x K minors of M.
```

Set `Delta_K(M)=0` when every top minor is zero. For a prime `ell`, use the
convention that `ell` divides zero.

**Lemma 1 (rank through determinantal divisors).**

```text
rank_Q(M) = K       iff Delta_K(M) != 0,
rank_F_ell(M) = K   iff ell does not divide Delta_K(M).
```

The first equivalence is the maximal-minor rank criterion. The second follows
by reducing every maximal minor modulo `ell`. Equivalently, it follows from the
Smith normal form of `M`.

Let `G` be a full-row-rank `K x N` generator matrix. For
`D subseteq [N]`, let `G_{not D}` contain the columns outside `D`. Then

```text
d_F = min { |D| : rank_F(G_{not D}) < K }.
```

Therefore, if `d_2=d`, then every `D` with `|D|<d` has an odd, nonzero
determinantal divisor. An odd-prime loss below the binary distance occurs
exactly when an odd prime divides one of these odd divisors:

```text
d_p < d_2
iff
there exists D with |D| < d_2 and p | Delta_K(G_{not D}).
```

Call such a set a **p-torsion deletion set**. This terminology refers to the
torsion exposed by the Smith form of the retained-column lattice.

The reverse inequality has a different cause. A binary-minimal deletion set
can have a nonzero even determinantal divisor. The retained matrix then loses
rank over `F_2` but can keep full rank over `Q` and every odd prime not dividing
that divisor. Thus `d_p>d_2` is not an anomaly. It is the expected outcome when
the first extremal dependencies are characteristic-two dependencies.

These observations separate three mechanisms:

1. `Delta=0`: a structural dependency present over every field;
2. `Delta` even and nonzero: a possible binary-only dependency;
3. `p | Delta` with `Delta` odd: an odd-characteristic dependency below a
   binary full-rank certificate.

Exact coupled equality requires the extremal deletion threshold to coincide.
It does not require the same deletion set or the same codeword in both fields.

## 4. A sufficient condition that is too strong

If every square minor of `G` lies in `{0,1,-1}`, then every column restriction
has the same rank over every field. Hence every field has the same minimum
distance. More generally, the conclusion holds when the represented column
matroid is regular.

The accumulator matrix is totally unimodular, and permutations preserve total
unimodularity. Products do not preserve it. Cauchy--Binet expresses a minor of
a product as a signed sum over intermediate path families. Several unit terms
can combine into an integer divisible by `2`, by `p`, or by no small prime.

This explains both sides of the data. Binary rank often drops because an even
signed path count vanishes modulo two. An odd field retains that rank unless
its characteristic divides the same count. Odd-characteristic failures need a
more specific congruence and are less frequent for larger primes.

Random diagonal variables avoid this collision in the generic-rank analysis
of BAA. Distinct path families receive distinct monomials. Setting every
diagonal entry to one collapses those monomials back into signed integer path
counts. The no-scaling problem is therefore a specialization problem, not a
generic-rank problem.

Total unimodularity is stronger than necessary. A one-sided theorem only needs
the absence of `p`-torsion deletion sets below the binary distance. The useful
property is **support-resilient p-torsion freeness**, restricted to large
retained-column sets.

## 5. Exact experiments

The experiment harness constructs the encoder over the integers. It computes
exact distances by exhaustive deletion ranks or exhaustive field-valued
messages. The two methods agree in unit tests. Minimum-weight messages and
codewords are retained as witnesses.

### 5.1 Standard `PAPA`, repetition factor three

The local code is `[3,1,3]`. The table reports sample means of exact relative
distance. `lower` is the fraction of samples with `d_3<d_2`.

| `N` | samples | mean `d_2/N` | mean `d_3/N` | `lower` |
|---:|---:|---:|---:|---:|
| 6  | 10,000 | 0.375 | 0.398 | 0.244 |
| 12 | 1,000  | 0.261 | 0.322 | 0.132 |
| 15 | 100    | 0.226 | 0.294 | 0.140 |
| 24 | 100    | 0.198 | 0.261 | 0.150 |
| 36 | 100    | 0.183 | 0.247 | 0.090 |
| 48 | 2      | 0.135 | 0.250 | 0.000 |

The length-48 row is diagnostic only because it has two samples. The preceding
rows show a stable positive mean gap. They do not support two-sided equality.

The event `d_3<d_2` is less concerning than its raw frequency suggests. At
length 36, eight of the nine lower samples have deficit one. Only one sample
has normalized deficit at least `0.05`. Minimum messages in lower samples have
mean outer weight `2.22`, compared with `4.49` in the other samples. Sparse
messages dominate the observed failures.

For `F_5`, the mean normalized gap over `F_2` is `0.22, 0.18, 0.18, 0.17` at
lengths `6,12,15,24`. No length-15 or length-24 sample has `d_5<d_2`.

Repetition factor five gives the same qualitative result. At length 60, 50
exact `F_2/F_3` samples have mean relative distances `0.255` and `0.369`.
Every ternary distance exceeds its coupled binary distance; the smallest
absolute gap is three.

### 5.2 Other field-stable local codes

The binary simplex generator `[7,3,4]` has the same complete deletion-rank
profile over `F_2`, `F_3`, `F_5`, and `Q`. Thus its local represented matroid is
identical in these four fields.

After two `PAPA` stages with two blocks and length 14, 200 exact samples give

| field | mean relative distance | mean gap over binary | fraction below binary |
|:---|---:|---:|---:|
| `F_2` | 0.176 | 0 | 0 |
| `F_3` | 0.272 | 0.096 | 0.020 |
| `F_5` | 0.333 | 0.157 | 0 |
| `Q`   | 0.377 | 0.201 | 0 |

Local field stability therefore does not imply global distance equality. It
only removes an easy source of counterexamples.

### 5.3 Schedule sensitivity

For repetition factor three, length 12, and 1,000 exact samples:

| schedule | field | mean relative distance | mean gap over binary | fraction below binary |
|:---|:---|---:|---:|---:|
| `PA` | `F_3` | 0.276 | 0.020 | 0.177 |
| `PAPA` | `F_3` | 0.322 | 0.061 | 0.132 |
| `PAPAPA` | `F_3` | 0.354 | 0.098 | 0.080 |
| `APAP` | `F_3` | 0.135 | -0.085 | 0.650 |

The `APAP` row is a genuine characteristic-three degradation. It does not yet
refute normalized inheritance because the binary `APAP` family also appears
to have zero asymptotic relative distance.

At lengths `12,15,36`, the exact mean binary distances for `APAP` are
`2.64,2.94,3.55`. The ternary means are `1.62,1.63,1.63`. Both normalized
means decrease, while the ternary absolute distance remains tight.

### 5.4 Torsion certificates and sparse residual cores

The first classifier covered every sampled loss `d_p<d_2`. Let `D` be the
support of the odd-field minimum word and let `M=G_{not D}`. Since
`|D|<d_2`, the matrix `M` has full row rank over `F_2`. For these losses, the
classifier records the following exact data.

1. `K-rank_F_p(M)`, which is the number of Smith invariant factors divisible
   by `p`;
2. a maximal minor selected by binary pivoting, whose determinant is nonzero,
   odd, and divisible by `p`;
3. the exact top determinantal divisor when the number of maximal minors is
   within a specified budget; and
4. the residual core after unit-equation elimination in the sparse encoder
   circuit.

The sparse circuit introduces variables for the message and every intermediate
word. Its equations encode the local map, permutation equalities, accumulator
recurrences, and the zero output coordinates outside `D`. A degree-one
equation forces its variable to zero. Repeating this operation preserves the
solution space up to the forced zero coordinates over every field. Therefore,
every field-specific witness must leave at least one message variable in the
residual core.

The current harness classifies every odd-field minimum, not only losses. It
distinguishes distance-beating `p`-torsion, nonbeating `p`-torsion, modular
dependencies shared with characteristic two, and structural dependencies over
the rationals. It also records exact intermediate circuit traces and a colored
graph fingerprint for the factor core. The resulting tables and their
interpretation appear in [`CLEAN_THEORY.md`](CLEAN_THEORY.md). The main new
finding is that successful samples usually still contain odd-characteristic
torsion; the torsion minimum simply lies above the binary distance.

For a residual bipartite core, the reported cycle rank is
`edges-variables-equations+components`.

The following table summarizes all classified losses. The top-divisor column
reports exact computations only. The length-36 retained matrices have between
51,895,935 and 834,451,800 maximal minors, so only their explicit pivot-minor
certificates were computed.

| field and length | losses | Smith `p`-nullity | exact top divisors | mean core message variables | mean core variables | mean cycle rank |
|:---|---:|:---|:---|---:|---:|---:|
| `F_3`, `N=6`  | 2,441 | 1 in every case | `3` in 2,435; `9` in 6 | 1.31 | 16.77 | 3.75 |
| `F_5`, `N=6`  | 190   | 1 in every case | `5` in 190 | 1.54 | 22.34 | 4.73 |
| `F_3`, `N=15` | 14    | 1 in every case | `3` in 14 | 1.43 | 19.21 | 4.14 |
| `F_3`, `N=24` | 15    | 1 in every case | `3` in 15 | 2.00 | 28.07 | 5.33 |
| `F_3`, `N=36` | 9     | 1 in every case | not enumerated | 2.22 | 34.00 | 5.89 |

Every residual factor core in this data is one connected component and has no
isolated variable. In 2,650 of the 2,669 losses, the surviving message
variables are exactly the support of the minimum witness. Thus the observed
losses are almost uniformly rank-one torsion events carried by one small
connected circuit trap. They are not diffuse rank collapses.

This observation strengthens the sparse-alignment explanation, but it does
not prove that the core size is bounded as `N` grows. The classifier sees the
minimum witness in a small sample of encoders. A linear-size family of rarer
cores could still control an asymptotic tail. The forward retained-matrix core
is also much less informative: it becomes nearly the entire message space by
length 24. The sparse circuit representation is essential.

## 6. A fully explained finite failure

Consider `PAPA` with five repetition-three blocks at length 15. In one sampled
encoder, the binary distance is four and the ternary distance is one. The
ternary minimum message activates a single repetition block.

The first permutation places its three equal symbols in three consecutive
positions. The first accumulator over `F_3` maps

```text
(1,1,1) -> (1,2,0).
```

The second permutation places `1,2` consecutively. The final accumulator maps
them to one nonzero symbol. Over `F_2`, three equal increments leave a nonzero
state after the third increment. The binary word therefore acquires a suffix
and cannot use the same cancellation.

This failure succeeds because two independent rare alignments occur. For one
fixed repetition block, the probability that its three coordinates become a
consecutive triple is of order `N^-2`. The second alignment contributes another
inverse power of `N`. There are only order `N` single-block messages up to
scale. This mechanism therefore vanishes at a fixed linear-distance threshold.

The example is still valuable. It proves that samplewise monotonicity is false,
even for a field-stable repetition block and the standard schedule.

## 7. Why `APAP` fails dramatically

For repetition factor three over `F_3`, the first accumulator resets at every
local block:

```text
(x,x,x) -> (x,-x,0).
```

Before the random permutation, the first-stage code is therefore a direct sum
of opposite-valued pairs. After permutation and the second accumulator, one
generator row becomes an interval between the two positions of its pair. Its
weight is the pair separation.

There are `N/3` disjoint random pairs. Their minimum separation is tight on a
constant scale. In particular, the ternary minimum distance is `o(N)` with
high probability. The exact experiments show mean distance about `1.63` from
length 12 through length 36.

The binary first accumulator does not reset after one odd repetition block.
It does reset after suitable pairs of blocks, which supplies constant-weight
inputs to the remaining single random accumulator. Classical one-accumulator
results predict sublinear rather than linear binary distance. Thus `APAP` can
separate absolute growth rates without separating limiting relative distances.

This example gives a design rule: an accumulator placed before the first
random interleaver can turn a characteristic resonance inside each local block
into a permanent low-weight subcode.

## 8. Counterexample taxonomy

### 8.1 Local dimension or distance instability

A binary local generator can lose rank or minimum distance after reduction
modulo `p`. Repeating that defect creates an immediate global obstruction.
Any theorem must assume or prove local field stability.

This condition is necessary but not sufficient. The simplex experiment has an
identical local matroid in all tested fields, yet its global distances differ.

### 8.2 An explicit architecture-free counterexample

No theorem can cover arbitrary binary generator matrices. Let

```text
M := J_4-I_4,
```

where `J_4` is the all-one matrix. The binary matrix `M` has determinant `-3`.
For `L>=1`, form a `4 x (4L+1)` binary matrix `G_L` by repeating each column
of `M` exactly `L` times and appending one all-one column.

Over `F_2`, `M` is invertible. If `y=xM`, the last coordinate of `xG_L` is
the parity of `y`. Therefore

```text
wt(xG_L) = L wt(y) + (wt(y) mod 2),
d_2(G_L) = L+1.
```

Over `F_3`, the all-one message annihilates every column of `M`, because each
column has sum three. It evaluates to one on the appended column. Hence

```text
d_3(G_L) = 1.
```

Both matrices have row rank four. Over `F_3`, the appended column removes the
one-dimensional kernel of `M`. Consequently,

```text
d_2(G_L)/(4L+1) -> 1/4,
d_3(G_L)/(4L+1) -> 0.
```

This is a proved asymptotic counterexample for unrestricted binary matrices.
Its rate tends to zero, and its local distance is not field-stable. It does not
refute Conjecture A, which fixes a positive-rate, field-stable local code and a
specific random encoder architecture. It shows that those hypotheses are
substantive.

A fixed-rate extension appears plausible. Choose `K` growing and sample dense
binary columns of an odd weight `w` divisible by `p`. Every sampled column lies
in the `F_p` hyperplane orthogonal to the all-one message. One appended column
can restore full `F_p` rank while leaving that message with weight one. A
constant-weight random-coding argument should give positive binary distance at
some positive rates. Completing this extension requires a uniform bound on the
relevant Krawtchouk coefficients and a span estimate. It remains a candidate,
not a result of this note.

### 8.3 Sparse alignment events

Fixed-support messages can exploit characteristic-specific prefix returns.
Two independent interleavers usually force several independent geometric
alignments before the final word becomes short. The number of fixed-weight
messages grows polynomially, so enough inverse powers of `N` suppress the
event at a fixed linear threshold.

These events explain most exact samples with `d_3<d_2`. They affect tail
exponents and concrete failure probabilities. They need not change the
limiting relative distance.

### 8.4 Opposite-valued messages without random scalings

Over every odd field, the message `e_u-e_v` can pair repeated copies with
opposite signs. Characteristic two uses `e_u+e_v` for the same support pattern.
Recent large-field RAA analysis proves that this creates a no-scaling
low-distance event of order `N^(2-r)` for repetition factor `r`. The mechanism
is field-independent at the support level, although its values differ.

This is a real reason that removing diagonal scalings worsens the sparse tail.
It is not, by itself, a counterexample to one-sided relative-distance
inheritance.

### 8.5 Linear-support torsion cores

The most serious unresolved mechanism uses messages and deletion sets of
linear size. After peeling all degree-one constraints, a residual core can
have several path families. Their signed count can vanish modulo `p` but not
modulo two, creating `p`-torsion on exponentially many candidate supports.

Fixed-support rank universality does not control this event. Minimum distance
chooses the support after seeing the encoder. A proof needs a resilient rank
statement uniform over all deletions below a threshold.

No experiment has isolated a linear-support torsion core that lowers the odd
relative distance below the binary relative distance.

## 9. Why the successful cases succeed

Three effects favor the odd field.

First, many binary minimum words are characteristic-two dependencies. Their
maximal minors are even but nonzero over the integers. These words disappear
over `Q` and usually disappear over odd fields.

Second, a nonbinary accumulator has more nonzero states. Repeated equal
increments alternate between zero and nonzero over `F_2`, but traverse several
nonzero states over `F_p`. Unless the input values are chosen to cancel, the
odd-field output is heavier.

Third, a `K`-dimensional prime-field code contains `p^K` messages, but its
random-code benchmark is also the `p`-ary benchmark. At the same symbol rate,
the `p`-ary Gilbert--Varshamov relative distance exceeds the binary value.
Binary coefficients do not force prime-field messages to behave like binary
messages.

The rational comparison exposes the first effect cleanly. For every fixed
integer encoder, only finitely many primes can differ from its rational column
matroid. In the length-six smoke test, `F_101` and `F_103` match `Q` in all ten
samples. Fixed small primes remain special as `N` grows because new minors
introduce new divisibility events.

## 10. Why random permutations do not replace random scalings

A permutation makes the positions of a fixed value composition uniform. It
does not change the composition. If a message produces `w` copies of `a` and
`w` copies of `-a`, every permutation preserves those multiplicities.

A random nonzero diagonal changes both positions and active values. In the
weighted nonbinary RMA analysis, this distinction is exactly what permits a
scalar input-output weight enumerator. Without the weighter, the analysis must
retain a vector enumerator indexed by all nonzero symbol multiplicities.

Rosnes and Graell i Amat make this reduction explicit: their random weighter
turns the output composition into a uniform composition conditional on Hamming
weight. See [arXiv:1101.5966, Theorem 1 and equations (1)--(7)]
(https://arxiv.org/html/1101.5966v1#S2.SS1).

This is the main analytic cost of the addition-only proposal. Position
randomness survives. Value randomness does not.

## 11. Relation to existing rank results

Sparse random-matrix rank can be field-universal at first order. Coja-Oghlan,
Erguer, Gao, Hetterich, and Rolvien derive the same limiting rank formula over
an arbitrary field for broad prescribed-degree models. Their result is driven
by the random graph core and permits arbitrary nonzero coefficients in the
stated model. See [The rank of sparse random matrices]
(https://arxiv.org/abs/1906.05757).

That theorem does not settle the present question. It concerns the rank of the
sampled matrix, not the minimum rank after an adversary deletes a prescribed
fraction of columns. The desired statement is a resilient, extremal version of
field universality.

The closest useful analogy is:

```text
ordinary rank universality
    + uniform resilience under column deletion
    + control of short torsion relations
    => one-sided distance inheritance.
```

The second and third inputs are currently missing.

## 12. Literature constraints on the conjecture

Binary RAA with at least two accumulators has linear distance for suitable
repetition factors. The standard uniform-interleaver analysis is developed in
[Kliewer et al., arXiv:0810.3422](https://arxiv.org/abs/0810.3422).

Weighted nonbinary RMA also has linear distance, but every encoder is followed
by a random nonzero weighter. Its distance growth depends on the field and is
not asserted to equal the binary growth rate. See
[Rosnes and Graell i Amat, arXiv:1101.5966]
(https://arxiv.org/abs/1101.5966).

The recent large-field RAA tail analysis treats
`R P_1 A P_2 A` with random diagonal scalings. It proves that large-field and
binary low-distance tails have genuinely different polynomial exponents. It
also proves a larger `N^(2-r)` obstruction when the scalings are removed. See
[Khabbazian, Asymptotically Optimal Distance-Tail Bounds for Large-Field RAA
Codes](https://sites.ualberta.ca/~mkhabbaz/assets/pdf/RAA.pdf), especially
Theorem 4.

These results support addition-only encoding as a serious separate ensemble.
They do not support exact binary/prime-field equality.

## 13. Revised conjecture hierarchy

### Conjecture A: one-sided relative-distance inheritance

Fix an odd prime `p`, a field-stable local binary code, and the standard
two-interleaver schedule. For every `epsilon>0`,

```text
Pr[d_p < d_2 - epsilon N] -> 0.
```

This is the main target.

### Conjecture B: strict odd-field improvement

For some local codes and primes, both normalized distances converge in
probability and

```text
lim d_p/N > lim d_2/N.
```

The repetition-three experiments support this statement for `p=3` and `p=5`.
They are far too small to establish it.

### Conjecture C: resilient full-rank inheritance

For an integer `t`, define

```text
r_F,N(t) := min_{|D|=t} rank_F(G_{not D}).
```

A useful theorem would compare the profiles exactly below the binary cogirth.
For every fixed `epsilon>0`, with probability tending to one,

```text
for every t <= d_2-epsilon N,
r_p,N(t) = r_2,N(t) = K.
```

This statement is a uniform absence of `p`-torsion deletion sets. A normalized
rank comparison is insufficient because a rank loss of one already creates a
codeword.

Two-sided profile equality is not expected.

## 14. Proof program

1. Prove Lemma 1 and the deletion-set equivalences in the eventual Lean
   development for this project.
2. Express every retained-column rank loss as torsion in a backward sparse
   constraint matrix. Separate the peelable part from its residual core.
3. Prove that peeling creates unit minors. All field dependence must then live
   in the core.
4. Classify fixed-message sparse failures by value composition and accumulator
   return trace. Show that every deficit of order `epsilon N` costs enough
   independent placement constraints.
5. Develop a vector-composition enumerator for the unweighted nonbinary
   accumulator. Do not import the scalar weighted enumerator silently.
6. For linear message weights, derive a type-based first moment that retains
   symbol composition. Compare its exponent with the binary exponent.
7. Seek a uniform bound on `p`-torsion cores below the binary distance. A
   successful bound would prove Conjecture A.
8. In parallel, search for a planted residual core whose entropy exceeds its
   placement cost. Such a construction would refute Conjecture A.

## 15. Experimental next steps

The harness now records message composition, exact support spectra,
Smith-theoretic witness certificates, circuit traces, graph fingerprints,
exact top divisors under a work budget, and both dense and sparse unit-equation
cores. The first three items in the earlier experimental plan are now partly
or fully implemented. The next work should use these diagnostics rather than
add more undirected distance tables.

1. Turn the exact repetition-three span-two count into a uniform bound over all
   sparse message supports.
2. Relate trace and core fingerprints to exact embedding counts for further
   recurrent trap families.
3. Reproduce the ternary interval certificate independently and connect its
   `0.19` bulk bound to a proved binary threshold.
4. Extend exact message enumeration with branch-and-bound for dimensions above
   16.
5. Compare `PAPA`, `PAPAPA`, and sign-only diagonal variants at the same
   lengths and seeds.
6. Search local binary codes whose represented matroid is stable over
   `F_2,F_3,F_5,Q` but whose accumulated core has frequent odd torsion.

The decisive negative result would be a family of linear-support witnesses
with `d_p <= d_2-epsilon N` and nonvanishing probability. The decisive positive
result would be a uniform torsion-free bound for all deletion sets below the
binary threshold.

## 16. Reproducibility and limitations

The versioned harness lives in `src/main.rs`. Raw CSV files under `results/`
are intentionally ignored. Every row contains the sample seed, permutations,
minimum message, and minimum codeword.

The principal scaling command was

```text
cargo run --release -- \
  --local examples/repetition3.matrix \
  --blocks 12 \
  --schedule PAPA \
  --samples 100 \
  --primes 3 \
  --distance-only \
  --no-rational \
  --classify-witnesses \
  --support-spectrum \
  --max-minors 0 \
  --max-n 60 \
  --output results/rep3-papa-b12-s100-classified.csv
```

All reported distances are exact for the sampled encoders. Sampling error is
substantial for the larger lengths. The experiments do not estimate an
asymptotic threshold, prove concentration, or justify extrapolation from
length 36. The two length-48 samples are witnesses, not statistics.
