# Dense supports for GF128 RS followed by plain scans

The target is minimum distance at least 209716 for length n=2097152,
dimension n/2, and the existing RS[16,8,9] block code followed by two
independent uniform permutations and plain prefix sums. The statement
must hold simultaneously for every nonzero message after the permutations
are revealed. The existing complete certificate remains 6.6%. This note
proves additional counting ingredients; it does not certify 10%.

The field is F=GF(2^128) in the implementation's polynomial basis. The
local code evaluates polynomials of degree at most seven at the sixteen
field elements whose integer representations are 0,...,15. This is the
code used through `AdditivePoints` and `AdditiveRs8`; systematic message
coordinates do not change its code space. All subsequent statements about
the constituent refer to these points, not arbitrary MDS codes.

## An exact local multiplicity bound

Every weight-nine local word is a nonzero scalar multiple of

    ( product(a - z : z in Z) : a in {0,...,15} ),  |Z|=7.

An exact enumeration of all binom(16,7)=11440 choices of Z gives:

| Largest multiplicity of a nonzero value | Number of zero sets |
| --- | ---: |
| 1 | 8848 |
| 2 | 2544 |
| 3 | 48 |

Multiplying the word by a nonzero scalar preserves these multiplicities.
Thus every weight-nine word has nonzero-value multiplicity at most three.
The corresponding sums of squared multiplicities are 9, 11, and 15,
with the same counts. The enumeration uses binary polynomial products of
degree at most 21, so no modular reduction of degree 128 occurs. It checks
the actual field values exactly, without floating point or random samples.
`certificates/verify_rs_plain_dense.py` reproduces the enumeration and
hashes all the evaluated words.

For a local word with support size s, the following bounds on a nonzero
value's multiplicity will suffice:

| s | Shortening dimension d=s-8 | Multiplicity bound a(s) |
| --- | ---: | ---: |
| 9 | 1 | 3 |
| 10 | 2 | 6 |
| 11,...,15 | 3,...,7 | 7 |
| 16 | 8 | 16 |

For s<16 the polynomial has a root and is not constant. A nonzero value
therefore has at most seven preimages. For s=10, seven occurrences of a
nonzero value c would make f-c a weight-nine word. At the six zeros of f,
the word f-c takes the same nonzero value -c, contradicting the preceding
weight-nine bound. The s=16 bound permits constant polynomials. In every
case a(s)<=3d. These bounds need not be sharp for s>=10.

## Equal source forms after imposing equations

Fix a source support S, with s=|S|, and its shortened message space V_S.
Let D=dim(V_S). A nonempty block has support size s_b>=9 and shortening
dimension d_b=s_b-8; V_S is the direct sum of these block spaces. Write
f_i in V_S^* for the active coordinate forms. Let W be an r-dimensional
space of imposed zero equations and assume every f_i has nonzero image
in V_S^*/W. A branch violating that assumption can be discarded.

**Lemma.** Every class of equal nonzero images among the f_i has at most
3(r+1) labels.

**Proof.** Let T be one class and t_b its number of labels in block b.
Since |F|>s, the kernel of W contains a vector on which every active f_i
is nonzero: at most s proper hyperplanes cannot cover that kernel. On
this vector the labels in T all take the same nonzero value. The local
multiplicity bound gives t_b<=a(s_b)<=3d_b.

The shortened local code is MDS of dimension d_b. Consequently the rank
of any t_b of its coordinate forms is min(t_b,d_b). The block spaces are
independent, so the rank of all forms indexed by T is

    sum_b min(t_b,d_b).

All these forms lie in W plus the span of one representative of their
common quotient image. Their rank is therefore at most r+1. Finally,
t_b<=3 min(t_b,d_b): for t_b<=d_b this is immediate, and otherwise it
follows from the multiplicity bound. Summing proves the lemma. QED.

This lemma applies after arbitrary zero equations, not only fresh RS
inputs. It concerns the original source coordinate forms. It does not
assert the same multiplicity bound for the intermediate prefix-sum forms
used as inputs to the second scan.

## A finite rank bound for prescribed pair returns

Suppose m unexposed source labels remain and the imposed equations have
rank r. For their equal-image class sizes c_i, a uniformly sampled ordered
pair of distinct labels has equal images with probability

    sum_i c_i(c_i-1) / (m(m-1))
        <= min(1, (3r+2)/(m-1)).

In characteristic two these are exactly the free pair-return equations.
An unequal pair either adds one independent equation, or is inconsistent
with an active source coordinate remaining nonzero. The latter branch dies.

Fix a collection of j disjoint two-input returns in the first scan. The
permutation gives their labels a uniform order without replacement.
Expose these pairs before the other first-scan equations; equations can
be reordered for this proof even though the encoder's order is unchanged.
Ignore all longer-return equations for now. Starting at rank zero, use
the upper chain with

    rho(l,r) = min(1, (3r+2)/(s-2l-1))

at pair number l, counting from zero. The chain stays at r with probability
rho(l,r) and otherwise moves to r+1. Rank D is dead.

This chain stochastically underestimates actual rank until death. If the
two ranks coincide, couple the stays using the same uniform variable;
the chain's stay probability is an upper bound on the actual free-return
probability. If the ranks differ, their order cannot reverse in a step
that increases either rank by at most one. Extra actual deaths only help.
Thus it upper-bounds the expectation of every nonnegative decreasing
terminal weight, including tau^-r for tau>=1, with value zero at rank D.

Define M(s,D,j;tau) by the following exact recurrence. Initially w_0=1
and w_r=0 for r>0. At step l put

    w'_r = rho(l,r) w_r
           + [r>0] (1-rho(l,r-1)) w_(r-1)/tau,  0<=r<D.

Then M is the sum of w_r after j steps. It upper-bounds the rank-weighted
count for the prescribed pair equations, with every original active
source coordinate required to remain nonzero. The verifier implements
this recurrence over rational numbers. The separate C++ screening program
evaluates the same recurrence in floating point at full sizes; those
numerical results are not certified upper enclosures.

## Counting the first-scan patterns exactly

For s,h>=1, let c be the number of completed nonzero-output runs in the
first scan, and j the number of those runs containing exactly two active
inputs, including the input that returns the feedback to zero. An open
last run is allowed. The number of admissible input/output support-pattern
pairs of length n, input weight s, and output weight h is

    B(n,s,h,c,j) = binom(n-h,c) binom(h-1,s-c-1)
                   binom(c,j) binom(s-2c,c-j).

Out-of-range binomial coefficients are zero. To derive the formula,
choose the zero-output gaps in binom(n-h,c) ways. For fixed active counts
of the runs, summing their positive lengths and internal active placements
gives binom(h-1,s-c-1). Choose the j two-input completed runs. Every other
completed run has at least three active inputs. Counting the remaining
active allocations separately for a closed and an open ending, then using
Pascal's identity, gives the last binomial factor. The boundary case in
which every run is a completed pair gives the same expression.

The verifier checks this identity against exhaustive enumeration of the
support transition rules for n<=9. This checks a combinatorial identity;
it is not a small-code distance experiment.

## Combining this bound with the second scan

Let A(s,D) count source supports of size s and dimension D. Exactly,

    A(s,D) = [z^s u^D]
      (1 + sum(w=9..16, binom(16,w) z^w u^(w-8)))^(n/16).

Choose tau>=1 and a uniform second-scan bound K_2(h), with markers x>0
and 0<y<=1, as in the existing composition lemma. For fixed s,D,h the
bad-event probability is bounded by

    A(s,D) tau^(D-1) K_2(h)
    ------------------------------------------
      binom(n,s) binom(n,h) x^h y^e

    times sum_(c,j) B(n,s,h,c,j) M(s,D,j;tau).

The first permutation's support placements contribute 1/binom(n,s);
M averages over its distinct active labels. The second permutation remains
independent after the complete first-scan history is fixed. Apply its
uniform bound in that history's quotient. Dropping the longer first-scan
return equations only increases the rank weight and retains more branches,
so replacing the first-scan rank moment by M is valid. Sum over all
admissible s,D,h to obtain a whole-code bound. Numerical use still needs
a complete finite cover and directed rounding, as for the existing ledger.

This is a valid candidate upper bound, not a completed evaluation at 10%.

## What the investigation did and did not close

The characteristic-independent sparse-source calculation already closes
source weights through 31 at the 10% target, below 2^-47.919. Dense
supports remain unresolved.

A full-size floating rank-chain screen used s=419430, D=59117, 180000
prescribed pair returns, and tau=2. Its log2 moment was about -59743.224.
Combining this scale of bound with one previously problematic continuous
source/run profile still gave a positive diagnostic exponent (about 0.075
per output coordinate). This is neither a failure-probability lower bound
nor a proof that optimizing the new bound cannot close. It establishes
that the first tested use of the largest-class lemma is insufficient.

The chain only tracks rank and the largest class; it permits implausibly
many large equal-image classes without recording their shared origin.
For a source consisting entirely of weight-nine blocks, pair equations
have extra structure: they link the blocks' one-dimensional variables.
On a surviving connected component these variables are proportional.
A component containing b blocks has equality energy at most 15 b^2,
by the verified local energy bound and Cauchy--Schwarz. Independent
components have no equal nonzero forms. Tracking component sizes therefore
gives more information than a largest-class bound. Extending this accounting
to mixed local dimensions and carrying it through the first scan's repeated
prefix values is the next proof direction. No such full transfer or finite
10% certificate is claimed here.

For this restricted all-weight-nine source family, the component argument
gives an explicit improvement to the recurrence. With B=s/9=D active
blocks, rank r means B-r surviving connected components. Their squared
sizes sum to at most (r+1)^2+B-r-1. The full source pool's equality energy
is therefore at most 15(r(r+1)+B). Subtract its s diagonal pairs to count
ordered distinct equal pairs; deleting labels cannot increase that count.
Thus the recurrence may use the smaller cap

    min(1, (3r+2)/(m-1), (15r(r+1)+6B)/(m(m-1))).

Every consistent edge relates two nonzero block scalars. A spanning tree
gives one independent equation per edge; a cycle either adds no equation
or kills that component. This proves the rank/component relationship
used above. Both verification programs implement this restricted cap.
It must not be used for sources containing any higher-dimensional block.

A floating full-size screen of this restricted chain used s=419427,
B=D=46603, j=180000, and tau=2. It returned log2 M approximately
-94635.682. That is stronger than the largest-class calculation, but a
sampled continuous source/run profile still did not close. The recurrence
allows cycles to be consistent whenever its collision cap permits it.
In the actual minimum-word components, an edge imposes a ratio of two
fixed nonzero RS coordinates on the block scalars. Every cycle requires
the product of these ratios to equal one. Counting these consistency
conditions, including dependence and depletion when cycles overlap, is
the next concrete refinement to attempt before extending the bound to
mixed local dimensions. Merely extending the current coarse recurrence
has not been shown sufficient.

The exact ingredient checker output is saved in
`certificates/rs-plain-dense-ingredients.json`. Floating screens and their
scope are recorded in `results/rs-plain-dense-screen.json`; exploratory
scripts and checker sources are archived in
`results/rs-plain-dense-exploration-20261114.tar.gz`.

No encoder, selector, default, or benchmark setting was changed for this
proof investigation.
