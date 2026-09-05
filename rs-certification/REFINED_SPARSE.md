# Tightening the sparse-support calculation

At the fixed lengths and distances in README.md, the earlier sparse bound lost
most of its precision by applying one envelope to t=1,...,31 active blocks.
This calculation isolates t=1, evaluates its important traces directly, and
retains proved bounds for all remaining terms. The bulk certificate is unchanged.

Let m=16, sigma in {4,8}, h0=m-sigma, d=h0+1, B=n/m, and
L=floor(delta*n). For an exact support touching t blocks and having size s,
write r=s-h0*t for its shortening dimension. The number of such supports is

    binom(B,t) [z^s] (sum_{ell=d}^m binom(m,ell) z^ell)^t.

The coefficient is computed by integer convolution. For Reed--Solomon codes,
every indicated local support is possible: the polynomial vanishing exactly
at its complement has degree m-ell<=sigma-1 and is nonzero on that support.

The trace kernel, with impossible binomials interpreted as zero, is

    G_n(s,H,a) = binom(s-a,a) binom(n-H,a)
                binom(H-1,s-a-1) / binom(n,s).

The master summand for this support is the sum of
G_n(s,H,a) G_n(H,j,b) chi_q(r,a+b) over a,H,b and 1<=j<=L.
All probability assumptions are those in README.md.

## Exact summation over the final weight

For H>=1 and 0<=b<=floor(H/2),

    sum_{j=1}^L G_n(H,j,b)
    = binom(H-b,b)/binom(n,H)
      * sum_{v=0}^b binom(L,H-v) binom(n-L,v).                 (1)

To see this, select H positions from {1,...,n}. Counting sets whose
(H-b)-th smallest position is j gives
binom(j-1,H-b-1) binom(n-j,b). That order statistic is at most L precisely
when at most b selected positions lie above L. Counting by that number v
gives the right side of (1), before its common prefactor.

For t=1, direct_sparse.py evaluates (1) and the first kernel for every
1<=H<=512, 0<=a<=floor(s/2), and 0<=b<r. Integer binomial coefficients
are exact; Arb intervals enclose the divisions and additions. There is no
floating-point truncation rule.

For uniformity over q>=q_min=2^127, the cancellation factor can be bounded
by its value at q_min. It is exactly 1 when c=a+b<r. When c>=r, putting
Q=q-1 gives

    chi_q(r,c) = sum_{j=1}^r binom(r,j) Q^(j-1-c).

All powers are negative, so this expression decreases with q. The script
uses the exact rational value (q_min^r-1)/(q_min-1)^(c+1).

## A finite formula for an infinite positive series

Set k=s-a and, for 0<u<1, define

    R_{k,b}(u) = sum_{H>=1} binom(H-1,k-1) binom(H,2b) u^H.

For k>=1 and b>=0 this equals the following finite rational expression:

    sum_{i=0}^{min(k,2b)}
       [k/(k+2b-i)] * [(k+2b-i)!/(i!(k-i)!(2b-i)!)]
       * [u/(1-u)]^(k+2b-i).                                (2)

Proof: use binom(H-1,k-1)=k binom(H,k)/H and count a pair of subsets of
sizes k and 2b by their intersection size i. Their union has size N=k+2b-i.
Finally binom(H,N)/H=binom(H-1,N-1)/N, whose generating series is
[u/(1-u)]^N/N. The tests independently check (2) with forward differences.

## Every intermediate weight above 512

In this paragraph let delta0=L/n and D_s=1-delta0-s/n>0. The inequalities
from SPARSE_BOUND.md, keeping the exact first trace coefficient, imply for b<=s

    G_n(s,H,a) sum_{j<=L} G_n(H,j,b)
    <= A_{s,a,b} binom(H-1,s-a-1) binom(H,2b) delta0^H,

where

    A_{s,a,b} = [binom(s-a,a) binom(n,a)/binom(n,s)]
                * binom(2b,b)/(D_s*delta0)^b.

Here binom(H-b,b) binom(H,b)=binom(2b,b) binom(H,2b).
Choose u=(1+delta0)/2. For H>M=512,

    delta0^H <= (delta0/u)^(M+1) u^H.

Thus the sum over every H>M is at most

    A_{s,a,b} (delta0/u)^(M+1) R_{s-a,b}(u).                  (3)

The script multiplies (3) by the appropriate cancellation factor and sums
every a and b<r. Formula (3) is a proved upper bound on the entire omitted
tail. It is evaluated with rational arithmetic before conversion to Arb.

## The other sparse traces and supports

For t=1 and r<=b<=s, use the geometric bound in SPARSE_BOUND.md, section 3,
with each support's own K_s and D_s. For b>s, use that document's section 4
and its aggregate support count from section 5, for every 1<=t<=T.

For 2<=t<=T=floor(n/1000), section 5 already gives the all-b<=s bound

    (c2*t^2+c0) x(t)^t,   x(t)=J(t/n)^(h0-1).

Use the disjoint slabs [2,3], [4,7], [8,15], [16,31], [32,63], and [64,T].
For a slab [A,E], monotonicity gives x(t)<=x(E)<1. Its contribution is at most

    c2 sum_{t>=A} t^2 x(E)^t + c0 sum_{t>=A} x(E)^t.

Both sums have closed rational formulas. Extending each slab to infinity
adds nonnegative terms. In particular the final slab covers every growing
sublinear block count up to T; the existing bulk certificate covers t>T.

refined_sparse.py also provides a simpler independent upper calculation:
use (2) with u=delta0 for all H and b<r, instead of the finite sum and (3).
This gives about 209.599 and 137.567 bits. The exact finite sum improves
these to about 209.601 and 137.568 bits.

## Result and the limit of further numerical tightening

At k=2^20 and q>=2^127 the full verified upper bounds have logarithms

    RS[16,4,13], delta=3/5:   -209.6008601171179...
    RS[16,8,9],  delta=3/10:  -137.5682486336414...

The verifier checks the safely rounded bounds 2^-209.60 and 2^-137.56
directly using intervals. These also imply the weaker RS probabilities
previously displayed in the manuscript. RA configurations are not covered.

The directly summed t=1, H<=512, a+b<r terms are exact nonnegative terms
of the support-restricted master sum, since their projective factor is 1.
They therefore also give a lower bound on that sum, for every q in scope.
The ratio of the complete upper bound to this partial sum is less than
1+5e-13 for RS[16,4,13] and 1+3.3e-7 for RS[16,8,9]. Fewer than one
millionth of a bit remains available by more accurate evaluation of this
same sum. This is a limitation of this upper-bound method, **not a lower
bound on the actual failure probability**.

The new identities and reductions have exact small-case checks and interval
verification, but have not yet been formalized in Lean or reviewed by a coauthor.
