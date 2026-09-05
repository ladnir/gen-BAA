# Finite entropy certificate for the remaining supports

This bound covers every exact outer support touching t>T=floor(n/1000)
local blocks. SPARSE_BOUND.md covers t<=T. The field size satisfies q>=2^127,
the local code is RS[16,sigma,17-sigma], and n=16*2^20/sigma.

## Statement reduced to a finite optimization

Let B=n/m, m=16, h0=m-sigma, and d=h0+1. A support type consists of counts
k_0,k_d,...,k_m summing to B. Set p_l=k_l/B. The number of supports of this
type is exactly

    multinomial(B;k_0,k_d,...,k_m) prod_{l=d}^m binom(m,l)^k_l.

Thus it is at most exp(n S(p)), where

    S(p)=[H(p)+sum_{l=d}^m p_l log binom(m,l)]/m.

Only exact supports whose local intersections are empty or have at least d
coordinates need be included: every other exact-support bad event is empty.
This restriction follows from local minimum distance, even though the paper's
unrestricted master sum also includes those zero events with loose bounds.

Normalize the outer support s, intermediate weight H, returns a,b, and final
weight j by n, writing alpha, eta, x,y,zeta respectively. Set

    alpha=sum_l l p_l/m,
    beta=sum_l (l-h0)_+ p_l/m=r/n.

There are six binomial-entropy perspective terms f(A,B)=A H(B/A):

    f(alpha-x,x), f(1-eta,x), f(eta,alpha-x),
    f(eta-y,y),   f(1-zeta,y), f(zeta,eta-y).

Let their sum be J. Feasibility requires that every B and A-B above is
nonnegative, p is a probability vector, 0<=zeta<=delta, and

    p_0 <= 1-m(T+1)/n.

Consequently alpha>=d(T+1)/n and eta>=alpha/2. Define

    F=S(p)+J-H(alpha)-H(eta)-log(q-1)*max(0,x+y-beta).

Each support-type/trace summand of the master bound is at most

    2 (n+1)^2 exp(n F).

Proof: use binom(v,u)<=exp(v H(u/v)) in each numerator and
binom(n,u)>=exp(n H(u/n))/(n+1) in the two denominators. Replacing
binom(H-1,s-a-1) by binom(H,s-a), and the analogous final binomial, only
increases the numerators. The projective factor is at most

    2 min(1,(q-1)^(r-1-a-b))
    <=2 exp[-n log(q-1) max(0,x+y-beta)].

The factor 2 follows from q-1>=2n, as proved in SPARSE_BOUND.md. In the bulk
we intentionally discard the additional minus one in the exponent.
For a uniform result over q>=q_min, replace log(q-1) by log(q_min-1).
Its multiplier is nonpositive, so this only increases F. The complete
verifier uses q_min=2^127.

There are at most (B+1)^sigma outer support types: each of the sigma positive
local support counts is an integer from 0 to B, and k_0 is then determined.
There are at most (n+1)^4 choices of a,H,b,j. Therefore any uniform bound
F<=U on the feasible region yields the complete finite bulk bound

    2 (B+1)^sigma (n+1)^6 exp(n U).                    (BULK)

No trace, support type, or growing sublinear range in t>T is omitted.

## Why a two-dimensional covering suffices

All terms of F are concave in their arguments except -H(alpha) and -H(eta).
On an interval [l,u], convexity bounds -H(v) above by the secant joining its
endpoint values. We partition only the (alpha,eta) rectangle

    [d(T+1)/n,1] x [d(T+1)/(2n),1].

The remaining variables need no discretization. To encode the cancellation
term, introduce c with c>=x+y-beta and c>=0, and use -c log(q-1). For the
original objective choose c=max(0,x+y-beta). This choice always lies in [0,1]
because x<=alpha/2<=1/2 and y<=eta/2<=1/2.

Every binomial perspective has a global linear upper bound: for any rational
theta in (0,1),

    f(A,B) <= -B log(theta)-(A-B)log(1-theta).

For any strictly positive rational probability vector p*,

    H(p) <= -sum_l p_l log(p*_l).

Combining these inequalities with the two secants gives a linear function
of all variables which dominates F throughout the rectangle. A numerical
optimizer only chooses p*, the six theta values, and linear-program dual
multipliers. Their accuracy or optimality is not trusted.

## Independent verification of a leaf

The verifier reconstructs all inequalities and equalities with rational
coefficients. If the linear upper function is c0+c.v, the constraints are
A v<=b and E v=e, and the coordinate bounds are l<=v<=u, then ANY rational
mu>=0 and nu give

    c0+c.v
    <= c0+mu.b+nu.e
       +sum_i [r_i*l_i + max(0,r_i)*(u_i-l_i)],
    r=c-A^T mu-E^T nu.

This is the bound checked by verify_bulk.py using 192-bit Arb intervals.
The verifier normalizes the submitted rational p* exactly, checks theta and
multiplier signs, and uses log(q_min-1), not the approximate log(2^128) used
by the optimizer. The same proposed multipliers remain valid for this
smaller field-size lower bound. Residual LP errors increase the verified upper
bound instead of being ignored.

All box boundaries are recovered by exact rational bisection from the root.
The verifier checks that every internal node has both children and that no
node or leaf is missing or extraneous. Infeasible leaves are accepted only
when one of these exact necessary conditions fails:

    alpha_upper>=d(T+1)/n,
    alpha_lower<=2*eta_upper,
    eta_lower<=2*delta.

Floating-point feasibility declarations alone are not certificates.

## Current verified data

The RS[16,4,13] covering has 21 leaves. The verified bulk upper bound from
(BULK) is below 2^-1078. The RS[16,8,9] covering has 19 leaves, and its bulk
upper bound is below 2^-5634. These intentionally loose bounds use a single
worst exponent over all leaves; no optimized remainder constant is supplied.

The Python verification establishes the numerical inequalities from the
analytic reduction above. It is not a Lean formalization of that reduction.
