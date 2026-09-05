# Finite bound for a slab of sparse supports

This derivation covers all supports touching at most T blocks and all zero-return
counts. BULK_BOUND.md covers t>T. Together they give the complete bound checked
by verify_complete.py. The new reductions have not yet had coauthor review.

Let the local code be MDS with parameters [m,sigma,d], let h0=m-sigma=d-1,
and let n=mB. Fix an exact support meeting t blocks. Its size s and shortening
dimension r satisfy

    dt <= s <= mt,  r=s-h0*t,  1 <= r <= sigma*t.

Write Q=q-1 and assume Q>=2n. For r<=n, the projective count satisfies

    P_r=(q^r-1)/(q-1) <= 2 Q^(r-1).

Indeed P_r <= Q^(r-1)(1+1/Q)^r, and the last factor is at most
sum_{k>=0}(r/Q)^k <=2. Consequently

    chi_q(r,c) <= 2 min(1,Q^(r-1-c)).

For a uniform result over q>=q_min, use Q=q_min-1 throughout the envelope.
Indeed min(1,Q^e) is constant for e>=0 and decreases with Q for e<0.
The preceding bound at the actual field size is therefore at most the
same envelope at q_min. The complete verifier uses q_min=2^127.

Let L=floor(delta*n), choose z>0 with rho=delta(1+z)^2<1, and suppose
delta+mT/n<1. Put

    D=1-delta-mT/n,
    V=1/(delta*D*z^2),
    K_s=[s/(n-s+1)]*[rho/(1-rho)].

We require K_s<1, K_s Q>1, and V<Q. All these conditions are checked in the
script through uniform extrema over d<=s<=mT.

## 1. Sum the second output weight

For a fixed intermediate weight H and b returns, the placement formula and
the hockey-stick identity give

    sum_{j=1}^L G_n(H,j,b)
    <= binom(H-b,b) binom(n,b) binom(L,H-b) / binom(n,H).

Impossible terms are zero. For a possible term with b<=s, H<=L+b<=L+s, so
(n-H+1)/n >= D. The binomial ratio implies

    sum_{j=1}^L G_n(H,j,b)
    <= binom(H,b)^2 D^(-b) delta^(H-b)
    <= rho^H V^b.

The second inequality uses binom(H,b) <= (1+z)^H/z^b. For the first,
binom(L,H-b)/binom(n,H-b) <= (L/n)^(H-b) <=delta^(H-b), and

    binom(n,b) binom(n,H-b)/binom(n,H)
    <= binom(H,b) [n/(n-H+1)]^b.

## 2. Sum the intermediate weight

For a feasible first return count a,

    G_n(s,H,a)
    <= 2^s [s/(n-s+1)]^(s-a) binom(H-1,s-a-1).

Here binom(s-a,a)<=2^s, binom(n-H,a)<=binom(n,a), and
binom(n,a)/binom(n,s)<=[s/(n-s+1)]^(s-a).
Summing the product of the preceding two bounds over H and extending the
positive series to infinity gives

    sum_H G_n(s,H,a) sum_{j<=L}G_n(H,j,b)
    <= 2^s K_s^(s-a) V^b                  (b<=s),

because sum_{H>=k}binom(H-1,k-1)rho^H=[rho/(1-rho)]^k.
Only the feasible H terms need the preceding bound; adding the other terms of
the positive series is valid.

## 3. Include every cancellation layer for b<=s

Write K=K_s and A=h0*t+1. For b<r put e=a+b-r+1. Then

    K^(s-a)V^b = K^A (KV)^b K^(-e).

Extending the feasible e range to all integers gives

    sum_a K^(s-a)V^b chi_q(r,a+b)
    <= 2 K^A (KV)^b [1/(1-K) + 1/(KQ-1)].

The first geometric series covers e<=0; the second covers e>=1.
For b>=r, every term is in the surplus region. Extending the a and b ranges
to nonnegative integers gives the additional bound

    2 K^s V^r / [Q(1-1/(KQ))(1-V/Q)].

Thus a bound for all b<=s is 2^s times

    2 K^(h0*t+1) [1/(1-K)+1/(KQ-1)] sum_{b=0}^{r-1}(KV)^b
    + 2 K^(h0*t) (KV)^r / [Q(1-1/(KQ))(1-V/Q)].

No surplus-return layer within this range is omitted.

## 4. The remaining b>s tail

For every H,b, sum_{j<=L}G_n(H,j,b) <=binom(H-b,b)<=binom(n,b).
Also sum_H G_n(s,H,a)=binom(s-a,a); summing these trace counts over a is at
most 2^s. Since b>s>=r, the untruncated projective bound yields

    tail_{b>s}
    <= 2^(s+1) Q^(r-1) (n/Q)^(s+1)/(1-n/Q)
     = 2^(s+1) n^(s+1) Q^(-h0*t-2)/(1-n/Q).

This bound includes every intermediate and final weight in the tail.

## 5. Count supports and sum all t<=T

Set U=sum_{l=d}^m binom(m,l). At most binom(B,t) U^t supports touch t
blocks. For the present large field this counts all feasible local supports;
even if it includes additional candidates, it remains a valid upper bound.
Use binom(B,t)<=(3B/t)^t, replacing e by the rational upper bound 3.

Define

    C=[mn/(n-mT+1)] rho/(1-rho),
    Kmax=C*T/n,  Kmin=[d/(n-d+1)]rho/(1-rho),
    W=max(1,Kmax V),  M=2^m U,
    J=(3/m) M C^h0 W^sigma,
    x(t)=J(t/n)^(h0-1).

The part with b<=s, after summing over supports at a fixed t, is at most

    [c2*t^2+c0] x(t)^t,

where

    c2=2[1/(1-Kmax)+1/(Kmin Q-1)] C sigma/n,
    c0=2/[Q(1-1/(Kmin Q))(1-V/Q)].

This follows from K_s<=Ct/n, r<=sigma*t, and
sum_{b=0}^{r-1}(K_s V)^b <=sigma*t*W^(sigma*t).

Choose an integer A0 between 2 and T+1. Bound x(t) by x(A0-1) below A0 and
by x(T) at or above A0. If x(T)<1, the resulting sums are rational geometric
series and their second derivatives. The script evaluates both parts exactly
using fractions. Extending each finite t interval to its infinite geometric
tail only increases the bound.

The b>s contribution is at most

    [2n/(Q^2(1-n/Q))] * sum_{t>=1} Y^t,
    Y=3 B M n^m / Q^h0,

using 1/t<=1. The script checks Y<1 and evaluates this series exactly.

For n=2^22, RS[16,4,13], delta=3/5, q_min=2^127, T=floor(n/1000),
z=1/10, A0=32, the bound is approximately 2^-95.93.
For n=2^21, RS[16,8,9], delta=3/10, z=1/5, the analogous value is
approximately 2^-63.65. These are sparse-slab bounds only, not complete code
failure bounds. The comparison with 2^-40 is exact rational arithmetic;
decimal logarithms are diagnostics.
