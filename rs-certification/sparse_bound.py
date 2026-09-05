"""Exact rational evaluation of the candidate sparse-slab bound.

See SPARSE_BOUND.md for the derivation. This covers t<=T, not the full code.
"""
from fractions import Fraction as F
from math import comb, log2
import json


def geom(x):
    return x / (1-x)


def squares(x):
    return x * (1+x) / (1-x)**3


def geom_tail(x, start):
    return x**start / (1-x)


def squares_tail(x, start):
    return x**start * (F(start*start)/(1-x) +
                       F(2*start)*x/(1-x)**2 + x*(1+x)/(1-x)**3)


def sparse_bound(n, m, sigma, delta, q, T=None, split=32, z=None):
    assert n % m == 0
    h = m-sigma
    d = h+1
    B = n//m
    T = n//1000 if T is None else T
    assert 1 <= T <= B
    split = min(split, T+1)
    assert split >= 2
    z = (F(1,10) if delta > F(1,2) else F(1,5)) if z is None else z
    Q = q-1
    rho = delta*(1+z)**2
    assert 0 < rho < 1
    D = 1-delta-F(m*T,n)
    assert D > 0
    C = F(m*n, n-m*T+1)*rho/(1-rho)
    Kmax = C*F(T,n)
    Kmin = F(d,n-d+1)*rho/(1-rho)
    V = 1/(delta*D*z*z)
    assert Kmax < 1 and Kmin*Q > 1 and V < Q and n < Q
    assert q-1 >= 2*n  # implies P_r <= 2 Q^(r-1), for r<=n
    W = max(F(1), Kmax*V)
    support_count = sum(comb(m,s) for s in range(d,m+1))
    M = support_count * 2**m
    base = F(3,m)*M*C**h*W**sigma
    xsmall = base*F(split-1,n)**(h-1)
    xlarge = base*F(T,n)**(h-1)
    assert 0 <= xsmall <= xlarge < 1
    S0 = geom(xsmall)
    S2 = squares(xsmall)
    if split <= T:
        S0 += geom_tail(xlarge, split)
        S2 += squares_tail(xlarge, split)
    coeff2 = 2*(1/(1-Kmax) + 1/(Kmin*Q-1))*C*sigma/n
    coeff0 = F(2,Q)/((1-1/(Kmin*Q))*(1-V/Q))
    low_b = coeff2*S2 + coeff0*S0
    high_base = F(3*B*M*n**m, Q**h)
    assert high_base < 1
    high_b = F(2*n,Q**2)/(1-F(n,Q))*geom(high_base)
    return {'bound': low_b+high_b, 'low_b': low_b, 'high_b': high_b,
            'T': T, 'split': split, 'C': C, 'Kmax': Kmax,
            'V': V, 'xsmall': xsmall, 'xlarge': xlarge,
            'base': base, 'coeff2': coeff2, 'coeff0': coeff0}


def log2frac(x):
    return log2(x.numerator)-log2(x.denominator) if x else float('-inf')


if __name__ == '__main__':
    out = []
    for sigma,delta in [(4,F(3,5)),(8,F(3,10))]:
        n = 2**20 * 16//sigma
        r = sparse_bound(n,16,sigma,delta,2**128)
        out.append({'m':16,'sigma':sigma,'n':n,'q':str(2**128),
                    'delta':str(delta),'T':r['T'],'split':r['split'],
                    'log2_bound_diagnostic':log2frac(r['bound']),
                    'log2_low_b_diagnostic':log2frac(r['low_b']),
                    'log2_high_b_diagnostic':log2frac(r['high_b']),
                    'xlarge_diagnostic':float(r['xlarge']),
                    'proved_rational_comparison_below_2^-40':r['bound']<F(1,2**40),
                    'scope':'sparse slab only; derivation requires independent review'})
    print(json.dumps(out,indent=2))
