"""Tighter rational sparse bound; see REFINED_SPARSE.md for all reductions."""
from fractions import Fraction as F
from functools import lru_cache
from math import comb, factorial
from sparse_bound import sparse_bound, geom_tail, squares_tail, log2frac


@lru_cache(maxsize=None)
def binomial_moment(k, b, delta):
    """Sum C(H-1,k-1) C(H,2b) delta^H over every H>=1."""
    assert k>=1 and b>=0 and 0<delta<1
    v=2*b
    odds=delta/(1-delta)
    result=F(0)
    for overlap in range(min(k,v)+1):
        size=k+v-overlap
        coefficient=factorial(size)//(factorial(overlap)*factorial(k-overlap)*factorial(v-overlap))
        result+=F(k*coefficient,size)*odds**size
    return result


def support_low(n, L, s, r, q, z):
    """Bound every a,H,j and 0<=b<=s for one exact support."""
    delta=F(L,n);Q=q-1
    assert 1<=r<=s<=n and Q>=2*n
    D=1-delta-F(s,n)
    rho=delta*(1+z)**2
    assert D>0 and 0<rho<1
    K=F(s,n-s+1)*rho/(1-rho)
    V=1/(delta*D*z*z)
    assert K<1 and K*Q>1 and V<Q
    projective=F(q**r-1,Q)
    no_surplus=F(0);surplus=F(0)
    denominator=comb(n,s)
    for b in range(r):
        second_factor=F(comb(2*b,b))/(D*delta)**b
        for aa in range(s//2+1):
            first_factor=F(comb(s-aa,aa)*comb(n,aa),denominator)
            term=first_factor*second_factor*binomial_moment(s-aa,b,delta)
            if aa+b<r:
                no_surplus+=term
            else:
                surplus+=term*projective/Q**(aa+b)
    # Every b>=r is surplus. The existing geometric envelope covers this
    # whole range, even after extending a,b to infinity.
    high_b=F(2**(s+1))*K**s*V**r/(Q*(1-1/(K*Q))*(1-V/Q))
    return {'no_surplus':no_surplus,'surplus':surplus,'b_ge_r':high_b}


def support_counts(m,sigma,blocks):
    """Coefficients of (sum_{ell=d}^m C(m,ell) z^ell)^blocks."""
    d=m-sigma+1
    counts={0:1}
    for _ in range(blocks):
        new={}
        for size,count in counts.items():
            for ell in range(d,m+1):
                new[size+ell]=new.get(size+ell,0)+count*comb(m,ell)
        counts=new
    return counts


def refined_sparse(n,m,sigma,delta,q,direct_blocks=1):
    baseline=sparse_bound(n,m,sigma,delta,q)
    T=baseline['T'];h=m-sigma;B=n//m;L=int(delta*n)
    assert 1<=direct_blocks<T
    z=F(1,10) if delta>F(1,2) else F(1,5)
    direct={key:F(0) for key in ('no_surplus','surplus','b_ge_r')}
    by_t=[]
    for t in range(1,direct_blocks+1):
        contribution=F(0)
        for s,count in support_counts(m,sigma,t).items():
            multiplicity=comb(B,t)*count
            values=support_low(n,L,s,s-h*t,q,z)
            for key,value in values.items():
                direct[key]+=multiplicity*value
                contribution+=multiplicity*value
        by_t.append(contribution)
    # Disjoint t slabs; extending each slab's constant-base geometric
    # series to infinity adds positive terms and is an upper bound.
    remainder=F(0);slabs=[];start=direct_blocks+1
    while start<=T:
        end=min(T,2*start-1) if start<64 else T
        x=baseline['base']*F(end,n)**(h-1)
        assert 0<x<1
        value=baseline['coeff2']*squares_tail(x,start)+baseline['coeff0']*geom_tail(x,start)
        remainder+=value
        slabs.append((start,end,value))
        start=end+1
    total=sum(direct.values(),F(0))+remainder+baseline['high_b']
    return {'bound':total,'direct':direct,'direct_by_t':by_t,'slabs':slabs,
            'remainder':remainder,'b_gt_s':baseline['high_b'],'T':T}


if __name__=='__main__':
    import json
    result=[]
    for sigma,delta in [(4,F(3,5)),(8,F(3,10))]:
        data=refined_sparse(2**20*16//sigma,16,sigma,delta,2**127)
        result.append({'sigma':sigma,'log2_bound_diagnostic':log2frac(data['bound']),
                       'direct':{key:log2frac(v) for key,v in data['direct'].items()},
                       'remainder':log2frac(data['remainder']),
                       'slabs':[(lo,hi,log2frac(v)) for lo,hi,v in data['slabs']]})
    print(json.dumps(result,indent=2))
