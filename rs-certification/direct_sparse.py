"""Finite sparse trace sum plus a proved bound for every omitted term."""
from fractions import Fraction as F
from math import comb
from flint import arb,ctx
from refined_sparse import refined_sparse, support_counts, binomial_moment
from verify_bulk import a


def second_tail_table(n,L,maximum,bmax):
    """Exact order-statistic identity, evaluated with enclosing Arb intervals."""
    left=[comb(L,i) if i<=L else 0 for i in range(maximum+1)]
    right=[comb(n-L,i) if i<=n-L else 0 for i in range(bmax+1)]
    result={}
    for H in range(1,maximum+1):
        denominator=comb(n,H)
        for b in range(min(bmax,H//2)+1):
            numerator=comb(H-b,b)*sum(left[H-j]*right[j] for j in range(b+1))
            result[H,b]=arb(numerator)/arb(denominator)
    return result


def support_direct(n,L,s,r,q,maximum,second):
    """Cover b<r, with exact finite H and an infinite positive-series tail."""
    Q=q-1;delta=F(L,n);D=1-delta-F(s,n)
    assert 1<=r<=s<=n and D>0 and 0<maximum<=n and Q>=2*n
    projective=F(q**r-1,Q)
    no_surplus=arb(0);surplus=arb(0);tail=F(0)
    denominator=comb(n,s)
    tilt=(1+delta)/2
    for b in range(r):
        for aa in range(s//2+1):
            cancellation=F(1) if aa+b<r else projective/Q**(aa+b)
            trace=arb(0)
            for H in range(max(s-aa,2*b,1),maximum+1):
                numerator=comb(s-aa,aa)*comb(n-H,aa)*comb(H-1,s-aa-1) if aa<=n-H else 0
                trace+=arb(numerator)/arb(denominator)*second[H,b]
            if aa+b<r:
                no_surplus+=trace
            else:
                surplus+=trace*a(cancellation)
            if maximum<n:
                first=F(comb(s-aa,aa)*comb(n,aa),denominator)
                second_factor=F(comb(2*b,b))/(D*delta)**b
                tail+=cancellation*first*second_factor*(delta/tilt)**(maximum+1)*binomial_moment(s-aa,b,tilt)
    return {'no_surplus':no_surplus,'surplus':surplus,'H_tail':a(tail)}


def direct_sparse(n,m,sigma,delta,q,maximum=512,direct_blocks=1,precision=192):
    ctx.prec=precision
    bounds=refined_sparse(n,m,sigma,delta,q,direct_blocks=direct_blocks)
    L=int(delta*n);h=m-sigma;B=n//m
    assert maximum<=n
    second=second_tail_table(n,L,maximum,sigma*direct_blocks-1)
    direct={key:arb(0) for key in ('no_surplus','surplus','H_tail')}
    for t in range(1,direct_blocks+1):
        for s,count in support_counts(m,sigma,t).items():
            multiplicity=comb(B,t)*count
            values=support_direct(n,L,s,s-h*t,q,maximum,second)
            for key,value in values.items():direct[key]+=multiplicity*value
    # Replace only b<r in the refined bound. All other regions remain covered.
    direct['b_ge_r']=a(bounds['direct']['b_ge_r'])
    direct['remaining_sparse']=a(bounds['remainder'])
    direct['b_gt_s']=a(bounds['b_gt_s'])
    return {'bound':sum(direct.values(),arb(0)),'parts':direct,'T':bounds['T'],
            'maximum_intermediate_weight':maximum,'direct_blocks':direct_blocks}


if __name__=='__main__':
    import json
    result=[]
    for sigma,delta in [(4,F(3,5)),(8,F(3,10))]:
        data=direct_sparse(2**20*16//sigma,16,sigma,delta,2**127)
        result.append({'sigma':sigma,'log2_bound_upper':str((data['bound'].log()/arb(2).log()).upper()),
                       'parts':{key:str((v.log()/arb(2).log()).upper()) for key,v in data['parts'].items()}})
    print(json.dumps(result,indent=2))
