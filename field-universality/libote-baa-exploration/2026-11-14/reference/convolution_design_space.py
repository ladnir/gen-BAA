"""Matched numerical dense analysis for widths one/two and MDS local blocks.

Exploration only: no sparse sum, finite prefactors, or global certificate.
"""
import argparse
import json
from fractions import Fraction as F
from functools import lru_cache
from math import comb, log
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, xlogy
from convolution_quotient import row_polynomials

ROOT=Path(__file__).parent/'results'
LN2=log(2)


def rows(width,q):
    if width=='projective_pairs':
        # Proportional nonopposite inputs kill the zero branch. Counting
        # all projective lines bounds free opposite pairs and dead pairs
        # together; the finite m/(m-1) correction is outside this transfer.
        assert q==1
        return [[[(F(1),0,0,0,0),(F(1),1,1,0,1)]],
                [[(F(1),0,1,0,1),(F(1),1,1,0,2),(F(1),1,0,-1,0)],
                 [(F(1),0,1,0,1),(F(1,2),1,1,0,2),(F(1,2),1,0,0,0)]],
                [[(F(1),0,1,0,2),(F(1),1,1,0,2),(F(1),1,0,-1,0)],
                 [(F(1),0,1,0,2),(F(1),1,0,0,0)]]]
    if width=='fresh_pairs':
        # A macro bound at the first two active inputs of a run. It is not
        # a conditional transition after exposing just the first label.
        # A finite sampling-without-replacement factor is charged outside.
        assert q==1
        return [[[(F(1),0,0,0,0),(F(1),1,1,0,1)]],
                [[(F(1),0,1,0,1),(F(1),1,1,0,2),(F(1),1,0,-1,0)],
                 [(F(1),0,1,0,1),(F(1,2),1,1,0,2),
                  (F(1,2),1,0,0,0),(F(1,2),1,0,-1,0)]],
                [[(F(1),0,1,0,2),(F(1),1,1,0,2),(F(1),1,0,-1,0)],
                 [(F(1),0,1,0,2),(F(1),1,0,0,0)]]]
    if width=='pair_returns':
        # Three support states: zero; one active input in the current run;
        # at least two active inputs. A two-input closed run gains the
        # rank factor (1+1/tau)/2, with the finite matching factor outside.
        assert q==1
        return [[[(F(1),0,0,0,0),(F(1),1,1,0,1)]],
                [[(F(1),0,1,0,1),(F(1),1,1,0,2),
                  (F(1,2),1,0,0,0),(F(1,2),1,0,-1,0)]],
                [[(F(1),0,1,0,2),(F(1),1,1,0,2),(F(1),1,0,0,0)]]]
    if width==3:
        from convolution_matroid import row_polynomials as matroid_rows
        return matroid_rows(width,q)
    if width==2:return row_polynomials(q)
    if isinstance(width,int) and width>=4:
        from convolution_rank_mask_exact import exact_rows
        return exact_rows(width,q)
    assert width==1
    # Terms: coefficient, input marker, output marker, rank marker, target.
    idle=(F(1),0,0,0,0); emit=(F(1),0,1,0,1)
    return [[[idle,(F(1),1,1,0,1)]],
            [[emit,(F(1),1,1,0,1),(F(1),1,0,-1,0)],
             [emit,(F(q-1,q),1,1,0,1),(F(1,q),1,0,0,0)]]]


@lru_cache(None)
def source_coefficients(s,r=4,n=1<<21):
    """Exact coefficient of z^s in Q_r(z,tau)^(n/(2r))."""
    assert n%(2*r)==0
    blocks=n//(2*r);table=[{0:1}]
    for weight in range(1,s+1):
        total={}
        for d in range(r+1,2*r+1):
            if d>weight:continue
            for dimension,value in table[weight-d].items():
                key=dimension+d-r
                total[key]=total.get(key,0)+((blocks+1)*d-weight)*comb(2*r,d)*value
        assert all(c%weight==0 for c in total.values())
        table.append({d:c//weight for d,c in total.items() if c})
    return table[s]


@lru_cache(None)
def constraints(width,q):
    if width=='feedback_pairs':
        assert q==1
        from all_one_feedback_pairs import timed_rows
        rr=timed_rows()
    else:rr=rows(width,q)
    states=len(rr); stride=states+2; size=2+2*stride
    terms=[]
    for base in (2,2+stride):
        for row,choices in enumerate(rr):
            for poly in choices:
                vectors=[];coeff=[]
                for term in poly:
                    c,x,y,t,j=term[:5];duration=term[5] if len(term)==6 else 1
                    if not c:
                        continue
                    v=np.zeros(size);v[1]=t;v[base]=x;v[base+1]=y
                    if j:v[base+1+j]+=1
                    if row:v[base+1+row]-=1
                    v[base+states+1]-=duration
                    vectors.append(v);coeff.append(log(float(c)))
                terms.append((vectors,coeff))
    count=len(terms);length=max(len(t[0]) for t in terms)
    vectors=np.zeros((count,length,size));coeff=np.full((count,length),-np.inf)
    for i,(v,c) in enumerate(terms):vectors[i,:len(v)]=v;coeff[i,:len(c)]=c
    order=[]
    if width in (2,3):
        from convolution_matroid import order_pairs
        for base in (2,2+stride):
            for high,low in order_pairs(width):
                v=np.zeros(size)
                if high:v[base+1+high]=1
                if low:v[base+1+low]-=1
                order.append(v)
    elif isinstance(width,int) and width>=4:
        from convolution_rank_mask import layout
        L=layout(width)
        for base in (2,2+stride):
            for high,(mask,r) in enumerate(zip(L['mask'],L['rank'])):
                if r>1:
                    low=int(L['lookup'][mask,r-1]);v=np.zeros(size)
                    v[base+1+high]=1;v[base+1+low]=-1;order.append(v)
    return states,stride,size,vectors,coeff,np.array(order).reshape((-1,size))


def propose(width,q,r,delta,a,b,start=None,exact_s=None,n=1<<21,source_terms=None):
    states,stride,size,vectors,coeff,order=constraints(width,q)
    bases=(2,2+stride)
    degrees=np.array([0]+list(range(r+1,2*r+1)))
    dims=np.array([0]+list(range(1,r+1)))
    source_logs=np.array([0]+[log(comb(2*r,d)) for d in range(r+1,2*r+1)])
    divisor=2*r
    if source_terms is not None:
        assert exact_s is None
        degrees=np.array([0]+[v['weight'] for v in source_terms])
        dims=np.array([0]+[v['dimension'] for v in source_terms])
        source_logs=np.array([0]+[log(v['count']) for v in source_terms])
    if exact_s is not None:
        source=source_coefficients(exact_s,r,n)
        assert source and abs(a*n-exact_s)<1e-6
        degrees=np.zeros(len(source));dims=np.array(list(source))
        source_logs=np.array([log(c) for c in source.values()]);divisor=n
    linear=np.zeros(size);linear[0]=-a
    linear[bases[0]]=-a;linear[bases[0]+1]=-b
    linear[bases[1]]=-b;linear[bases[1]+1]=-delta
    for base in bases:linear[base+states+1]=1
    if exact_s is not None:linear[0]=0;linear[1]=-1/n
    entropy=sum(xlogy(t,t) for t in (a,1-a,b,1-b))
    def objective(point):
        vals=source_logs+degrees*point[0]+dims*point[1]
        total=logsumexp(vals);prob=np.exp(vals-total)
        grad=linear.copy();grad[0]+=prob@degrees/divisor;grad[1]+=prob@dims/divisor
        value=total/divisor+linear@point+entropy
        if exact_s is not None:
            for base in bases:
                tail=np.r_[0.,-point[base+2:base+states+1]];tail_sum=logsumexp(tail)
                value+=tail_sum/n;grad[base+2:base+states+1]-=np.exp(tail[1:]-tail_sum)/n
        scale=10 if a<1e-4 else 1
        return value*scale,grad*scale
    def feasible(point):
        vals=vectors@point+coeff;total=logsumexp(vals,axis=1)
        prob=np.exp(vals-total[:,None])
        grad=-np.einsum('ij,ijk->ik',prob,vectors)
        return np.r_[-total,order@point],np.r_[grad,order]
    if start is None:
        start=np.zeros(size);start[0]=log(a)/(r+1)-1;start[1]=3
        for base,density in zip(bases,(a,b)):
            start[base]=log(density)/2;start[base+1]=-1;start[base+states+1]=1
    bounds=[(-35,12),(0,25)]
    for i in range(2):bounds += [(-35,15),(-35,12 if i==0 else 0)]+[(-40,40)]*(states-1)+[(-10,20)]
    if exact_s is not None:
        bounds[0]=(0,0);start=list(start);start[0]=0
    result=minimize(objective,start,jac=True,method='SLSQP',bounds=bounds,
                    constraints=[dict(type='ineq',fun=lambda p:feasible(p)[0],jac=lambda p:feasible(p)[1])],
                    options=dict(ftol=1e-11,maxiter=700))
    point=result.x
    gradient=np.array([log(a/(1-a))-point[0]-point[bases[0]],
                       log(b/(1-b))-point[bases[0]+1]-point[bases[1]]])/LN2
    return dict(width=width,q=q,block_k=r,delta=delta,a=a,b=b,
                exponent_per_n=float(result.fun/LN2/(10 if a<1e-4 else 1)),values=point.tolist(),
                gradient=gradient.tolist(),success=bool(result.success),
                min_constraint=float(min(feasible(point)[0])))


def critical(width,q,r,delta,initial=(.65,.54)):
    warm=[None];records=[]
    def objective(ab):
        a,b=ab
        p=propose(width,q,r,delta,a,b,warm[0])
        if not p['success']:
            cold=propose(width,q,r,delta,a,b)
            if cold['exponent_per_n']<p['exponent_per_n']:p=cold
        warm[0]=p['values'];records.append(p)
        return -p['exponent_per_n'],-np.array(p['gradient'])
    fits=[]
    for seed in (initial,(.97,.75)):
        warm[0]=None
        fits.append(minimize(objective,seed,jac=True,method='L-BFGS-B',
                        bounds=[(.15,1-1e-8),(.15,1-1e-8)],
                        options=dict(ftol=1e-12,gtol=2e-7,maxiter=80)))
    result=min(fits,key=lambda fit:fit.fun)
    a,b=result.x;p=propose(width,q,r,delta,a,b,warm[0])
    p.update(outer_success=bool(result.success),evaluations=len(records))
    return p


def threshold(width,q,r):
    calls=[]
    def value(d):
        p=critical(width,q,r,d);calls.append(p)
        print('candidate',width,q,r,d,p['exponent_per_n'],p['a'],p['b'],flush=True)
        return p['exponent_per_n']
    root=brentq(value,.22,.49,xtol=2e-7)
    final=critical(width,q,r,root)
    assert abs(final['exponent_per_n'])<2e-5,'root search did not find a reliable zero'
    result=dict(width=width,q=q,block_k=r,block_n=2*r,estimated_dense_threshold=root,
                critical=final,search=calls,scope='local numerical dense threshold; margin not yet evaluated')
    path=ROOT/f'convolution-design-w{width}-q{q}-r{r}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('THRESHOLD',width,q,r,root,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--widths',type=int,nargs='+',default=[1,2])
    p.add_argument('--alphabets',type=int,nargs='+',default=[2048])
    p.add_argument('--blocks',type=int,nargs='+',default=[4,8,16])
    args=p.parse_args()
    for w in args.widths:
        for q in args.alphabets:
            for r in args.blocks:threshold(w,q,r)
