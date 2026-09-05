"""Verify entropy/LP upper bounds with Arb interval arithmetic.

No numerical optimization is used here. Every proposed tangent and dual
multiplier yields an upper bound even if the optimizer was inaccurate.
"""
from fractions import Fraction as F
from math import comb
from flint import arb,ctx
import json
import argparse


def a(x):
    if not isinstance(x,F): x=F(str(x))
    return arb(x.numerator)/arb(x.denominator)


def H(x):
    if x==0 or x==1: return arb(0)
    return -a(x)*a(x).log()-a(1-x)*a(1-x).log()


def setup(sigma):
    if sigma not in (4,8):raise ValueError('Unsupported constituent dimension')
    m=16;n=2**20*m//sigma;delta=F(3,5) if sigma==4 else F(3,10)
    h=m-sigma;ells=[0]+list(range(h+1,m+1));P=len(ells);N=P+6
    ai,ei,xi,yi,zi,ci=range(P,N)
    def row(terms):
        r=[F(0)]*N
        for i,v in terms: r[i]=F(v)
        return r
    pairs=[(row([(ai,1),(xi,-1)]),F(0),row([(xi,1)]),F(0)),
           (row([(ei,-1)]),F(1),row([(xi,1)]),F(0)),
           (row([(ei,1)]),F(0),row([(ai,1),(xi,-1)]),F(0)),
           (row([(ei,1),(yi,-1)]),F(0),row([(yi,1)]),F(0)),
           (row([(zi,-1)]),F(1),row([(yi,1)]),F(0)),
           (row([(zi,1)]),F(0),row([(ei,1),(yi,-1)]),F(0))]
    A=[];b=[]
    for ar,ac,br,bc in pairs:
        A.append([-v for v in br]);b.append(bc)
        A.append([v-u for u,v in zip(ar,br)]);b.append(ac-bc)
    penalty=row([(xi,1),(yi,1),(ci,-1)])
    for i,l in enumerate(ells): penalty[i]-=F(max(0,l-h),m)
    A.append(penalty);b.append(F(0))
    E=[[F(1) if i<P else F(0) for i in range(N)],row([(ai,1)])]
    for i,l in enumerate(ells):E[1][i]=-F(l,m)
    e=[F(1),F(0)]
    lo=[F(0)]*N;hi=[F(1)]*N
    hi[0]=1-F(m*(n//1000+1),n);hi[zi]=delta
    amin=F((h+1)*(n//1000+1),n)
    return dict(m=m,n=n,delta=delta,h=h,ells=ells,P=P,N=N,ai=ai,ei=ei,
                ci=ci,pairs=pairs,A=A,b=b,E=E,e=e,lo=lo,hi=hi,amin=amin)


def impossible(model,box):
    al,au,el,eu=box
    return au<model['amin'] or 2*eu<al or el>2*model['delta']


def verify_leaf(sigma,box,data,precision=192,q_min=2**127):
    ctx.prec=precision
    md=setup(sigma)
    box=[v if isinstance(v,F) else F(str(v)) for v in box]
    if data.get('status')=='infeasible':
        if not impossible(md,box):raise ValueError('Unproved infeasible box')
        return None
    P,N=md['P'],md['N']
    probs=[F(str(v)) for v in data['probs']]
    if len(probs)!=P or any(p<=0 for p in probs):raise ValueError('Bad probabilities')
    psum=sum(probs);probs=[p/psum for p in probs]
    theta=[F(str(v)) for v in data['theta']]
    if len(theta)!=6 or any(not 0<t<1 for t in theta):raise ValueError('Bad tangent')
    c=[arb(0) for _ in range(N)];constant=arb(0)
    for i,l in enumerate(md['ells']):
        count=1 if l==0 else comb(16,l)
        c[i]=(arb(count).log()-a(probs[i]).log())/16
    for (ar,ac,br,bc),t in zip(md['pairs'],theta):
        u=-a(1-t).log();w=a((1-t)/t).log()
        for i in range(N):c[i]+=u*a(ar[i])+w*a(br[i])
        constant+=u*a(ac)+w*a(bc)
    for idx,low,high in [(md['ai'],box[0],box[1]),(md['ei'],box[2],box[3])]:
        if not 0<=low<high<=1:raise ValueError('Invalid secant interval')
        slope=(H(low)-H(high))/a(high-low)
        intercept=-H(low)-slope*a(low)
        c[idx]+=slope;constant+=intercept
    if q_min-1 < 2*md['n']:raise ValueError('Field too small for projective envelope')
    c[md['ci']]-=arb(q_min-1).log()
    mu=[F(str(v)) for v in data['ineq_dual']]
    nu=[F(str(v)) for v in data['eq_dual']]
    if len(mu)!=len(md['A']) or len(nu)!=len(md['E']):raise ValueError('Dual size')
    if any(v<0 for v in mu):raise ValueError('Negative inequality multiplier')
    for v,row,rhs in zip(mu,md['A'],md['b']):
        constant+=a(v*rhs)
        for i in range(N):c[i]-=a(v*row[i])
    for v,row,rhs in zip(nu,md['E'],md['e']):
        constant+=a(v*rhs)
        for i in range(N):c[i]-=a(v*row[i])
    lo,hi=md['lo'],md['hi']
    lo[md['ai']],hi[md['ai']],lo[md['ei']],hi[md['ei']]=box
    upper=constant
    for ci,li,ui in zip(c,lo,hi):
        positive=max(arb(0),ci.upper())
        upper+=ci*a(li)+positive*a(ui-li)
    return upper.upper()


def threshold(sigma,bits=42):
    md=setup(sigma);n=md['n'];B=n//16
    logpref=arb(2).log()+sigma*arb(B+1).log()+6*arb(n+1).log()
    return -(logpref+bits*arb(2).log())/n


def verify_covering(data,q_min=2**127,precision=192):
    """Return the checked model, leaf count, and enclosing bulk log bound."""
    ctx.prec=precision
    sigma=data['sigma'];md=setup(sigma)
    leaves=data['leaves'];nodes=data['nodes']
    root=[md['amin'],F(1),md['amin']/2,F(1)]
    visited=set();uppers=[]
    def walk(path,box):
        if path in visited:raise ValueError('Duplicate path')
        visited.add(path)
        if path in leaves:
            if path in nodes:raise ValueError('Leaf/internal collision')
            u=verify_leaf(sigma,box,leaves[path],q_min=q_min,precision=precision)
            if u is not None:uppers.append(u)
            return
        if path not in nodes:raise ValueError('Missing leaf '+path)
        axis=nodes[path]
        if axis not in (0,1):raise ValueError('Bad axis')
        ix=2*axis;mid=(box[ix]+box[ix+1])/2
        left=box.copy();right=box.copy();left[ix+1]=mid;right[ix]=mid
        walk(path+'0',left);walk(path+'1',right)
    walk('',root)
    if visited!=set(leaves)|set(nodes):raise ValueError('Extraneous tree nodes')
    if not uppers:raise ValueError('No feasible leaves')
    u=max(uppers)
    n=md['n'];B=n//16
    logbound=arb(2).log()+sigma*arb(B+1).log()+6*arb(n+1).log()+n*u
    return md,len(leaves),u,logbound


def verify_tree(data,q_min=2**127,precision=192):
    _,leaves,u,logbound=verify_covering(data,q_min=q_min,precision=precision)
    return {'leaves':leaves,'exponent_upper':str(u),
            'log2_bulk_bound_upper':str((logbound/arb(2).log()).upper()),
            'below_2^-42':bool(logbound < -42*arb(2).log())}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('certificate');args=ap.parse_args()
    with open(args.certificate) as f:data=json.load(f)
    print(json.dumps(verify_tree(data),indent=2))
