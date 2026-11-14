"""Direct finite-alphabet width-one certificates with configurable local MDS.

The quotient transfer counts actual surviving solution spaces. Generic
rank and the separate repair scan are not premises of this certificate.
Numerical proposals are accepted only after rational endpoint verification.
"""
from fractions import Fraction as F
from math import comb,exp,isqrt,log,ceil
from pathlib import Path
import argparse
import json
from convolution_design_space import rows,propose,source_coefficients
from repeat_accumulate_precode import log_upper,cached_logs

N=1<<21
ROOT=Path(__file__).parent/'results'


def binary_method_cost(q):
    """Exact average and worst multiplication additions for 1,...,q."""
    assert q>=1
    lengths=ones=0;candidates=[q]
    for j in range(q.bit_length()):
        first=1<<j;last=min(q,(first<<1)-1)
        lengths+=(j+1)*(last-first+1)
        period=first<<1
        ones+=(q+1)//period*first+max(0,(q+1)%period-first)
        if q&first:
            candidate=(q&~(period-1))+(first-1)
            if candidate:candidates.append(candidate)
    worst=max(a.bit_length()+a.bit_count()-2 for a in candidates)
    return F(lengths+ones-2*q,q),worst


def collatz(q,tau,x,y,v):
    assert q>=1 and tau>=1 and min(x,y,*v)>0 and len(v)==2
    return max(max(sum(c*x**a*y**b*tau**d*v[j] for c,a,b,d,j in polynomial)
                   for polynomial in choices)/v[i]
               for i,choices in enumerate(rows(1,q)))


def terminal(q,tau,x,y,v,steps):
    evaluated=[[[ (c*x**a*y**b*tau**d,j) for c,a,b,d,j in poly]
                for poly in choices] for choices in rows(1,q)]
    current=[F(1),F(1)]
    for _ in range(steps):
        current=[max(sum(c*current[j] for c,j in poly) for poly in choices)
                 for choices in evaluated]
    return max(a/b for a,b in zip(current,v))


def rational_values(proposal):
    values=[F(str(exp(v))) for v in proposal['values']]
    values[1]=max(F(1),values[1]);values[7]=min(F(1),values[7])
    return [str(values[i]) for i in (0,1,2,3,4,6,7,8)]


def decode(values):
    z,tau,x1,y1,v1,x2,y2,v2=map(F,values)
    assert min(z,x1,y1,v1,x2,y2,v2)>0 and tau>=1 and y2<=1
    return z,tau,x1,y1,v1,x2,y2,v2


def vertices(rect):
    sl,sh,hl,hh=rect
    assert 0<sl<=sh<=N and 0<hl<=hh<=N
    points=[(F(sl),F(hl)),(F(sh),F(hl)),(F(sh),F(hh)),(F(sl),F(hh))]
    out=[]
    for a,b in zip(points[-1:]+points[:-1],points):
        va,vb=2*a[1]-a[0],2*b[1]-b[0]
        if (va>=0)!=(vb>=0):
            ratio=va/(va-vb);out.append(tuple(x+ratio*(y-x) for x,y in zip(a,b)))
        if vb>=0:out.append(b)
    return list(dict.fromkeys(out))


def entropy_terms(weight,n=N):
    assert 0<=weight<=n
    return [(a,a/n) for a in (F(weight),F(n-weight)) if a]


def dense_bound(q,r,e,rect,values):
    z,tau,x1,y1,v1,x2,y2,v2=decode(values)
    source=1+sum(comb(2*r,d)*z**d*tau**(d-r) for d in range(r+1,2*r+1))
    l1=collatz(q,tau,x1,y1,[F(1),v1]);l2=collatz(q,tau,x2,y2,[F(1),v2])
    c1=max(F(1),1/v1);c2=max(F(1),1/v2)
    sl,sh,hl,hh=rect
    def factor(lo,hi):
        maximum=min(hi,N-lo);u=isqrt(maximum)
        if u*u<maximum:u+=1
        return F(16*u+4,3)
    common=[(1,factor(sl,sh)*factor(hl,hh)*c1*c2/tau),(N//(2*r),source),
            (N,l1),(N,l2),(-e,y2)]
    bounds=[log_upper(common+[(-s,z*x1),(-h,y1*x2)]+entropy_terms(s)+entropy_terms(h))
            for s,h in vertices(rect)]
    assert bounds
    return max(bounds)+cached_logs(F((sh-sl+1)*(hh-hl+1)))[1]


def sparse_bound(q,r,e,s,lo,hi,values,steps=16,n=N):
    assert (s+1)//2<=lo<=hi<=min(n,2*e) and n%(2*r)==0 and 0<=steps<=n
    z,tau,x1,y1,v1,x2,y2,v2=decode(values)
    source=sum(c*tau**d for d,c in source_coefficients(s,r,n).items())
    assert source>0
    common=[(1,source/tau),(-1,F(comb(n,s))),(-s,x1),(-e,y2)]
    for x,y,v in ((x1,y1,[F(1),v1]),(x2,y2,[F(1),v2])):
        common.extend([(n-steps,collatz(q,tau,x,y,v)),(1,terminal(q,tau,x,y,v,steps))])
    bounds=[]
    for h in (lo,hi):
        denom=[(-1,F(comb(n,h)))] if hi<=1024 else [(1,F(n+1))]+entropy_terms(h,n)
        bounds.append(log_upper(common+[(-h,y1*x2)]+denom))
    return max(bounds)+cached_logs(F(hi-lo+1))[1]


def verify(data):
    n,q,r,e=data['n'],data['q'],data['block_k'],data['e'];cut=data['source_min']
    assert n==N and n%(2*r)==0 and 1<=q<(1<<128)-159 and r>=1 and 0<e<n//2
    assert r+1<=cut<=min(n,4*e) and data['steps']==16 and data['complete']
    expected=[s for s in range(r+1,cut) if source_coefficients(s,r,n)]
    total=F(0);sparse_intervals=0
    if 'sparse_support_moment' in data:
        # The ordinary support count is stronger than the rank potential on
        # very small sources at q=1. It bounds the whole range, including
        # every intermediate support, rather than a sample of profiles.
        from block_support_precode import small_bound
        moment=data['sparse_support_moment'];t=F(moment['t']);rho=F(e,n-e)
        assert q==1 and not data['sparse'] and moment['cutoff']==cut-1
        assert 0<rho<t<1 and (t-rho)**2>=4*rho
        poly={0:1,**{d:comb(2*r,d) for d in range(r+1,2*r+1)}}
        total=small_bound(poly,n//(2*r),n,t,cut-1)
        assert str(total)==moment['failure_upper']
    else:
        assert [row['s'] for row in data['sparse']]==expected
    for row in data['sparse']:
        s=row['s'];following=(s+1)//2
        for leaf in row['intervals']:
            lo,hi=leaf['lo'],leaf['hi'];assert lo==following
            upper=sparse_bound(q,r,e,s,lo,hi,leaf['values'])
            assert upper==F(leaf['upper'])
            total+=F(2)**max(-160,ceil(upper));sparse_intervals+=1;following=hi+1
        assert following==min(n,2*e)+1
    sparse_total=total;dense_leaves=0
    def visit(node,rect):
        nonlocal total,dense_leaves
        assert node['rect']==rect
        sl,sh,hl,hh=rect
        if node.get('empty'):
            assert not vertices(rect)
        elif 'children' in node:
            assert len(node['children'])==2
            left,right=node['children'];a,b=left['rect'],right['rect']
            assert (a[0]==sl and b[1]==sh and a[1]+1==b[0] and a[2:]==b[2:]==[hl,hh]) or (
                a[:2]==b[:2]==[sl,sh] and a[2]==hl and b[3]==hh and a[3]+1==b[2])
            visit(left,a);visit(right,b)
        else:
            upper=dense_bound(q,r,e,rect,node['values'])
            assert upper==F(node['upper'])
            total+=F(2)**max(-160,ceil(upper));dense_leaves+=1
    visit(data['dense'],[cut,min(n,4*e),(cut+1)//2,min(n,2*e)])
    assert total<F(1,1<<40)
    source_additions=(r-1)*(3*r-2)//2*(n//(2*r))
    average,worst=binary_method_cost(q)
    return dict(n=n,k=n//2,q=q,block_k=r,minimum_distance_at_least=e+1,
                failure_upper=str(total),failure_log2_upper=str(cached_logs(total)[1]),
                sparse_failure_upper=str(sparse_total),sparse_intervals=sparse_intervals,
                dense_leaves=dense_leaves,scan_word_products=0 if q==1 else 2*(n-1),
                additions_including_newton_source=2*(n-1)+(2*r-1)*(r-1)*(n//(2*r)),
                source_map='Extended Newton RS: evaluations at 0,...,2r-2 and leading difference',
                source_field_additions=source_additions,
                field_additions_with_extended_rs=2*(n-1)+source_additions,
                binary_method_additions_expected=str(source_additions+2*(n-1)*(1+average)),
                binary_method_additions_worst=source_additions+2*(n-1)*(1+worst),
                prime=(1<<128)-159,
                permutations=2,scans=2,complete_distance_certificate=True,
                scope='All nonzero messages; finite independent coefficients; no repair or generic-rank premise; no benchmark.')


def generate(q,r,e,source_min=32,max_nodes=1500,sparse_bits=55,support_moment=False):
    stem=f'convolution-width-one-direct-r{r}-q{q}-e{e}'
    sparse_path=ROOT/f'{stem}-sparse-progress.json';records=[]
    moment=None
    if support_moment:
        from block_support_precode import small_bound
        from precode_tradeoffs import moment_t
        assert q==1
        t=moment_t(N,e)
        poly={0:1,**{d:comb(2*r,d) for d in range(r+1,2*r+1)}}
        small=small_bound(poly,N//(2*r),N,t,source_min-1)
        assert small<F(1,1<<40)
        moment=dict(t=str(t),cutoff=source_min-1,failure_upper=str(small))
        print('SPARSE MOMENT',q,r,e,float(cached_logs(small)[1]),flush=True)
    elif sparse_path.exists():
        old=json.loads(sparse_path.read_text());assert (old['q'],old['r'],old['e'],old['source_min'])==(q,r,e,source_min)
        records=old['records']
    done={row['s'] for row in records}
    for s in range(r+1,source_min):
        if support_moment:break
        if s in done or not source_coefficients(s,r,N):continue
        bottom=(s+1)//2;leaves=[]
        pool=[rational_values(propose(1,q,r,e/N,s/N,h/N,exact_s=s))
              for h in sorted({bottom,2*bottom,4*bottom,10*bottom})]
        def interval(lo,hi,parent=None):
            h=isqrt(lo*hi);p=propose(1,q,r,e/N,s/N,h/N,start=parent,exact_s=s)
            values=rational_values(p);bound=sparse_bound(q,r,e,s,lo,hi,values)
            for seed in pool:
                other=sparse_bound(q,r,e,s,lo,hi,seed)
                if other<bound:values,bound=seed,other
            target=-sparse_bits-(s-(r+1))
            if bound<target:
                leaves.append(dict(lo=lo,hi=hi,values=values,upper=str(bound)));return
            if lo==hi:raise ValueError(f'Unclosed sparse s={s},h={lo},bits={float(bound)}')
            mid=min(hi-1,max(lo,isqrt(lo*hi)))
            interval(lo,mid,p['values']);interval(mid+1,hi,p['values'])
        interval(bottom,min(N,2*e))
        records.append(dict(s=s,intervals=leaves))
        sparse_path.write_text(json.dumps(dict(q=q,r=r,e=e,source_min=source_min,records=records),indent=2)+'\n')
        print('SPARSE',q,r,e,s,'leaves',len(leaves),flush=True)
    visited=0;leaves=0
    def box(rect,parent=None):
        nonlocal visited,leaves
        visited+=1
        if visited>max_nodes:raise ValueError('Bounded dense discovery exhausted its node limit')
        polygon=vertices(rect)
        if not polygon:return dict(rect=rect,empty=True)
        a=float(sum(s for s,h in polygon)/(len(polygon)*N));b=float(sum(h for s,h in polygon)/(len(polygon)*N))
        p=propose(1,q,r,e/N,a,b,start=parent);values=rational_values(p)
        upper=dense_bound(q,r,e,rect,values)
        if upper<-70:
            leaves+=1
            if leaves%10==0:print('DENSE',q,r,e,'leaves',leaves,'nodes',visited,flush=True)
            return dict(rect=rect,values=values,upper=str(upper))
        sl,sh,hl,hh=rect
        if sl==sh and hl==hh:raise ValueError(f'Unclosed dense {rect}: {float(upper)}')
        source=log(sh/sl)>=log(hh/hl)
        if sl==sh:source=False
        if hl==hh:source=True
        lo,hi=(sl,sh) if source else (hl,hh);mid=min(hi-1,max(lo,isqrt(lo*hi)))
        children=([[sl,mid,hl,hh],[mid+1,sh,hl,hh]] if source else
                  [[sl,sh,hl,mid],[sl,sh,mid+1,hh]])
        return dict(rect=rect,children=[box(child,p['values']) for child in children])
    tree=box([source_min,min(N,4*e),(source_min+1)//2,min(N,2*e)])
    data=dict(n=N,q=q,block_k=r,e=e,source_min=source_min,steps=16,
              sparse=records,dense=tree,complete=True)
    if moment is not None:data['sparse_support_moment']=moment
    result=verify(data)
    (ROOT/f'{stem}.json').write_text(json.dumps(data,indent=2)+'\n')
    (ROOT/f'{stem}-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',q,r,e,float(F(result['failure_log2_upper'])),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--q',type=int,default=16)
    parser.add_argument('--block-k',type=int,default=8);parser.add_argument('--distance',default='.255')
    parser.add_argument('--source-min',type=int,default=32);parser.add_argument('--max-nodes',type=int,default=1500)
    parser.add_argument('--sparse-bits',type=int,default=55)
    parser.add_argument('--support-only-sparse',action='store_true')
    parser.add_argument('--verify',type=Path);args=parser.parse_args()
    if args.verify:print(json.dumps(verify(json.loads(args.verify.read_text())),indent=2))
    else:generate(args.q,args.block_k,int(F(args.distance)*N),args.source_min,args.max_nodes,args.sparse_bits,args.support_only_sparse)
