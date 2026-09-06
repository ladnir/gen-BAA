"""Propose finite-entropy upper certificates for the nonsparse region.

The only nonconcave terms are -H(alpha) and -H(eta). We replace them by
secants on a rectangle. Every other entropy term has a global linear upper
bound from a probability vector. A linear-program dual certifies the resulting
linear upper bound. Optimizers propose data; independent interval verification
will be required before publication use.
"""
import argparse
from dataclasses import dataclass
from math import comb,log
import json
import numpy as np
from scipy.optimize import minimize,linprog,LinearConstraint,Bounds
from scipy.special import xlogy


def entropy(x):
    return -xlogy(x,x)-xlogy(1-x,1-x)


@dataclass
class Model:
    m:int=16
    sigma:int=4
    delta:float=.60
    n:int=2**22
    qbits:int=128

    def __post_init__(self):
        self.h=self.m-self.sigma
        self.d=self.h+1
        self.ells=np.array([0]+list(range(self.d,self.m+1)))
        self.P=len(self.ells)
        self.N=self.P+6
        self.ai,self.ei,self.xi,self.yi,self.zi,self.ci=range(self.P,self.N)
        self.counts=np.array([1]+[comb(self.m,int(l)) for l in self.ells[1:]])
        self.logcounts=np.log(self.counts)
        self.logQ=self.qbits*log(2)  # proposal only; verifier uses log(2^qbits-1)
        self.lambdamin=self.m*(self.n//1000+1)/self.n
        self.amin=self.d*(self.n//1000+1)/self.n
        self.emax=min(1,2*self.delta)
        def row(**kwargs):
            r=np.zeros(self.N)
            for k,v in kwargs.items(): r[getattr(self,k)]=v
            return r
        self.pairs=[(row(ai=1,xi=-1),0,row(xi=1),0),
                    (row(ei=-1),1,row(xi=1),0),
                    (row(ei=1),0,row(ai=1,xi=-1),0),
                    (row(ei=1,yi=-1),0,row(yi=1),0),
                    (row(zi=-1),1,row(yi=1),0),
                    (row(zi=1),0,row(ei=1,yi=-1),0)]
        self.A=[]; self.b=[]
        for ar,ac,br,bc in self.pairs:
            self.A.append(-br); self.b.append(bc)  # B>=0
            self.A.append(br-ar); self.b.append(ac-bc)  # A-B>=0
        penalty=row(xi=1,yi=1,ci=-1)
        penalty[:self.P]-=np.maximum(0,self.ells-self.h)/self.m
        self.A.append(penalty); self.b.append(0)
        self.A=np.array(self.A);self.b=np.array(self.b)
        self.E=np.zeros((2,self.N));self.e=np.array([1.,0.])
        self.E[0,:self.P]=1
        self.E[1,self.ai]=1;self.E[1,:self.P]=-self.ells/self.m

    def bounds(self,box):
        lo=np.zeros(self.N);hi=np.ones(self.N)
        hi[0]=1-self.lambdamin
        lo[self.ai],hi[self.ai],lo[self.ei],hi[self.ei]=box
        hi[self.zi]=self.delta
        return lo,hi

    def chord(self,lo,hi):
        if hi-lo<1e-15:
            return 0.,-float(entropy(lo))
        slope=(float(entropy(lo))-float(entropy(hi)))/(hi-lo)
        return slope,-float(entropy(lo))-slope*lo

    def value_grad(self,v,box):
        p=v[:self.P]
        value=float(np.sum(-xlogy(p,p)+p*self.logcounts))/self.m
        grad=np.zeros(self.N)
        grad[:self.P]=(-np.log(np.maximum(p,1e-14))-1+self.logcounts)/self.m
        for ar,ac,br,bc in self.pairs:
            aa=max(0.,float(ar@v+ac));bb=float(br@v+bc)
            bb=min(aa,max(0.,bb))
            value+=float(xlogy(aa,aa)-xlogy(bb,bb)-xlogy(aa-bb,aa-bb))
            theta=np.clip(bb/max(aa,1e-14),1e-12,1-1e-12)
            grad+=-log(1-theta)*ar+log((1-theta)/theta)*br
        for idx,low,high in [(self.ai,box[0],box[1]),(self.ei,box[2],box[3])]:
            slope,intercept=self.chord(low,high)
            value+=slope*v[idx]+intercept;grad[idx]+=slope
        value-=self.logQ*v[self.ci];grad[self.ci]-=self.logQ
        return value,grad

    def tangent(self,v,box):
        probs=np.maximum(v[:self.P],1e-12);probs=probs/probs.sum()
        c=np.zeros(self.N);constant=0.
        c[:self.P]=(self.logcounts-np.log(probs))/self.m
        theta=[]
        for ar,ac,br,bc in self.pairs:
            aa=float(ar@v+ac);bb=float(br@v+bc)
            t=float(np.clip(bb/max(aa,1e-14),1e-10,1-1e-10))
            theta.append(t)
            u=-log(1-t);w=log((1-t)/t)
            c+=u*ar+w*br;constant+=u*ac+w*bc
        for idx,low,high in [(self.ai,box[0],box[1]),(self.ei,box[2],box[3])]:
            slope,intercept=self.chord(low,high)
            c[idx]+=slope;constant+=intercept
        c[self.ci]-=self.logQ
        return c,constant,probs,theta

    def solve(self,box,start=None):
        lo,hi=self.bounds(box)
        feasible=linprog(np.zeros(self.N),A_ub=self.A,b_ub=self.b,
                         A_eq=self.E,b_eq=self.e,bounds=list(zip(lo,hi)),method='highs')
        if not feasible.success:
            return {'box':list(box),'status':'infeasible_proposal'}
        v0=feasible.x
        if start is not None and np.all(start>=lo) and np.all(start<=hi):
            v0=start
        opt=minimize(lambda v:-self.value_grad(v,box)[0],v0,
                     jac=lambda v:-self.value_grad(v,box)[1],method='SLSQP',
                     bounds=Bounds(lo,hi),
                     constraints=[LinearConstraint(self.A,-np.inf,self.b),
                                  LinearConstraint(self.E,self.e,self.e)],
                     options={'ftol':1e-14,'maxiter':600})
        v=opt.x
        c,constant,probs,theta=self.tangent(v,box)
        upper=linprog(-c,A_ub=self.A,b_ub=self.b,A_eq=self.E,b_eq=self.e,
                      bounds=list(zip(lo,hi)),method='highs')
        if not upper.success:
            return {'box':list(box),'status':'lp_failed'}
        return {'box':list(box),'status':'candidate','upper':float(-upper.fun+constant),
                'value':float(self.value_grad(v,box)[0]),'optimizer_success':bool(opt.success),
                'point':v.tolist(),'probs':probs.tolist(),'theta':theta,
                'ineq_dual':np.maximum(0.,-upper.ineqlin.marginals).tolist(),
                'eq_dual':(-upper.eqlin.marginals).tolist()}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--sigma',type=int,default=4)
    ap.add_argument('--box',nargs=4,type=float)
    args=ap.parse_args()
    model=Model(sigma=args.sigma,delta=.6 if args.sigma==4 else .3,
                n=2**20*16//args.sigma)
    box=args.box or [model.amin,1.,model.amin/2,model.emax]
    print(json.dumps(model.solve(box),indent=2))
