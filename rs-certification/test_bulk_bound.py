"""Checks of finite prefactors, dual signs, and certificate coverage."""
import copy,json
from pathlib import Path
from fractions import Fraction as F
from math import comb,factorial
import random
import unittest
from flint import arb,ctx
from test_sparse_bound import kernel
from verify_bulk import verify_tree,a,H,verify_leaf,setup
from bulk_bound import Model
from verify_complete import complete
from scipy.optimize import linprog
import numpy as np


HERE=Path(__file__).resolve().parent


def perspective(A,B):
    assert 0<=B<=A
    return arb(0) if A==0 else a(A)*H(B/A)


class BulkBoundTests(unittest.TestCase):
    def test_complete_bounds_at_minimum_field(self):
        for sigma,bits in [(4,'209.60'),(8,'137.56')]:
            result=complete(sigma)
            self.assertEqual(result['field_size_lower_bound'],str(2**127))
            self.assertEqual(result['certified_failure_bound_bits_decimal'],bits)
            self.assertEqual(result['sparse_max_active_blocks']+1,result['bulk_min_active_blocks'])

    def test_complete_bounds_with_higher_precision_and_field(self):
        for sigma in (4,8):
            result=complete(sigma,precision=256,q_min=2**128)
            self.assertEqual(result['arithmetic_precision_bits'],256)
            self.assertTrue(result['failure_bound_below_2^-40'])

    def test_exact_finite_terms_below_entropy_envelope(self):
        ctx.prec=192;rng=random.Random(1729);checked=0
        n=12;m=4;sigma=2;h0=2;B=n//m;q=257;Q=q-1
        for k3 in range(B+1):
            for k4 in range(B-k3+1):
                t=k3+k4
                if not t:continue
                k0=B-t;s=3*k3+4*k4;r=s-h0*t
                count=factorial(B)//(factorial(k0)*factorial(k3)*factorial(k4))*4**k3
                p=[F(k0,B),F(k3,B),F(k4,B)]
                S=sum((-a(v)*a(v).log() for v in p if v),arb(0))/m+a(p[1])*arb(4).log()/m
                for _ in range(400):
                    aa=rng.randrange(s//2+1);HH=rng.randrange(1,n+1)
                    bb=rng.randrange(HH//2+1);jj=rng.randrange(1,n+1)
                    k1=kernel(n,s,HH,aa);k2=kernel(n,HH,jj,bb)
                    if not k1 or not k2:continue
                    exact=count*k1*k2*min(F(1),F(q**r-1,Q**(aa+bb+1)))
                    alpha,eta,x,y,zeta=[F(v,n) for v in (s,HH,aa,bb,jj)]
                    terms=[(alpha-x,x),(1-eta,x),(eta,alpha-x),
                           (eta-y,y),(1-zeta,y),(zeta,eta-y)]
                    exponent=S+sum((perspective(A,bb_) for A,bb_ in terms),arb(0))
                    exponent-=H(alpha)+H(eta)+a(max(F(0),x+y-F(r,n)))*arb(Q).log()
                    bound=arb(2*(n+1)**2)*(n*exponent).exp()
                    self.assertTrue(a(exact)<=bound,(k3,k4,aa,HH,bb,jj))
                    checked+=1
        self.assertGreater(checked,100)

    def test_interval_coverings(self):
        for sigma in (4,8):
            data=json.loads((HERE/f'bulk-rs16-{sigma}.json').read_text())
            self.assertTrue(verify_tree(data)['below_2^-42'])

    def test_missing_leaf_is_rejected(self):
        data=json.loads((HERE/'bulk-rs16-4.json').read_text())
        del data['leaves'][next(iter(data['leaves']))]
        with self.assertRaises(ValueError):verify_tree(data)

    def test_false_infeasibility_is_rejected(self):
        md=setup(4)
        with self.assertRaises(ValueError):
            verify_leaf(4,[md['amin'],F(1),md['amin']/2,F(1)],{'status':'infeasible'})

    def test_negative_dual_is_rejected(self):
        data=json.loads((HERE/'bulk-rs16-4.json').read_text())
        for leaf in data['leaves'].values():
            if leaf['status']=='candidate':
                leaf['ineq_dual'][0]=-1
                break
        with self.assertRaises(ValueError):verify_tree(data)

    def test_leaf_upper_bounds_independent_feasible_points(self):
        rng=np.random.default_rng(42)
        for sigma in (4,8):
            data=json.loads((HERE/f'bulk-rs16-{sigma}.json').read_text())
            md=setup(sigma);model=Model(sigma=sigma,n=md['n'],delta=float(md['delta']))
            for leaf in data['leaves'].values():
                if leaf['status']!='candidate':continue
                box=leaf['box'];bound=float(verify_leaf(sigma,box,leaf))
                lo,hi=model.bounds(box)
                for _ in range(3):
                    point=linprog(rng.normal(size=model.N),A_ub=model.A,b_ub=model.b,
                                  A_eq=model.E,b_eq=model.e,bounds=list(zip(lo,hi)),method='highs')
                    if not point.success:continue
                    v=point.x
                    beta=np.dot(np.maximum(0,model.ells-model.h),v[:model.P])/model.m
                    v[model.ci]=max(0,v[model.xi]+v[model.yi]-beta)
                    value=model.value_grad(v,box)[0]
                    self.assertLessEqual(value,bound+1e-8)


if __name__=='__main__':unittest.main()
