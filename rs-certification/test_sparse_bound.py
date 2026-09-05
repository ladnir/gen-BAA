"""Independent exact finite-kernel checks of the sparse envelope steps."""
from fractions import Fraction as F
from math import comb
import unittest
from sparse_bound import sparse_bound


def choose(n,k):
    return comb(n,k) if 0<=k<=n else 0


def kernel(n,s,h,a):
    if s<1 or h<1 or h>n or a<0 or s-a-1<0:
        return F(0)
    return F(choose(s-a,a)*choose(n-h,a)*choose(h-1,s-a-1),choose(n,s))


def exact_support_terms(n,L,s,r,q):
    P=F(q**r-1,q-1)
    low,high=F(0),F(0)
    for H in range(1,n+1):
        for a in range(s//2+1):
            first=kernel(n,s,H,a)
            if not first:
                continue
            for b in range(H//2+1):
                tail=sum((kernel(n,H,j,b) for j in range(1,L+1)),F(0))
                term=first*tail*min(F(1),P/F((q-1)**(a+b)))
                if b<=s: low+=term
                else: high+=term
    return low,high


def support_envelope(n,L,s,r,q,z):
    delta=F(L,n)
    D=1-delta-F(s,n)
    rho=delta*(1+z)**2
    if not (D>0 and 0<rho<1):
        return None
    K=F(s,n-s+1)*rho/(1-rho)
    V=1/(delta*D*z*z)
    Q=q-1
    if not (K<1 and K*Q>1 and V<Q):
        return None
    low=2**(s+1)*(K**(s-r+1)*(1/(1-K)+1/(K*Q-1))*
                  sum(((K*V)**b for b in range(r)),F(0))+
                  K**s*V**r/(Q*(1-1/(K*Q))*(1-V/Q)))
    high=F(2**(s+1)*n**(s+1),Q**(s-r+2))/(1-F(n,Q))
    return low,high


class SparseBoundTests(unittest.TestCase):
    def test_trace_mass_identity(self):
        for n in range(1,13):
            for s in range(1,n+1):
                for a in range(s//2+1):
                    self.assertEqual(sum((kernel(n,s,h,a) for h in range(1,n+1)),F(0)),
                                     choose(s-a,a))

    def test_support_bound_against_full_exact_sums(self):
        count=0
        for n in [8,10,12]:
            for L in [1,n//3,n//2]:
                for s in range(2,min(6,n-L)):
                    for r in range(1,s):
                        bound=support_envelope(n,L,s,r,65537,F(1,10))
                        if bound is None:
                            continue
                        low,high=exact_support_terms(n,L,s,r,65537)
                        self.assertLessEqual(low,bound[0],(n,L,s,r,'low'))
                        self.assertLessEqual(high,bound[1],(n,L,s,r,'high'))
                        count+=1
        self.assertGreater(count,50)

    def test_target_sparse_slabs_exact_rational(self):
        for sigma,delta in [(4,F(3,5)),(8,F(3,10))]:
            result=sparse_bound(2**20*16//sigma,16,sigma,delta,2**128)
            self.assertLess(result['bound'],F(1,2**60))


if __name__=='__main__':
    unittest.main()
