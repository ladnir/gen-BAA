"""Independent identities and exact small cases for tighter sparse bounds."""
from fractions import Fraction as F
from math import comb
import unittest
from flint import arb,ctx
from refined_sparse import binomial_moment,support_low,support_counts
from direct_sparse import second_tail_table,support_direct
from test_sparse_bound import choose,kernel,exact_support_terms
from verify_bulk import a


class RefinedSparseTests(unittest.TestCase):
    def test_moment_against_forward_differences(self):
        for k in range(1,11):
            for b in range(9):
                degree=k-1+2*b
                values=[choose(H-1,k-1)*choose(H,2*b) for H in range(1,degree+2)]
                differences=[]
                while values:
                    differences.append(values[0])
                    values=[right-left for left,right in zip(values,values[1:])]
                for delta in (F(1,3),F(3,5)):
                    reference=sum((delta*d*delta**j/(1-delta)**(j+1)
                                   for j,d in enumerate(differences)),F(0))
                    self.assertEqual(binomial_moment(k,b,delta),reference)

    def test_order_statistic_identity(self):
        for n in range(2,15):
            for L in range(1,n):
                for H in range(1,n+1):
                    for b in range(H//2+1):
                        original=sum(choose(n-j,b)*choose(j-1,H-b-1) for j in range(1,L+1))
                        rearranged=sum(choose(L,H-v)*choose(n-L,v) for v in range(b+1))
                        self.assertEqual(original,rearranged)

    def test_exact_supports_below_refined_envelope(self):
        checked=0
        for n,L in [(10,2),(12,3),(14,4)]:
            for s in range(2,6):
                for r in range(1,s):
                    envelope=support_low(n,L,s,r,65537,F(1,10))
                    low,_=exact_support_terms(n,L,s,r,65537)
                    self.assertLessEqual(low,sum(envelope.values(),F(0)))
                    checked+=1
        self.assertGreater(checked,25)

    def test_direct_finite_sum_and_intermediate_tail(self):
        ctx.prec=192
        for n,L,s,r in [(10,2,3,2),(12,3,4,3),(14,4,5,4),(12,3,5,1)]:
            q=65537;projective=F(q**r-1,q-1)
            exact=F(0)
            for H in range(1,n+1):
                for aa in range(s//2+1):
                    for b in range(min(r-1,H//2)+1):
                        exact+=kernel(n,s,H,aa)*sum((kernel(n,H,j,b) for j in range(1,L+1)),F(0))*min(F(1),projective/(q-1)**(aa+b))
            for maximum in (3,5,n):
                second=second_tail_table(n,L,maximum,r-1)
                result=support_direct(n,L,s,r,q,maximum,second)
                bound=sum(result.values(),arb(0))
                if maximum<n:
                    self.assertTrue(a(exact)<bound)
                else:
                    # Equal rationals, evaluated through different interval expressions.
                    self.assertTrue((bound-a(exact)).contains(0))
                    self.assertEqual(result['H_tail'],0)

    def test_support_coefficients(self):
        for sigma in (4,8):
            U=sum(comb(16,ell) for ell in range(17-sigma,17))
            for blocks in (1,2,3):
                counts=support_counts(16,sigma,blocks)
                self.assertEqual(sum(counts.values()),U**blocks)
                self.assertEqual(min(counts),blocks*(17-sigma))
                self.assertEqual(max(counts),16*blocks)


if __name__=='__main__':unittest.main()
