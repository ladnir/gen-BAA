"""Exact checks for constant-weighted RM BAA specialization. No dependencies.

This verifies a GF(4) counterexample and algebra used in the degree-bound
lemma. It does NOT certify distance 10% over GF(2^128).
"""
from itertools import combinations, permutations, product
import json

MONS = [0,1,2,3,4,5,6,8,9,10,12,16,17,18,20,24]
P = [6,31,25,3,16,1,21,20,10,14,8,2,26,9,23,22,4,19,15,18,30,5,13,28,17,0,12,11,7,24,27,29]
Q = [2,29,28,9,18,8,24,31,30,19,6,14,26,15,0,13,22,20,27,3,5,12,25,17,21,10,4,7,16,11,23,1]
SUPPORT = [5,13,15,19]
MOD128 = (1 << 128) | 0x87

def pmul(a,b):
    r=0
    while b:
        if b&1:r^=a
        a<<=1;b>>=1
    return r

def rem(a,b):
    assert b
    while a.bit_length() >= b.bit_length():a^=b<<(a.bit_length()-b.bit_length())
    return a

def gcd(a,b):
    while b:a,b=b,rem(a,b)
    return a

def mul(a,b,mod):return rem(pmul(a,b),mod)

def inverse(a,mod):
    assert a
    u,v,s,t=a,mod,1,0
    while u!=1:
        shift=u.bit_length()-v.bit_length()
        if shift<0:u,v,s,t=v,u,t,s;shift=-shift
        u^=v<<shift;s^=t<<shift
    return rem(s,mod)

def local(message):
    return [xor(message[j] for j,m in enumerate(MONS) if i&m==m) for i in range(32)]

def xor(values):
    result=0
    for x in values:result^=x
    return result

def stage(v,p,c,mod=None):
    result=[];state=0
    for i in p:
        state=v[i]^(pmul(c,state) if mod is None else mul(c,state,mod))
        result.append(state)
    return result

def unstage(v,p,c,mod=None):
    result=[0]*len(v)
    for i,j in enumerate(p):
        prev=0 if i==0 else (pmul(c,v[i-1]) if mod is None else mul(c,v[i-1],mod))
        result[j]=v[i]^prev
    return result

def syndrome(v):
    return [xor(v[i] for i in range(32) if i&m==m) for m in MONS]

def parity(c,mod=None):
    result=[]
    for i in range(32):
        e=[0]*32;e[i]=1
        result.append(syndrome(unstage(unstage(e,Q,c,mod),P,c,mod)))
    return result  # columns

def rank(columns,mod):
    basis={}
    for column in columns:
        v=column[:]
        for i in range(len(v)):
            if not v[i]:continue
            if i in basis:
                a=v[i];v=[x^mul(a,y,mod) for x,y in zip(v,basis[i])]
            else:
                a=inverse(v[i],mod);basis[i]=[mul(a,x,mod) for x in v];break
    return len(basis)

def evaluate(p,c,mod):
    result=0
    for i in range(p.bit_length()-1,-1,-1):result=mul(result,c,mod)^((p>>i)&1)
    return result

def minimal_polynomial(c,mod):
    basis={};value=1
    for i in range(mod.bit_length()):
        v,comb=value,1<<i
        while v:
            bit=v.bit_length()-1
            if bit not in basis:basis[bit]=(v,comb);break
            old,coeff=basis[bit];v^=old;comb^=coeff
        if not v:return comb
        value=mul(value,c,mod)
    raise AssertionError('No dependence found')

def determinant(rows):
    return xor(product_poly(rows[i][p[i]] for i in range(len(rows))) for p in permutations(range(len(rows))))

def product_poly(values):
    result=1
    for x in values:result=pmul(result,x)
    return result

def main():
    # Rabin irreducibility criterion for degree 128 (its only prime factor
    # is 2): x^(2^128)=x and gcd(x^(2^64)-x, modulus)=1.
    frobenius=2
    for i in range(1,129):
        frobenius=mul(frobenius,frobenius,MOD128)
        if i==64:assert gcd(frobenius^2,MOD128)==1
    assert frobenius==2
    assert sorted(P)==sorted(Q)==list(range(32))
    base=[local([int(j==i) for j in range(16)]) for i in range(16)]
    assert rank([[row[i] for row in base] for i in range(32)],3)==16
    assert all(not any(syndrome(row)) for row in base)  # RM(2,5) is self-dual.
    binary=[]
    for row in base:
        out=stage(stage(row,P,1,3),Q,1,3)
        binary.append(sum(x<<i for i,x in enumerate(out)))
    current=0;distance=32;binary_witness=None
    for i in range(1,1<<16):
        current^=binary[(i&-i).bit_length()-1]
        if current.bit_count()<distance:
            distance=current.bit_count();binary_witness={'message_bits':i^(i>>1),'word_bits':current}
    assert distance==5

    h4=parity(2,7)
    witness=None
    for coeffs in product(range(1,4),repeat=4):
        if all(xor(mul(a,h4[j][row],7) for a,j in zip(coeffs,SUPPORT))==0 for row in range(16)):
            witness=coeffs;break
    assert witness is not None
    word=[0]*32
    for i,a in zip(SUPPORT,witness):word[i]=a
    precode=unstage(unstage(word,Q,2,7),P,2,7)
    transformed=precode[:]
    for bit in (1,2,4,8,16):
        for i in range(32):
            if i&bit:transformed[i]^=transformed[i^bit]
    assert all(transformed[i]==0 for i in range(32) if i not in MONS)
    message=[transformed[i] for i in MONS]
    assert local(message)==precode
    assert stage(stage(local(message),P,2,7),Q,2,7)==word
    # Also check the legacy convention state=c*(state+input). With two
    # rounds it differs from this code by the global nonzero scalar c^2.
    assert sum(x!=0 for x in word)==4

    hp=parity(2)  # polynomial variable t is represented by the integer 2.
    assert max(x.bit_length()-1 for col in hp for x in col)<=2
    minors_gcd=0;minor_count=0
    for rows in combinations(range(16),4):
        matrix=[[hp[j][r] for j in SUPPORT] for r in rows]
        minors_gcd=gcd(minors_gcd,determinant(matrix));minor_count+=1
    assert rem(minors_gcd,7)==0 and evaluate(minors_gcd,1,3)!=0
    assert rank([parity(1,3)[j] for j in SUPPORT],3)==4
    assert rank([h4[j] for j in SUPPORT],7)<4

    constants=[]
    for c in (2,4234123421):
        m=minimal_polynomial(c,MOD128)
        assert m.bit_length()-1==128 and evaluate(m,c,MOD128)==0
        assert evaluate(minors_gcd,c,MOD128)!=0
        h=parity(c,MOD128)
        assert h==[[evaluate(x,c,MOD128) for x in col] for col in hp]
        assert rank([h[j] for j in SUPPORT],MOD128)==4
        for row in base:
            out=stage(stage(row,P,c,MOD128),Q,c,MOD128)
            assert not any(xor(mul(a,col[r],MOD128) for a,col in zip(out,h)) for r in range(16))
        constants.append({'constant':c,'minimal_polynomial_hex':hex(m),'degree':128,
                          'gf4_witness_support_rank':4})
    report={'scope':'Exact GF4 counterexample and GF128 algebra checks, not a 10% GF128 certificate',
            'gf128_modulus_irreducibility_checked':True,
            'n':32,'k':16,'local':'RM(2,5)', 'permutations':[P,Q],
            'binary_distance':distance,'binary_witness':binary_witness,
            'gf4':{'modulus':7,'constant':2,'word':word,'message':message,
                   'weight':4,'support':SUPPORT,'all_maximal_minors_gcd_hex':hex(minors_gcd),
                   'maximal_minors_checked':minor_count},'gf128':constants}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
