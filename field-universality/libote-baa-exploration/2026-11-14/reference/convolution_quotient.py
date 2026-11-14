"""Width-two quotient-rank transfer, with exclusive rank-one outcomes.

Floating convex proposals are not certificates. Transfer rows themselves
are positive rational polynomials and admit exact Collatz verification.
"""
from fractions import Fraction as F
from itertools import product
from math import log, exp


def row_polynomials(q):
    """Each row is a list of alternative positive-polynomial upper bounds.

    A term is (coefficient, x power, y power, tau power, target state).
    States: 00/rank0, 01/rank1, 10/rank1, 11/rank1, 11/rank2.
    Required comparison: v[4]>=v[3].
    """
    assert q >= 2
    def term(c=1,x=0,y=0,t=0,j=0): return (F(c),x,y,t,j)
    def scaled(poly,c=1,x=0):
        return [(a*c,b+x,d,e,j) for a,b,d,e,j in poly]
    B = [term(y=1,j=3)]
    C = [term(F(q-1,q),y=1,j=3),term(F(1,q),j=2)]
    A = [term(y=1,j=4),term(t=-1,j=2)]
    A0 = [term(y=1,j=1),term(t=-1,j=0)]
    C0 = [term(F(q-1,q),y=1,j=1),term(F(1,q),j=0)]
    A2rare = scaled(A,F(q*q-1,q*q))+[term(F(1,q*q),j=2)]
    return [
        [[term(j=0),term(x=1,y=1,j=1)]],
        [[term(y=1,j=3)]+scaled(v,x=1) for v in (A,C)],
        [[term(y=1,j=1)]+scaled(v,x=1) for v in (A0,C0)],
        [u+scaled(v,x=1) for u,v in product((B,C),(A,C))],
        [A+scaled(v,x=1) for v in (A,A2rare)],
    ]


def collatz(q,tau,x,y,v):
    assert tau >= 1 and min(x,y,*v) > 0 and v[4] >= v[3]
    rows = row_polynomials(q)
    values = [max(sum(c*x**a*y**b*tau**d*v[j] for c,a,b,d,j in poly)
                  for poly in choices) for choices in rows]
    return max(a/b for a,b in zip(values,v))


def terminal_bound(q,tau,x,y,v,steps=16):
    """Exact finite terminal iteration, followed by a Collatz bound."""
    rows = row_polynomials(q)
    evaluated = [[[(c*x**a*y**b*tau**d,j) for c,a,b,d,j in poly]
                  for poly in choices] for choices in rows]
    current = [F(1)]*5
    for _ in range(steps):
        current = [max(sum(c*current[j] for c,j in poly) for poly in choices)
                   for choices in evaluated]
        # The local row bound replaces unknown rank-one continuations by
        # rank-two continuations. Preserve its admissibility condition.
        current[4] = max(current[3],current[4])
    return max(a/b for a,b in zip(current,v))


def source_coefficients(s,blocks=1 << 18):
    """Coefficient [z^s]Q(z,tau)^blocks, as a polynomial in tau."""
    from collections import defaultdict
    rows = [{0:1}]
    for weight in range(1,s+1):
        total = defaultdict(int)
        for d,c,dimension in [(5,56,1),(6,28,2),(7,8,3),(8,1,4)]:
            if d <= weight:
                for D,value in rows[weight-d].items():
                    total[D+dimension] += ((blocks+1)*d-weight)*c*value
        assert all(c % weight == 0 for c in total.values())
        rows.append({D:c//weight for D,c in total.items() if c})
    return rows[s]


def rational_profile(proposal,n=1 << 21,e=209715,steps=16):
    """Exact rational upper bound for ONE (source,middle) weight pair.

    Uses the exact local source coefficient and binomials for weights
    <=1024. It is not a sum over the full profile domain.
    """
    from math import comb
    from repeat_accumulate_precode import log_upper
    q = proposal['q']
    s = round(proposal['source_density']*n)
    h = round(proposal['middle_density']*n)
    assert 1 <= s <= 1024 and 1 <= h < n and n % 8 == 0
    values = [F(str(exp(x))) for x in proposal['values']]
    tau = max(F(1),values[1])
    terms = [(-1,tau)]
    data = dict(q=q,n=n,e=e,s=s,h=h,tau=str(tau),steps=steps,passes=[])
    source = sum(c*tau**D for D,c in source_coefficients(s,n//8).items())
    if not source:
        return dict(**data,log2_upper=str(-10**9))
    terms += [(1,source),(-1,F(comb(n,s)))]
    if h <= 1024:
        terms += [(-1,F(comb(n,h)))]
    else:
        terms += [(1,F(n+1)),(h,F(h,n)),(n-h,F(n-h,n))]
    for stage,base in enumerate((2,9)):
        x,y = values[base:base+2]
        if stage: y = min(F(1),y)
        v = [F(1)]+values[base+2:base+6]
        v[4] = max(v[3],v[4])
        lam = collatz(q,tau,x,y,v)
        terminal = terminal_bound(q,tau,x,y,v,steps)
        terms += [(n-steps,lam),(1,terminal),(-(s if not stage else h),x),
                  (-(h if not stage else e),y)]
        data['passes'].append(dict(x=str(x),y=str(y),v=list(map(str,v)),
                                   lam=str(lam),terminal=str(terminal)))
    data['log2_upper'] = str(log_upper(terms))
    return data


def propose_two(q,delta,a,b,start=None,exact_s=None,n=1 << 21):
    """Geometric-program proposal at fixed source/middle densities.

    Variables are log z, log tau, then (log x,log y,log v1..4,log lambda)
    for each pass. Constraints are convex log-sum-exp inequalities.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    size = 16
    degrees = np.array([0,5,6,7,8])
    dims = np.array([0,1,2,3,4])
    coeff = np.log([1,56,28,8,1])
    divisor=8
    if exact_s is not None:
        source=source_coefficients(exact_s,n//8)
        assert source and abs(a*n-exact_s)<1e-6
        degrees=np.zeros(len(source))
        dims=np.array(list(source))
        coeff=np.array([log(c) for c in source.values()])
        divisor=n
    linear = np.zeros(size)
    linear[0] = -a
    linear[2] = -a
    linear[3] = -b
    linear[8] = 1
    linear[9] = -b
    linear[10] = -delta
    linear[15] = 1
    if exact_s is not None:
        linear[0]=0
        linear[1]=-1/n
    assert 0<a<=1 and 0<b<=1
    entropy = sum(v*log(v) for v in (a,1-a,b,1-b) if v)
    constraints = []
    for base in (2,9):
        for row,choices in enumerate(row_polynomials(q)):
            for poly in choices:
                vectors,logs = [],[]
                for c,x,y,t,j in poly:
                    vector = np.zeros(size)
                    vector[1] = t
                    vector[base] = x
                    vector[base+1] = y
                    if j: vector[base+1+j] += 1
                    if row: vector[base+1+row] -= 1
                    vector[base+6] -= 1
                    vectors.append(vector)
                    logs.append(log(float(c)))
                constraints.append((np.array(vectors),np.array(logs)))
    def objective(point):
        weights = coeff+degrees*point[0]+dims*point[1]
        total = logsumexp(weights)
        probabilities = np.exp(weights-total)
        grad = linear.copy()
        grad[0] += probabilities@degrees/divisor
        grad[1] += probabilities@dims/divisor
        value=total/divisor+linear@point+entropy
        if exact_s is not None:
            for base in (2,9):
                tail=np.array([0]+[-point[j] for j in range(base+2,base+6)])
                tail_sum=logsumexp(tail)
                value+=tail_sum/n
                grad[base+2:base+6]-=np.exp(tail[1:]-tail_sum)/n
        return value,grad
    def feasible(point):
        values,jac = [],[]
        for vectors,logs in constraints:
            weights = vectors@point+logs
            total = logsumexp(weights)
            values.append(-total)
            jac.append(-np.exp(weights-total)@vectors)
        for base in (2,9):
            values.append(point[base+5]-point[base+4])
            vector = np.zeros(size)
            vector[base+5],vector[base+4] = 1,-1
            jac.append(vector)
        return np.array(values),np.array(jac)
    if start is None:
        start = np.zeros(size)
        start[0],start[1] = log(a)/5-1,3
        for base,density in ((2,a),(9,b)):
            start[base] = log(density)/2
            start[base+1] = -1
            start[base+6] = 1
    bounds = [(-35,12),(0,25)]
    if exact_s is not None:
        bounds[0]=(0,0)
        start=list(start)
        start[0]=0
    for stage in range(2):
        bounds += [(-35,15),(-35,12 if stage == 0 else 0)]+[(-40,40)]*4+[(-10,20)]
    scale=10 if a<1e-4 else 1
    result = minimize(lambda v:scale*objective(v)[0],start,
        jac=lambda v:scale*objective(v)[1],method="SLSQP",bounds=bounds,
        constraints=[dict(type="ineq",fun=lambda v:feasible(v)[0],jac=lambda v:feasible(v)[1])],
        options=dict(ftol=1e-11,maxiter=700))
    return dict(q=q,delta=delta,source_density=a,middle_density=b,exact_source=exact_s,
                exponent_per_n=float(result.fun/(scale*log(2))),
                values=result.x.tolist(),success=bool(result.success),
                message=result.message,min_constraint=float(min(feasible(result.x)[0])))


if __name__ == "__main__":
    import json
    from pathlib import Path
    records = []
    for q in (2,4,8,16):
        row = [propose_two(q,.1,a,b) for a,b in
               [(.0001,.001),(.001,.01),(.01,.05),(.05,.1),(.1,.1),
                (.2,.15),(.3,.2),(.4,.2),(.4,.3)]]
        records.extend(row)
        worst = max(row,key=lambda r:r['exponent_per_n'])
        print(q,worst['exponent_per_n'],worst['source_density'],worst['middle_density'],
              all(r['success'] for r in row),flush=True)
    path = Path(__file__).parent/'results/convolution-quotient-screen.json'
    path.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
