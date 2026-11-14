"""Support-count certificates for short addition-only blocks followed by AA.

No encoders or field elements are sampled. Numerical optimization proposes
witnesses; exact rational interval bounds verify the selected construction.
"""

from fractions import Fraction as F
from math import comb, exp, log, sqrt
from pathlib import Path
import json

from repeat_accumulate_precode import cached_logs, log_upper, moment_upper
from sparse_mixing_optimization import stage_certificate

ROOT = Path(__file__).parent


def block_polynomial(m):
    # (a,b,a+b,...,a+(m-2)b): full support or precisely one zero.
    if m < 3:
        raise ValueError('block length must be at least three')
    return {0: 1, m-1: m, m: 1}


def polynomial_value(poly, z):
    return sum(c*z**j for j,c in poly.items())


def support_coefficients(poly, blocks, cutoff):
    """Exact initial coefficients of U(z)^blocks using U F'=blocks U'F."""
    if poly.get(0) != 1:
        raise ValueError('constant coefficient must be one')
    coefficients = [1]
    for s in range(1, cutoff+1):
        numerator = sum(((blocks+1)*j-s)*c*coefficients[s-j]
                        for j,c in poly.items() if 0 < j <= s)
        assert numerator % s == 0
        coefficients.append(numerator//s)
    return coefficients


def small_bound(poly, blocks, length, t, cutoff):
    coefficients = support_coefficients(poly, blocks, cutoff)
    return 2*sum(F(c,comb(length,s))*moment_upper(length,s,t)
                 for s,c in enumerate(coefficients) if s and c)


def interval_certificate(poly, blocks, length, erasures, t,
                         lower, upper, x, y, v, z):
    if not (1 <= lower <= upper < length and x > 0 and v > 0
            and z > 0 and 0 < y <= t < 1):
        raise ValueError('invalid witness')
    lam = max(1+x*y*v, (x+(1+x)*y*v)/v)
    common = [(1,2*(length+1)*max(F(1),1/v)), (length,lam),
              (2*erasures,t/y), (blocks,polynomial_value(poly,z))]
    endpoints = []
    for s in (lower,upper):
        alpha = F(s,length)
        endpoints.append(log_upper(common+[
            (-s,x*z), (s,alpha), (length-s,1-alpha)]))
    bound = max(endpoints)+cached_logs(F(upper-lower+1))[1]
    return dict(lower=lower,upper=upper,x=str(x),y=str(y),v=str(v),z=str(z),
                interval_sum_log2_upper=str(bound))


def floating_witness(poly, blocks, length, erasures, t, support):
    from scipy.optimize import minimize, brentq
    alpha,cap = support/length,2*erasures/length
    def outer_mean(logz):
        # Proposal search only: normalize log weights so the root bracket
        # works for large block degrees without overflowing z**degree.
        terms = [(j,log(c)+j*logz) for j,c in poly.items() if c]
        shift = max(value for _,value in terms)
        weights = [(j,exp(value-shift)) for j,value in terms]
        return sum(j*w for j,w in weights)/sum(w for _,w in weights)
    logz = brentq(lambda z: outer_mean(z)-support/blocks,-40,20)
    def objective(point):
        x,y = exp(point[0]),exp(point[1])
        lam = (1+(1+x)*y+sqrt((1-(1+x)*y)**2+4*x*x*y))/2
        return log(lam)-alpha*log(x)+cap*log(float(t)/y)
    result = minimize(objective,[log(max(.0001,sqrt(alpha))),log(float(t))],
                      method='L-BFGS-B',bounds=[(-25,8),(-25,log(float(t)))],
                      options=dict(ftol=1e-14,gtol=1e-10,maxiter=250))
    x,y,z = exp(result.x[0]),exp(result.x[1]),exp(logz)
    lam = (1+(1+x)*y+sqrt((1-(1+x)*y)**2+4*x*x*y))/2
    v = x/(lam-(1+x)*y)
    entropy = -alpha*log(alpha)-(1-alpha)*log(1-alpha)
    exponent = (length*result.fun+blocks*log(polynomial_value(poly,z))
                -support*logz-length*entropy)/log(2)
    return (x,y,v,z),exponent


def propose_interval(poly, blocks, length, erasures, t, lower, upper):
    values,_ = floating_witness(poly,blocks,length,erasures,t,(lower+upper)/2)
    scale = 10**12
    x,y,v,z = (F(round(value*scale),scale) for value in values)
    return interval_certificate(poly,blocks,length,erasures,t,lower,upper,
                                x,min(t,y),v,z)


def generate_cover(m, delta, cutoff=128):
    n = 2**20
    blocks = n//4
    length = m*blocks
    erasures = int(delta*length)
    ratio = F(erasures,length-erasures)
    # A decimal rational upper bound on ratio+2sqrt(ratio), verified exactly.
    from math import ceil
    t = F(ceil((float(ratio)+2*sqrt(float(ratio)))*10**8),10**8)
    assert t > ratio and (t-ratio)**2 >= 4*ratio and t < 1
    poly = block_polynomial(m)
    intervals = []
    lower,last = cutoff+1,4*erasures
    while lower <= last:
        upper = min(last,max(lower,int(lower*1.06)))
        record = propose_interval(poly,blocks,length,erasures,t,lower,upper)
        while F(record['interval_sum_log2_upper']) >= -70 and upper > lower:
            upper = (lower+upper)//2
            record = propose_interval(poly,blocks,length,erasures,t,lower,upper)
        if F(record['interval_sum_log2_upper']) >= -70:
            raise ValueError(f'uncertified support {lower}: {record}')
        intervals.append(record)
        lower = upper+1
    return dict(n=n,dimension=n//2,block_length=m,blocks=blocks,length=length,
                erasures=erasures,moment_t=str(t),cutoff=cutoff,intervals=intervals)


def verify_cover(data):
    n,m,L,N = data['n'],data['block_length'],data['blocks'],data['length']
    assert n == 2**20 and data['dimension'] == n//2 and L == n//4 and N == m*L
    e,t = data['erasures'],F(data['moment_t'])
    ratio = F(e,N-e)
    assert 0 < e and 4*e < N and t > ratio and (t-ratio)**2 >= 4*ratio and t < 1
    poly = block_polynomial(m)
    small = small_bound(poly,L,N,t,data['cutoff'])
    next_support = data['cutoff']+1
    for record in data['intervals']:
        assert record['lower'] == next_support
        checked = interval_certificate(poly,L,N,e,t,record['lower'],record['upper'],
                                       *(F(record[v]) for v in ('x','y','v','z')))
        assert checked == record and F(checked['interval_sum_log2_upper']) < -70
        next_support = record['upper']+1
    assert next_support == 4*e+1
    return small,small+F(len(data['intervals']),2**70)


def report():
    data = json.loads((ROOT/'results/block-support-cover.json').read_text())
    small,total = verify_cover(data)
    n,N,e = data['n'],data['length'],data['erasures']
    assert data['block_length'] == 13 and e+1 == 323748
    assert F(e+1,N) > F(19,200)
    final = stage_certificate(n,n//2,115343,F(19,200),33,16,
                              F(11,89),F(11,89),F(29,4))
    # The binary expression returned by the shared helper is not a claim here.
    final.pop('binary_failure_log2_upper')
    final.pop('binary_bias_upper')
    final.pop('binary_t')
    final.pop('two_sided_binary')
    assert F(final['all_128_bit_primes_failure_log2_upper']) < -128
    assert total < F(1,2**62)
    assert F(1,2**62)+F(1,2**128) < F(1,2**61)
    precode_additions = (data['block_length']-2)*data['blocks']+2*(N-1)
    additions = precode_additions+final['additions']
    assert additions == 221*n//4-2
    baseline = stage_certificate(n,n//2,115343,F(3,40),42,16,
                                 F(11,89),F(11,89),F(36,5))
    assert F(baseline['all_128_bit_primes_failure_log2_upper']) < -128
    assert baseline['additions']+6*n-2 == 61*n-2
    return dict(n=n,k=n//2,minimum_distance_at_least=115344,
                block_length=13,block_dimension=2,block_distance=12,
                block_additions=11,precode_length=N,precode_distance=e+1,
                precode_failure_log2_upper=str(cached_logs(total)[1]),
                small_support_failure_log2_upper=str(cached_logs(small)[1]),
                bulk_intervals=len(data['intervals']),precode_additions=precode_additions,
                final_stage=final,worst_case_additions=additions,
                additions_per_n_upper='55.25',combined_failure_less_than='2^-61',
                simultaneous_every_128_bit_prime=True,binary_guarantee=False,
                random_prime_transfer_theorem=False,sampled_encoders=0,
                prime_only_repetition_baseline_additions=61*n-2)


def screen():
    """Heuristic search over rigorous bounds, never an encoder experiment."""
    rows = []
    n = 2**20
    for m in range(5,17):
        L,N,poly = n//4,m*n//4,block_polynomial(m)
        best = None
        for numerator in range(40,136,5):
            delta = F(numerator,1000)
            e = int(N*delta)
            ratio = F(e,N-e)
            t = F(round((float(ratio)+2*sqrt(float(ratio)))*10**8)+1,10**8)
            small = small_bound(poly,L,N,t,16)
            if small >= F(1,2**22):
                continue
            supports = sorted(set([129,4*e]+[int(129*(4*e/129)**(i/55)) for i in range(56)]))
            exponents = [floating_witness(poly,L,N,e,t,s)[1] for s in supports]
            if max(exponents) >= -120:
                continue
            # Prime-only final mixer: uniform coefficients 0,...,q-1.
            best_stage = None
            for q in range(2,42,2):
                for d in range(1,140):
                    beta = (1-float(delta))**d*(1-1/q)+1/q
                    retained = n-115343
                    failures = retained-n//2+1
                    a = failures/retained
                    if beta >= a:
                        continue
                    rt = a*(1-beta)/(beta*(1-a))
                    bound = 128+(n*log(1+11/89)-115343*log(11/89)
                        +retained*log(1+(rt-1)*beta)-failures*log(rt))/log(2)
                    if bound < -40:
                        cost = d+q-3
                        if best_stage is None or cost < best_stage[0]:
                            best_stage = (cost,d,q,rt,bound)
                        break
            if best_stage is None:
                continue
            total = (3*m-2)/4+best_stage[0]
            row = dict(m=m,delta=str(delta),small_log2=float(cached_logs(small)[1]),
                       grid_worst_log2=max(exponents),mixer=best_stage,total_per_n=total)
            if best is None or total < best['total_per_n']:
                best = row
        rows.append(best or dict(m=m,no_candidate=True))
        print(json.dumps(rows[-1]),flush=True)
    (ROOT/'results/block-support-screen.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'screen':
        screen()
    elif len(sys.argv) > 1 and sys.argv[1] == 'generate':
        data = generate_cover(int(sys.argv[2]),F(sys.argv[3]))
        small,total = verify_cover(data)
        data['verified_small_log2_upper'] = str(cached_logs(small)[1])
        data['verified_total_log2_upper'] = str(cached_logs(total)[1])
        (ROOT/'results/block-support-cover.json').write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({k:v for k,v in data.items() if k != 'intervals'},indent=2))
        print('intervals',len(data['intervals']))
    else:
        result = report()
        (ROOT/'results/block-support-precode.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
