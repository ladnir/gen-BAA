"""Finite stopping-pattern certificates for an addition-only R6AA precode.

No encoder is sampled. Floating optimization only proposes rational witnesses;
verification of each witness and every probability bound is exact.
"""

from fractions import Fraction as F
from functools import lru_cache
from math import comb
from pathlib import Path
import json

from sparse_mixing_optimization import log2_bounds, stage_certificate

ROOT = Path(__file__).parent


def binom(n,r):
    return comb(n,r) if 0 <= r <= n else 0


def support_enumerator(length,input_weight,output_weight):
    """Unterminated accumulator: fixed zero initial state, either end state."""
    if input_weight == 0:
        return int(output_weight == 0)
    if output_weight == 0:
        return 0
    s,h,N = input_weight,output_weight,length
    return sum(binom(h-1,r-1)*(
        binom(N-h,r)*binom(h-r,s-2*r)
        +binom(N-h,r-1)*binom(h-r,s-2*r+1))
        for r in range(1,h+1))


def moment_upper(length,input_weight,t):
    """Sum_h A(input_weight,h)*t^h, bounded by extending run lengths."""
    B = t/(1-t)
    s,N = input_weight,length
    if s == 0:
        return F(1)
    closed = sum(F(binom(N,r)*binom(s-r-1,r-1))*B**(s-r)
                 for r in range(1,s//2+1))
    open_run = sum(F(binom(N,r-1)*binom(s-r,r-1))*B**(s-r+1)
                   for r in range(1,(s+1)//2+1))
    return closed+open_run


def small_support_bound(dimension,repetitions,t,cutoff):
    N = dimension*repetitions
    return 2*sum(F(comb(dimension,w),comb(N,repetitions*w))
                 *moment_upper(N,repetitions*w,t)
                 for w in range(1,cutoff+1))


@lru_cache(None)
def cached_logs(value):
    return log2_bounds(value)


def log_upper(terms):
    total = F(0)
    for coefficient,argument in terms:
        lo,hi = cached_logs(F(argument))
        total += coefficient*(hi if coefficient>=0 else lo)
    return total


def interval_certificate(dimension,repetitions,erasures,t,lower,upper,x,y,v):
    if not (1<=lower<=upper<=dimension and x>0 and v>0 and 0<y<=t<1):
        raise ValueError('invalid interval witness')
    N = dimension*repetitions
    lam = max(1+x*y*v, (x+(1+x)*y*v)/v)
    prefactor = max(F(1),1/v)
    common = [(1,2*(N+1)*prefactor),(N,lam),(2*erasures,t/y)]
    endpoint_bounds=[]
    for w in (lower,upper):
        alpha = F(w,dimension)
        endpoint_bounds.append(log_upper(common+[
            (-repetitions*w,x),
            ((N-dimension)*alpha,alpha),
            ((N-dimension)*(1-alpha),1-alpha)]))
    # Convexity of -H_2(alpha) bounds every interior integer input support.
    per_weight = max(endpoint_bounds)
    grouped = per_weight+cached_logs(F(upper-lower+1))[1]
    return dict(lower=lower,upper=upper,x=str(x),y=str(y),v=str(v),
                lambda_upper=str(lam),per_weight_log2_upper=str(per_weight),
                interval_sum_log2_upper=str(grouped))


def propose_witness(dimension,repetitions,erasures,t,lower,upper):
    """Heuristic parameters only; interval_certificate proves their validity."""
    from math import exp,log,sqrt
    from scipy.optimize import minimize
    N = dimension*repetitions
    alpha = (lower+upper)/(2*dimension)
    cap = 2*erasures/N
    def objective(point):
        x,y = exp(point[0]),exp(point[1])
        lam = (1+(1+x)*y+sqrt((1-(1+x)*y)**2+4*x*x*y))/2
        return log(lam)-alpha*log(x)+cap*log(float(t)/y)
    result = minimize(objective,[log(max(.001,sqrt(alpha))),log(float(t))],
                      method='L-BFGS-B',bounds=[(-20,5),(-20,log(float(t)))],
                      options=dict(ftol=1e-14,gtol=1e-10,maxiter=300))
    xf,yf = exp(result.x[0]),exp(result.x[1])
    lam = (1+(1+xf)*yf+sqrt((1-(1+xf)*yf)**2+4*xf*xf*yf))/2
    vf = (lam-1)/(xf*yf)
    scale = 10**9
    x,y,v = F(round(xf*scale),scale),min(t,F(round(yf*scale),scale)),F(round(vf*scale),scale)
    return interval_certificate(dimension,repetitions,erasures,t,lower,upper,x,y,v)


def generate_cover():
    dimension,repetitions,t,cutoff = 2**19,6,F(2,3),31
    N = dimension*repetitions
    erasures = 3*N//40
    last = min(dimension,4*erasures//repetitions)
    intervals=[]
    lower=cutoff+1
    while lower<=last:
        upper=min(last,max(lower,int(lower*1.08)))
        record=propose_witness(dimension,repetitions,erasures,t,lower,upper)
        while F(record['interval_sum_log2_upper']) >= -80 and upper>lower:
            upper=(lower+upper)//2
            record=propose_witness(dimension,repetitions,erasures,t,lower,upper)
        if F(record['interval_sum_log2_upper']) >= -80:
            raise ValueError(f'uncertified weight {lower}: {record}')
        intervals.append(record)
        lower=upper+1
    return dict(dimension=dimension,repetitions=repetitions,length=N,
                erasures=erasures,moment_t=str(t),small_support_cutoff=cutoff,
                intervals=intervals)


def verify_cover(data):
    k,q,e,t = data['dimension'],data['repetitions'],data['erasures'],F(data['moment_t'])
    N=k*q
    assert data['length']==N
    z=F(e,N-e)
    assert t>z and (t-z)**2>=4*z
    small=small_support_bound(k,q,t,data['small_support_cutoff'])
    next_weight=data['small_support_cutoff']+1
    for recorded in data['intervals']:
        assert recorded['lower']==next_weight
        checked=interval_certificate(k,q,e,t,recorded['lower'],recorded['upper'],
                                     F(recorded['x']),F(recorded['y']),F(recorded['v']))
        assert checked==recorded
        assert F(checked['interval_sum_log2_upper']) < -80
        next_weight=checked['upper']+1
    assert next_weight==min(k,4*e//q)+1
    total=small+F(len(data['intervals']),2**80)
    return small,total


def report():
    data=json.loads((ROOT/'results/repeat-accumulate-cover.json').read_text(encoding='utf-8'))
    small,total=verify_cover(data)
    n=2**20
    final=stage_certificate(n,n//2,115343,F(3,43),66,12,
                            F(11,89),F(11,89),F(8),True)
    assert (data['dimension'],data['repetitions'],data['length'],data['erasures']) == (
        2**19,6,3*2**20,235929)
    assert total < F(1,2**34)
    assert F(final['binary_failure_log2_upper']) < -31
    assert F(final['all_128_bit_primes_failure_log2_upper']) < -128
    assert F(1,2**34)+F(1,2**31)+F(1,2**128) < F(1,2**30)
    padded=data['length']+data['erasures']+1
    assert F(data['erasures']+1,padded) > F(3,43)
    assert 2*(data['length']-1)+final['additions'] == 83*n-2
    return dict(n=n,k=n//2,precode_length=data['length'],
                precode_distance=data['erasures']+1,
                zero_padding=data['erasures']+1,
                padded_precode_length=padded,
                final_input_relative_distance='3/43',
                precode_additions=2*(data['length']-1),
                small_support_failure_log2_upper=str(cached_logs(small)[1]),
                precode_failure_log2_upper=str(cached_logs(total)[1]),
                bulk_intervals=len(data['intervals']),final_stage=final,
                worst_case_additions=2*(data['length']-1)+final['additions'],
                additions_per_n=str(F(2*(data['length']-1)+final['additions'],n)),
                combined_failure_less_than='2^-30',
                minimum_distance_at_least=115344,rate='1/2',
                simultaneous_binary_and_every_128_bit_prime=True,
                precode_support_certificate_covers_every_field=True,
                addition_only=True,subtractions_required=False,
                practical_100n_budget_met=True,
                randomness_is_encoder_setup=True,random_prime_transfer_theorem=False,
                concrete_encoder_sampled=False,sampled_encoders=0,
                linear_time_asymptotic_gv_theorem_proved_here=False)


if __name__ == '__main__':
    import sys
    if '--generate' in sys.argv:
        cover=generate_cover()
        (ROOT/'results/repeat-accumulate-cover.json').write_text(
            json.dumps(cover,indent=2)+'\n',encoding='utf-8')
        print('generated intervals:',len(cover['intervals']))
    result=report()
    (ROOT/'results/repeat-accumulate-precode.json').write_text(
        json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
