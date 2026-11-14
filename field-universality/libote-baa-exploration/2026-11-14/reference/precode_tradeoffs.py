"""Analytical searches and exact support certificates for addition circuits.

The search screens probability bounds, never sampled encoder distances.
"""
from fractions import Fraction as F
from itertools import combinations
from math import comb, ceil, sqrt, log
from pathlib import Path
import json

from block_support_precode import (polynomial_value,small_bound,floating_witness,
    propose_interval,interval_certificate,cached_logs)
from sparse_mixing_optimization import stage_certificate

ROOT = Path(__file__).parent
N_OUT = 2**20


def rref(rows, dimension):
    matrix = [list(map(F,row)) for row in rows]
    rank = 0
    for column in range(dimension):
        pivot = next((i for i in range(rank,len(matrix)) if matrix[i][column]),None)
        if pivot is None:
            continue
        matrix[rank],matrix[pivot] = matrix[pivot],matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value/scale for value in matrix[rank]]
        for i in range(len(matrix)):
            if i != rank and matrix[i][column]:
                multiple = matrix[i][column]
                matrix[i] = [a-multiple*b for a,b in zip(matrix[i],matrix[rank])]
        rank += 1
    return tuple(tuple(row) for row in matrix[:rank])


def subset_sum_polynomial(dimension):
    """Enumerate the proper flats of the nonzero Boolean-row configuration."""
    rows = [tuple((mask>>i)&1 for i in range(dimension))
            for mask in range(1,2**dimension)]
    subspaces = {rref(basis,dimension)
                 for size in range(dimension)
                 for basis in combinations(rows,size)}
    supports = set()
    for basis in subspaces:
        rank = len(basis)
        supports.add(tuple(len(rref(basis+(row,),dimension)) > rank for row in rows))
    poly = {0:1}
    for support in supports:
        weight = sum(support)
        poly[weight] = poly.get(weight,0)+1
    return dict(sorted(poly.items()))


def candidate(kind, dimension, length):
    if kind == 'mds':
        if length < dimension+1:
            raise ValueError('require redundant block')
        poly = {0:1,**{length-j:comb(length,j) for j in range(dimension)}}
        # Leading difference (point at infinity) and length-1 finite evaluations.
        gates = (dimension-1)*length-(dimension-1)*(dimension+2)//2
    elif kind == 'subset':
        assert length == 2**dimension-1
        poly = subset_sum_polynomial(dimension)
        gates = length-dimension
    else:
        raise ValueError(kind)
    blocks = (N_OUT//2+dimension-1)//dimension
    return dict(kind=kind,block_dimension=dimension,block_length=length,
                block_gates=gates,poly=poly,blocks=blocks,length=length*blocks,
                padded_input_dimension=dimension*blocks,
                precode_gates=(gates+2*length)*blocks-2)


def best_mixer(delta, distance=115344):
    n,k,e = N_OUT,N_OUT//2,distance-1
    retained,f = n-e,n-e-k+1
    a = f/retained
    volume_t = e/(n-e)
    best = None
    for q in range(2,42,2):
        for d in range(1,180):
            beta = (1-float(delta))**d*(1-1/q)+1/q
            if beta >= a:
                continue
            rt = a*(1-beta)/(beta*(1-a))
            bound = 128+(n*log(1+volume_t)-e*log(volume_t)
                +retained*log(1+(rt-1)*beta)-f*log(rt))/log(2)
            if bound < -40:
                result = dict(cost=d+q-3,samples=d,alphabet=q,
                              rank_t=str(F(round(rt*10**8),10**8)),
                              volume_t=str(F(e,n-e)),distance=distance)
                if best is None or result['cost'] < best['cost']:
                    best = result
                break
    return best


def moment_t(length, erasures):
    r = F(erasures,length-erasures)
    t = F(ceil((float(r)+2*sqrt(float(r)))*10**8),10**8)
    assert t > r and (t-r)**2 >= 4*r and t < 1
    return t


def screen_one(spec):
    L,N,poly = spec['blocks'],spec['length'],spec['poly']
    best = None
    for numerator in range(35,126,5):
        delta = F(numerator,1000)
        e = int(N*delta)
        t = moment_t(N,e)
        small = small_bound(poly,L,N,t,16)
        if small >= F(1,2**22):
            continue
        supports = sorted(set([129,4*e]+[int(129*(4*e/129)**(i/40)) for i in range(41)]))
        worst = max(floating_witness(poly,L,N,e,t,s)[1] for s in supports)
        if worst >= -110:
            continue
        mixer = best_mixer(delta)
        if mixer is None:
            continue
        total = spec['precode_gates']/N_OUT+mixer['cost']
        if best is None or total < best['total_per_n']:
            best = dict(delta=str(delta),small_log2=float(cached_logs(small)[1]),
                        grid_worst=worst,mixer=mixer,total_per_n=total)
    return dict(**spec,screen=best)


def generate(spec, delta, distance=115344, max_intervals=None):
    L,N,poly = spec['blocks'],spec['length'],spec['poly']
    e = int(N*delta)
    t = moment_t(N,e)
    cutoff = 128
    intervals = []
    lower,last = cutoff+1,4*e
    while lower <= last:
        if max_intervals is not None and len(intervals) >= max_intervals:
            raise ValueError('proposal exceeds interval search budget')
        upper = min(last,max(lower,int(lower*1.06)))
        record = propose_interval(poly,L,N,e,t,lower,upper)
        while F(record['interval_sum_log2_upper']) >= -70 and upper > lower:
            upper = (lower+upper)//2
            record = propose_interval(poly,L,N,e,t,lower,upper)
        if F(record['interval_sum_log2_upper']) >= -70:
            raise ValueError(f'uncertified support {lower}')
        intervals.append(record)
        lower = upper+1
    mixer = best_mixer(delta,distance)
    if mixer is None:
        raise ValueError('no mixer')
    return dict(spec=spec,delta=str(delta),erasures=e,moment_t=str(t),
                cutoff=cutoff,intervals=intervals,mixer=mixer)


def verify(data):
    spec = data['spec']
    expected = candidate(spec['kind'],spec['block_dimension'],spec['block_length'])
    spec = dict(spec,poly={int(j):c for j,c in spec['poly'].items()})
    assert spec == expected
    L,N,poly = spec['blocks'],spec['length'],spec['poly']
    e,t,delta = data['erasures'],F(data['moment_t']),F(data['delta'])
    assert 0 < e and 4*e < N and F(e+1,N) > delta
    r = F(e,N-e)
    assert r < t < 1 and (t-r)**2 >= 4*r
    small = small_bound(poly,L,N,t,data['cutoff'])
    following = data['cutoff']+1
    for record in data['intervals']:
        assert record['lower'] == following
        checked = interval_certificate(poly,L,N,e,t,record['lower'],record['upper'],
                                       *(F(record[v]) for v in ('x','y','v','z')))
        assert checked == record and F(checked['interval_sum_log2_upper']) < -70
        following = record['upper']+1
    assert following == 4*e+1
    pre_failure = small+F(len(data['intervals']),2**70)
    assert pre_failure < F(1,2**21)
    m = data['mixer']
    final = stage_certificate(N_OUT,N_OUT//2,m['distance']-1,delta,
                             m['samples'],m['alphabet'],F(1,2),
                             F(m['volume_t']),F(m['rank_t']))
    assert F(final['all_128_bit_primes_failure_log2_upper']) < -40
    assert final['additions'] == m['cost']*N_OUT
    assert pre_failure+F(1,2**40) < F(1,2**20)
    return dict(kind=spec['kind'],block_dimension=spec['block_dimension'],
                block_length=spec['block_length'],block_gates=spec['block_gates'],
                support_polynomial={str(j):c for j,c in poly.items()},
                precode_length=N,expansion_over_n=str(F(N,N_OUT)),
                precode_distance=e+1,precode_relative_distance_lower=str(delta),
                precode_failure_log2_upper=str(cached_logs(pre_failure)[1]),
                final_distance=m['distance'],final_samples=m['samples'],
                final_alphabet=m['alphabet'],final_additions=final['additions'],
                final_failure_log2_upper=final['all_128_bit_primes_failure_log2_upper'],
                additions=spec['precode_gates']+final['additions'],
                additions_over_n=str(F(spec['precode_gates']+final['additions'],N_OUT)),
                setup_failure_less_than='2^-20',simultaneous_all_128_bit_primes=True)


def screen():
    specifications = [candidate('mds',b,m) for b in (3,4)
                      for m in range(b+4,4*b+1,2)]
    specifications += [candidate('subset',b,2**b-1) for b in (3,4)]
    rows = []
    for spec in specifications:
        row = screen_one(spec)
        rows.append(row)
        print(json.dumps(row),flush=True)
        (ROOT/'results/precode-tradeoff-screen.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'screen':
        screen()
    elif sys.argv[1] == 'generate':
        spec = candidate(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
        delta = F(sys.argv[5])
        distance = int(sys.argv[6]) if len(sys.argv)>6 else 115344
        data = generate(spec,delta,distance)
        result = verify(data)
        name = f"tradeoff-{spec['kind']}-{spec['block_dimension']}-{spec['block_length']}-{distance}"
        (ROOT/f'results/{name}-cover.json').write_text(json.dumps(data,indent=2)+'\n')
        (ROOT/f'results/{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    elif sys.argv[1] == 'verify':
        print(json.dumps(verify(json.loads(Path(sys.argv[2]).read_text())),indent=2))
