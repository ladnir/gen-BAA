"""Check a distance-search certificate with all sparse and bulk terms."""
import argparse
import json
from fractions import Fraction as F
from pathlib import Path
from flint import arb
from direct_sparse import direct_sparse
from verify_bulk import verify_covering,a


def verify_distance(path,precision=192,q_min=2**127,reported_bits='40'):
    data=json.loads(Path(path).read_text())
    if 'delta' not in data:raise ValueError('Distance certificate must specify delta')
    md,leaves,_,logbulk=verify_covering(data,q_min=q_min,precision=precision)
    sigma=data['sigma'];n=md['n'];delta=md['delta']
    sparse=direct_sparse(n,16,sigma,delta,q_min,precision=precision)
    total=sparse['bound']+logbulk.exp()
    if not total<a(F(1,2**40)):raise ValueError('Complete bound does not meet 40-bit target')
    bits=F(str(reported_bits))
    if bits<40:raise ValueError('Reported margin must be at least 40 bits')
    if not total.log() < -a(bits)*arb(2).log():
        raise ValueError('Complete bound does not meet reported margin')
    return {'constituent':f'RS[16,{sigma},{17-sigma}]','message_dimension':2**20,
            'length':n,'delta':str(delta),'L':int(delta*n),'field_size_lower_bound':str(q_min),
            'bulk_leaves':leaves,'sparse_max_active_blocks':sparse['T'],
            'bulk_min_active_blocks':sparse['T']+1,
            'sparse_log2_upper':str((sparse['bound'].log()/arb(2).log()).upper()),
            'sparse_parts_log2_upper':{key:str((value.log()/arb(2).log()).upper())
                                       for key,value in sparse['parts'].items()},
            'bulk_log2_upper':str((logbulk/arb(2).log()).upper()),
            'total_log2_upper':str((total.log()/arb(2).log()).upper()),
            'failure_bound_below_2^-40':True,'arithmetic_precision_bits':precision,
            'certified_failure_bound_bits_decimal':str(reported_bits),
            'scope':'Independent uniform permutations and independent uniform nonzero diagonal entries; complete support/trace bound. Written analytic reductions, not Lean-formalized.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('certificate')
    parser.add_argument('--precision',type=int,default=192)
    parser.add_argument('--bits',default='40',help='Margin to check, as an exact decimal or rational')
    args=parser.parse_args()
    print(json.dumps(verify_distance(args.certificate,precision=args.precision,reported_bits=args.bits),indent=2))
