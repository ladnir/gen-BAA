"""Verify all regions and the complete RS bounds stated in the manuscript."""
from pathlib import Path
from fractions import Fraction as F
import json
from flint import arb,ctx
from direct_sparse import direct_sparse
from verify_bulk import verify_covering,a


def complete(sigma,precision=192,q_min=2**127):
    ctx.prec=precision
    directory=Path(__file__).resolve().parent
    data=json.loads((directory/f'bulk-rs16-{sigma}.json').read_text())
    if data['sigma']!=sigma:raise ValueError('Certificate constituent mismatch')
    # Reconstruct and verify every leaf and the covering from scratch.
    md,leaves,_,logbulk=verify_covering(data,q_min=q_min,precision=precision)
    n=md['n']
    bulk=logbulk.exp()
    sparse=direct_sparse(n,16,sigma,md['delta'],q_min,precision=precision)
    total=sparse['bound']+bulk
    target=a(F(1,2**40))
    if not total<target:raise ValueError('Complete bound did not meet target')
    reported_bits=209 if sigma==4 else 137
    if not total<a(F(1,2**reported_bits)):
        raise ValueError('Complete bound did not meet reported dyadic bound')
    decimal_bits='209.60' if sigma==4 else '137.56'
    if not total.log() < -a(F(decimal_bits))*arb(2).log():
        raise ValueError('Complete bound did not meet reported decimal bound')
    relative_gap=total/sparse['parts']['no_surplus']-1
    gap_limit=F(5,10**13) if sigma==4 else F(33,10**8)
    if not relative_gap<a(gap_limit):
        raise ValueError('Master-sum enclosure did not meet reported precision')
    return {'constituent':f'RS[16,{sigma},{17-sigma}]','field_size_lower_bound':str(q_min),
            'message_dimension':2**20,'length':n,'delta':str(md['delta']),
            'L':int(md['delta']*n),'sparse_max_active_blocks':sparse['T'],
            'bulk_min_active_blocks':sparse['T']+1,'bulk_leaves':leaves,
            'sparse_log2_upper':str((sparse['bound'].log()/arb(2).log()).upper()),
            'sparse_parts_log2_upper':{key:str((value.log()/arb(2).log()).upper())
                                       for key,value in sparse['parts'].items()},
            'direct_maximum_intermediate_weight':sparse['maximum_intermediate_weight'],
            'direct_maximum_active_blocks':sparse['direct_blocks'],
            'master_bound_log2_lower':str((sparse['parts']['no_surplus'].log()/arb(2).log()).lower()),
            'master_bound_relative_gap_upper':str(relative_gap.upper()),
            'bulk_log2_upper':str((logbulk/arb(2).log()).upper()),
            'total_log2_upper':str((total.log()/arb(2).log()).upper()),
            'failure_bound_below_2^-40':True,'certified_failure_bound_bits':reported_bits,
            'certified_failure_bound_bits_decimal':decimal_bits,
            'arithmetic_precision_bits':precision,
            'scope':'Uniform independent permutations and nonzero diagonal entries; analytic reductions in SPARSE_BOUND.md, REFINED_SPARSE.md, and BULK_BOUND.md, not Lean-formalized. The master-bound lower estimate is not a lower bound on the true failure probability.'}


if __name__=='__main__':
    print(json.dumps([complete(4),complete(8)],indent=2))
