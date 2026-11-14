"""Replay the direct q=2 certificate for the nonbinary low-tier candidate.

Usage: python verify_q2_field_transfer.py /path/to/gen-BAA
The field-transfer argument is in ../NonbinaryLowTier.md. The upstream
verifier also reports prime-field implementation costs; those costs do not
transfer to the GF128 implementation and are deliberately omitted here.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(sys.argv[1]) / 'field-universality/expander-lift'))
import convolution_width_one_finite as certificate

ledger = gzip.decompress((root / 'q2-direct.json.gz').read_bytes())
data = json.loads(ledger)
assert (data['n'], data['q'], data['block_k'], data['e']) == (2097152, 2, 8, 209715)
assert 'sparse_support_moment' not in data
assert data['complete'] and data['steps'] == 16
result = certificate.verify(data)
assert result['sparse_intervals'] == 167 and result['dense_leaves'] == 45
assert result['minimum_distance_at_least'] == 209716
assert result['failure_upper'] == (
    '135807966222723081714348558404059809/'
    '1461501637330902918203684832716283019655932542976')
print(json.dumps({
    'construction': 'MDS [16,8,9]; two independent uniform permutations; two weighted scans',
    'field': 'one fixed field admitting the MDS code and two distinct nonzero coefficients',
    'gf128_coefficients': ['1', 'x in the canonical polynomial basis'],
    'n': result['n'], 'k': result['k'],
    'minimum_distance_at_least': result['minimum_distance_at_least'],
    'failure_upper': result['failure_upper'],
    'failure_log2_upper': result['failure_log2_upper'],
    'sparse_intervals': result['sparse_intervals'],
    'dense_leaves': result['dense_leaves'],
    'ledger_sha256': hashlib.sha256(ledger).hexdigest(),
}, indent=2))
