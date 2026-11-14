"""Replay the MDS [16,8,9] + two plain scans distance certificate.

Usage: python verify_q1_field_transfer.py /path/to/gen-BAA
The mathematical field-transfer argument is in ../RsPlainScans.md.
The original verifier's Newton implementation costs do not transfer to GF128.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

research = Path(sys.argv[1]) / 'field-universality/expander-lift'
sys.path.insert(0, str(research))
import convolution_width_one_finite as certificate

ledger = (research / 'results/convolution-width-one-direct-r8-q1-e138412.json').read_bytes()
assert hashlib.sha256(ledger).hexdigest() == '0b4a3577c9cd20f0eedd8aa02f255a5b096d7a0cf50ffc9de738e9e866b674f2'
data = json.loads(ledger)
assert (data['n'], data['q'], data['block_k'], data['e']) == (2097152, 1, 8, 138412)
assert data['complete'] and data['steps'] == 16
assert 'sparse_support_moment' in data and not data['sparse']
result = certificate.verify(data)
assert result['dense_leaves'] == 137 and result['sparse_intervals'] == 0
assert result['minimum_distance_at_least'] == 138413
assert Fraction(result['failure_log2_upper']) == Fraction(-13490343832277,250000000000)
print(json.dumps({
    'construction': 'MDS [16,8,9]; two independent uniform permutations; two plain accumulators',
    'field': 'one fixed field admitting the local MDS code, including GF(2^128)',
    'n': result['n'], 'k': result['k'],
    'minimum_distance_at_least': result['minimum_distance_at_least'],
    'failure_upper': result['failure_upper'],
    'failure_log2_upper': result['failure_log2_upper'],
    'dense_leaves': result['dense_leaves'],
    'ledger_sha256': hashlib.sha256(ledger).hexdigest(),
}, indent=2))
