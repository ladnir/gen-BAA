"""Exact local RS and counting checks for RsPlainDenseSupport.md.

This verifies proof ingredients, not a complete 10% distance certificate.
Run with Python 3; no third-party packages are needed.
"""
from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
from math import comb
import json


def clmul(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def local_catalog():
    multiplicity = Counter()
    energy = Counter()
    digest = sha256()
    witness = None
    for zeros in combinations(range(16), 7):
        word = []
        for point in range(16):
            value = 1
            for root in zeros:
                value = clmul(value, point ^ root)
            # Degree <=21: no reduction modulo the degree-128 field
            # polynomial occurs. These are exact GF128 products.
            assert value.bit_length() <= 22
            word.append(value)
        assert tuple(i for i, v in enumerate(word) if not v) == zeros
        counts = Counter(v for v in word if v)
        largest = max(counts.values())
        multiplicity[largest] += 1
        energy[sum(c * c for c in counts.values())] += 1
        digest.update(bytes(zeros))
        for v in word:
            digest.update(v.to_bytes(4, 'little'))
        if witness is None or largest > witness['multiplicity']:
            witness = dict(zeros=zeros, word=word, multiplicity=largest)
    assert dict(multiplicity) == {1: 8848, 2: 2544, 3: 48}
    assert dict(energy) == {9: 8848, 11: 2544, 15: 48}
    assert sum(multiplicity.values()) == comb(16, 7)
    return dict(supports=comb(16, 7), multiplicity_histogram=dict(multiplicity),
                energy_histogram=dict(energy), digest=digest.hexdigest(), witness=witness)


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def support_count(n, s, h, closed, pairs):
    """Exact admissible support-pattern count, including an open final run."""
    assert n >= 1 and s >= 1 and h >= 1
    return (choose(n - h, closed) * choose(h - 1, s - closed - 1)
            * choose(closed, pairs) * choose(s - 2 * closed, closed - pairs))


def enumerate_supports(n):
    counts = Counter()

    def visit(left, feedback, active_in_run, s, h, closed, pairs):
        if not left:
            if s:
                counts[s, h, closed, pairs] += 1
            return
        if not feedback:
            visit(left - 1, 0, 0, s, h, closed, pairs)
            visit(left - 1, 1, 1, s + 1, h + 1, closed, pairs)
        else:
            visit(left - 1, 1, active_in_run, s, h + 1, closed, pairs)
            visit(left - 1, 1, active_in_run + 1, s + 1, h + 1, closed, pairs)
            visit(left - 1, 0, 0, s + 1, h, closed + 1,
                  pairs + (active_in_run == 1))
    visit(n, 0, 0, 0, 0, 0, 0)
    return counts


def pair_rank_moment(s, dimension, pairs, tau=F(1), minimum_only=False):
    """Exact upper chain with killing at full rank; fixed source support.

    The proof of stochastic domination is in RsPlainDenseSupport.md.
    It depends on the verified RS constituent, not arbitrary MDS codes.
    """
    assert 1 <= dimension <= s and 0 <= 2 * pairs <= s and tau >= 1
    assert not minimum_only or s == 9 * dimension
    mass = [F(1)] + [F(0)] * (dimension - 1)
    for j in range(pairs):
        remaining = s - 2 * j
        after = [F(0)] * dimension
        for rank, value in enumerate(mass):
            rho = min(F(1), F(3 * rank + 2, remaining - 1))
            if minimum_only:
                rho = min(rho, F(15 * rank * (rank + 1) + 6 * dimension,
                                 remaining * (remaining - 1)))
            after[rank] += value * rho
            if rank + 1 < dimension:
                after[rank + 1] += value * (1 - rho) / tau
        mass = after
    return sum(mass)


def check_counts():
    checked = 0
    for n in range(1, 10):
        actual = enumerate_supports(n)
        for s in range(1, n + 1):
            for h in range(1, n + 1):
                for closed in range(s // 2 + 1):
                    for pairs in range(closed + 1):
                        assert actual[s, h, closed, pairs] == support_count(n, s, h, closed, pairs)
                        checked += 1
    # Algebra checks of the chain, not distance measurements on small codes.
    assert pair_rank_moment(9, 1, 0) == 1
    assert pair_rank_moment(9, 1, 1) == F(1, 4)
    assert pair_rank_moment(9, 1, 4) == F(1, 24)
    assert pair_rank_moment(9, 1, 1, minimum_only=True) == F(1, 12)
    for blocks in range(1, 5):
        for pairs in range(9 * blocks // 2 + 1):
            assert pair_rank_moment(9 * blocks, blocks, pairs, minimum_only=True) <= pair_rank_moment(9 * blocks, blocks, pairs)
    for s in range(9, 19):
        for dimension in range(1, s // 2 + 1):
            for pairs in range(s // 2 + 1):
                one = pair_rank_moment(s, dimension, pairs)
                two = pair_rank_moment(s, dimension, pairs, F(2))
                assert 0 <= two <= one <= 1
    return checked


if __name__ == '__main__':
    print(json.dumps(dict(scope='Verified ingredients only; no complete 10% certificate.',
                          local_rs=local_catalog(), support_count_cases=check_counts()), indent=2))
