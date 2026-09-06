# Increased RS distances with more than 40 failure bits

At the same rates, lengths, and message dimension as the manuscript, the
complete finite bound certifies the following larger relative distances.
Here L=floor(delta*n), and failure means that some nonzero codeword has
final Hamming weight at most L.

| Constituent | Length n | Previous delta | New delta | L | Complete failure upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| RS[16,4,13] | 4,194,304 | 0.60 | 0.604395 | 2,535,016 | < 2^-44.36 |
| RS[16,8,9] | 2,097,152 | 0.30 | 0.30523 | 640,113 | < 2^-42.37 |

Both rows have message dimension k=2^20 and hold over every finite field
of size q>=2^127. The local RS code is fixed. The probability is over
independent uniform permutations and independent uniform nonzero diagonal
entries in the two weighted accumulator stages. Thus, except with the
stated probability, the minimum distance is at least L+1.

The distance thresholds increase by 18,434 and 10,968 coordinates,
respectively. The relative-distance gains are 0.7325% and approximately
1.7433% relative to the previous targets.

The existing analytic reductions apply with delta as a parameter.
The sparse calculation checks its convergence inequalities anew, including
every cancellation layer. The bulk verifier reconstructs the complete
rectangle covering with the new constraint on final weight. The cutoff
T=floor(n/1000), the direct intermediate cutoff 512, and the field lower
bound are unchanged.

| Constituent | Sparse log2 upper bound, rounded upward | Bulk log2 upper bound, rounded upward | Bulk leaves |
| --- | ---: | ---: | ---: |
| RS[16,4,13] | -209.25 | -44.36 | 57 |
| RS[16,8,9] | -137.24 | -42.37 | 47 |

The complete bounds in the first table are checked against the sum of
sparse and bulk contributions using interval arithmetic. They are not
obtained by discarding the smaller contribution. The bulk contribution
now controls the reported failure margin.

## Reproduce

With only the dependencies in `requirements.txt`, run from the repository root:

```text
python rs-certification/verify_distance.py rs-certification/bulk-distance-rs16-4.json --bits 44.36
python rs-certification/verify_distance.py rs-certification/bulk-distance-rs16-8.json --bits 42.37
```

The full output, including separate sparse contributions, is saved in
[distance-results.json](distance-results.json). Verification succeeded in
a clean environment containing python-flint 0.9.0 without SciPy or NumPy.
The tests repeat both checks at 192-bit and 256-bit interval precision.
All twenty tests pass, including the original manuscript bounds and
rejection of an overstated margin, a mismatched checkpoint, and a false
infeasibility claim under an enlarged distance domain.

Optional certificate generation requires `requirements-generation.txt`:

```text
python rs-certification/generate_bulk.py --sigma 4 --delta 0.604395 --output tmp/distance-rs4.json --seconds 45
python rs-certification/generate_bulk.py --sigma 8 --delta 0.30523 --output tmp/distance-rs8.json --seconds 45
```

The optimizer proposes tangents and dual multipliers. Only the independent
interval verifier establishes an upper bound. A nonzero pending count or
an unresolved-box error leaves an incomplete certificate.

These are the largest distances certified in this search, not proven
optimal distances. Attempts at 0.604399 and 0.305235 did not complete the
covering with the search's 42-bit bulk target. That outcome does not prove
that a 40-bit bound fails at those distances. Improving the proposals,
refining the cover, or reducing its target toward 40 bits could yield
further small gains.

The manuscript retains the original distances and stronger failure margins.
These certificates provide an additional distance-versus-failure tradeoff
for coauthor review. The analytic reductions remain written proofs rather
than new Lean formalizations; the scope and review items in
[AUDIT.md](AUDIT.md) still apply.
