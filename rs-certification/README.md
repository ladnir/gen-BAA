# Complete finite-length bounds for weighted RS BAA

The derivations and executable certificates here give a code-sampling failure
bound below **2^-209.60 and 2^-137.56 for the two RS configurations**, at
message dimension k=2^20. They use exact rational arithmetic and Arb interval
arithmetic, with proved bounds for every omitted term. No manually supplied
remainder is used.

| Constituent | Length n | Target delta | L=floor(delta*n) | Certified failure upper bound |
| --- | ---: | ---: | ---: | ---: |
| RS[16,4,13] | 4,194,304 | 3/5 | 2,516,582 | < 2^-209.60 |
| RS[16,8,9] | 2,097,152 | 3/10 | 629,145 | < 2^-137.56 |

The failure event is that a nonzero outer codeword has final Hamming weight
at most L after the two weighted accumulator stages. The bound holds over
any finite field of size q>=2^127, including every 128-bit prime field, when
the two permutations are independent and uniform and all diagonal entries
are independent and uniform in the nonzero field elements. The local RS
constituent is fixed. The proof only uses its MDS support properties.

These bounds improve the first complete certificates (2^-95 and 2^-63) by
tightening the sparse-support calculation. They now imply the manuscript's
earlier RS values 2^-208.6 and 2^-136.5 under the stated ensemble assumptions.
They do not certify RA rows or a particular pseudorandom permutation implementation.

## Proof and trust boundary

The starting point is the support/trace master bound, whose concrete Lean
theorem is `concrete_nonbinaryBAA_probability_le_masterTraceBound_proved` in
`../lean/BAA/NonbinaryProjectiveBound.lean`. For an exact support of size s
meeting t nonempty RS blocks, its shortening dimension is
r=s-(16-sigma)t. Impossible local exact supports have empty bad events and
can be omitted before applying the union bound.

* [SPARSE_BOUND.md](SPARSE_BOUND.md) bounds every support with
  1<=t<=T=floor(n/1000), including every cancellation layer. Geometric
  envelopes reduce the calculation to exact rational operations.
* [REFINED_SPARSE.md](REFINED_SPARSE.md) tightens that bound by evaluating
  one-block traces directly and bounding all remaining terms. An exact
  partial sum also shows that less than a millionth of a bit remains
  available by further numerical tightening of this same master sum.
* [BULK_BOUND.md](BULK_BOUND.md) bounds all t>=T+1 by finite entropy
  inequalities with explicit counting prefactors. A complete rectangle
  covering and rational linear-program multipliers give a global upper
  bound; Arb checks its numerical value.
* `verify_complete.py` checks both ranges and their sum. Certificate
  generation uses floating-point optimization, but verification does not
  trust the optimizer's objective, feasibility decisions, or accuracy.

The new analytic reductions are written mathematical proofs, supported by
small exact checks; they are **not yet Lean-formalized or independently
reviewed by a coauthor**. The existing Lean project was not rebuilt during
this investigation. The results are integrated into Theorem `thm:rs-finite`
in `main.tex` and the complete appendix in [finite-rs.tex](finite-rs.tex).

## Reproduce the result

From the repository root, using a Python environment with python-flint:

```text
python -m pip install -r rs-certification/requirements.txt
python rs-certification/verify_complete.py
```

The output is recorded in [verified-results.json](verified-results.json).
The comparisons against the reported probability bounds are checked directly
with intervals, rather than by rounding the displayed logarithms. The default field-size lower bound is
2^127 and the default interval precision is 192 bits.

For the reference tests and optional regeneration:

```text
python -m pip install -r rs-certification/requirements-generation.txt
python -m unittest discover -s rs-certification -p "test_*.py" -v
python rs-certification/generate_bulk.py --sigma 4 --output tmp/rs16-4.json --seconds 60
python rs-certification/generate_bulk.py --sigma 8 --output tmp/rs16-8.json --seconds 60
python rs-certification/verify_bulk.py tmp/rs16-4.json
python rs-certification/verify_bulk.py tmp/rs16-8.json
```

Generation saves a checkpoint. If it reports a nonzero `pending` count,
repeat that generation command to resume; an incomplete covering will be
rejected by the verifier. Generation is optional for the saved certificates.

The sixteen tests include complete verification at the minimum field size,
verification at 256-bit precision over q=2^128, exact small-length trace sums, the order-statistic
identity for the final weight, generating-function moments, the intermediate
tail envelope, finite entropy prefactors, both certificate coverings, and
rejection of missing leaves, unsupported infeasibility declarations, and
negative dual multipliers.
The checked environment was Python 3.14, python-flint 0.9.0, NumPy 2.4.4,
and SciPy 1.18.0. SciPy and NumPy are unnecessary for verifying the saved
certificates.

The next paper step is a coauthor review of the reductions and release of
these scripts and certificate data with the paper. Meaningfully more failure bits at these
parameters would require improving the underlying probabilistic bound;
another option is to change length, distance, or constituent. Random outer
blocks are unnecessary to reach the requested failure target.

## Build the paper

With TeX Live and latexmk installed, run from the repository root:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/rs-paper main.tex
```

The reviewed release copy is `output/pdf/chosen-block-baa.pdf`.
[AUDIT.md](AUDIT.md) records the proof review, validation, and remaining
coauthor review items. The addition-only project is outside this artifact
and is not included by the manuscript.
