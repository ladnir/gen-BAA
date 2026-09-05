# RS finite-length certificate audit

Reviewed September 5, 2026, in the worktree based on Overleaf commit
`f929ad7`, with the local manuscript and certificate changes recorded here.
The audit was completed before committing and pushing this revision.
No ePrint submission was made as part of the audit.

The audit retained both complete bounds at message dimension k=2^20:

| Local code | Output length | Relative distance | Verified failure upper bound |
| --- | ---: | ---: | ---: |
| RS[16,4,13] | 4,194,304 | 3/5 | < 2^-209.60 |
| RS[16,8,9] | 2,097,152 | 3/10 | < 2^-137.56 |

The probability is over independent uniform permutations and independent
uniform nonzero diagonal entries. The bounds hold over every finite field
of size q>=2^127. They include all nonzero outer codewords, support sizes,
intermediate weights, final weights up to floor(delta*n), and cancellation
layers. No conclusion here transfers the bound to a Feistel permutation
family or certifies a weighted RA configuration.

## Proof coverage checked

| Obligation | Evidence and audit result |
| --- | --- |
| Starting finite master bound | Inspected the concrete theorem and discharged per-trace bound in `../lean/BAA/NonbinaryProjectiveBound.lean`, and the corresponding paper argument. The new numerical reduction is a written proof, not a new Lean theorem. |
| Exact support restriction | Impossible local supports have empty events and are omitted before the union bound. For RS, every size ell>=d occurs. Integer convolution counts all patterns by (s,t); shortening dimension is r=s-(16-sigma)t. |
| Uniformity over field size | Checked the projective polynomial's monotonicity in q for surplus traces and its value 1 in the no-surplus region. Every envelope uses q_min=2^127 and checks q_min-1>=2n. |
| Direct one-block terms | Checked the order-statistic identity summing every final weight. All feasible first-return counts and b<r are evaluated for 1<=H<=512. |
| Intermediate tail | The positive-series tilt bounds every H>512. The finite formula for its binomial moment was checked independently by forward differences. |
| One-block surplus traces | The geometric envelope covers r<=b<=s; the separate high-b bound covers b>s. No cancellation range is discarded. |
| Growing sparse supports | The disjoint slabs [2,3], [4,7], [8,15], [16,31], [32,63], [64,T] cover every 2<=t<=T. Monotone bases and positive geometric tails give rational upper bounds. |
| Remaining supports | For t>=T+1, the type entropy and six trace entropies retain all variables, including surplus cancellations. The exact finite counting prefactor is 2(B+1)^sigma(n+1)^6. |
| Covering verification | All rectangle boundaries are reconstructed by rational bisection. Every leaf is checked with global entropy upper bounds and residual-corrected dual inequalities. Missing leaves and unsupported infeasibility declarations are rejected. |
| Arithmetic | Rational operations handle coefficients and geometric tails. Arb encloses logarithms, exponentials, and direct sums. Optimization only proposes data; it is not part of the trusted numerical verification. |
| Final comparison | The direct and remainder contributions are added before checking the stated decimal bounds. The exact one-block partial sum also brackets the restricted master sum within less than one millionth of a bit. This does not lower-bound actual failure probability. |

The complete written reduction is included as Appendix C in
`finite-rs.tex`; the three Markdown derivations provide a convenient source
view of the same calculation.

## Corrections made during the audit

1. Added Theorem 7.1 (`thm:rs-finite`) and Appendix C
   (`app:rs-certificate`), including the region contributions and executable
   verification commands. Updated the abstract, contributions, and RS table.
2. Defined the zero-dimensional boundary value chi_q(0,c)=0 and stated the
   admissible exact-support restriction explicitly.
3. Removed unsupported RA failure probabilities from the implementation
   table and prose. The complete formal RA sum remains, but its numerical
   evaluation is explicitly uncertified. Removed the missing large-field
   script citation and the unsupported one-bit-rounding description.
4. Separated uniform-ensemble probabilities from Feistel implementation
   timings and from the Lean-verified finite counting claims.
5. Corrected the old negative-global-supremum claim: both exponent domains
   approach zero at the sparse boundary. The binary sufficient condition
   now restricts outer weight to at least epsilon*n; the reported RS
   stationary-branch values are labeled as diagnostics. Corrected the
   natural-log-to-bits factor in the RS margin table.
6. Unified the complete verifier's tree traversal and propagated its
   requested interval precision through every leaf. Standalone bulk
   verification now also defaults to q_min=2^127. Invalid field-size and
   secant conditions fail explicitly.
7. Synchronized `abstract.tex` with the manuscript. Shortened the abstract
   so its keywords stay on the first page, and used a section-level table
   of contents to remove the contents overflow page.

## Reproduction and document validation

These commands passed from the source root:

```text
python -m unittest discover -s rs-certification -p "test_*.py" -v
python rs-certification/verify_complete.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/rs-paper main.tex
```

All 16 tests passed, including full verification at the minimum field size
and a separate 256-bit-precision check with q_min=2^128. The saved default
output is `verified-results.json`.

A fresh virtual environment installed only `requirements.txt`
(`python-flint==0.9.0`) and reproduced that JSON exactly. NumPy and SciPy
were confirmed absent. They are optional dependencies for generation and
the diagnostic tests, not for verifying the saved certificates.

The release PDF is `../output/pdf/chosen-block-baa.pdf`, rebuilt from the
current source: 54 pages. The PDF metadata matches the title and authors.
The final build has no overfull/underfull boxes, undefined citations or
references, or multiply defined labels. Remaining TeX warnings concern the
existing vec macro, float-placement fallback, and a mathematical bookmark.
All pages were inspected in rendered contact sheets, with detailed checks
of the revised theorem, probability table, appendix formulas, and component
table. The release copy is byte-identical to the inspected build.
`manifest-sha256.json` records this source and artifact snapshot. Text hashes
normalize line endings to LF; the PDF hash uses its unchanged binary bytes.

The manuscript and its recorded LaTeX inputs contain no addition-only
project material. The separate `field-universality/` directory was not
changed. No benchmarks were run.

## Remaining coauthor and release review

* Have a coauthor review the new analytic reductions, especially moving
  from the exact-support events to the finite entropy envelope. The checks
  above are an agent audit and numerical verification, not independent
  human review or Lean formalization of the appendix.
* Publish the entire `rs-certification/` directory with the source. The
  certificate files and dependencies are reproducible from the source
  snapshot recorded in the manifest.
* The existing Lean development was inspected but not rebuilt here. Its
  source was unchanged; this audit does not replace its documented
  `lake build BAA.AxiomAudit` release check.
* The asymptotic transition searches and structural concentration problem
  remain outside this finite certificate. The corrected stationary-branch
  diagnostics were not independently regenerated during this audit.
* The separate binary certificate cited as
  `scripts/binary_first_moment_certificate.py` is still absent from this
  checkout. Locate and include it for whole-paper reproducibility before
  upload; the RS artifact does not certify the binary calculations.
* The RA rows require complete numerical bounds before probabilities can
  be restored. The current paper labels them uncertified and retains their
  runtime measurements only as configuration context.

Recommended next turn: coauthor review of Theorem 7.1 and Appendix C, then
resolve the binary artifact location and perform the release checks before
uploading to ePrint.
