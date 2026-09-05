# ePrint readiness review

Historical review of `c471c12`; findings below describe that revision.
See `rs-certification/AUDIT.md` for the subsequent RS certificate audit,
manuscript corrections, and current remaining review items.

Reviewed September 5, 2026, at Overleaf commit `c471c12` (fast-forward from `96e2e0a`). Manuscript and bibliography unchanged. Recommendation: address the claim and definition issues below before uploading.

## 1. P1: The binary security match lacks the reported full failure bound

Location: `main.tex:2437–2451`, `2520–2526`; contribution at `242–246`.

The comparison is called security-matched at relative distance 0.10 and code-sampling failure 2^-20. The evidence actually reported for the chosen codes is an asymptotic transition and the one-active-block contributions, approximately 2^-47.4 and 2^-48.1. A small contribution to a nonnegative union sum does not bound the complete sum. In particular, it does not control words activating multiple blocks, including growing sublinear numbers of blocks. Naming the complete first moment as the comparison criterion does not report its evaluation.

Correction: give an upper bound for the complete cumulative first moment for both constituents at n=2,097,024 and L=floor(0.10n)=209,702. Include the numerical method, truncation bounds, and reproducible output. If this computation has not been completed, describe these as comparisons at a common proposed distance/failure target, and remove the claim that the target is established. This is a missing evidentiary step, not a demonstration that either chosen code fails the target.

CWC: 08, 09, 11; hard distinction between a partial first moment and a failure upper bound.

## 2. P1: The negative global supremum is impossible as written

Location: `main.tex:1109–1116`, `1564–1565`, `2011–2015`, `2136–2146`; compare the interior-maximum language at `1848–1851`.

The binary criterion asks for sup Phi < 0 over the feasible domain. Take alpha=eta=t and let t decrease to zero. This remains feasible for any fixed positive target delta and sufficiently small t. The local coefficient exponent and both accumulator exponents tend to zero. Therefore the supremum is at least zero, even if the all-zero point itself is excluded.

The same problem occurs for the large-field exponent: take p_d=t, p_0=1-t, alpha=eta=dt/m, beta=t/m, and x=y=0. These points are feasible for sufficiently small t, and all terms tend to zero. Consequently a negative value such as -0.017 cannot be the global maximum on the stated domain.

Correction: explicitly define the compact domain bounded away from the sparse boundary on which a uniform negative margin is claimed. Define the reported numerical crossing as the crossing of a specified interior branch, consistent with the existing binary explanation. Keep fixed-size sparse, growing sublinear, and linear regimes separate. The existing asymptotic corollary can still establish some positive distance through the cited result; it does not itself certify the numerical near-GV threshold.

CWC: 03, 08, 09, 11; hard mathematical domain error. This finding does not establish that the interior numerical values are wrong.

## 3. P1: The large-field benchmark samples different interleavers from the analyzed ensemble

Location: `main.tex:752–759`, `1252–1254`, `2477–2489`.

The finite analysis assumes independent uniform permutations. The rate-1/2 benchmark uses four-round Feistel interleavers, while repetition-3 RAA uses stored uniform permutations. A seeded Feistel family is not the uniform distribution on Sym([n]), and no transfer argument for the low-distance event is given. This also means the comparison changes the interleaver implementation as well as rate and outer code.

Correction: for the comparison supporting the analyzed ensemble, use stored uniform permutations in all rows and repeat that campaign. Alternatively, retain the practical Feistel timings but identify the ensemble substitution explicitly, avoid presenting the uniform-ensemble distance analysis as a guarantee for it, and add a comparison with a common interleaver implementation. Do not assume that a generic claim of pseudorandomness proves preservation of this global code property.

Related wording: Table 3 already correctly calls the large-field sparse bits parameter-selection proxies. Carry that qualification into the abstract and the statement at `2591–2593` about reaching each minimum security target. These terms are not complete failure bounds or protocol-security levels.

CWC: 04, 08, 09; unaccounted change of probability space.

## 4. P2: Accumulator orientation and the timed operation need consistent interfaces

Location: `main.tex:767–784`, `1583–1592`, `2414–2427`, `2459–2466`.

With row vectors and prefix sums, A must be upper triangular: A_ij=1 when i<=j. The source calls A lower triangular. For example, multiplying the row (1,0,0) by the stated lower-triangular all-ones matrix gives (1,0,0), whereas the defined prefix accumulator gives (1,1,1). The later column-vector backward derivation uses the lower-triangular accumulator and successor difference neighborhoods, so the convention change must include the transpose of the matrix representation.

The implementation section also introduces multiplication by the k-by-n generator, but reports an n-to-k operation. That is transpose multiplication, not the forward encoder u -> uG. Both are useful, but their interfaces should be explicit before assigning timings to them.

Correction: distinguish the row and column accumulator matrices, for example A_row=A_col^T; retain the lower-triangular A_col in the backward analysis. Define the benchmark as z -> zG^T (with its actual coordinate/order convention), and identify its relevance to the PCG application. Specify whether forward encoding was separately measured before using these timings as forward-encoder performance.

CWC: 03, 04, 05, 07; hard matrix-convention error and incomplete performance interface.

## 5. P2: A binary matrix does give an extension-field-linear map

Location: `main.tex:696–703`.

The paragraph says a binary matrix cannot silently replace an F_q-linear encoder when the proof system expects F_q-linearity. For q=2^s, a binary matrix, interpreted entrywise in F_q, is already F_q-linear. Applying it coordinatewise to the binary representation implements that same map. For G over F_2 and a,b in F_q, (ax+by)G=a(xG)+b(yG).

What may fail is a required distance/rate guarantee or another protocol-specific property. Binary extension also preserves the binary code's minimum symbol distance: any nonzero extension-field codeword has a nonzero binary component whose support is contained in its support; the reverse inequality follows by including binary codewords.

Correction: explain the actual gap as the alphabet-specific distance benchmark or protocol requirements, while retaining the distinction from arbitrary characteristic. Do not use lack of extension-field linearity as the reason.

CWC: 03, 07, 09; hard algebraic distinction.

## 6. P2: The numerical and implementation evidence is not reproducible from this release

Location: `main.tex:1806–1810`, `1848–1851`, `2196–2199`, `2414–2416`, `2628–2630`, `2725–2729`.

This checkout includes the paper, figures, and Lean sources, but no numerical optimization scripts, C++ implementation, benchmark drivers, or raw measurements. The paper gives no release URL or implementation commit. It also omits the optimizer, restart/grid strategy, tolerances, and unrounded optima. The Lean version/artifact sentence is commented out.

This is especially important for the 99.99%-of-GV headline, the claimed finite binary security match, and the large-field sparse proxies. The QR table's coefficients do sum to 2^84, and the stated gap 1.33e-5 is arithmetically consistent with 99.99% after rounding; those checks do not independently reproduce the optimization.

Correction: add an accessible, versioned artifact with the exact implementation revision, benchmark commands/seeds/raw process medians, optimization scripts/settings/unrounded results, and Lean toolchain/build instructions. If some components remain unpublished, state that limitation and include enough numerical detail to audit each principal claim.

CWC: 03, 09, 11; missing evidence and dependencies.

## 7. P2: The title and abstract combine properties established in different settings

Location: `main.tex:52`, `75–104`, `1888–1921`.

“Fast, Near the Gilbert–Varshamov Bound, and Field-Agnostic” can be read as a joint claim. The near-GV result is a binary interior-exponent estimate for the unbenchmarked QR constituent. The measured binary speedups use Golay and Reed–Muller. The large-field analysis reports a substantial gap to its own GV benchmark. The abstract also calls 0.11 a relative-distance threshold without the numerical qualifier supplied later and ends with an unqualified constituent-code verification claim, although QR is excluded from Lean.

Correction: make the separation visible in the title and abstract. A possible title is “Chosen-Block BAA Codes: Near-GV Binary Estimates and Fast Encoding over Finite Fields.” Label the QR result as an estimated asymptotic transition; state explicitly that the compact implemented constituents provide the measured timings and that QR spectra are externally sourced rather than Lean-certified. Shorten the abstract by moving individual baseline timing values into the evaluation section.

CWC: 01, 02, 09; claim-calibration recommendation, not a counterexample to the reported measurements.

## 8. P2: The tracked PDF and submission text are stale

Location: `output/pdf/chosen-block-baa.pdf`, `abstract.tex`, `main.tex:16–24`.

The tracked PDF has 39 pages, the old title, and the old 21–24 ms abstract figures. A clean build of current main.tex produces 47 pages with the new title, table of contents, 24–27 ms abstract figures, and large-field timings. Even the fresh build retains the old title in PDF metadata because both pdftitle branches are stale. The standalone abstract.tex contains an older abstract and is not included by main.tex.

Correction: after substantive revisions, build and replace the release PDF from the exact reviewed source, update PDF metadata, and synchronize or clearly retire the standalone abstract before preparing the ePrint form. The current tracked PDF should not be used for upload.

CWC: 05, 09; release consistency.

## Lower-priority cleanup

- The fresh table of contents spills only two appendix entries onto page 3, leaving most of that page blank. Fit the contents on one page or omit subsection entries.
- At `main.tex:1021–1025`, the fixed-h Theta estimate needs feasibility: if h<ceil(w/2), the transition is identically zero.
- The master bound sums supports with r_S=0 although chi_q was introduced only for r>=1. Define chi_q(0,c)=0, or restrict the sum to positive-dimensional shortenings.
- The “bits” heading at `main.tex:2141` needs division by log 2, consistent with the preceding column and the stated natural-log convention.
- Add the exact theorem/section pointer in the original BAA paper for the asymptotic corollary, so the fixed-block specialization is easy to audit.

## Verification performed and limits

- Fetched origin and fast-forwarded this clean detached checkout to origin/main at c471c12.
- Built current source using `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/build main.tex`.
- Final build has no undefined citations, undefined references, or multiply defined labels. Remaining warnings are a 1.43532 pt overfull line, the existing vec warning, float placement, and math in a PDF bookmark.
- Visually inspected the title/abstract page, contents overflow, a structural-count page, and the methodology and binary benchmark pages. This was a targeted visual review, not a page-by-page layout certification.
- Checked the QR coefficient sum, binary rate-1/2 GV value (0.11002786443835955), and the three headline speedup ratios. These arithmetic checks pass.
- Scanned Lean sources for sorry/admit/project-local axiom declarations; the only text matches were the English word “admit” in comments. Did not rebuild Lean or rerun its axiom audit; this checkout has no local dependency/build cache. A source scan is not kernel verification.
- Checked the primary landing pages for [original BAA](https://eprint.iacr.org/2025/1828) and [Khabbazian's scaled-RAA result](https://eprint.iacr.org/2026/1075). The latter states r>=4 and q-1>=gamma N, consistent with the manuscript's caveat. Full PDF retrieval was unavailable, so the original theorem specialization was not independently checked against its full proof. Current ePrint submission/conditions pages also could not be retrieved; no claim of complete venue-policy verification is made.
- No benchmarks were run. No manuscript, bibliography, tracked PDF, or Lean source was edited.

Recommended next turn: resolve the binary full-first-moment evidence and the uniform-versus-Feistel benchmark scope first, then correct the exponent domains and matrix conventions. After claim wording is settled, make a focused manuscript revision and regenerate the release PDF.
