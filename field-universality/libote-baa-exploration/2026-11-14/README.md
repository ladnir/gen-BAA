# libOTe BAA exploration, 2026-11-14

This archive preserves the experiments before the public split into BinaryBAA
(binary-only, 10% tier) and ChosenBAA30 (large-field default).

The production libOTe base is recorded in manifest.json. snapshot/ contains
exact source, test, proof-checker, note, and result copies. The patch records
changes to tracked files relative to that base; snapshot/ also includes new
files and the complete BAA header directory. Paths and API names in those
historical files are intentionally unchanged. They do not describe the current
public API. SHA-256 hashes cover every snapshot file.

Start with snapshot/libOTe_Tests/benchmarks/RsPlainDenseSupport.md for the latest
proof work. RsPlainScans.md records the 6.6% certificate and plain-scan timings;
NonbinaryLowTier.md records the q=2 candidate and profiling;
ConstantMixingSpecialization.md records the constant-mixing investigation.
No 10% certificate was obtained for RS plus two addition-only scans.

Run the standalone exact ingredient checker with:

    python snapshot/libOTe_Tests/benchmarks/certificates/verify_rs_plain_dense.py

Other field-transfer replays take the gen-BAA root as an argument. The reference/
directory preserves the immediate q1 replay dependencies and original ledger.
Some reference scripts import additional research utilities when invoked in
generation modes; the full expander-lift working tree remains the research
source for those modes. Archived benchmark tarballs contain their own runner,
source snapshots, compiler flags, checksums, and host observations. Benchmarks
must run sequentially. Floating proof screens are not distance certificates.
