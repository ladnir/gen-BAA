# Field Universality for Addition-Only Encoders

This directory contains an independent research project on addition-only linear
encoders over finite fields. It does not form part of the BAA paper or the
`lean/BAA` formalization. Results may be transferred to another project only by
an explicit later decision.

## Question

Fix a family of binary local maps `B_N`, a constant number of additive linear
transforms, and independent uniform coordinate permutations. Interpret the same
integer circuit over a field `F`, without random diagonal multipliers, and let
`C_{F,N}` be its row-space code.

The central question is whether random interleaving makes the normalized
minimum distance asymptotically independent of the field characteristic.

For an odd prime `p`, define

```text
Delta_{p,N} := d_min(C_{F_p,N}) / N.
```

The binary code uses the same sampled permutations and defines `Delta_{2,N}`.

## Conjecture hierarchy

The first target is one-sided distance inheritance. For every fixed odd prime
`p` and every `epsilon > 0`,

```text
Pr[Delta_{p,N} < Delta_{2,N} - epsilon] -> 0.
```

This statement is sufficient for the intended application: an encoder over a
prime field would inherit the binary relative-distance guarantee while using
only additions, permutations, and fixed local binary maps.

The stronger coupled-universality conjecture is

```text
Delta_{p,N} - Delta_{2,N} -> 0 in probability.
```

The strongest intended generalization replaces the two-accumulator circuit by
any fixed-depth composition of field-stable binary local maps, independent
uniform permutations, and fixed additive recursions with sparse inverses.

## Rank formulation

For a full-row-rank generator matrix `G` over `F`,

```text
d_min(rowspan_F(G))
  = min { |T| : rank_F(G restricted to the complement of T) < dimension(G) }.
```

Equivalently, minimum distance is the cogirth of the column matroid represented
by `G`. The conjecture therefore asks for universality of an extremal deletion
rank, not a correspondence between individual codewords.

For a backward constraint matrix `Q_F`, define the worst-support rank profile

```text
r_{F,N}(t) := min_{T subseteq [N], |T| = t} rank_F((Q_F)_T).
```

The distance is the first `t` for which `r_{F,N}(t) < t`. A field-independent
limiting rank profile would imply coupled distance universality.

## Necessary qualifications

Sample-by-sample equality is false. With the length-three repetition input
`(1,1,1)`, identity permutations, and two prefix accumulators, the output has
weight two over `F_2` and weight one over `F_3`.

The conjectures must therefore quantify over the random interleavers. Initial
work will also fix `p` before taking `N -> infinity`. Uniformity over primes,
adversarial characteristics, deterministic interleavers, and arbitrary binary
local maps are outside the first scope.

The local maps must preserve dimension and minimum distance over the fields
being compared. This property is an assumption until proved for a particular
family.

## Proof program

1. Prove exact distance preservation for scalar extension from `F_2` to
   `F_{2^r}`. This gives a baseline theorem and isolates the odd-characteristic
   obstruction.
2. Establish the rank/cogirth formulation for the complete encoder circuit.
3. Prove field-independent rank for supports whose backward constraint graph
   can be eliminated by degree-one peeling.
4. Bound the probability of a non-peelable low-weight residual core.
5. Prove one-sided distance inheritance uniformly over all supports below a
   target relative weight.
6. Investigate the matching upper bound required for coupled universality.
7. Generalize only after the two-layer model is understood.

The uniform quantifier over exponentially many supports is the main obstacle.
A fixed-support rank theorem is useful but does not by itself imply a distance
theorem.

## Experimental program

Experiments must use the same permutations for every field. Random-message
searches are insufficient because minimum distance is an adversarial property.
The first implementation should compute or certify:

1. exact minimum distance for small instances over `F_2`, several odd prime
   fields, and the rationals;
2. the gap `(d_p - d_2) / N` for coupled samples;
3. minimum rank-deficient deletion sets;
4. Smith normal forms of candidate integer submatrices, which identify the
   characteristics that create additional dependencies; and
5. the effect of no diagonal multipliers, sign-only diagonals, and unrestricted
   field multipliers.

The rational comparison is important. If the rational distance remains above
the binary distance by a linear amount, sufficiently large characteristics may
improve the distance rather than match it.

## Project boundary

All notes, experiments, proofs, and future formalization for this conjecture
belong under `field-universality/`. In particular:

- do not modify `main.tex` for exploratory results;
- do not import project files into `lean/BAA`;
- do not describe numerical evidence as a theorem; and
- do not transfer a claim to the BAA paper before its assumptions and proof
  status are explicit.
