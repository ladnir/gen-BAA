# Raising the quinary and septenary distance bounds

## Objective

The current paper proves relative-distance lower bounds `0.05` over `F_5` and
`0.018` over `F_7`. These constants measure the loss in the sign-orbit
relaxation. They do not describe the apparent distance of the BAA ensemble.

The next theorem should prove the common bound

```text
delta_5 >= 0.22  and  delta_7 >= 0.22.
```

This target is deliberately below the numerical full-composition transitions.
It would show that the first three odd prime fields all beat the certified
binary bound `0.1323`. It would also give increasing certified bounds from
`F_3` to `F_5` and a nondecreasing bound from `F_5` to `F_7`.

## Numerical landscape

All values in this section come from floating-point multistart optimization.
They guide certificate construction but do not prove a global maximum.

The full vector-composition exponent gives the following brackets.

| field | message cutoff | negative endpoint | exponent | positive endpoint | exponent |
|---:|---:|---:|---:|---:|---:|
| `F_5` | `0.05` | `0.260` | `-0.000353` | `0.261` | `+0.000227` |
| `F_7` | `0.05` | `0.270` | `-0.000730` | `0.274` | `+0.001081` |

Thus the expected enumerator predicts much larger distances than the current
theorems. The quinary transition is about `81%` of the rate-`1/3` quinary GV
distance. The septenary value `0.270` is about `75%` of the corresponding GV
distance.

## Two complementary composition bounds

For one accumulator, let `P` be the increment--source law. Let `Q` be the law
obtained by making the two symbols conditionally independent given their
zero/nonzero indicators. Data processing gives

```text
I(D;S)-I(1[D!=0];1[S!=0]) = D(P || Q)
                          >= D(Phi(P) || Phi(Q))
```

for every deterministic projection `Phi`. The cell-target projection records
the zero/nonzero support cell and the target symbol `D+S`. It retains more of
the forced cancellation structure than the earlier two-level projection.

The cell-target projection does not use the full stationarity constraint. Map
`(D,S)` bijectively to `(S,T)`, where `T=S+D`, and let `K` be the image of
`Q` under this map. The true transition law has source and target marginal
`beta`. Therefore

```text
D(P || Q)
  >= S(alpha,beta,x)
  := inf { D(R || K) : row(R)=beta, column(R)=beta }.
```

This stationary-coupling bound is a matrix-scaling problem. Since `K` already
has row marginal `beta`, it has the dual form

```text
S(alpha,beta,x)
  = sup_v [ <beta,v>
            - sum_s beta_s log(sum_t (K_st/beta_s) exp(v_t)) ].
```

The equality uses the continuous extension on zero-mass rows. It supplies
symbol-dependent target potentials while fixing them before any maximization
over the transition law. It therefore does not repeat the quantifier error in
the first eliminated-composition checker.

Neither projection dominates the other. Their maximum is nevertheless a
valid lower bound on `D(P || Q)`. Applying this hybrid payment independently
to the two accumulators gives a rigorous composition-only upper bound.
Floating-point multistart maximization currently gives:

| field | message cutoff | output fraction | hybrid upper bound found |
|---:|---:|---:|---:|
| `F_5` | `0.0020` | `0.220` | `-0.00118836` |
| `F_7` | `0.0008` | `0.220` | `-0.00004311` |

The quinary search relocates to a ridge where the two payments coincide at the
second accumulator. This behavior supports the use of their maximum: each
projection blocks an escape available to the other.

The septenary value has almost no numerical margin. A broader search found a
dense sign-symmetric branch that an earlier cell-target run had missed. The
previous value `-0.00036564` is therefore withdrawn as an estimate of the
maximum. The new negative value remains evidence, not a global certificate;
further restarts or a verified cover may still find a positive point.

## Exact-checker audit

The stationary dual now has an exact cut representation. A cut stores
rational source and target potentials. Directed intervals bound every term
`exp(u_s+v_t)-1`. Two independent regression paths check the same cut: one
against an arbitrary stationary coupling, and one through the pure-branch
specialization.

The current five-variable eliminated checker does not exploit these cuts. At the
quinary numerical support point, its exact upper remains `+0.00118844`.
Adding 32 orbit-complete local stationary cuts does not change this value. On
the worst box after 20 splits, both the cell-target and hybrid cut banks give
the same upper bound `0.336818`.

The observed obstruction comes from the elimination step. A fixed cut makes the
conditional symbol laws multilinear, so their maximum occurs at pure laws.
A common stationary cut must control all pure branches. The symbol-sensitive
gain observed at an interior composition disappears under that requirement.

The retained-composition checker preserves the gain. At the quinary numerical
maximizer, a degenerate rational box has exact upper bound `-0.00115832`.
The global cover remains far from closing: after 200 splits, a broad
conditional box has upper bound `0.902508`, and both KL payments vanish. The
remaining task is therefore a global envelope for the conditional simplexes,
not another bank of support-only cuts.

## Sparse endpoint

The exact spacing argument controls message support below `10^-5` at the
target output fraction.

For `F_5`, the existing rational specialization works for every
`delta <= 13/50`; its geometric base is less than `4/5`.

For `F_7` and `delta <= 11/50`,

```text
sqrt(4 delta (1-delta)) < 21/25,
exp(D(1/2 || delta))-1 > 4/21.
```

Together with `e<3` and `eta<=10^-5`, an exact integer comparison makes the
geometric base less than `7/10`. The script `scripts/sparse_spacing_bound.py`
checks this comparison, and `scripts/test_sparse_spacing.py` contains the
independent integer regression.

## Remaining certificate problem

The missing domain is

```text
10^-5 <= message support <= 1,
output support <= 0.22.
```

The quinary checker eliminates the conditional symbol compositions and leaves
five support variables `(A,B,C,x_1,x_2)`. An audit found that its first version
selected Fenchel cuts after fixing a pure symbol branch. That operation
reversed the required minimax order. The old global-cover diagnostics are
withdrawn.

The corrected eliminated checker fixes one common rational cut mixture before maximizing
over the symbol branches. At `delta=0.1323`, `eta=0.002`, and 20 splits, its
exact upper bound is `0.140051`. The stationary-coupling lemma supplies the
missing valid marginals: a certificate may use source and target potentials
chosen for a whole support box, without choosing them after a pure symbol
branch.

The septenary proof also needs a specialization of the eliminated-composition
checker. Multiplication by `F_7^*/{+1,-1}` should reduce its nonzero branches
to three orbit images of one certificate family.

## Certification plan

1. Replace naive interval bounds on each conditional simplex with a joint
   convex envelope or a verified local Taylor model.
2. Use the exact support-only bound before subdividing symbol coordinates.
   Split a symbol coordinate only when the active KL cut is too loose.
3. Re-run the quinary cover at `delta=0.22`. Require every retained box to
   replay with rational potentials and directed transcendental intervals.
4. Re-audit `F_7` numerically with analytic gradients. If the endpoint remains
   too close to zero, lower the theorem target before attempting a global
   cover.
5. Generalize the checker to `F_7`, preserving the same trusted arithmetic
   boundary, and combine each bulk certificate with the exact spacing result.

The paper should continue to state `0.05` and `0.018` until the exact covers
close. The numerical targets belong in an investigation or methodology
section, with their non-rigorous status stated explicitly.
