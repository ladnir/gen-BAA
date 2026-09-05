# Run profiles and first-round peeling

## 1. Purpose

This note proves the first run-sensitive statement for the sparse dual matrix.
The result does not prove minimum-distance inheritance. It identifies a large
class of fixed deletions for which the two interleavers create linearly many
degree-one checks in the first peeling round.

The argument has three exact parts:

1. inverse difference decomposes a deletion set into zero-sum spaces;
2. the run profile determines the number of active rows before the second
   interleaver; and
3. the two interleavers turn row expansion into degree-one dual checks.

All algebraic statements hold over the integers. The peeling theorem applies
over every field of odd characteristic.

## 2. Difference matrix and deletion runs

Fix a length `N>=1` and write `[N]:={0,...,N-1}`. Let

```text
E_D : Z^D -> Z^N
```

be the coordinate embedding for a nonempty set `D subseteq [N]`. Define the
inverse-transpose accumulator matrix `T_N` by

```text
T_N e_j = e_j-e_(j+1)  for j<N-1,
T_N e_(N-1) = e_(N-1).
```

A run of `D` is a maximal interval contained in `D`. Let `r(D)` denote the
number of runs, and define

```text
tau(D) := 1[N-1 in D].
```

### Lemma 1 (integer run decomposition)

Let the runs of `D` be the intervals

```text
[a_1,b_1], ..., [a_r,b_r].
```

For a nonterminal run `[a_i,b_i]`, define

```text
Z_i := {v in Z^N :
        supp(v) subseteq [a_i,b_i+1] and
        sum_j v_j = 0}.
```

If the final run ends at `N-1`, define

```text
Z_r := {v in Z^N : supp(v) subseteq [a_r,N-1]}.
```

Then

```text
T_N E_D(Z^D) = direct_sum_i Z_i.
```

Each summand is a saturated sublattice. In particular,
`T_N E_D(Z^D)` is saturated and has rank `|D|`.

**Proof.** Consider a nonterminal run `[a,b]`. Its columns are

```text
e_a-e_(a+1), ..., e_b-e_(b+1).
```

Their span is contained in the zero-sum lattice on `[a,b+1]`. Conversely, fix
a zero-sum vector `v` on this interval and set

```text
c_j := sum_(i=a)^j v_i  for a<=j<=b.
```

Then

```text
v = sum_(j=a)^b c_j(e_j-e_(j+1)).
```

Hence the columns generate the full zero-sum lattice over `Z`. This lattice is
the kernel of a surjective integer sum map, so it is saturated.

For a terminal run `[a,N-1]`, the columns end with `e_(N-1)`. Back substitution
expresses every coordinate basis vector on `[a,N-1]` as an integer combination
of these columns. The terminal summand is therefore the full coordinate
lattice. Distinct runs use disjoint extended intervals, so their sum is direct.
∎

### Corollary 2 (active-row count)

Let `S(D)` be the set of nonzero rows of `T_N E_D`. Then

```text
|S(D)| = |D| + r(D) - tau(D).
```

This count is the first run-profile parameter. Write

```text
d := |D|,
r := r(D),
tau := tau(D),
s := d+r-tau.
```

## 3. Exact profile count

The run profile also determines the deletion entropy.

### Lemma 3 (number of deletion sets by run profile)

For `d>=1` and `r>=1`, the number of `d`-element subsets of `[N]` with exactly
`r` runs is

```text
binom(d-1,r-1) binom(N-d+1,r).
```

The terminal flag refines this count as follows:

```text
tau=0: binom(d-1,r-1) binom(N-d,r),
tau=1: binom(d-1,r-1) binom(N-d,r-1).
```

**Proof.** First compose the `d` deleted coordinates into `r` positive run
lengths. This gives `binom(d-1,r-1)` choices. Next distribute the undeleted
coordinates among the gaps. Every internal gap is nonempty. The terminal gap
is nonempty exactly when `tau=0`. Stars and bars gives the two refined counts.
Pascal's identity gives their sum. ∎

Fix constants `alpha=d/N` and `rho=r/N`. Stirling's formula gives the profile
entropy

```text
E(alpha,rho)
  := alpha h_2(rho/alpha)
     +(1-alpha) h_2(rho/(1-alpha)).
```

Thus the number of sets in this profile is
`2^(N E(alpha,rho)+o(N))`. Maximizing over `rho` recovers `h_2(alpha)`.

## 4. Row expansion under the second interleaver

Let `P_2` be a uniform coordinate permutation, independent of `D`, and define

```text
A_D := T_N P_2 T_N E_D.
```

The matrix `A_D` has `d` columns. It is the portion of the sparse dual before
the first interleaver and the repetition checks.

The nonzero rows of `T_N E_D` are pairwise distinct over `Z`. Indeed, every
row contains the labeled boundary of one or two adjacent deleted coordinates.
Two singleton rows associated with the same deleted coordinate have opposite
signs rather than equal signs.

Let `Q:=P_2(S(D))`. The set `Q` is a uniform `s`-element subset of `[N]`. A row
of `A_D` is zero exactly when the corresponding row and its predecessor are
both outside `Q`. Consequently,

```text
L(Q) := number of nonzero rows of A_D
      = |Q union (Q+1)|
      = 2s-X(Q),
```

where `X(Q)` is the number of adjacent pairs contained in `Q`, plus
`1[N-1 in Q]`. Direct counting gives

```text
E[X(Q)] = s^2/N,
E[L(Q)] = 2s-s^2/N.
```

### Lemma 4 (row-expansion tail)

For every `a>=0`,

```text
Pr_P2[L(Q) < 2s-s^2/N-a] <= exp(-a^2/(72N)).
```

**Proof.** Represent `Q` by the first `s` images of a uniform permutation.
Transposing two images changes membership of at most two coordinates. Removing
or adding one coordinate changes at most two adjacency indicators and one
terminal indicator. Hence a transposition changes `X` by at most six.

Expose the permutation one image at a time. Coupling two conditional
completions by one transposition bounds every Doob-martingale increment by six.
Azuma's inequality gives the stated upper tail for `X`. ∎

## 5. Singleton rows force degree-one checks

Let `l:=L(Q)`. Every column of `A_D` has at most four nonzero entries. Hence
the support matrix of `A_D` has at most `4d` incidences.

Let `u` be the number of rows of `A_D` that contain exactly one nonzero entry.
Counting one incidence for a singleton row and at least two for every other
nonzero row gives

```text
u >= 2l-4d.
```

There are `z:=N-l` zero rows.

Let `P_1` be an independent uniform coordinate permutation. Let `H_0` contain
the two adjacent differences in every repetition triple. The fixed-deletion
dual matrix is

```text
H_D := H_0 P_1 A_D.
```

The support graph of `H_0` is a disjoint union of length-two paths. It has
`2N/3` edges and maximum vertex degree two. A path edge gives a degree-one row
of `H_D` whenever `P_1` places a singleton row of `A_D` at one endpoint and a
zero row at the other endpoint.

### Lemma 5 (degree-one check tail)

Condition on `P_2` and hence on `A_D`. Let `C` count the singleton-zero edges
after sampling `P_1`. Then

```text
E[C | A_D] = 4uz/(3(N-1)).
```

For every `b>=0`,

```text
Pr_P1[C < E[C | A_D]-b | A_D] <= exp(-b^2/(32N)).
```

**Proof.** A fixed path edge receives one singleton and one zero row with
probability `2uz/(N(N-1))`. Summing over the `2N/3` edges gives the expectation.

A transposition in `P_1` changes only edges incident to the two transposed
positions. At most four path edges are affected. The permutation-exposure
martingale therefore has increments bounded by four. Azuma's inequality gives
the tail bound. ∎

One column of `A_D` occupies at most four coordinate rows. Each coordinate is
incident to at most two path edges. Hence one column accounts for at most eight
of the `C` candidate checks. At least `C/8` distinct columns therefore have a
degree-one check.

The nonzero coefficient in such a check is `+1`, `-1`, `+2`, or `-2`.
Consequently, every candidate check is a valid pivot over a field of odd
characteristic.

## 6. First-round peeling theorem

### Theorem 6 (run-sensitive first-round peeling)

Fix `N=3K`, a nonempty deletion set `D subseteq [N]`, and a field `F` of odd
characteristic. Define `d,r,tau,s` as above. Sample `P_1` and `P_2`
independently and uniformly from the coordinate permutations.

Fix `a,b>=0` and define

```text
l_*  := 2s-s^2/N-a,
u_*  := max(0,2l_*-4d),
z_*  := max(0,N-2s),
mu_* := 4u_*z_*/(3(N-1)).
```

With probability at least

```text
1-exp(-a^2/(72N))-exp(-b^2/(32N)),
```

the first peeling round removes at least

```text
floor(max(0,mu_*-b)/8)
```

columns of `(H_F)_D`.

**Proof.** Lemma 4 gives `l>=l_*` except with the first stated failure
probability. On this event,

```text
u >= u_*,
z = N-l >= N-2s >= z_*.
```

Lemma 5 therefore gives `C>=mu_*-b` except with the second failure
probability. At least `C/8` distinct columns have degree-one checks. Their
coefficients are nonzero in `F`, so the peeling algorithm removes all selected
columns. The union bound completes the proof. ∎

The theorem certifies a rank contribution, not full column rank. Later rounds
reuse the same first interleaver, so they are not independent repetitions of
Theorem 6.

## 7. Two rounds with deferred decisions

Theorem 6 uses every repetition check at once. It gives the strongest
one-round count, but it exposes the full first interleaver. A staged argument
must reserve untouched coordinates for later rounds.

Partition the `K=N/3` repetition triples into two fixed sets `J_1` and `J_2`.
Let `|J_1|=gamma K` and `|J_2|=(1-gamma)K`. In each triple, retain only the
check joining its first two coordinates. These retained checks form a
matching.

The first stage reveals the `2|J_1|` row identities assigned to the endpoints
of the matching on `J_1`. It removes every distinct column exposed by a
singleton-zero edge. The second stage uses the matching on `J_2`.

### Lemma 7 (uniform conditional completion)

Let a uniform permutation assign `N` distinct row identities to `N` coordinate
slots. Fix a set `E` of slots and reveal every identity assigned to `E`.
Conditioned on this revelation, the remaining identities are assigned
uniformly to the remaining slots.

If a set of columns is a deterministic function of the revealed identities,
then restricting all unexposed row vectors to the remaining columns preserves
this uniform conditional assignment.

**Proof.** Every bijection between the remaining identities and slots has the
same number of extensions consistent with the revelation: exactly one. The
column restriction is deterministic after the revealed identities are fixed,
so it does not reweight the remaining bijections. ∎

The following theorem uses the parameters `l_*`, `u_*`, and `z_*` from
Theorem 6. Thus

```text
l_* := 2s-s^2/N-a,
u_* := max(0,2l_*-4d),
z_* := max(0,N-2s).
```

### Theorem 8 (two-round deferred peeling)

Fix the setting of Theorem 6 and a fraction `0<gamma<1` such that `gamma K` is
an integer and `N-2gamma K>1`. Define

```text
m_1 := gamma K,
n_e := 2m_1,
N'  := N-n_e,
m_2 := (1-gamma)K.
```

Fix an integer stopping cap `q_bar>=0` and slacks `b_1,b_2,c>=0`. Define

```text
mu_1 := 2m_1 u_* z_*/(N(N-1)),
q_1  := floor(max(0,mu_1-b_1)/4),

u_2  := max(0,(N'/N)u_*-c-4q_bar),
z_2  := max(0,(N'/N)z_*-c),
mu_2 := 2m_2 u_2 z_2/(N'(N'-1)),
q_2  := floor(max(0,mu_2-b_2)/4),

q_*  := min(q_bar,q_1+q_2).
```

The two-stage procedure peels at least `q_*` columns except with probability
at most

```text
exp(-a^2/(72N))
+ exp(-b_1^2/(8N))
+ 2exp(-c^2/(2N))
+ exp(-b_2^2/(8N')).
```

The probability is over the independent interleavers `(P_1,P_2)`.

**Proof.** Lemma 4 supplies at least `u_*` singleton rows and `z_*` zero rows,
except with the first failure probability. The first reserved matching has
`m_1` edges. Its expected number of singleton-zero edges is at least `mu_1`.
A transposition affects at most two matching edges, so permutation exposure
and Azuma's inequality give at least `mu_1-b_1` candidates except with the
second failure probability.

One column occupies at most four coordinate rows. A matching uses each row at
most once. Hence one column accounts for at most four candidates, and the
first stage peels at least `q_1` distinct columns.

Let `E` contain the `n_e` exposed slots. Before column removal, the expected
numbers of singleton and zero rows outside `E` are at least

```text
(N'/N)u_* and (N'/N)z_*.
```

Transposing two row identities changes either count by at most one. Azuma's
inequality therefore gives simultaneous lower bounds within `c`, except with
probability at most `2exp(-c^2/(2N))`.

If the first stage peels at least `q_bar` columns, the claimed total already
holds. Otherwise it removes fewer than `q_bar` columns. Removing one column
can eliminate at most four original singleton rows. Original zero rows remain
zero. The unexposed residual matrix therefore has at least `u_2` singleton
rows and `z_2` zero rows.

Lemma 7 makes their assignment to the unexposed slots uniform, conditioned on
the complete first-stage history. The second reserved matching has `m_2`
edges. The same matching argument gives at least `q_2` new pivots except with
conditional probability `exp(-b_2^2/(8N'))`. The bound is uniform in the
first-stage history, so the union bound gives the stated failure probability.
∎

This theorem resolves the main conditioning issue for two rounds. It does not
yet provide a stronger numerical count than Theorem 6 because reserving
untouched triples discards checks.

## 8. Arbitrarily many reservoirs

Lemma 7 applies inductively to any fixed sequence of disjoint reservoirs. The
procedure must truncate each round to a deterministic number of pivots. This
choice prevents an unusually successful early round from invalidating later
category bounds.

Fix `L>=1`. For each round `t`, reserve `m_t` repetition triples, with

```text
sum_t m_t <= K.
```

Use the first check edge in every reserved triple. Define the initial values

```text
n_1 := N,
u_1 := u_*,
z_1 := z_*.
```

For slacks `b_t>=0`, define

```text
mu_t := 2m_t u_t z_t/(n_t(n_t-1)),
q_t  := floor(max(0,mu_t-b_t)/4).
```

After round `t<L`, fix a category slack `c_t>=0` and set

```text
n_(t+1) := n_t-2m_t,
u_(t+1) := max(0,(n_(t+1)/n_t)u_t-c_t-4q_t),
z_(t+1) := max(0,(n_(t+1)/n_t)z_t-c_t).
```

### Theorem 9 (multi-reservoir peeling)

Fix the setting of Theorem 6. Suppose that every `n_t>1`. In round `t`, expose
the matching on its reserved triples. If at least `q_t` distinct columns are
exposed, remove exactly the first `q_t` columns in a fixed canonical order.

The procedure removes at least

```text
sum_(t=1)^L q_t
```

columns except with probability at most

```text
exp(-a^2/(72N))
+ sum_(t=1)^L exp(-b_t^2/(8n_t))
+ 2 sum_(t=1)^(L-1) exp(-c_t^2/(2n_t)).
```

**Proof.** Proceed by induction over the reservoirs. At the start of round
`t`, Lemma 7 makes the residual row identities uniform over the `n_t`
unexposed slots, conditioned on the complete preceding history. The reserved
matching therefore has expected candidate count at least `mu_t`. Matching
concentration gives at least `4q_t` candidates outside the stated event. Each
column accounts for at most four matching edges, so at least `q_t` distinct
columns are available.

The algorithm removes exactly `q_t` columns. Among the unexposed slots, the
singleton and zero counts concentrate around the sampling fractions shown in
the recurrence. Removing one column destroys at most four original singleton
rows, while original zero rows remain zero. Thus the recurrence supplies valid
category bounds for round `t+1`. Lemma 7 then restores the uniform conditional
assignment. Summing the conditional failure bounds completes the induction.
∎

### 8.1 Finite evaluation

At the intended parameters, split the `K=349525` triples as evenly as possible
among four reservoirs. Set every matching and category tail to `10^-4`. The
four rounds certify

```text
4356, 3935, 3467, 2924
```

pivots, respectively. The total is 14,682 columns, or 10.58% of the deletion.
The combined failure probability is below `0.003952`.

For equal reservoirs and the fixed per-event tail used here, additional
reservoirs reduce the certified count. They expose the same total matching but
pay more concentration slacks and more worst-case singleton losses.

### 8.2 A ceiling for the recurrence

The recurrence itself cannot certify a complete peel, regardless of the
reservoir sizes. Both `u_t/n_t` and `z_t/n_t` are nonincreasing. Moreover,
`n_t>=N-2K=N/3`. Dropping the floors and all slacks therefore gives

```text
sum_t q_t
  <= sum_t (m_t/2)(u_t/n_t)(z_t/n_t)(n_t/(n_t-1))
  <= (K/2)(u_1/N)(z_1/N)((N/3)/(N/3-1)).
```

At the intended parameters, this upper bound on what the recurrence can
certify is 26,216.04 columns, or 18.90% of the deletion. This is not an upper
bound on the actual peeling algorithm. It proves only that no choice of
reservoir partition can close the proof while retaining the present category
bounds and the four-singleton loss charge.

### 8.3 Zero-slack fluid limit

The obstruction remains when every concentration slack is set to zero. Split
all `K` matching edges into equal reservoirs, drop the integer floors, and
take the large-`N` limit followed by a vanishing reservoir mesh. Let `x` be the
fraction of row identities that remain unexposed. Let `u(x)` and `z(x)` be the
singleton and zero lower bounds, normalized by `N`, and write

```text
u_bar := u_1/N,
z_bar := z_1/N.
```

The recurrence converges to

```text
z(x) = z_bar x,
u(x) = u_bar x^(1+z_bar).
```

After all matching edges are exposed, `x=1/3`. The limiting peeled fraction is

```text
Q_infinity
  = u_bar z_bar/(4(1+z_bar)) * (1-3^(-(1+z_bar))).
```

Indeed, zero rows undergo only uniform thinning, which gives the first
identity. Dividing the singleton recurrence by `x` and passing to the limit
gives `d log(u(x)/x) = z_bar d log x`, which gives the second identity. The
candidate recurrence is then integrated from `x=1` to `x=1/3`.

For the intended run profile,

```text
u_bar = 0.296255...,
z_bar = 0.506351...,
Q_infinity = 0.0201382N = 0.152216d.
```

This value is not an upper bound on the actual peeling process. It is the
fluid limit of the present lower-bound recurrence under ideal conditions. The
general ceiling in Section 8.2, rather than the fluid calculation alone,
proves that adding reservoirs cannot turn this recurrence into a
complete-peel proof.

## 9. Consequences at the binary benchmark

Ignore lower-order endpoint and slack terms. Write

```text
alpha := d/N,
rho := r/N,
sigma := alpha+rho.
```

The singleton-row lower bound becomes positive when

```text
2sigma-sigma^2 > 2alpha.
```

Equivalently,

```text
rho > 1-sqrt(1-2alpha)-alpha.
```

At `alpha=0.1323`, the threshold run fraction is

```text
rho_0 = 0.0101453...
```

A uniform deletion of size `alpha N` has approximately
`alpha(1-alpha)N=0.114797N` runs. The first-round theorem therefore applies
far below the typical run count.

The exceptional few-run profiles also have reduced entropy. At `rho=rho_0`,

```text
E(0.1323,rho_0) = 0.131315...,
```

whereas all deletions cost `h_2(0.1323)=0.563712...` bits per coordinate. For
a 128-bit prime, a hypothetical `p^(-gamma N)` saturation bound would pay for
the few-run profiles once

```text
gamma > 0.131315/128 = 0.0010259...
```

This calculation isolates the remaining task. Many-run profiles admit direct
peeling progress. Few-run profiles have much lower entropy and the exact
zero-sum structure from Lemma 1.

For the actual fixed random deletion used at `N=1048575`, the parameters are

```text
d = 138726,
r = 120088,
tau = 0.
```

Set `a=b=0.02N`. Theorem 6 proves that the first round peels at least 23,594
columns with probability at least `0.997046`. This is 17.0% of the deletion.
Across ten interleaver samples, the observed first round removed between
120,466 and 120,731 columns. Every sample then peeled completely. The theorem
is deliberately conservative: it certifies only 19.6% of the smallest
observed first-round count and does not certify the later rounds.

For a genuinely staged certificate, set

```text
gamma K = 174762,
(1-gamma)K = 174763,
q_bar = floor(0.017N),
c = 0.005N,
b_1 = b_2 = sqrt(8N log(1000)).
```

Theorem 8 certifies 11,204 pivots in the first reserved matching and 6,237
additional pivots in the second. It therefore peels at least 17,441 columns,
or 12.57% of the deletion, with failure probability below `0.003988`.

The staged count is smaller than the one-round count because each stage uses
only half of one matching. Its value is logical rather than numerical: the
second-round probability is valid after conditioning on the columns selected
in the first round.

The script
[`scripts/run_profile_bound.py`](scripts/run_profile_bound.py) verifies the
finite profile counts through length ten and evaluates these constants.

## 10. Next proof obligation

Theorem 9 completes the formal multi-reservoir extension. Its continuum limit
does not contract to complete peeling. Further reservoir splitting is
therefore not useful without a stronger structural estimate.

One possible refinement would track how a pivot column meets unexposed rows.
The present recurrence charges four destroyed singleton rows per pivot. A
conditional occupancy theorem may replace that charge by its typical value.
Without such a refinement, the proof should move to rank anti-concentration
for the residual matrix rather than continue the peeling trajectory.

Profiles below `rho_0` should use Lemma 1 instead. Each deletion becomes a
small direct sum of zero-sum spaces embedded by `P_2`. The remaining objective
is an anti-concentration bound for the intersection of this embedded space
with the permuted repetition kernel.
