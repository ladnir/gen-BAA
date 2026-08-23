import BAA.BinaryRuns
import BAA.FiniteProbability

/-!
# Nonbinary accumulator zero-return traces

For a trace with `a` nonadjacent returns among `s` active symbols, deleting
the mandatory separator before each return gives an `a`-subset of `s-a`
positions.  The other two components are the positive compositions of the
zero-state and nonzero-state gap lengths.
-/

namespace BAA

section TraceSemantics

variable {F : Type*} [Field F] [DecidableEq F] {s : ℕ}

/-- Accumulator state immediately after active symbol `i`. -/
def accumulatorActiveState (u : Fin s → F) (i : Fin s) : F :=
  ∑ j ∈ Finset.Iic i, u j

/-- Active-symbol indices after which the accumulator returns to zero. -/
def accumulatorZeroReturnTrace (u : Fin s → F) : Finset (Fin s) :=
  Finset.univ.filter fun i => accumulatorActiveState u i = 0

theorem accumulatorActiveState_zero (u : Fin s → F) (hs : 0 < s) :
    accumulatorActiveState u ⟨0, hs⟩ = u ⟨0, hs⟩ := by
  have hset : Finset.Iic (⟨0, hs⟩ : Fin s) = {⟨0, hs⟩} := by
    ext j
    simp only [Finset.mem_Iic, Finset.mem_singleton]
    constructor
    · intro hj
      apply Fin.ext
      change j.val = 0
      change j.val ≤ 0 at hj
      omega
    · intro hj
      rw [hj]
  rw [accumulatorActiveState, hset]
  simp

theorem first_not_mem_zeroReturnTrace (u : Fin s → F) (hs : 0 < s)
    (hu : ∀ i, u i ≠ 0) :
    ⟨0, hs⟩ ∉ accumulatorZeroReturnTrace u := by
  simp [accumulatorZeroReturnTrace, accumulatorActiveState_zero u hs, hu]

theorem accumulatorActiveState_succ (u : Fin s → F) (i : Fin (s - 1)) :
    accumulatorActiveState u ⟨i.val + 1, by omega⟩ =
      accumulatorActiveState u ⟨i.val, by omega⟩ + u ⟨i.val + 1, by omega⟩ := by
  let current : Fin s := ⟨i.val, by omega⟩
  let next : Fin s := ⟨i.val + 1, by omega⟩
  have hset : Finset.Iic next = insert next (Finset.Iic current) := by
    ext j
    simp only [Finset.mem_Iic, Finset.mem_insert]
    constructor
    · intro hj
      by_cases heq : j.val = i.val + 1
      · left
        apply Fin.ext
        exact heq
      · right
        change j.val ≤ i.val
        change j.val ≤ i.val + 1 at hj
        omega
    · rintro (hj | hj)
      · rw [hj]
      · change j.val ≤ i.val at hj
        change j.val ≤ i.val + 1
        omega
  have hnmem : next ∉ Finset.Iic current := by
    simp [current, next]
  simp only [accumulatorActiveState]
  rw [hset, Finset.sum_insert hnmem]
  simp [current, next, add_comm]

theorem zeroReturnTrace_nonadjacent (u : Fin s → F) (hu : ∀ i, u i ≠ 0)
    (i : Fin (s - 1)) :
    ¬ (⟨i.val, by omega⟩ ∈ accumulatorZeroReturnTrace u ∧
      ⟨i.val + 1, by omega⟩ ∈ accumulatorZeroReturnTrace u) := by
  intro h
  have hcurrent : accumulatorActiveState u ⟨i.val, by omega⟩ = 0 := by
    simpa [accumulatorZeroReturnTrace] using h.1
  have hnext : accumulatorActiveState u ⟨i.val + 1, by omega⟩ = 0 := by
    simpa [accumulatorZeroReturnTrace] using h.2
  rw [accumulatorActiveState_succ, hcurrent, zero_add] at hnext
  exact hu _ hnext

end TraceSemantics

/-- The compressed encoding of a nonadjacent zero-return trace. -/
abbrev CompressedZeroReturnTrace (s a : ℕ) := FixedCardFinset (s - a) a

theorem card_compressedZeroReturnTrace (s a : ℕ) :
    Fintype.card (CompressedZeroReturnTrace s a) = Nat.choose (s - a) a :=
  card_fixedCardFinset (s - a) a

/-- Exact finite parameter space counted by the numerator of equation (1).
The zero-state composition has one shifted leading block, so its total is
`N-h+1` and its length is `a+1`. -/
abbrev NonbinaryAccumulatorPlacements (N s h a : ℕ) :=
  CompressedZeroReturnTrace s a ×
    FixedLengthComposition (N - h + 1) (a + 1) ×
    FixedLengthComposition h (s - a)

theorem card_nonbinaryAccumulatorPlacements {N s h a : ℕ}
    (hs : 0 < s) (hhs : a < s) (hh : 0 < h) (hhN : h ≤ N) :
    Fintype.card (NonbinaryAccumulatorPlacements N s h a) =
      Nat.choose (s - a) a * Nat.choose (N - h) a *
        Nat.choose (h - 1) (s - a - 1) := by
  rw [Fintype.card_prod, Fintype.card_prod,
    card_compressedZeroReturnTrace,
    card_fixedLengthComposition (by omega : 0 < N - h + 1) (by omega : 0 < a + 1),
    card_fixedLengthComposition hh (by omega : 0 < s - a)]
  have hz : N - h + 1 - 1 = N - h := by omega
  have ha : a + 1 - 1 = a := by omega
  rw [hz, ha]
  ring

/-- Exact rational support-placement kernel `G_N(s,h,a)` from equation (1). -/
noncomputable def nonbinaryAccumulatorKernel (N s h a : ℕ) : ℚ :=
  (Fintype.card (NonbinaryAccumulatorPlacements N s h a) : ℚ) /
    Nat.choose N s

theorem nonbinaryAccumulatorKernel_formula {N s h a : ℕ}
    (hs : 0 < s) (hhs : a < s) (hh : 0 < h) (hhN : h ≤ N) :
    nonbinaryAccumulatorKernel N s h a =
      ((Nat.choose (s - a) a : ℕ) : ℚ) *
        Nat.choose (N - h) a * Nat.choose (h - 1) (s - a - 1) /
        Nat.choose N s := by
  rw [nonbinaryAccumulatorKernel,
    card_nonbinaryAccumulatorPlacements hs hhs hh hhN]
  push_cast
  ring

/-- The denominator of (1) is exactly the number of size-`s` supports in
length `N`. -/
theorem card_inputSupports (N s : ℕ) :
    Fintype.card (FixedCardFinset N s) = Nat.choose N s :=
  card_fixedCardFinset N s

end BAA
