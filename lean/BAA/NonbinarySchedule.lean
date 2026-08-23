import BAA.NonbinaryTraceDecode

/-!
# Concrete accumulator timelines from placement parameters

The interval following active symbol `i` has the accumulator state produced by
that symbol.  Return intervals therefore contribute zero output coordinates;
all other intervals contribute to the output weight.
-/

namespace BAA

def fixedCompositionPart {n k : ℕ} (c : FixedLengthComposition n k)
    (i : Fin k) : ℕ :=
  c.1.blocksFun (finCongr c.2.symm i)

theorem fixedCompositionPart_pos {n k : ℕ}
    (c : FixedLengthComposition n k) (i : Fin k) :
    0 < fixedCompositionPart c i := by
  exact c.1.one_le_blocksFun (finCongr c.2.symm i)

theorem sum_fixedCompositionPart {n k : ℕ}
    (c : FixedLengthComposition n k) :
    ∑ i : Fin k, fixedCompositionPart c i = n := by
  calc
    (∑ i : Fin k, fixedCompositionPart c i) =
        ∑ j : Fin c.1.length, c.1.blocksFun j :=
      (finCongr c.2.symm).sum_comp c.1.blocksFun
    _ = n := c.1.sum_blocksFun

theorem fixedCompositionPart_injective {n k : ℕ} :
    Function.Injective (@fixedCompositionPart n k) := by
  intro c d hcd
  apply Subtype.ext
  apply Composition.ext
  have hlen : c.1.blocks.length = d.1.blocks.length := by
    exact c.2.trans d.2.symm
  apply List.ext_get hlen
  intro i hic hid
  let j : Fin k := ⟨i, by simpa [← c.2] using hic⟩
  have hj := congrFun hcd j
  simpa [fixedCompositionPart, Composition.blocksFun, j] using hj

/-- Build a fixed-length composition from a positive indexed family with the
prescribed total. -/
def fixedLengthCompositionOfFn {n k : ℕ} (f : Fin k → ℕ)
    (hpos : ∀ i, 0 < f i) (hsum : ∑ i, f i = n) :
    FixedLengthComposition n k := by
  let c : Composition n :=
    { blocks := List.ofFn f
      blocks_pos := by
        intro x hx
        rw [List.mem_ofFn] at hx
        obtain ⟨i, rfl⟩ := hx
        exact hpos i
      blocks_sum := by
        rw [List.sum_ofFn]
        exact hsum }
  exact ⟨c, by simp only [c, Composition.length, List.length_ofFn]⟩

@[simp]
theorem fixedCompositionPart_fixedLengthCompositionOfFn {n k : ℕ}
    (f : Fin k → ℕ) (hpos : ∀ i, 0 < f i) (hsum : ∑ i, f i = n)
    (i : Fin k) :
    fixedCompositionPart (fixedLengthCompositionOfFn f hpos hsum) i = f i := by
  unfold fixedCompositionPart fixedLengthCompositionOfFn
  simp only [Composition.blocksFun, List.get_ofFn]
  congr 1

def decodedReturnRank {s a : ℕ} (t : CompressedZeroReturnTrace s a)
    (i : {i : Fin s // i ∈ decodedZeroReturnTrace t}) : Fin a :=
  ((decodedZeroReturnTrace t).orderIsoOfFin
    (card_decodedZeroReturnTrace t)).symm i

def decodedNonreturnRank {s a : ℕ} (t : CompressedZeroReturnTrace s a)
    (i : {i : Fin s // i ∉ decodedZeroReturnTrace t}) : Fin (s - a) := by
  let A := decodedZeroReturnTrace t
  have hcard : Aᶜ.card = s - a := by
    rw [Finset.card_compl, Fintype.card_fin, card_decodedZeroReturnTrace]
  exact (Aᶜ.orderIsoOfFin hcard).symm
    ⟨i.1, Finset.mem_compl.mpr i.2⟩

/-- Number of initial zero-state coordinates before the first active symbol. -/
def scheduleLeadingZero {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) : ℕ :=
  fixedCompositionPart p.2.1 ⟨0, Nat.zero_lt_succ a⟩ - 1

/-- Length of the constant-state interval beginning at active symbol `i`. -/
def scheduleActiveInterval {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (i : Fin s) : ℕ :=
  if hi : i ∈ decodedZeroReturnTrace p.1 then
    fixedCompositionPart p.2.1 (Fin.succ (decodedReturnRank p.1 ⟨i, hi⟩))
  else
    fixedCompositionPart p.2.2 (decodedNonreturnRank p.1 ⟨i, hi⟩)

theorem scheduleActiveInterval_pos {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (i : Fin s) :
    0 < scheduleActiveInterval p i := by
  unfold scheduleActiveInterval
  split <;> apply fixedCompositionPart_pos

theorem sum_return_scheduleIntervals {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) :
    (∑ i : {i : Fin s // i ∈ decodedZeroReturnTrace p.1},
      scheduleActiveInterval p i.1) =
      ∑ j : Fin a, fixedCompositionPart p.2.1 j.succ := by
  let e := ((decodedZeroReturnTrace p.1).orderIsoOfFin
    (card_decodedZeroReturnTrace p.1)).symm.toEquiv
  calc
    _ = ∑ i : {i : Fin s // i ∈ decodedZeroReturnTrace p.1},
        fixedCompositionPart p.2.1 (Fin.succ (e i)) := by
      apply Fintype.sum_congr
      intro i
      simp [scheduleActiveInterval, e, decodedReturnRank]
    _ = _ := e.sum_comp (fun j : Fin a =>
      fixedCompositionPart p.2.1 j.succ)

theorem sum_nonreturn_scheduleIntervals {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) :
    (∑ i : {i : Fin s // i ∉ decodedZeroReturnTrace p.1},
      scheduleActiveInterval p i.1) = h := by
  let A := decodedZeroReturnTrace p.1
  have hcard : Aᶜ.card = s - a := by
    rw [Finset.card_compl, Fintype.card_fin, card_decodedZeroReturnTrace]
  let e₀ : {i : Fin s // i ∉ A} ≃ ↥(Aᶜ) := {
    toFun i := ⟨i.1, Finset.mem_compl.mpr i.2⟩
    invFun i := ⟨i.1, Finset.mem_compl.mp i.2⟩
    left_inv _ := rfl
    right_inv _ := rfl }
  let e := e₀.trans (Aᶜ.orderIsoOfFin hcard).symm.toEquiv
  calc
    _ = ∑ i : {i : Fin s // i ∉ A},
        fixedCompositionPart p.2.2 (e i) := by
      apply Fintype.sum_congr
      intro i
      rw [scheduleActiveInterval, dif_neg i.2]
      rfl
    _ = ∑ j : Fin (s - a), fixedCompositionPart p.2.2 j :=
      e.sum_comp (fixedCompositionPart p.2.2)
    _ = h := sum_fixedCompositionPart p.2.2

/-- The active intervals marked non-return have total length exactly `h`. -/
theorem scheduleOutputWeight {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) :
    (∑ i : Fin s, if i ∈ decodedZeroReturnTrace p.1 then 0
      else scheduleActiveInterval p i) = h := by
  have hnon := sum_nonreturn_scheduleIntervals p
  have hsplit := Fintype.sum_subtype_add_sum_subtype
    (fun i : Fin s => i ∈ decodedZeroReturnTrace p.1)
    (fun i => if i ∈ decodedZeroReturnTrace p.1 then 0
      else scheduleActiveInterval p i)
  have hzero :
      (∑ i : {i : Fin s // i ∈ decodedZeroReturnTrace p.1},
        (if i.1 ∈ decodedZeroReturnTrace p.1 then 0
          else scheduleActiveInterval p i.1)) = 0 := by
    apply Finset.sum_eq_zero
    intro i _
    simp [i.2]
  have hkeep :
      (∑ i : {i : Fin s // i ∉ decodedZeroReturnTrace p.1},
        (if i.1 ∈ decodedZeroReturnTrace p.1 then 0
          else scheduleActiveInterval p i.1)) =
        ∑ i : {i : Fin s // i ∉ decodedZeroReturnTrace p.1},
          scheduleActiveInterval p i.1 := by
    apply Fintype.sum_congr
    intro i
    simp [i.2]
  calc
    _ = (∑ i : {i : Fin s // i ∉ decodedZeroReturnTrace p.1},
        scheduleActiveInterval p i.1) := by
      have hcombine := congrArg₂ (· + ·) hzero hkeep
      have hsplitSymm := hsplit.symm
      exact hsplitSymm.trans (by simpa using hcombine)
    _ = h := hnon

/-- The leading zero gap and all active intervals fill exactly `N`
coordinates. -/
theorem scheduleTotalLength {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a)
    (hhN : h ≤ N) :
    scheduleLeadingZero p + ∑ i : Fin s, scheduleActiveInterval p i = N := by
  have hreturn := sum_return_scheduleIntervals p
  have hnon := sum_nonreturn_scheduleIntervals p
  have hzsum := sum_fixedCompositionPart p.2.1
  rw [Fin.sum_univ_succ] at hzsum
  have hz0 := fixedCompositionPart_pos p.2.1 ⟨0, Nat.zero_lt_succ a⟩
  have hzsub :
      fixedCompositionPart p.2.1 ⟨0, Nat.zero_lt_succ a⟩ - 1 + 1 =
        fixedCompositionPart p.2.1 ⟨0, Nat.zero_lt_succ a⟩ :=
    Nat.sub_add_cancel hz0
  let A := decodedZeroReturnTrace p.1
  have hsplitFin :
      (∑ i ∈ A, scheduleActiveInterval p i) +
        (∑ i ∈ Aᶜ, scheduleActiveInterval p i) =
          ∑ i : Fin s, scheduleActiveInterval p i :=
    Finset.sum_add_sum_compl A (scheduleActiveInterval p)
  have hreturnFin : (∑ i ∈ A, scheduleActiveInterval p i) =
      ∑ j : Fin a, fixedCompositionPart p.2.1 j.succ := by
    calc
      _ = ∑ i : ↥A, scheduleActiveInterval p i.1 := by
        simpa [Finset.univ_eq_attach] using
          (Finset.sum_attach A (scheduleActiveInterval p)).symm
      _ = _ := hreturn
  have hnonFin : (∑ i ∈ Aᶜ, scheduleActiveInterval p i) = h := by
    let e₀ : {i : Fin s // i ∉ A} ≃ ↥(Aᶜ) := {
      toFun i := ⟨i.1, Finset.mem_compl.mpr i.2⟩
      invFun i := ⟨i.1, Finset.mem_compl.mp i.2⟩
      left_inv _ := rfl
      right_inv _ := rfl }
    calc
      _ = ∑ i : ↥(Aᶜ), scheduleActiveInterval p i.1 := by
        simpa [Finset.univ_eq_attach] using
          (Finset.sum_attach Aᶜ (scheduleActiveInterval p)).symm
      _ = ∑ i : {i : Fin s // i ∉ A},
          scheduleActiveInterval p i.1 := by
        exact (e₀.sum_comp (fun i : ↥(Aᶜ) =>
          scheduleActiveInterval p i.1)).symm
      _ = h := by simpa [A] using hnon
  have hzsum' :
      fixedCompositionPart p.2.1 ⟨0, Nat.zero_lt_succ a⟩ +
        (∑ i : Fin a, fixedCompositionPart p.2.1 i.succ) = N - h + 1 := by
    simpa using hzsum
  unfold scheduleLeadingZero
  omega

end BAA
