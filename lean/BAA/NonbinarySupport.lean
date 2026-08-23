import BAA.NonbinarySchedule

/-!
# Concrete supports decoded from accumulator placement parameters

The placement parameters determine the coordinate of every active input
symbol.  Positivity of all schedule intervals makes these coordinates strictly
increasing, so their image is an actual support of cardinality `s` in `Fin N`.
-/

namespace BAA

/-- Natural-valued coordinate of active input symbol `i`. -/
def scheduleActivePositionNat {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (i : Fin s) : ℕ :=
  scheduleLeadingZero p +
    ∑ j ∈ Finset.Iio i, scheduleActiveInterval p j

theorem scheduleActivePositionNat_lt {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) : scheduleActivePositionNat p i < N := by
  have hsum :
      (∑ j ∈ Finset.Iio i, scheduleActiveInterval p j) <
        ∑ j : Fin s, scheduleActiveInterval p j := by
    apply Finset.sum_lt_sum_of_subset (s := Finset.Iio i)
      (t := Finset.univ) (i := i)
    · exact Finset.subset_univ _
    · simp
    · simp
    · exact scheduleActiveInterval_pos p i
    · intro j _ _
      exact Nat.zero_le _
  have htotal := scheduleTotalLength p hhN
  unfold scheduleActivePositionNat
  omega

/-- Coordinate of active input symbol `i` in the length-`N` word. -/
def scheduleActivePosition {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) : Fin N :=
  ⟨scheduleActivePositionNat p i, scheduleActivePositionNat_lt p hhN i⟩

theorem scheduleActivePositionNat_strictMono {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) :
    StrictMono (scheduleActivePositionNat p) := by
  intro i j hij
  have hsum :
      (∑ k ∈ Finset.Iio i, scheduleActiveInterval p k) <
        ∑ k ∈ Finset.Iio j, scheduleActiveInterval p k := by
    apply Finset.sum_lt_sum_of_subset (s := Finset.Iio i)
      (t := Finset.Iio j) (i := i)
    · intro k hk
      simp only [Finset.mem_Iio] at hk ⊢
      exact hk.trans hij
    · exact Finset.mem_Iio.mpr hij
    · simp
    · exact scheduleActiveInterval_pos p i
    · intro k _ _
      exact Nat.zero_le _
  unfold scheduleActivePositionNat
  omega

theorem scheduleActivePositionNat_succ {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (i : Fin (s - 1)) :
    scheduleActivePositionNat p ⟨i.val + 1, by omega⟩ =
      scheduleActivePositionNat p ⟨i.val, by omega⟩ +
        scheduleActiveInterval p ⟨i.val, by omega⟩ := by
  let current : Fin s := ⟨i.val, by omega⟩
  let next : Fin s := ⟨i.val + 1, by omega⟩
  have hset : Finset.Iio next = insert current (Finset.Iio current) := by
    ext j
    simp only [Finset.mem_Iio, Finset.mem_insert]
    constructor
    · intro hj
      by_cases heq : j.val = i.val
      · left
        exact Fin.ext heq
      · right
        change j.val < i.val
        change j.val < i.val + 1 at hj
        omega
    · rintro (rfl | hj)
      · change i.val < i.val + 1
        omega
      · change j.val < i.val at hj
        change j.val < i.val + 1
        omega
  have hnmem : current ∉ Finset.Iio current := by simp
  unfold scheduleActivePositionNat
  rw [hset, Finset.sum_insert hnmem]
  simp only [current, next]
  omega

theorem scheduleActivePositionNat_add_interval_le {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) {i j : Fin s}
    (hij : i < j) :
    scheduleActivePositionNat p i + scheduleActiveInterval p i ≤
      scheduleActivePositionNat p j := by
  let q : Fin (s - 1) := ⟨i.val, by omega⟩
  let next : Fin s := ⟨i.val + 1, by omega⟩
  have hnext : next ≤ j := by
    change i.val + 1 ≤ j.val
    omega
  have hmono := (scheduleActivePositionNat_strictMono p).monotone hnext
  have hstep := scheduleActivePositionNat_succ p q
  change scheduleActivePositionNat p i + scheduleActiveInterval p i ≤
    scheduleActivePositionNat p j
  calc
    _ = scheduleActivePositionNat p next := by
      simpa [q, next] using hstep.symm
    _ ≤ _ := hmono

theorem scheduleActivePosition_strictMono {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    StrictMono (scheduleActivePosition p hhN) := by
  intro i j hij
  exact scheduleActivePositionNat_strictMono p hij

/-- The concrete length-`N` support selected by placement parameters. -/
def scheduleInputSupport {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    Finset (Fin N) :=
  Finset.univ.image (scheduleActivePosition p hhN)

@[simp]
theorem card_scheduleInputSupport {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    (scheduleInputSupport p hhN).card = s := by
  rw [scheduleInputSupport, Finset.card_image_of_injective _
    (scheduleActivePosition_strictMono p hhN).injective]
  simp

theorem scheduleActivePosition_mem_support {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) :
    scheduleActivePosition p hhN i ∈ scheduleInputSupport p hhN := by
  exact Finset.mem_image.mpr ⟨i, Finset.mem_univ _, rfl⟩

section Values

variable {F : Type*} [Field F] [DecidableEq F]

/-- Place the active-symbol sequence `u` at the support coordinates decoded
from `p`, and put zero at every other coordinate. -/
def scheduleInputWord {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) : Word F N :=
  fun k => ∑ i : Fin s,
    if scheduleActivePosition p hhN i = k then u i else 0

@[simp]
theorem scheduleInputWord_active {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (i : Fin s) :
    scheduleInputWord p hhN u (scheduleActivePosition p hhN i) = u i := by
  classical
  unfold scheduleInputWord
  calc
    _ = if scheduleActivePosition p hhN i =
          scheduleActivePosition p hhN i then u i else 0 := by
      apply Fintype.sum_eq_single i
      intro j hji
      have hne : scheduleActivePosition p hhN j ≠
          scheduleActivePosition p hhN i := by
        exact fun h => hji ((scheduleActivePosition_strictMono p hhN).injective h)
      simp [hne]
    _ = u i := by simp

theorem support_scheduleInputWord {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (hu : ∀ i, u i ≠ 0) :
    support (scheduleInputWord p hhN u) = scheduleInputSupport p hhN := by
  classical
  ext k
  rw [mem_support]
  constructor
  · intro hk
    by_contra hkmem
    apply hk
    unfold scheduleInputWord
    apply Finset.sum_eq_zero
    intro i _
    have hne : scheduleActivePosition p hhN i ≠ k := by
      intro heq
      apply hkmem
      rw [← heq]
      exact scheduleActivePosition_mem_support p hhN i
    simp [hne]
  · intro hkmem
    rw [scheduleInputSupport, Finset.mem_image] at hkmem
    obtain ⟨i, _, rfl⟩ := hkmem
    simpa [scheduleInputWord_active] using hu i

theorem hammingNorm_scheduleInputWord {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (hu : ∀ i, u i ≠ 0) :
    hammingNorm (scheduleInputWord p hhN u) = s := by
  rw [← card_support_eq_hammingNorm, support_scheduleInputWord p hhN u hu,
    card_scheduleInputSupport]

/-- Prefix summation of the placed word is the sum of precisely the active
symbols whose scheduled coordinates have already occurred. -/
theorem accum_scheduleInputWord_apply {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (k : Fin N) :
    accumLinearEquiv F N (scheduleInputWord p hhN u) k =
      ∑ i : Fin s, if scheduleActivePosition p hhN i ≤ k then u i else 0 := by
  classical
  rw [accum_apply_eq_sum_Iic]
  unfold scheduleInputWord
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : scheduleActivePosition p hhN i ≤ k
  · rw [if_pos hi]
    calc
      _ = if scheduleActivePosition p hhN i =
          scheduleActivePosition p hhN i then u i else 0 := by
        apply Finset.sum_eq_single (scheduleActivePosition p hhN i)
        · intro j _ hji
          simp [Ne.symm hji]
        · intro hnot
          exact (hnot (Finset.mem_Iic.mpr hi)).elim
      _ = u i := by simp
  · rw [if_neg hi]
    apply Finset.sum_eq_zero
    intro j hj
    have hne : scheduleActivePosition p hhN i ≠ j := by
      intro hij
      apply hi
      rw [hij]
      exact Finset.mem_Iic.mp hj
    simp [hne]

/-- Every coordinate in scheduled active interval `i` sees exactly the prefix
state after active symbol `i`. -/
theorem accum_scheduleInputWord_on_interval {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (i : Fin s) (k : Fin N)
    (hlow : scheduleActivePositionNat p i ≤ k.val)
    (hhigh : k.val < scheduleActivePositionNat p i +
      scheduleActiveInterval p i) :
    accumLinearEquiv F N (scheduleInputWord p hhN u) k =
      accumulatorActiveState u i := by
  rw [accum_scheduleInputWord_apply]
  unfold accumulatorActiveState
  have hiff : ∀ j : Fin s,
      scheduleActivePosition p hhN j ≤ k ↔ j ≤ i := by
    intro j
    constructor
    · intro hj
      by_contra hji
      have hij : i < j := lt_of_not_ge hji
      have hnext := scheduleActivePositionNat_add_interval_le p hij
      change scheduleActivePositionNat p j ≤ k.val at hj
      omega
    · intro hji
      have hpos := (scheduleActivePositionNat_strictMono p).monotone hji
      change scheduleActivePositionNat p j ≤ k.val
      omega
  classical
  calc
    (∑ j : Fin s,
        if scheduleActivePosition p hhN j ≤ k then u j else 0) =
        ∑ j : Fin s, if j ≤ i then u j else 0 := by
      apply Fintype.sum_congr
      intro j
      simp only [hiff j]
    _ = ∑ j ∈ Finset.Iic i, u j := by
      have hiic : Finset.Iic i =
          Finset.univ.filter (fun j : Fin s => j ≤ i) := by
        ext j
        simp
      rw [hiic]
      exact (Finset.sum_filter (s := Finset.univ)
        (p := fun j : Fin s => j ≤ i) (f := u)).symm

theorem accum_scheduleInputWord_before_first {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (k : Fin N)
    (hk : k.val < scheduleLeadingZero p) :
    accumLinearEquiv F N (scheduleInputWord p hhN u) k = 0 := by
  rw [accum_scheduleInputWord_apply]
  apply Finset.sum_eq_zero
  intro i _
  have hpos : scheduleLeadingZero p ≤ scheduleActivePositionNat p i := by
    unfold scheduleActivePositionNat
    omega
  have hne : ¬ scheduleActivePosition p hhN i ≤ k := by
    change ¬ scheduleActivePositionNat p i ≤ k.val
    omega
  simp [hne]

end Values

end BAA
