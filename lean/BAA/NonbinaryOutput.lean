import BAA.NonbinarySupport

/-!
# Output semantics of decoded nonbinary accumulator placements

The active intervals form a disjoint partition of all coordinates after the
leading zero gap.  Consequently, when the active prefix states vanish exactly
at the decoded return indices, the actual accumulated word has weight `h`.
-/

namespace BAA

theorem scheduleActivePositionNat_add_interval_le_N {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) :
    scheduleActivePositionNat p i + scheduleActiveInterval p i ≤ N := by
  have hset : Finset.Iic i = insert i (Finset.Iio i) := by
    ext j
    simp only [Finset.mem_Iic, Finset.mem_insert, Finset.mem_Iio]
    omega
  have hnmem : i ∉ Finset.Iio i := by simp
  have hle :
      (∑ j ∈ Finset.Iic i, scheduleActiveInterval p j) ≤
        ∑ j : Fin s, scheduleActiveInterval p j := by
    apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
    intro _ _ _
    exact Nat.zero_le _
  have htotal := scheduleTotalLength p hhN
  unfold scheduleActivePositionNat
  rw [hset, Finset.sum_insert hnmem] at hle
  omega

/-- Coordinate at offset `o` in active interval `i`. -/
def scheduleIntervalCoordinate {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) (o : Fin (scheduleActiveInterval p i)) : Fin N :=
  ⟨scheduleActivePositionNat p i + o.val, by
    have hend := scheduleActivePositionNat_add_interval_le_N p hhN i
    omega⟩

theorem scheduleIntervalCoordinate_injective {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) : Function.Injective (scheduleIntervalCoordinate p hhN i) := by
  intro x y hxy
  apply Fin.ext
  have := congrArg Fin.val hxy
  change scheduleActivePositionNat p i + x.val =
    scheduleActivePositionNat p i + y.val at this
  omega

/-- Coordinates occupied by the constant state following active symbol `i`. -/
def scheduleActiveIntervalSet {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) : Finset (Fin N) :=
  Finset.univ.image (scheduleIntervalCoordinate p hhN i)

theorem mem_scheduleActiveIntervalSet_iff {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) (k : Fin N) :
    k ∈ scheduleActiveIntervalSet p hhN i ↔
      scheduleActivePositionNat p i ≤ k.val ∧
        k.val < scheduleActivePositionNat p i + scheduleActiveInterval p i := by
  constructor
  · intro hk
    rw [scheduleActiveIntervalSet, Finset.mem_image] at hk
    obtain ⟨o, _, rfl⟩ := hk
    change scheduleActivePositionNat p i ≤
        scheduleActivePositionNat p i + o.val ∧
      scheduleActivePositionNat p i + o.val <
        scheduleActivePositionNat p i + scheduleActiveInterval p i
    omega
  · rintro ⟨hlow, hhigh⟩
    let o : Fin (scheduleActiveInterval p i) :=
      ⟨k.val - scheduleActivePositionNat p i, by omega⟩
    apply Finset.mem_image.mpr
    refine ⟨o, Finset.mem_univ _, ?_⟩
    apply Fin.ext
    change scheduleActivePositionNat p i +
      (k.val - scheduleActivePositionNat p i) = k.val
    omega

@[simp]
theorem card_scheduleActiveIntervalSet {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (i : Fin s) :
    (scheduleActiveIntervalSet p hhN i).card = scheduleActiveInterval p i := by
  rw [scheduleActiveIntervalSet, Finset.card_image_of_injective _
    (scheduleIntervalCoordinate_injective p hhN i)]
  simp

theorem scheduleActiveIntervalSet_disjoint {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    {i j : Fin s} (hij : i ≠ j) :
    Disjoint (scheduleActiveIntervalSet p hhN i)
      (scheduleActiveIntervalSet p hhN j) := by
  apply Finset.disjoint_left.mpr
  intro k hki hkj
  rw [mem_scheduleActiveIntervalSet_iff] at hki hkj
  rcases lt_or_gt_of_ne hij with hij | hji
  · have hsep := scheduleActivePositionNat_add_interval_le p hij
    omega
  · have hsep := scheduleActivePositionNat_add_interval_le p hji
    omega

/-- Union of all active intervals. -/
def scheduleActiveRegion {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    Finset (Fin N) :=
  Finset.univ.biUnion (scheduleActiveIntervalSet p hhN)

theorem card_scheduleActiveRegion {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    (scheduleActiveRegion p hhN).card =
      ∑ i : Fin s, scheduleActiveInterval p i := by
  unfold scheduleActiveRegion
  rw [Finset.card_biUnion]
  · simp
  · intro i _ j _ hij
    exact scheduleActiveIntervalSet_disjoint p hhN hij

/-- Coordinates after the leading zero gap. -/
def scheduleTailRegion {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) : Finset (Fin N) :=
  Finset.univ.filter fun k => scheduleLeadingZero p ≤ k.val

theorem scheduleActiveRegion_eq_tail {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    scheduleActiveRegion p hhN = scheduleTailRegion p := by
  apply Finset.eq_of_subset_of_card_le
  · intro k hk
    rw [scheduleActiveRegion, Finset.mem_biUnion] at hk
    obtain ⟨i, _, hki⟩ := hk
    rw [mem_scheduleActiveIntervalSet_iff] at hki
    simp only [scheduleTailRegion, Finset.mem_filter, Finset.mem_univ, true_and]
    unfold scheduleActivePositionNat at hki
    omega
  · have htotal := scheduleTotalLength p hhN
    rw [card_scheduleActiveRegion]
    have hcard : (scheduleTailRegion p).card = N - scheduleLeadingZero p := by
      have hlead : scheduleLeadingZero p ≤ N := by omega
      by_cases hlt : scheduleLeadingZero p < N
      · have hset : scheduleTailRegion p =
            Finset.Ici (⟨scheduleLeadingZero p, hlt⟩ : Fin N) := by
          ext k
          simp only [scheduleTailRegion, Finset.mem_filter, Finset.mem_univ,
            true_and, Finset.mem_Ici]
          change scheduleLeadingZero p ≤ k.val ↔
            scheduleLeadingZero p ≤ k.val
          rfl
        rw [hset]
        simpa using
          (Fin.card_Ici (n := N) (⟨scheduleLeadingZero p, hlt⟩ : Fin N))
      · have heq : scheduleLeadingZero p = N := by omega
        simp [scheduleTailRegion, heq]
    rw [hcard]
    omega

/-- The union of the intervals whose active state is nonzero according to the
decoded return trace. -/
def scheduleNonzeroRegion {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    Finset (Fin N) :=
  (Finset.univ.filter fun i => i ∉ decodedZeroReturnTrace p.1).biUnion
    (scheduleActiveIntervalSet p hhN)

theorem card_scheduleNonzeroRegion {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N) :
    (scheduleNonzeroRegion p hhN).card = h := by
  unfold scheduleNonzeroRegion
  rw [Finset.card_biUnion]
  · rw [Finset.sum_filter]
    simpa using scheduleOutputWeight p
  · intro i _ j _ hij
    exact scheduleActiveIntervalSet_disjoint p hhN hij

section Field

variable {F : Type*} [Field F] [DecidableEq F]

/-- The value-level trace condition needed independently of the support
placement count in equation (1). -/
def MatchesDecodedReturnTrace {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (u : Fin s → F) : Prop :=
  accumulatorZeroReturnTrace u = decodedZeroReturnTrace p.1

theorem support_accum_scheduleInputWord {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (htrace : MatchesDecodedReturnTrace p u) :
    support (accumLinearEquiv F N (scheduleInputWord p hhN u)) =
      scheduleNonzeroRegion p hhN := by
  classical
  unfold MatchesDecodedReturnTrace at htrace
  ext k
  rw [mem_support]
  constructor
  · intro hk
    have htail : k ∈ scheduleTailRegion p := by
      by_contra hnot
      have hbefore : k.val < scheduleLeadingZero p := by
        simpa [scheduleTailRegion] using hnot
      exact hk (accum_scheduleInputWord_before_first p hhN u k hbefore)
    rw [← scheduleActiveRegion_eq_tail p hhN,
      scheduleActiveRegion, Finset.mem_biUnion] at htail
    obtain ⟨i, _, hki⟩ := htail
    have hi := (mem_scheduleActiveIntervalSet_iff p hhN i k).mp hki
    have hstate := accum_scheduleInputWord_on_interval p hhN u i k hi.1 hi.2
    have hinon : i ∉ decodedZeroReturnTrace p.1 := by
      intro hireturn
      have : accumulatorActiveState u i = 0 := by
        have : i ∈ accumulatorZeroReturnTrace u := by
          rw [htrace]
          exact hireturn
        simpa [accumulatorZeroReturnTrace] using this
      exact hk (hstate.trans this)
    rw [scheduleNonzeroRegion, Finset.mem_biUnion]
    exact ⟨i, by simp [hinon], hki⟩
  · intro hk
    rw [scheduleNonzeroRegion, Finset.mem_biUnion] at hk
    obtain ⟨i, hi, hki⟩ := hk
    have hinon : i ∉ decodedZeroReturnTrace p.1 := by simpa using hi
    have hstateNonzero : accumulatorActiveState u i ≠ 0 := by
      intro hz
      apply hinon
      rw [← htrace]
      simpa [accumulatorZeroReturnTrace] using hz
    have hik := (mem_scheduleActiveIntervalSet_iff p hhN i k).mp hki
    rw [accum_scheduleInputWord_on_interval p hhN u i k hik.1 hik.2]
    exact hstateNonzero

/-- Equation (1)'s schedule has actual accumulator output weight `h` once its
field values realize the decoded return trace. -/
theorem hammingNorm_accum_scheduleInputWord {N s h a : ℕ}
    (p : NonbinaryAccumulatorPlacements N s h a) (hhN : h ≤ N)
    (u : Fin s → F) (htrace : MatchesDecodedReturnTrace p u) :
    hammingNorm (accumLinearEquiv F N (scheduleInputWord p hhN u)) = h := by
  rw [← card_support_eq_hammingNorm,
    support_accum_scheduleInputWord p hhN u htrace,
    card_scheduleNonzeroRegion]

end Field

end BAA
