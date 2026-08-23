import BAA.FiniteProbability
import BAA.UniformPermutation
import BAA.WeightEnumerator

/-!
# Exact finite binary probabilities and first moments

This module connects the integer accumulator counts to the exact rational
finite-probability interface.  It contains B3 and is the home of the B4--B5
formalization.
-/

namespace BAA

def binaryAccumulatorSliceEvent (N w h : ℕ) :
    Finset (BinaryWeightSlice N w) :=
  Finset.univ.filter fun x =>
    hammingNorm (accumLinearEquiv (ZMod 2) N x.1) = h

noncomputable def binaryAccumulatorConditionalProbability (N w h : ℕ) : ℚ :=
  uniformEventProbability (Omega := BinaryWeightSlice N w) fun x =>
    hammingNorm (accumLinearEquiv (ZMod 2) N x.1) = h

private def accumulatorSliceEventEquiv (N w h : ℕ) :
    {x : BinaryWeightSlice N w //
      hammingNorm (accumLinearEquiv (ZMod 2) N x.1) = h} ≃
    {x : BinaryWord N //
      hammingNorm x = w ∧
        hammingNorm (accumLinearEquiv (ZMod 2) N x) = h} where
  toFun x := ⟨x.1.1, x.1.2, x.2⟩
  invFun x := ⟨⟨x.1, x.2.1⟩, x.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl

theorem card_binaryAccumulatorSliceEvent (N w h : ℕ) :
    (binaryAccumulatorSliceEvent N w h).card =
      binaryAccumulatorCount N w h := by
  classical
  unfold binaryAccumulatorSliceEvent binaryAccumulatorCount
  rw [← Fintype.card_of_subtype
      (p := fun x : BinaryWeightSlice N w =>
        hammingNorm (accumLinearEquiv (ZMod 2) N x.1) = h)
      _ (by simp),
    ← Fintype.card_of_subtype
      (p := fun x : BinaryWord N =>
        hammingNorm x = w ∧
          hammingNorm (accumLinearEquiv (ZMod 2) N x) = h)
      _ (by simp)]
  exact Fintype.card_congr (accumulatorSliceEventEquiv N w h)

/-- B3 as an equality between the actual uniform finite event and the exact
rational transition count. -/
theorem binaryAccumulatorConditionalProbability_eq (N w h : ℕ) :
    binaryAccumulatorConditionalProbability N w h =
      binaryAccumulatorProbability N w h := by
  unfold binaryAccumulatorConditionalProbability uniformEventProbability
  change ((binaryAccumulatorSliceEvent N w h).card : ℚ) /
      Fintype.card (BinaryWeightSlice N w) = binaryAccumulatorProbability N w h
  rw [card_binaryAccumulatorSliceEvent, card_binaryWeightSlice,
    binaryAccumulatorProbability]

/-- B3 with B2's closed form substituted. -/
theorem binaryAccumulatorConditionalProbability_eq_formula {N w h : ℕ}
    (hB2 : binaryAccumulatorCount N w h = binaryAccumulatorFormula N w h) :
    binaryAccumulatorConditionalProbability N w h =
      (binaryAccumulatorFormula N w h : ℚ) / Nat.choose N w := by
  rw [binaryAccumulatorConditionalProbability_eq]
  exact binaryAccumulatorProbability_eq_formula hB2

theorem uniform_permuted_accumulator_weight_probability {N : ℕ}
    (x : BinaryWord N) (h : ℕ) :
    uniformEventProbability (Omega := Equiv.Perm (Fin N)) (fun sigma =>
      hammingNorm (accumLinearEquiv (ZMod 2) N
        (permLinearEquiv (ZMod 2) sigma x)) = h) =
      binaryAccumulatorProbability N (hammingNorm x) h := by
  calc
    _ = uniformEventProbability
        (Omega := BinaryWeightSlice N (hammingNorm x)) (fun y =>
          hammingNorm (accumLinearEquiv (ZMod 2) N y.1) = h) :=
      uniform_permutation_event_probability x (fun y =>
        hammingNorm (accumLinearEquiv (ZMod 2) N y) = h)
    _ = _ := binaryAccumulatorConditionalProbability_eq N (hammingNorm x) h

def binaryFirstStage {N : ℕ} (sigma : Equiv.Perm (Fin N))
    (x : BinaryWord N) : BinaryWord N :=
  accumLinearEquiv (ZMod 2) N (permLinearEquiv (ZMod 2) sigma x)

def binarySecondStage {N : ℕ} (sigma : Equiv.Perm (Fin N))
    (x : BinaryWord N) : BinaryWord N :=
  accumLinearEquiv (ZMod 2) N (permLinearEquiv (ZMod 2) sigma x)

abbrev WeightIndex (N : ℕ) := Fin (N + 1)

def wordWeightIndex {N : ℕ} (x : BinaryWord N) : WeightIndex N :=
  ⟨hammingNorm x, by
    have hx : hammingNorm x ≤ N := by
      simpa using (hammingNorm_le_card_fintype (x := x))
    omega⟩

/-- The second-stage low-weight tail, represented directly as an exact finite
event.  A later theorem expands it as the B3 sum from `1` through `L`. -/
noncomputable def binaryAccumulatorTail (N h L : ℕ) : ℚ :=
  uniformEventProbability (Omega := BinaryWeightSlice N h) fun x =>
    hammingNorm (accumLinearEquiv (ZMod 2) N x.1) ≤ L

theorem uniform_permuted_accumulator_lowWeight_probability {N : ℕ}
    (x : BinaryWord N) (L : ℕ) :
    uniformEventProbability (Omega := Equiv.Perm (Fin N)) (fun sigma =>
      hammingNorm (binarySecondStage sigma x) ≤ L) =
      binaryAccumulatorTail N (hammingNorm x) L := by
  exact uniform_permutation_event_probability x (fun y =>
    hammingNorm (accumLinearEquiv (ZMod 2) N y) ≤ L)

theorem uniform_firstStage_weightIndex_probability {N : ℕ}
    (x : BinaryWord N) (h : WeightIndex N) :
    uniformEventProbability (Omega := Equiv.Perm (Fin N)) (fun sigma =>
      wordWeightIndex (binaryFirstStage sigma x) = h) =
      binaryAccumulatorProbability N (hammingNorm x) h.val := by
  calc
    _ = uniformEventProbability (Omega := Equiv.Perm (Fin N)) (fun sigma =>
        hammingNorm (binaryFirstStage sigma x) = h.val) := by
      apply uniformEventProbability_congr
      intro sigma
      simp [wordWeightIndex, Fin.ext_iff]
    _ = _ := uniform_permuted_accumulator_weight_probability x h.val

theorem binaryAccumulator_weightIndex_probability {N h : ℕ}
    (j : WeightIndex N) :
    uniformEventProbability (Omega := BinaryWeightSlice N h) (fun x =>
      wordWeightIndex (accumLinearEquiv (ZMod 2) N x.1) = j) =
      binaryAccumulatorProbability N h j.val := by
  calc
    _ = uniformEventProbability (Omega := BinaryWeightSlice N h) (fun x =>
        hammingNorm (accumLinearEquiv (ZMod 2) N x.1) = j.val) := by
      apply uniformEventProbability_congr
      intro x
      simp [wordWeightIndex, Fin.ext_iff]
    _ = _ := binaryAccumulatorConditionalProbability_eq N h j.val

theorem binaryAccumulatorTail_eq_bounded_sum {N h L : ℕ} (hh : h ≤ N) :
    binaryAccumulatorTail N h L =
      ∑ j : WeightIndex N,
        if j.val ≤ L then binaryAccumulatorProbability N h j.val else 0 := by
  classical
  letI : Nonempty (BinaryWeightSlice N h) := Fintype.card_pos_iff.mp <| by
    rw [card_binaryWeightSlice]
    exact Nat.choose_pos hh
  calc
    binaryAccumulatorTail N h L =
        uniformExpectationRat (Omega := BinaryWeightSlice N h) (fun x =>
          if hammingNorm (accumLinearEquiv (ZMod 2) N x.1) ≤ L
            then (1 : ℚ) else 0) :=
      uniformEventProbability_eq_expectationRat_indicator _
    _ = uniformExpectationRat (Omega := BinaryWeightSlice N h) (fun x =>
        if (wordWeightIndex (accumLinearEquiv (ZMod 2) N x.1)).val ≤ L
          then (1 : ℚ) else 0) := by
      apply uniformExpectationRat_congr
      intro x
      rfl
    _ = ∑ j : WeightIndex N,
        uniformEventProbability (fun x : BinaryWeightSlice N h =>
          wordWeightIndex (accumLinearEquiv (ZMod 2) N x.1) = j) *
            (if j.val ≤ L then (1 : ℚ) else 0) :=
      uniformExpectationRat_groupBy
        (fun x : BinaryWeightSlice N h =>
          wordWeightIndex (accumLinearEquiv (ZMod 2) N x.1))
        (fun j : WeightIndex N => if j.val ≤ L then (1 : ℚ) else 0)
    _ = _ := by
      apply Finset.sum_congr rfl
      intro j _
      rw [binaryAccumulator_weightIndex_probability]
      by_cases hj : j.val ≤ L <;> simp [hj]

noncomputable def binaryAccumulatorTransitionTail (N h L : ℕ) : ℚ :=
  ∑ j : WeightIndex N,
    if 1 ≤ j.val ∧ j.val ≤ L
      then binaryAccumulatorProbability N h j.val else 0

theorem binaryAccumulatorTail_eq_transitionTail {N h L : ℕ}
    (hhpos : 0 < h) (hhle : h ≤ N) :
    binaryAccumulatorTail N h L = binaryAccumulatorTransitionTail N h L := by
  rw [binaryAccumulatorTail_eq_bounded_sum hhle, binaryAccumulatorTransitionTail]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj0 : j.val = 0
  · simp [hj0, binaryAccumulatorProbability,
      binaryAccumulatorCount_output_zero hhpos]
  · have hjpos : 1 ≤ j.val := Nat.one_le_iff_ne_zero.mpr hj0
    by_cases hjL : j.val ≤ L <;> simp [hj0, hjpos, hjL]

abbrev BinaryEnsemble (N : ℕ) :=
  Equiv.Perm (Fin N) × Equiv.Perm (Fin N)

/-- The two-interleaver/two-accumulator binary transform. -/
def binaryBAALinearEquiv {N : ℕ} (omega : BinaryEnsemble N) :
    BinaryWord N ≃ₗ[ZMod 2] BinaryWord N :=
  permLinearEquiv (ZMod 2) omega.1 ≪≫ₗ
    accumLinearEquiv (ZMod 2) N ≪≫ₗ
      permLinearEquiv (ZMod 2) omega.2 ≪≫ₗ
        accumLinearEquiv (ZMod 2) N

theorem binaryBAALinearEquiv_apply {N : ℕ} (omega : BinaryEnsemble N)
    (x : BinaryWord N) :
    binaryBAALinearEquiv omega x =
      binarySecondStage omega.2 (binaryFirstStage omega.1 x) :=
  rfl

noncomputable def binaryTwoStageLowWeightProbability {N : ℕ}
    (x : BinaryWord N) (L : ℕ) : ℚ :=
  uniformEventProbability (Omega := BinaryEnsemble N) fun omega =>
    hammingNorm (binaryBAALinearEquiv omega x) ≤ L

/-- The two-stage law for one fixed source word, before deleting the impossible
zero intermediate weight for a nonzero input. -/
theorem binaryTwoStageLowWeightProbability_eq {N : ℕ}
    (x : BinaryWord N) (L : ℕ) :
    binaryTwoStageLowWeightProbability x L =
      ∑ h : WeightIndex N,
        binaryAccumulatorProbability N (hammingNorm x) h.val *
          binaryAccumulatorTail N h.val L := by
  rw [binaryTwoStageLowWeightProbability, uniformEventProbability_prod]
  have hinner : ∀ sigma : Equiv.Perm (Fin N),
      uniformEventProbability (Omega := Equiv.Perm (Fin N)) (fun tau =>
        hammingNorm (binaryBAALinearEquiv (sigma, tau) x) ≤ L) =
      binaryAccumulatorTail N
        (wordWeightIndex (binaryFirstStage sigma x)).val L := by
    intro sigma
    change uniformEventProbability (Omega := Equiv.Perm (Fin N)) (fun tau =>
        hammingNorm (binarySecondStage tau (binaryFirstStage sigma x)) ≤ L) =
      binaryAccumulatorTail N (hammingNorm (binaryFirstStage sigma x)) L
    exact uniform_permuted_accumulator_lowWeight_probability
      (binaryFirstStage sigma x) L
  rw [uniformExpectationRat_congr hinner]
  rw [uniformExpectationRat_groupBy
    (fun sigma : Equiv.Perm (Fin N) => wordWeightIndex (binaryFirstStage sigma x))
    (fun h : WeightIndex N => binaryAccumulatorTail N h.val L)]
  apply Finset.sum_congr rfl
  intro h _
  rw [uniform_firstStage_weightIndex_probability]

def binaryEnsembleCode {N : ℕ} (C : Code (ZMod 2) N)
    (omega : BinaryEnsemble N) : Code (ZMod 2) N :=
  C.map (binaryBAALinearEquiv omega).toLinearMap

noncomputable def binaryBadWordCount {N : ℕ} (C : Code (ZMod 2) N) (L : ℕ)
    (omega : BinaryEnsemble N) : ℕ :=
  (Finset.univ.filter fun x : C =>
    x ≠ 0 ∧ hammingNorm (binaryBAALinearEquiv omega x.1) ≤ L).card

noncomputable def binaryExpectedBadWords {N : ℕ} (C : Code (ZMod 2) N)
    (L : ℕ) : ℚ :=
  uniformExpectationNat (Omega := BinaryEnsemble N) (binaryBadWordCount C L)

theorem binaryExpectedBadWords_eq_sum {N : ℕ} (C : Code (ZMod 2) N)
    (L : ℕ) :
    binaryExpectedBadWords C L =
      ∑ x : C, if x ≠ 0 then binaryTwoStageLowWeightProbability x.1 L else 0 := by
  classical
  rw [binaryExpectedBadWords, uniformExpectationNat_eq_rat]
  calc
    _ = uniformExpectationRat (Omega := BinaryEnsemble N) (fun omega =>
        ∑ x : C, if x ≠ 0 ∧
          hammingNorm (binaryBAALinearEquiv omega x.1) ≤ L then (1 : ℚ) else 0) := by
      apply uniformExpectationRat_congr
      intro omega
      rw [binaryBadWordCount]
      have hcard : (Finset.univ.filter fun x : C =>
          x ≠ 0 ∧ hammingNorm (binaryBAALinearEquiv omega x.1) ≤ L).card =
          ∑ x : C, if x ≠ 0 ∧
            hammingNorm (binaryBAALinearEquiv omega x.1) ≤ L then 1 else 0 := by
        convert (Finset.card_eq_sum_ite
          (s := Finset.univ.filter fun x : C =>
            x ≠ 0 ∧ hammingNorm (binaryBAALinearEquiv omega x.1) ≤ L)
          (t := Finset.univ) (Finset.subset_univ _)) using 1 <;> simp
      rw [hcard, Nat.cast_sum]
      simp
    _ = ∑ x : C, uniformExpectationRat (Omega := BinaryEnsemble N) (fun omega =>
        if x ≠ 0 ∧ hammingNorm (binaryBAALinearEquiv omega x.1) ≤ L
          then (1 : ℚ) else 0) := uniformExpectationRat_sum _
    _ = _ := by
      apply Finset.sum_congr rfl
      intro x _
      by_cases hx : x = 0
      · simp [hx, uniformExpectationRat]
      · rw [← uniformEventProbability_eq_expectationRat_indicator, if_pos hx]
        unfold binaryTwoStageLowWeightProbability
        apply uniformEventProbability_congr
        intro omega
        exact and_iff_right hx

noncomputable def binaryFirstMomentFormula {N : ℕ} (C : Code (ZMod 2) N)
    (L : ℕ) : ℚ :=
  ∑ w : WeightIndex N,
    if w.val = 0 then 0 else
      ((weightEnumerator C).coeff w.val : ℚ) *
        ∑ h : WeightIndex N,
          binaryAccumulatorProbability N w.val h.val *
            binaryAccumulatorTail N h.val L

theorem card_code_weightIndex_fiber {N : ℕ} (C : Code (ZMod 2) N)
    (w : WeightIndex N) :
    (Finset.univ.filter fun x : C => wordWeightIndex x.1 = w).card =
      (weightEnumerator C).coeff w.val := by
  classical
  rw [coeff_weightEnumerator]
  congr 1
  ext x
  simp [wordWeightIndex, Fin.ext_iff]

/-- B4 in a bounded-index form.  The zero input-weight summand is explicitly
zero; expanding `binaryAccumulatorTail` yields the paper's nested `1..N` and
`1..L` sums. -/
theorem binaryExpectedBadWords_eq_firstMomentFormula {N : ℕ}
    (C : Code (ZMod 2) N) (L : ℕ) :
    binaryExpectedBadWords C L = binaryFirstMomentFormula C L := by
  classical
  rw [binaryExpectedBadWords_eq_sum, binaryFirstMomentFormula]
  let kernel : WeightIndex N → ℚ := fun w =>
    ∑ h : WeightIndex N,
      binaryAccumulatorProbability N w.val h.val *
        binaryAccumulatorTail N h.val L
  calc
    (∑ x : C, if x ≠ 0 then binaryTwoStageLowWeightProbability x.1 L else 0) =
        ∑ x : C, if (wordWeightIndex x.1).val = 0 then 0
          else kernel (wordWeightIndex x.1) := by
      apply Finset.sum_congr rfl
      intro x _
      by_cases hx : x = 0
      · simp [hx, wordWeightIndex]
      · have hxcoe : (x.1 : BinaryWord N) ≠ 0 := by
          intro hzero
          exact hx (Subtype.ext hzero)
        have hw : hammingNorm (x.1 : BinaryWord N) ≠ 0 :=
          hammingNorm_ne_zero_iff.mpr hxcoe
        rw [if_pos hx, if_neg]
        · exact binaryTwoStageLowWeightProbability_eq x.1 L
        · exact hw
    _ = ∑ w : WeightIndex N,
        ∑ x ∈ (Finset.univ.filter fun x : C => wordWeightIndex x.1 = w),
          (if w.val = 0 then 0 else kernel w) := by
      symm
      exact Finset.sum_fiberwise_of_maps_to'
        (s := Finset.univ) (t := Finset.univ)
        (g := fun x : C => wordWeightIndex x.1) (by simp)
        (fun w : WeightIndex N => if w.val = 0 then 0 else kernel w)
    _ = ∑ w : WeightIndex N,
        ((Finset.univ.filter fun x : C => wordWeightIndex x.1 = w).card : ℚ) *
          (if w.val = 0 then 0 else kernel w) := by
      apply Finset.sum_congr rfl
      intro w _
      simp
    _ = _ := by
      apply Finset.sum_congr rfl
      intro w _
      rw [card_code_weightIndex_fiber]
      by_cases hw : w.val = 0
      · simp [hw]
      · simp [hw, kernel, mul_comm]

noncomputable def binaryPaperFirstMomentFormula {N : ℕ}
    (C : Code (ZMod 2) N) (L : ℕ) : ℚ :=
  ∑ w : WeightIndex N,
    if w.val = 0 then 0 else
      ((weightEnumerator C).coeff w.val : ℚ) *
        ∑ h : WeightIndex N,
          if h.val = 0 then 0 else
            binaryAccumulatorProbability N w.val h.val *
              binaryAccumulatorTransitionTail N h.val L

theorem binaryFirstMomentFormula_eq_paper {N : ℕ}
    (C : Code (ZMod 2) N) (L : ℕ) :
    binaryFirstMomentFormula C L = binaryPaperFirstMomentFormula C L := by
  classical
  rw [binaryFirstMomentFormula, binaryPaperFirstMomentFormula]
  apply Finset.sum_congr rfl
  intro w _
  by_cases hw0 : w.val = 0
  · simp [hw0]
  · have hwpos : 0 < w.val := Nat.pos_of_ne_zero hw0
    simp only [hw0, if_false]
    congr 1
    apply Finset.sum_congr rfl
    intro h _
    by_cases hh0 : h.val = 0
    · simp [hh0, binaryAccumulatorProbability,
        binaryAccumulatorCount_output_zero hwpos]
    · have hhpos : 0 < h.val := Nat.pos_of_ne_zero hh0
      have hhle : h.val ≤ N := by omega
      rw [if_neg hh0, binaryAccumulatorTail_eq_transitionTail hhpos hhle]

/-- B4 exactly: the expected number of nonzero ensemble codewords of weight at
most `L` is the manuscript's two-transition sum. -/
theorem binaryExpectedBadWords_eq_paperFirstMoment {N : ℕ}
    (C : Code (ZMod 2) N) (L : ℕ) :
    binaryExpectedBadWords C L = binaryPaperFirstMomentFormula C L :=
  (binaryExpectedBadWords_eq_firstMomentFormula C L).trans
    (binaryFirstMomentFormula_eq_paper C L)

private theorem binaryEnsembleCode_nontrivial {N : ℕ} (C : Code (ZMod 2) N)
    [Nontrivial C] (omega : BinaryEnsemble N) :
    Nontrivial (binaryEnsembleCode C omega) := by
  obtain ⟨x, hx⟩ : ∃ x : C, x ≠ 0 := exists_ne 0
  let y : binaryEnsembleCode C omega :=
    ⟨binaryBAALinearEquiv omega x.1, ⟨x.1, x.2, rfl⟩⟩
  refine ⟨⟨y, 0, ?_⟩⟩
  intro hy
  apply hx
  apply Subtype.ext
  apply (binaryBAALinearEquiv omega).injective
  calc
    binaryBAALinearEquiv omega x.1 = 0 := by
      simpa [y] using congrArg Subtype.val hy
    _ = binaryBAALinearEquiv omega 0 :=
      (binaryBAALinearEquiv omega).map_zero.symm

theorem binaryBadWordCount_pos_iff_minimumDistance_le {N : ℕ}
    (C : Code (ZMod 2) N) [Nontrivial C] (L : ℕ) (omega : BinaryEnsemble N) :
    0 < binaryBadWordCount C L omega ↔
      minimumDistance (binaryEnsembleCode C omega) ≤ L := by
  classical
  letI := binaryEnsembleCode_nontrivial C omega
  constructor
  · intro hbad
    rw [binaryBadWordCount, Finset.card_pos] at hbad
    obtain ⟨x, hxmem⟩ := hbad
    have hx := (Finset.mem_filter.mp hxmem).2
    let y : binaryEnsembleCode C omega :=
      ⟨binaryBAALinearEquiv omega x.1, ⟨x.1, x.2, rfl⟩⟩
    have hy : y ≠ 0 := by
      intro hyzero
      apply hx.1
      apply Subtype.ext
      apply (binaryBAALinearEquiv omega).injective
      calc
        binaryBAALinearEquiv omega x.1 = 0 := by
          simpa [y] using congrArg Subtype.val hyzero
        _ = binaryBAALinearEquiv omega 0 :=
          (binaryBAALinearEquiv omega).map_zero.symm
    exact (minimumDistance_le_weight (binaryEnsembleCode C omega) y hy).trans hx.2
  · intro hdist
    obtain ⟨y, hy, hyweight⟩ :=
      minimumDistance_attained (binaryEnsembleCode C omega)
    obtain ⟨x, hxC, hxy⟩ := y.2
    let xs : C := ⟨x, hxC⟩
    have hxs : xs ≠ 0 := by
      intro hxzero
      apply hy
      apply Subtype.ext
      rw [← hxy]
      exact (congrArg (fun z => binaryBAALinearEquiv omega z.1) hxzero).trans
        (binaryBAALinearEquiv omega).map_zero
    rw [binaryBadWordCount, Finset.card_pos]
    refine ⟨xs, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hxs, ?_⟩⟩
    change hammingNorm (binaryBAALinearEquiv omega x) ≤ L
    calc
      _ = hammingNorm y.1 := congrArg hammingNorm hxy
      _ = minimumDistance (binaryEnsembleCode C omega) := hyweight
      _ ≤ L := hdist

/-- B5 in exact finite form.  B4 identifies the expectation on the right with
the paper's displayed transition sum. -/
theorem binary_minimumDistance_probability_le_expected {N : ℕ}
    (C : Code (ZMod 2) N) [Nontrivial C] (L : ℕ) :
    uniformEventProbability (Omega := BinaryEnsemble N) (fun omega =>
      minimumDistance (binaryEnsembleCode C omega) ≤ L) ≤
        binaryExpectedBadWords C L := by
  calc
    _ = uniformEventProbability (Omega := BinaryEnsemble N) (fun omega =>
        0 < binaryBadWordCount C L omega) := by
      have hfin :
          (Finset.univ.filter fun omega : BinaryEnsemble N =>
            minimumDistance (binaryEnsembleCode C omega) ≤ L) =
          (Finset.univ.filter fun omega : BinaryEnsemble N =>
            0 < binaryBadWordCount C L omega) := by
        ext omega
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact (binaryBadWordCount_pos_iff_minimumDistance_le C L omega).symm
      unfold uniformEventProbability
      rw [hfin]
    _ ≤ binaryExpectedBadWords C L := by
      rw [binaryExpectedBadWords]
      exact uniform_probability_positive_le_expectation (binaryBadWordCount C L)

end BAA
