import BAA.Construction
import BAA.MasterBound
import BAA.NonbinaryPlacementExtract
import BAA.Shortening
import BAA.SupportPropagation
import Mathlib.Data.Fintype.Perm

/-!
# The concrete finite nonbinary BAA ensemble

This module instantiates the sample space and trace-event cover used by the
master union bound (3).  Its four independent coordinates are the two
permutations and the two unit-valued diagonal maps.
-/

namespace BAA

/-- Two independent monomial maps for a length-`N` BAA construction. -/
structure NonbinaryBAAEnsemble (F : Type*) [CommRing F] (N : ℕ) where
  firstPermutation : Equiv.Perm (Fin N)
  firstDiagonal : Fin N → Fˣ
  secondPermutation : Equiv.Perm (Fin N)
  secondDiagonal : Fin N → Fˣ

noncomputable instance {F : Type*} [CommRing F] [Fintype F] [DecidableEq F]
    {N : ℕ} : Fintype (NonbinaryBAAEnsemble F N) := by
  letI : Fintype Fˣ := Fintype.ofFinite Fˣ
  exact Fintype.ofEquiv
    (Equiv.Perm (Fin N) × (Fin N → Fˣ) ×
      Equiv.Perm (Fin N) × (Fin N → Fˣ))
    { toFun omega := ⟨omega.1, omega.2.1, omega.2.2.1, omega.2.2.2⟩
      invFun omega := (omega.firstPermutation, omega.firstDiagonal,
        omega.secondPermutation, omega.secondDiagonal)
      left_inv omega := by rcases omega with ⟨p₁, d₁, p₂, d₂⟩; rfl
      right_inv omega := by rcases omega with ⟨p₁, d₁, p₂, d₂⟩; rfl }

instance {F : Type*} [CommRing F] [Fintype F] [DecidableEq F] {N : ℕ} :
    Nonempty (NonbinaryBAAEnsemble F N) :=
  ⟨⟨Equiv.refl _, fun _ => 1, Equiv.refl _, fun _ => 1⟩⟩

noncomputable local instance nonbinaryClassicalDecidablePred
    {α : Type*} (p : α → Prop) :
    DecidablePred p := Classical.decPred p

def nonbinaryFirstInput {F : Type*} [CommRing F] {N : ℕ}
    (omega : NonbinaryBAAEnsemble F N) (x : Word F N) : Word F N :=
  monomialLinearEquiv F omega.firstPermutation omega.firstDiagonal x

def nonbinaryFirstOutput {F : Type*} [CommRing F] {N : ℕ}
    (omega : NonbinaryBAAEnsemble F N) (x : Word F N) : Word F N :=
  accumLinearEquiv F N (nonbinaryFirstInput omega x)

def nonbinarySecondInput {F : Type*} [CommRing F] {N : ℕ}
    (omega : NonbinaryBAAEnsemble F N) (x : Word F N) : Word F N :=
  monomialLinearEquiv F omega.secondPermutation omega.secondDiagonal
    (nonbinaryFirstOutput omega x)

def nonbinaryFinalOutput {F : Type*} [CommRing F] {N : ℕ}
    (omega : NonbinaryBAAEnsemble F N) (x : Word F N) : Word F N :=
  accumLinearEquiv F N (nonbinarySecondInput omega x)

theorem nonbinaryFinalOutput_eq_baaTransform
    {F : Type*} [CommRing F] {N : ℕ}
    (omega : NonbinaryBAAEnsemble F N) (x : Word F N) :
    nonbinaryFinalOutput omega x =
      baaTransform F omega.firstPermutation omega.secondPermutation
        omega.firstDiagonal omega.secondDiagonal x := by
  rfl

/-- Number of active input coordinates after which this accumulator is in the
zero state. -/
def accumulatorReturnCount {F : Type*} [CommRing F] [DecidableEq F] {N : ℕ}
    (x : Word F N) : ℕ :=
  ((support x).filter fun i => accumLinearEquiv F N x i = 0).card

/-- The coordinate-level return count used by the concrete ensemble equals
the cardinality of the canonical active-symbol zero-return trace. -/
theorem accumulatorReturnCount_eq_activeTraceCard
    {F : Type*} [Field F] [DecidableEq F] {N : ℕ}
    (x : Word F N) (hx : x ≠ 0) :
    accumulatorReturnCount x =
      (accumulatorZeroReturnTrace (orderedSupportValues x rfl)).card := by
  obtain ⟨h, a, hh, p, hhweight, hareturn, htrace, hword, hsupport⟩ :=
    exists_placementOfWord x hx
  let S := support x
  let s := S.card
  let f : Fin s ↪o Fin N := S.orderEmbOfFin rfl
  let u : Fin s → F := orderedSupportValues x rfl
  have hposition : scheduleActivePosition p hh = f := by
    exact Finset.orderEmbOfFin_unique rfl
      (fun i => by
        have hm := scheduleActivePosition_mem_support p hh i
        simpa only [hsupport] using hm)
      (scheduleActivePosition_strictMono p hh)
  have hstate : ∀ i : Fin s,
      accumLinearEquiv F N x (f i) = accumulatorActiveState u i := by
    intro i
    rw [← hword, ← hposition]
    apply accum_scheduleInputWord_on_interval p hh u i
    · exact le_rfl
    · have hpos := scheduleActiveInterval_pos p i
      change scheduleActivePositionNat p i <
        scheduleActivePositionNat p i + scheduleActiveInterval p i
      omega
  have hfilter :
      (support x).filter (fun k => accumLinearEquiv F N x k = 0) =
        (Finset.univ.filter
          (fun i : Fin s => accumulatorActiveState u i = 0)).image f := by
    ext k
    constructor
    · intro hk
      rw [Finset.mem_filter] at hk
      have hkS : k ∈ S := hk.1
      rw [← S.image_orderEmbOfFin_univ rfl] at hkS
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hkS
      apply Finset.mem_image.mpr
      refine ⟨i, ?_, rfl⟩
      rw [hstate i] at hk
      simpa using hk.2
    · intro hk
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hk
      rw [Finset.mem_filter] at hi ⊢
      refine ⟨Finset.orderEmbOfFin_mem S rfl i, ?_⟩
      rw [hstate i]
      exact hi.2
  unfold accumulatorReturnCount accumulatorZeroReturnTrace
  rw [hfilter, Finset.card_image_of_injective _ f.injective]

/-- The return-count bridge without a nonzero hypothesis.  The zero word has
empty support and hence an empty active trace. -/
theorem accumulatorReturnCount_eq_activeTraceCard_all
    {F : Type*} [Field F] [DecidableEq F] {N : ℕ}
    (x : Word F N) :
    accumulatorReturnCount x =
      (accumulatorZeroReturnTrace (orderedSupportValues x rfl)).card := by
  by_cases hx : x = 0
  · subst x
    have hcard : (support (0 : Word F N)).card = 0 := by
      simp [support]
    have hempty :
        accumulatorZeroReturnTrace
            (orderedSupportValues (0 : Word F N) rfl) = ∅ := by
      ext i
      constructor
      · intro _
        exact Fin.elim0 (Fin.cast hcard i)
      · intro hi
        simp at hi
    rw [hempty, Finset.card_empty]
    simp [accumulatorReturnCount, support]
  · exact accumulatorReturnCount_eq_activeTraceCard x hx

/-- Return-count bridge at an explicitly chosen support-cardinality index. -/
theorem accumulatorReturnCount_eq_activeTraceCard_of_card
    {F : Type*} [Field F] [DecidableEq F] {N s : ℕ}
    (x : Word F N) (hcard : (support x).card = s) :
    accumulatorReturnCount x =
      (accumulatorZeroReturnTrace (orderedSupportValues x hcard)).card := by
  subst s
  exact accumulatorReturnCount_eq_activeTraceCard_all x

theorem accumulatorReturnCount_le_length
    {F : Type*} [CommRing F] [DecidableEq F] {N : ℕ} (x : Word F N) :
    accumulatorReturnCount x ≤ N := by
  unfold accumulatorReturnCount
  exact (Finset.card_le_card (Finset.filter_subset _ _)).trans
    (by simpa using Finset.card_le_univ (support x))

theorem accumulatorReturnCount_lt_support_card
    {F : Type*} [Field F] [DecidableEq F] {N : ℕ} (x : Word F N)
    (hx : x ≠ 0) :
    accumulatorReturnCount x < (support x).card := by
  let first := (support x).min' (support_nonempty_of_ne_zero x hx)
  have hfirstMem : first ∈ support x :=
    Finset.min'_mem _ (support_nonempty_of_ne_zero x hx)
  have hfirstNonzero : x first ≠ 0 := (mem_support x first).mp hfirstMem
  have haccum : accumLinearEquiv F N x first = x first := by
    rw [accum_apply_eq_sum_Iic]
    apply Finset.sum_eq_single first
    · intro j hj hne
      by_contra hjzero
      have hjmem : j ∈ support x := (mem_support x j).mpr hjzero
      have hmin : first ≤ j := Finset.min'_le _ _ hjmem
      have hjle : j ≤ first := Finset.mem_Iic.mp hj
      exact hne (le_antisymm hjle hmin)
    · intro hnot
      exact (hnot (Finset.mem_Iic.mpr le_rfl)).elim
  have hfirstNotReturn : first ∉
      (support x).filter (fun i => accumLinearEquiv F N x i = 0) := by
    simp only [Finset.mem_filter, hfirstMem, true_and, haccum]
    exact hfirstNonzero
  have hproper :
      (support x).filter (fun i => accumLinearEquiv F N x i = 0) ⊂
        support x := by
    apply (Finset.ssubset_iff_of_subset (Finset.filter_subset _ _)).mpr
    exact ⟨first, hfirstMem, hfirstNotReturn⟩
  exact Finset.card_lt_card hproper

theorem hammingNorm_monomialLinearEquiv
    {F : Type*} [Field F] [DecidableEq F] {N : ℕ}
    (sigma : Equiv.Perm (Fin N)) (d : Fin N → Fˣ) (x : Word F N) :
    hammingNorm (monomialLinearEquiv F sigma d x) = hammingNorm x := by
  rw [← card_support_eq_hammingNorm, support_monomialLinearEquiv,
    permuteCoordinateSet, Finset.card_image_of_injective _ sigma.injective,
    card_support_eq_hammingNorm]

theorem hammingNorm_pos_of_ne_zero
    {F : Type*} [Field F] [DecidableEq F] {N : ℕ} {x : Word F N}
    (hx : x ≠ 0) : 0 < hammingNorm x := by
  rw [← card_support_eq_hammingNorm]
  exact Finset.card_pos.mpr (support_nonempty_of_ne_zero x hx)

/-- The low-weight bad-code event for the concrete finite ensemble. -/
def nonbinaryBAABadEvent
    {F : Type*} [Field F] [Fintype F] [DecidableEq F]
    {m B : ℕ} (C0 : Code F m) (L : ℕ)
    (omega : NonbinaryBAAEnsemble F (B * m)) : Prop :=
  ∃ x : outerCode C0 B,
    x ≠ 0 ∧ hammingNorm (nonbinaryFinalOutput omega x.1) ≤ L

/-- Concrete trace event indexed by the five finite statistics in (3). -/
def nonbinaryBAATraceEvent
    {F : Type*} [Field F] [Fintype F] [DecidableEq F]
    {m B L : ℕ} (C0 : Code F m) (i : MasterTraceIndex (B * m) L)
    (omega : NonbinaryBAAEnsemble F (B * m)) : Prop :=
  ∃ x : outerCode C0 B,
    x ≠ 0 ∧
      support x.1 = i.support ∧
      accumulatorReturnCount (nonbinaryFirstInput omega x.1) =
        i.firstReturns.val ∧
      hammingNorm (nonbinaryFirstOutput omega x.1) =
        i.intermediateWeight.val ∧
      accumulatorReturnCount (nonbinarySecondInput omega x.1) =
        i.secondReturns.val ∧
      hammingNorm (nonbinaryFinalOutput omega x.1) = i.finalWeight.val

/-- Every concrete bad-code outcome has a trace index, and every indexed trace
event is bad.  Thus the abstract cover assumption in the master union theorem
is discharged for the actual BAA construction. -/
theorem nonbinaryBAABadEvent_iff_exists_trace
    {F : Type*} [Field F] [Fintype F] [DecidableEq F]
    {m B L : ℕ} (C0 : Code F m)
    (omega : NonbinaryBAAEnsemble F (B * m)) :
    nonbinaryBAABadEvent C0 L omega ↔
      ∃ i : MasterTraceIndex (B * m) L,
        nonbinaryBAATraceEvent C0 i omega := by
  constructor
  · rintro ⟨x, hx0, hlow⟩
    let firstInput := nonbinaryFirstInput omega x.1
    let firstOutput := nonbinaryFirstOutput omega x.1
    let secondInput := nonbinarySecondInput omega x.1
    let finalOutput := nonbinaryFinalOutput omega x.1
    have hlow' : hammingNorm finalOutput ≤ L := by
      simpa [finalOutput] using hlow
    let i : MasterTraceIndex (B * m) L :=
      { support := support x.1
        firstReturns := ⟨accumulatorReturnCount firstInput,
          by have := accumulatorReturnCount_le_length firstInput; omega⟩
        intermediateWeight := ⟨hammingNorm firstOutput, by
          have hw := hammingNorm_le_card_fintype (x := firstOutput)
          simp only [Fintype.card_fin] at hw
          omega⟩
        secondReturns := ⟨accumulatorReturnCount secondInput,
          by have := accumulatorReturnCount_le_length secondInput; omega⟩
        finalWeight := ⟨hammingNorm finalOutput, by omega⟩ }
    refine ⟨i, x, hx0, ?_⟩
    simp [i, firstInput, firstOutput, secondInput, finalOutput]
  · rintro ⟨i, x, hx0, -, -, -, -, hfinal⟩
    refine ⟨x, hx0, ?_⟩
    rw [hfinal]
    omega

/-- The shortening rank appearing in the concrete instance of (3). -/
noncomputable def outerShorteningRank
    {F : Type*} [Field F] {m B : ℕ} (C0 : Code F m)
    (S : Finset (Fin (B * m))) : ℕ :=
  shorteningDim (outerCode C0 B) S

/-- Every inhabited concrete trace event automatically satisfies the
admissibility hypotheses used to expand its two equation-(1) kernels. -/
theorem nonbinaryBAATraceEvent_feasible
    {F : Type*} [Field F] [Fintype F] [DecidableEq F]
    {m B L : ℕ} {C0 : Code F m} {i : MasterTraceIndex (B * m) L}
    {omega : NonbinaryBAAEnsemble F (B * m)}
    (hevent : nonbinaryBAATraceEvent C0 i omega) :
    0 < i.support.card ∧
      i.firstReturns.val < i.support.card ∧
      0 < i.intermediateWeight.val ∧
      i.intermediateWeight.val ≤ B * m ∧
      i.secondReturns.val < i.intermediateWeight.val ∧
      0 < i.finalWeight.val ∧ i.finalWeight.val ≤ B * m := by
  obtain ⟨x, hx0, hsupport, hreturn₁, hweight₁, hreturn₂, hweight₂⟩ := hevent
  have hxword : x.1 ≠ 0 := by
    intro hx
    apply hx0
    exact Subtype.ext hx
  have hinput₁ : nonbinaryFirstInput omega x.1 ≠ 0 := by
    simpa [nonbinaryFirstInput] using
      (monomialLinearEquiv F omega.firstPermutation
        omega.firstDiagonal).injective.ne hxword
  have houtput₁ : nonbinaryFirstOutput omega x.1 ≠ 0 := by
    simpa [nonbinaryFirstOutput] using
      (accumLinearEquiv F (B * m)).injective.ne hinput₁
  have hinput₂ : nonbinarySecondInput omega x.1 ≠ 0 := by
    simpa [nonbinarySecondInput] using
      (monomialLinearEquiv F omega.secondPermutation
        omega.secondDiagonal).injective.ne houtput₁
  have houtput₂ : nonbinaryFinalOutput omega x.1 ≠ 0 := by
    simpa [nonbinaryFinalOutput] using
      (accumLinearEquiv F (B * m)).injective.ne hinput₂
  have hsupportCard : (support x.1).card = i.support.card :=
    congrArg Finset.card hsupport
  have hfirstSupportCard :
      (support (nonbinaryFirstInput omega x.1)).card = i.support.card := by
    calc
      _ = (support x.1).card := by
        rw [nonbinaryFirstInput, support_monomialLinearEquiv,
          permuteCoordinateSet,
          Finset.card_image_of_injective _ omega.firstPermutation.injective]
      _ = i.support.card := hsupportCard
  have hsecondSupportCard :
      (support (nonbinarySecondInput omega x.1)).card =
        i.intermediateWeight.val := by
    calc
      _ = (support (nonbinaryFirstOutput omega x.1)).card := by
        rw [nonbinarySecondInput, support_monomialLinearEquiv,
          permuteCoordinateSet,
          Finset.card_image_of_injective _ omega.secondPermutation.injective]
      _ = hammingNorm (nonbinaryFirstOutput omega x.1) :=
        card_support_eq_hammingNorm _
      _ = i.intermediateWeight.val := hweight₁
  have hweight₁le := hammingNorm_le_card_fintype
    (x := nonbinaryFirstOutput omega x.1)
  have hweight₂le := hammingNorm_le_card_fintype
    (x := nonbinaryFinalOutput omega x.1)
  simp only [Fintype.card_fin] at hweight₁le hweight₂le
  constructor
  · rw [← hsupportCard]
    exact Finset.card_pos.mpr (support_nonempty_of_ne_zero x.1 hxword)
  constructor
  · rw [← hreturn₁, ← hfirstSupportCard]
    exact accumulatorReturnCount_lt_support_card _ hinput₁
  constructor
  · rw [← hweight₁]
    exact hammingNorm_pos_of_ne_zero houtput₁
  constructor
  · omega
  constructor
  · rw [← hreturn₂, ← hsecondSupportCard]
    exact accumulatorReturnCount_lt_support_card _ hinput₂
  constructor
  · rw [← hweight₂]
    exact hammingNorm_pos_of_ne_zero houtput₂
  · omega

theorem masterTraceTerm_formula_of_nonbinaryBAATraceEvent
    {F : Type*} [Field F] [Fintype F] [DecidableEq F]
    {m B L : ℕ} {C0 : Code F m} (i : MasterTraceIndex (B * m) L)
    {omega : NonbinaryBAAEnsemble F (B * m)}
    (hevent : nonbinaryBAATraceEvent C0 i omega) :
    masterTraceTerm F (outerShorteningRank (B := B) C0) i =
      (((Nat.choose (i.support.card - i.firstReturns.val)
            i.firstReturns.val : ℕ) : ℚ) *
          Nat.choose (B * m - i.intermediateWeight.val) i.firstReturns.val *
          Nat.choose (i.intermediateWeight.val - 1)
            (i.support.card - i.firstReturns.val - 1) /
          Nat.choose (B * m) i.support.card) *
      (((Nat.choose (i.intermediateWeight.val - i.secondReturns.val)
            i.secondReturns.val : ℕ) : ℚ) *
          Nat.choose (B * m - i.finalWeight.val) i.secondReturns.val *
          Nat.choose (i.finalWeight.val - 1)
            (i.intermediateWeight.val - i.secondReturns.val - 1) /
          Nat.choose (B * m) i.intermediateWeight.val) *
      projectiveCancellationBound F
        (outerShorteningRank (B := B) C0 i.support)
        (i.firstReturns.val + i.secondReturns.val) := by
  obtain ⟨hs, ha, hh, hhN, hah, hj, hjN⟩ :=
    nonbinaryBAATraceEvent_feasible hevent
  exact masterTraceTerm_formula F (outerShorteningRank (B := B) C0) i
    hs ha hh hhN hah hj hjN

/-- Equation (3) specialized to the actual finite BAA ensemble.  The cover is
proved above; the remaining hypothesis is precisely the per-trace estimate
assembled from equations (1) and (2). -/
theorem concrete_nonbinaryBAA_probability_le_masterTraceBound
    {F : Type*} [Field F] [Fintype F] [DecidableEq F]
    {m B L : ℕ} (C0 : Code F m)
    (hterm : ∀ i : MasterTraceIndex (B * m) L,
      uniformEventProbability (nonbinaryBAATraceEvent (B := B) C0 i) ≤
        masterTraceTerm F (outerShorteningRank (B := B) C0) i) :
    uniformEventProbability (nonbinaryBAABadEvent (B := B) C0 L) ≤
      masterTraceBound F (N := B * m) (L := L)
        (outerShorteningRank (B := B) C0) := by
  apply probability_badCode_le_masterTraceBound F
    (outerShorteningRank (B := B) C0)
    (nonbinaryBAABadEvent (B := B) C0 L)
    (nonbinaryBAATraceEvent (B := B) C0)
  · exact nonbinaryBAABadEvent_iff_exists_trace (B := B) C0
  · exact hterm

end BAA
