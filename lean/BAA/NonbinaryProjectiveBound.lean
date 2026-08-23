import BAA.NonbinaryMonomialUniform
import BAA.Projective
import BAA.Shortening

/-!
# Projective grouping for finite nonbinary trace events

Nonzero words in a shortening are grouped into one event per projective line.
The representative event records exact support explicitly, so representatives
from smaller shortenings contribute the empty event.
-/

namespace BAA

open scoped LinearAlgebra.Projectivization

section

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

noncomputable local instance projectiveBoundClassicalDecidablePred
    {α : Type*} (p : α → Prop) : DecidablePred p :=
  Classical.decPred p

abbrev ShorteningProjective {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) :=
  ℙ F (shortening C S)

noncomputable local instance shorteningProjectiveFintype
    {N : ℕ} (C : Code F N) (S : Finset (Fin N)) :
    Fintype (ShorteningProjective C S) :=
  Fintype.ofFinite (ShorteningProjective C S)

/-- The word chosen by mathlib for a projective point of a shortening. -/
noncomputable def shorteningProjectiveRepWord {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (D : ShorteningProjective C S) : Word F N :=
  D.rep.1

theorem shorteningProjectiveRepWord_ne_zero {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (D : ShorteningProjective C S) :
    shorteningProjectiveRepWord C S D ≠ 0 := by
  intro hzero
  apply D.rep_nonzero
  apply Subtype.ext
  exact hzero

theorem shorteningProjectiveRepWord_mem {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (D : ShorteningProjective C S) :
    shorteningProjectiveRepWord C S D ∈ C :=
  D.rep.2.1

theorem support_shorteningProjectiveRepWord_subset {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (D : ShorteningProjective C S) :
    support (shorteningProjectiveRepWord C S D) ⊆ S := by
  exact (mem_shortening_iff.mp D.rep.2).2

/-- Projectivizing a shortening only depends on its dimension. -/
theorem card_shorteningProjective_eq_fin {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) :
    Nat.card (ShorteningProjective C S) =
      Nat.card (ℙ F (Fin (shorteningDim C S) → F)) := by
  rw [Projectivization.card_of_finrank F (shortening C S) rfl,
    card_projectivization_fin_geom]
  rfl

/-- The exact-support fixed-statistic event before projective grouping. -/
def shorteningExactSupportStatEvent {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (h a j b : ℕ)
    (omega : NonbinaryBAAEnsemble F N) : Prop :=
  ∃ x : C, x ≠ 0 ∧ support x.1 = S ∧
    nonbinaryBAAFixedWordStatEvent x.1 h a j b omega

/-- Event attached to one projective line of the shortening. -/
def shorteningProjectiveStatEvent {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (h a j b : ℕ)
    (D : ShorteningProjective C S)
    (omega : NonbinaryBAAEnsemble F N) : Prop :=
  support (shorteningProjectiveRepWord C S D) = S ∧
    nonbinaryBAAFixedWordStatEvent
      (shorteningProjectiveRepWord C S D) h a j b omega

/-- Exact-support words are covered by, and can be recovered from, their
projective representative events. -/
theorem shorteningExactSupportStatEvent_iff_exists_projective
    {N : ℕ} (C : Code F N) (S : Finset (Fin N)) (h a j b : ℕ)
    (omega : NonbinaryBAAEnsemble F N) :
    shorteningExactSupportStatEvent C S h a j b omega ↔
      ∃ D : ShorteningProjective C S,
        shorteningProjectiveStatEvent C S h a j b D omega := by
  constructor
  · rintro ⟨x, hx, hsupport, hstat⟩
    have hxword : x.1 ≠ 0 := by
      intro hzero
      apply hx
      exact Subtype.ext hzero
    let xs : shortening C S :=
      ⟨x.1, (mem_shortening_iff.mpr
        ⟨x.2, by rw [hsupport]⟩)⟩
    have hxs : xs ≠ 0 := by
      intro hzero
      apply hxword
      exact congrArg Subtype.val hzero
    let D : ShorteningProjective C S := Projectivization.mk F xs hxs
    obtain ⟨c, hc⟩ := Projectivization.exists_smul_eq_mk_rep F xs hxs
    have hword :
        (c : F) • x.1 = shorteningProjectiveRepWord C S D := by
      exact congrArg Subtype.val hc
    refine ⟨D, ?_, ?_⟩
    · rw [← hword, support_smul_ne_zero (c : F) (Units.ne_zero c), hsupport]
    · rw [← hword]
      exact (nonbinaryBAAFixedWordStatEvent_smul
        (c : F) (Units.ne_zero c) x.1 h a j b omega).mpr hstat
  · rintro ⟨D, hsupport, hstat⟩
    let x : C := ⟨shorteningProjectiveRepWord C S D,
      shorteningProjectiveRepWord_mem C S D⟩
    refine ⟨x, ?_, hsupport, hstat⟩
    intro hzero
    apply shorteningProjectiveRepWord_ne_zero C S D
    exact congrArg Subtype.val hzero

theorem shorteningProjectiveStatEvent_probability_le
    {N s h a j b : ℕ} (C : Code F N) (S : Finset (Fin N))
    (hS : S.card = s) (hsN : s ≤ N)
    (ha : a < s) (hh : 0 < h) (hhN : h ≤ N)
    (hb : b < h) (hj : 0 < j) (hjN : j ≤ N)
    (D : ShorteningProjective C S) :
    uniformEventProbability
        (shorteningProjectiveStatEvent C S h a j b D) ≤
      (nonbinaryAccumulatorKernel N s h a *
        nonbinaryAccumulatorKernel N h j b) *
      (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ (a + b)) := by
  classical
  by_cases hsupport : support (shorteningProjectiveRepWord C S D) = S
  · calc
      uniformEventProbability
          (shorteningProjectiveStatEvent C S h a j b D) ≤
          uniformEventProbability (nonbinaryBAAFixedWordStatEvent
            (shorteningProjectiveRepWord C S D) h a j b) := by
        apply uniformEventProbability_mono
        intro omega hevent
        exact hevent.2
      _ ≤ _ := nonbinaryBAAFixedWordStatEvent_probability_le
        (shorteningProjectiveRepWord C S D)
        (shorteningProjectiveRepWord_ne_zero C S D)
        ((congrArg Finset.card hsupport).trans hS) hsN
        ha hh hhN hb hj hjN
  · have hempty : uniformEventProbability
        (shorteningProjectiveStatEvent C S h a j b D) = 0 := by
      unfold uniformEventProbability shorteningProjectiveStatEvent
      simp [hsupport]
    rw [hempty]
    unfold nonbinaryAccumulatorKernel
    positivity

/-- Projective union bound for an exact-support family.  This is the
non-truncated branch of the projective factor in equation (3). -/
theorem shorteningExactSupportStatEvent_probability_le_projective_mul
    {N s h a j b : ℕ} (C : Code F N) (S : Finset (Fin N))
    (hS : S.card = s) (hsN : s ≤ N)
    (ha : a < s) (hh : 0 < h) (hhN : h ≤ N)
    (hb : b < h) (hj : 0 < j) (hjN : j ≤ N) :
    uniformEventProbability
        (shorteningExactSupportStatEvent C S h a j b) ≤
      (Nat.card (ShorteningProjective C S) : ℚ) *
        ((nonbinaryAccumulatorKernel N s h a *
          nonbinaryAccumulatorKernel N h j b) *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ (a + b))) := by
  classical
  calc
    uniformEventProbability
        (shorteningExactSupportStatEvent C S h a j b) =
        uniformEventProbability (fun omega =>
          ∃ D : ShorteningProjective C S,
            shorteningProjectiveStatEvent C S h a j b D omega) := by
      apply uniformEventProbability_congr
      intro omega
      exact shorteningExactSupportStatEvent_iff_exists_projective
        C S h a j b omega
    _ ≤ ∑ D : ShorteningProjective C S,
          uniformEventProbability
            (shorteningProjectiveStatEvent C S h a j b D) :=
      uniformEventProbability_exists_le_sum
        (fun D : ShorteningProjective C S =>
          shorteningProjectiveStatEvent C S h a j b D)
    _ ≤ ∑ _D : ShorteningProjective C S,
          ((nonbinaryAccumulatorKernel N s h a *
            nonbinaryAccumulatorKernel N h j b) *
          (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ (a + b))) := by
      exact Finset.sum_le_sum fun D _ =>
        shorteningProjectiveStatEvent_probability_le
          C S hS hsN ha hh hhN hb hj hjN D
    _ = _ := by
      rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ,
        Nat.card_eq_fintype_card]

theorem nonbinaryBAATraceEvent_iff_shorteningExactSupportStatEvent
    {m B L : ℕ} (C0 : Code F m) (i : MasterTraceIndex (B * m) L)
    (omega : NonbinaryBAAEnsemble F (B * m)) :
    nonbinaryBAATraceEvent C0 i omega ↔
      shorteningExactSupportStatEvent (outerCode C0 B) i.support
        i.intermediateWeight.val i.firstReturns.val
        i.finalWeight.val i.secondReturns.val omega := by
  unfold nonbinaryBAATraceEvent shorteningExactSupportStatEvent
    nonbinaryBAAFixedWordStatEvent
  aesop

/-- The projective-count branch of the concrete per-trace estimate. -/
theorem nonbinaryBAATraceEvent_probability_le_projective_untruncated_of_feasible
    {m B L : ℕ} (C0 : Code F m) (i : MasterTraceIndex (B * m) L)
    (hs : 0 < i.support.card)
    (ha : i.firstReturns.val < i.support.card)
    (hh : 0 < i.intermediateWeight.val)
    (hhN : i.intermediateWeight.val ≤ B * m)
    (hb : i.secondReturns.val < i.intermediateWeight.val)
    (hj : 0 < i.finalWeight.val)
    (hjN : i.finalWeight.val ≤ B * m) :
    uniformEventProbability (nonbinaryBAATraceEvent C0 i) ≤
      (nonbinaryAccumulatorKernel (B * m) i.support.card
          i.intermediateWeight.val i.firstReturns.val *
        nonbinaryAccumulatorKernel (B * m) i.intermediateWeight.val
          i.finalWeight.val i.secondReturns.val) *
      ((Nat.card (ℙ F
          (Fin (outerShorteningRank C0 i.support) → F)) : ℕ) : ℚ) /
        ((Fintype.card F - 1 : ℕ) : ℚ) ^
          (i.firstReturns.val + i.secondReturns.val) := by
  classical
  calc
    uniformEventProbability (nonbinaryBAATraceEvent C0 i) =
        uniformEventProbability
          (shorteningExactSupportStatEvent (outerCode C0 B) i.support
            i.intermediateWeight.val i.firstReturns.val
            i.finalWeight.val i.secondReturns.val) := by
      apply uniformEventProbability_congr
      intro omega
      exact nonbinaryBAATraceEvent_iff_shorteningExactSupportStatEvent
        C0 i omega
    _ ≤ (Nat.card (ShorteningProjective (outerCode C0 B) i.support) : ℚ) *
        ((nonbinaryAccumulatorKernel (B * m) i.support.card
            i.intermediateWeight.val i.firstReturns.val *
          nonbinaryAccumulatorKernel (B * m) i.intermediateWeight.val
            i.finalWeight.val i.secondReturns.val) *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^
          (i.firstReturns.val + i.secondReturns.val))) :=
      shorteningExactSupportStatEvent_probability_le_projective_mul
        (outerCode C0 B) i.support rfl
        (by simpa only [Fintype.card_fin] using
          Finset.card_le_univ i.support)
        ha hh hhN hb hj hjN
    _ = _ := by
      rw [card_shorteningProjective_eq_fin]
      unfold outerShorteningRank
      ring

theorem uniform_monomial_support_probability {N : ℕ}
    (S T : Finset (Fin N)) (hcard : T.card = S.card) :
    uniformEventProbability (Omega := MonomialSample F N)
      (fun z => permuteCoordinateSet z.1 S = T) =
      1 / Nat.choose N S.card := by
  calc
    uniformEventProbability (Omega := MonomialSample F N)
        (fun z => permuteCoordinateSet z.1 S = T) =
        uniformEventProbability (Omega := Equiv.Perm (Fin N))
          (fun sigma => permuteCoordinateSet sigma S = T) :=
      uniformEventProbability_prod_left
        (fun sigma : Equiv.Perm (Fin N) =>
          permuteCoordinateSet sigma S = T)
    _ = _ := uniform_permutation_finset_probability S T hcard

/-- Support-only event for a pair of decoded accumulator placements.  It
depends on the two permutations but not on either diagonal assignment. -/
def twoStagePlacementSupportEvent {N s h a j b : ℕ}
    (S : Finset (Fin N)) (hh : h ≤ N)
    (p₁ : NonbinaryAccumulatorPlacements N s h a) (hj : j ≤ N)
    (p₂ : NonbinaryAccumulatorPlacements N h j b)
    (z : MonomialSample F N × MonomialSample F N) : Prop :=
  permuteCoordinateSet z.1.1 S = scheduleInputSupport p₁ hh ∧
    permuteCoordinateSet z.2.1 (scheduleNonzeroRegion p₁ hh) =
      scheduleInputSupport p₂ hj

theorem twoStagePlacementSupportEvent_probability_le
    {N s h a j b : ℕ} (S : Finset (Fin N)) (hS : S.card = s)
    (hh : h ≤ N) (p₁ : NonbinaryAccumulatorPlacements N s h a)
    (hj : j ≤ N) (p₂ : NonbinaryAccumulatorPlacements N h j b) :
    uniformEventProbability
        (twoStagePlacementSupportEvent (F := F) S hh p₁ hj p₂) ≤
      (1 / (Nat.choose N s : ℚ)) *
        (1 / (Nat.choose N h : ℚ)) := by
  classical
  let c : ℚ := 1 / (Nat.choose N h : ℚ)
  have hc : 0 ≤ c := by unfold c; positivity
  calc
    uniformEventProbability
        (twoStagePlacementSupportEvent (F := F) S hh p₁ hj p₂) =
        uniformEventProbability (fun z : MonomialSample F N × MonomialSample F N =>
          permuteCoordinateSet z.1.1 S = scheduleInputSupport p₁ hh ∧
            permuteCoordinateSet z.2.1 (scheduleNonzeroRegion p₁ hh) =
              scheduleInputSupport p₂ hj) := by
      apply uniformEventProbability_congr
      intro z
      rfl
    _ ≤
        uniformEventProbability (Omega := MonomialSample F N)
          (fun z => permuteCoordinateSet z.1 S =
            scheduleInputSupport p₁ hh) * c := by
      exact uniformEventProbability_and_prod_le
        (fun z : MonomialSample F N =>
          permuteCoordinateSet z.1 S = scheduleInputSupport p₁ hh)
        (fun _z₁ z₂ : MonomialSample F N =>
          permuteCoordinateSet z₂.1 (scheduleNonzeroRegion p₁ hh) =
            scheduleInputSupport p₂ hj)
        c hc (by
          intro _z₁ _
          unfold c
          rw [uniform_monomial_support_probability]
          · rw [card_scheduleNonzeroRegion]
          · rw [card_scheduleInputSupport, card_scheduleNonzeroRegion])
    _ = _ := by
      unfold c
      rw [uniform_monomial_support_probability]
      · rw [hS]
      · rw [card_scheduleInputSupport, hS]

/-- Any successful exact-support word determines two concrete placement
indices whose support-only event holds. -/
theorem twoStageMonomialStatEvent_exists_placement_support
    {N s h a j b : ℕ} (x : Word F N) (hx : x ≠ 0)
    (S : Finset (Fin N)) (hSx : support x = S) (hScard : S.card = s)
    (z : MonomialSample F N × MonomialSample F N)
    (hz : twoStageMonomialStatEvent x h a j b z) :
    ∃ hh : h ≤ N, ∃ p₁ : NonbinaryAccumulatorPlacements N s h a,
      ∃ hj : j ≤ N, ∃ p₂ : NonbinaryAccumulatorPlacements N h j b,
        twoStagePlacementSupportEvent (F := F) S hh p₁ hj p₂ z := by
  let x₁ := monomialSampleOutput x z.1
  let y₁ := accumLinearEquiv F N x₁
  let x₂ := monomialSampleOutput y₁ z.2
  have hx₁ : x₁ ≠ 0 := by
    unfold x₁ monomialSampleOutput
    simpa only [map_zero] using
      (monomialLinearEquiv F z.1.1 z.1.2).injective.ne hx
  have hy₁ : y₁ ≠ 0 := by
    unfold y₁
    simpa only [map_zero] using (accumLinearEquiv F N).injective.ne hx₁
  have hx₂ : x₂ ≠ 0 := by
    unfold x₂ monomialSampleOutput
    simpa only [map_zero] using
      (monomialLinearEquiv F z.2.1 z.2.2).injective.ne hy₁
  have hx₁card : (support x₁).card = s := by
    calc
      _ = hammingNorm x₁ := card_support_eq_hammingNorm _
      _ = hammingNorm x := hammingNorm_monomialSampleOutput x z.1
      _ = (support x).card := (card_support_eq_hammingNorm x).symm
      _ = S.card := congrArg Finset.card hSx
      _ = s := hScard
  have hx₂card : (support x₂).card = h := by
    calc
      _ = hammingNorm x₂ := card_support_eq_hammingNorm _
      _ = hammingNorm y₁ := hammingNorm_monomialSampleOutput y₁ z.2
      _ = h := hz.1.1
  have hreturn₁ :
      (accumulatorZeroReturnTrace
        (orderedSupportValues x₁ hx₁card)).card = a := by
    rw [← accumulatorReturnCount_eq_activeTraceCard_of_card x₁ hx₁card]
    exact hz.1.2
  have hreturn₂ :
      (accumulatorZeroReturnTrace
        (orderedSupportValues x₂ hx₂card)).card = b := by
    rw [← accumulatorReturnCount_eq_activeTraceCard_of_card x₂ hx₂card]
    exact hz.2.2
  obtain ⟨hh, p₁, htrace₁, hword₁, hsupport₁⟩ :=
    exists_placementOfWord_at_stats x₁ hx₁ hx₁card hz.1.1 hreturn₁
  obtain ⟨hj, p₂, _htrace₂, _hword₂, hsupport₂⟩ :=
    exists_placementOfWord_at_stats x₂ hx₂ hx₂card hz.2.1 hreturn₂
  refine ⟨hh, p₁, hj, p₂, ?_, ?_⟩
  · rw [hsupport₁]
    unfold x₁ monomialSampleOutput
    rw [support_monomialLinearEquiv, hSx]
  · rw [hsupport₂]
    unfold x₂ monomialSampleOutput
    rw [support_monomialLinearEquiv]
    have hySupport : support y₁ = scheduleNonzeroRegion p₁ hh := by
      unfold y₁
      rw [← hword₁]
      exact support_accum_scheduleInputWord p₁ hh
        (orderedSupportValues x₁ hx₁card) htrace₁
    rw [hySupport]

/-- Exact-support existential event on the product monomial sample. -/
def twoStageExactSupportMonomialEvent {N : ℕ} (C : Code F N)
    (S : Finset (Fin N)) (h a j b : ℕ)
    (z : MonomialSample F N × MonomialSample F N) : Prop :=
  ∃ x : C, x ≠ 0 ∧ support x.1 = S ∧
    twoStageMonomialStatEvent x.1 h a j b z

theorem shorteningExactSupportStatEvent_ensembleEquiv
    {N : ℕ} (C : Code F N) (S : Finset (Fin N)) (h a j b : ℕ)
    (omega : NonbinaryBAAEnsemble F N) :
    shorteningExactSupportStatEvent C S h a j b omega ↔
      twoStageExactSupportMonomialEvent C S h a j b
        (nonbinaryBAAEnsembleEquivMonomialProduct (F := F) N omega) := by
  unfold shorteningExactSupportStatEvent twoStageExactSupportMonomialEvent
  constructor
  · rintro ⟨x, hx, hsupport, hstat⟩
    exact ⟨x, hx, hsupport,
      (twoStageMonomialStatEvent_ensembleEquiv x.1 h a j b omega).2 hstat⟩
  · rintro ⟨x, hx, hsupport, hstat⟩
    exact ⟨x, hx, hsupport,
      (twoStageMonomialStatEvent_ensembleEquiv x.1 h a j b omega).1 hstat⟩

theorem twoStageExactSupportMonomialEvent_covered_by_placements
    {N s h a j b : ℕ} (C : Code F N) (S : Finset (Fin N))
    (hScard : S.card = s) (hhN : h ≤ N) (hjN : j ≤ N)
    (z : MonomialSample F N × MonomialSample F N)
    (hz : twoStageExactSupportMonomialEvent C S h a j b z) :
    ∃ p : NonbinaryAccumulatorPlacements N s h a ×
        NonbinaryAccumulatorPlacements N h j b,
      twoStagePlacementSupportEvent (F := F) S hhN p.1 hjN p.2 z := by
  obtain ⟨x, hx, hsupport, hstat⟩ := hz
  have hxword : x.1 ≠ 0 := by
    intro hzero
    apply hx
    exact Subtype.ext hzero
  obtain ⟨hh, p₁, hj, p₂, hp⟩ :=
    twoStageMonomialStatEvent_exists_placement_support
      x.1 hxword S hsupport hScard z hstat
  refine ⟨(p₁, p₂), ?_⟩
  simpa only [Subsingleton.elim hh hhN, Subsingleton.elim hj hjN] using hp

/-- Shared-placement branch: after forgetting all diagonal cancellation
constraints, the existential family is bounded by the product of the two
equation-(1) support kernels, with no projective multiplicity. -/
theorem shorteningExactSupportStatEvent_probability_le_kernels
    {N s h a j b : ℕ} (C : Code F N) (S : Finset (Fin N))
    (hScard : S.card = s) (hhN : h ≤ N) (hjN : j ≤ N) :
    uniformEventProbability
        (shorteningExactSupportStatEvent C S h a j b) ≤
      nonbinaryAccumulatorKernel N s h a *
        nonbinaryAccumulatorKernel N h j b := by
  classical
  let P := NonbinaryAccumulatorPlacements N s h a ×
    NonbinaryAccumulatorPlacements N h j b
  calc
    uniformEventProbability
        (shorteningExactSupportStatEvent C S h a j b) =
        uniformEventProbability (fun omega : NonbinaryBAAEnsemble F N =>
          twoStageExactSupportMonomialEvent C S h a j b
            (nonbinaryBAAEnsembleEquivMonomialProduct (F := F) N omega)) := by
      apply uniformEventProbability_congr
      intro omega
      exact shorteningExactSupportStatEvent_ensembleEquiv C S h a j b omega
    _ = uniformEventProbability
        (twoStageExactSupportMonomialEvent C S h a j b) :=
      uniformEventProbability_equiv
        (nonbinaryBAAEnsembleEquivMonomialProduct (F := F) N)
        (twoStageExactSupportMonomialEvent C S h a j b)
    _ ≤ uniformEventProbability (fun z =>
          ∃ p : P, twoStagePlacementSupportEvent
            (F := F) S hhN p.1 hjN p.2 z) := by
      apply uniformEventProbability_mono
      intro z hz
      exact twoStageExactSupportMonomialEvent_covered_by_placements
        C S hScard hhN hjN z hz
    _ ≤ ∑ p : P, uniformEventProbability
          (twoStagePlacementSupportEvent (F := F)
            S hhN p.1 hjN p.2) :=
      uniformEventProbability_exists_le_sum
        (fun p : P => twoStagePlacementSupportEvent
          (F := F) S hhN p.1 hjN p.2)
    _ ≤ ∑ _p : P,
          ((1 / (Nat.choose N s : ℚ)) *
            (1 / (Nat.choose N h : ℚ))) := by
      exact Finset.sum_le_sum fun p _ =>
        twoStagePlacementSupportEvent_probability_le
          S hScard hhN p.1 hjN p.2
    _ = _ := by
      unfold P nonbinaryAccumulatorKernel
      rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ,
        Fintype.card_prod]
      push_cast
      ring

/-- The two independent branches combine to the exact truncated projective
factor used by one summand of equation (3). -/
theorem nonbinaryBAATraceEvent_probability_le_masterTraceTerm_of_feasible
    {m B L : ℕ} (C0 : Code F m) (i : MasterTraceIndex (B * m) L)
    (hs : 0 < i.support.card)
    (ha : i.firstReturns.val < i.support.card)
    (hh : 0 < i.intermediateWeight.val)
    (hhN : i.intermediateWeight.val ≤ B * m)
    (hb : i.secondReturns.val < i.intermediateWeight.val)
    (hj : 0 < i.finalWeight.val)
    (hjN : i.finalWeight.val ≤ B * m) :
    uniformEventProbability (nonbinaryBAATraceEvent C0 i) ≤
      masterTraceTerm F (outerShorteningRank C0) i := by
  classical
  let K : ℚ :=
    nonbinaryAccumulatorKernel (B * m) i.support.card
        i.intermediateWeight.val i.firstReturns.val *
      nonbinaryAccumulatorKernel (B * m) i.intermediateWeight.val
        i.finalWeight.val i.secondReturns.val
  let R : ℚ :=
    ((Nat.card (ℙ F
      (Fin (outerShorteningRank C0 i.support) → F)) : ℕ) : ℚ) /
      ((Fintype.card F - 1 : ℕ) : ℚ) ^
        (i.firstReturns.val + i.secondReturns.val)
  have heq :
      uniformEventProbability (nonbinaryBAATraceEvent C0 i) =
        uniformEventProbability
          (shorteningExactSupportStatEvent (outerCode C0 B) i.support
            i.intermediateWeight.val i.firstReturns.val
            i.finalWeight.val i.secondReturns.val) := by
    apply uniformEventProbability_congr
    intro omega
    exact nonbinaryBAATraceEvent_iff_shorteningExactSupportStatEvent
      C0 i omega
  have hkernel :
      uniformEventProbability (nonbinaryBAATraceEvent C0 i) ≤ K := by
    rw [heq]
    exact shorteningExactSupportStatEvent_probability_le_kernels
      (outerCode C0 B) i.support rfl hhN hjN
  have hprojective :
      uniformEventProbability (nonbinaryBAATraceEvent C0 i) ≤ K * R := by
    dsimp [K, R]
    convert
      nonbinaryBAATraceEvent_probability_le_projective_untruncated_of_feasible
        C0 i hs ha hh hhN hb hj hjN using 1 <;> ring
  unfold masterTraceTerm projectiveCancellationBound
  change uniformEventProbability (nonbinaryBAATraceEvent C0 i) ≤
    K * min 1 R
  by_cases hR : 1 ≤ R
  · rw [min_eq_left hR, mul_one]
    exact hkernel
  · rw [min_eq_right (le_of_not_ge hR)]
    exact hprojective

/-- Fully discharged concrete per-trace obligation for equation (3).  Empty
trace events are handled directly; inhabited events supply all feasibility
conditions from the concrete construction. -/
theorem nonbinaryBAATraceEvent_probability_le_masterTraceTerm
    {m B L : ℕ} (C0 : Code F m) (i : MasterTraceIndex (B * m) L) :
    uniformEventProbability (nonbinaryBAATraceEvent C0 i) ≤
      masterTraceTerm F (outerShorteningRank C0) i := by
  classical
  by_cases hevent : ∃ omega, nonbinaryBAATraceEvent C0 i omega
  · obtain ⟨omega, homega⟩ := hevent
    obtain ⟨hs, ha, hh, hhN, hb, hj, hjN⟩ :=
      nonbinaryBAATraceEvent_feasible homega
    exact nonbinaryBAATraceEvent_probability_le_masterTraceTerm_of_feasible
      C0 i hs ha hh hhN hb hj hjN
  · have hempty : uniformEventProbability (nonbinaryBAATraceEvent C0 i) = 0 := by
      have hall : ∀ omega, ¬nonbinaryBAATraceEvent C0 i omega := by
        intro omega homega
        exact hevent ⟨omega, homega⟩
      unfold uniformEventProbability
      simp [hall]
    rw [hempty]
    unfold masterTraceTerm projectiveCancellationBound
      nonbinaryAccumulatorKernel
    positivity

/-- Equation (3) for the actual finite nonbinary BAA ensemble, with the
per-trace hypothesis eliminated. -/
theorem concrete_nonbinaryBAA_probability_le_masterTraceBound_proved
    {m B L : ℕ} (C0 : Code F m) :
    uniformEventProbability (nonbinaryBAABadEvent (B := B) C0 L) ≤
      masterTraceBound F (N := B * m) (L := L)
        (outerShorteningRank (B := B) C0) := by
  exact concrete_nonbinaryBAA_probability_le_masterTraceBound C0
    (nonbinaryBAATraceEvent_probability_le_masterTraceTerm C0)

end

end BAA
