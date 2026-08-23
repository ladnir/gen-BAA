import BAA.DiagonalReturns
import BAA.NonbinaryPlacementExtract
import BAA.UniformFinset

/-!
# One-stage finite nonbinary accumulator bound

This file separates a monomial stage into a uniformly chosen support and the
unit multipliers attached to its active coordinates.  Successful outcomes are
covered by the concrete placement objects counted in equation (1).
-/

namespace BAA

section

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

noncomputable local instance stageClassicalDecidablePred
    {α : Type*} (p : α → Prop) : DecidablePred p :=
  Classical.decPred p

/-- Put an active sequence on a fixed support in increasing coordinate order. -/
noncomputable def wordOnOrderedSupport {N s : ℕ}
    (T : FixedCardFinset N s) (u : Fin s → F) : Word F N :=
  fun k => if hk : k ∈ T.1 then
    u ((T.1.orderIsoOfFin T.2).symm ⟨k, hk⟩)
  else 0

theorem wordOnOrderedSupport_orderEmb_apply {N s : ℕ}
    (T : FixedCardFinset N s) (u : Fin s → F) (i : Fin s) :
    wordOnOrderedSupport T u (T.1.orderEmbOfFin T.2 i) = u i := by
  classical
  unfold wordOnOrderedSupport
  rw [dif_pos (Finset.orderEmbOfFin_mem T.1 T.2 i)]
  have hinv := (T.1.orderIsoOfFin T.2).symm_apply_apply i
  exact congrArg u hinv

theorem support_wordOnOrderedSupport {N s : ℕ}
    (T : FixedCardFinset N s) (u : Fin s → F) (hu : ∀ i, u i ≠ 0) :
    support (wordOnOrderedSupport T u) = T.1 := by
  classical
  ext k
  rw [mem_support]
  by_cases hk : k ∈ T.1
  · simp only [wordOnOrderedSupport, dif_pos hk, hk, iff_true]
    exact hu _
  · simp [wordOnOrderedSupport, hk]

theorem orderedSupportValues_wordOnOrderedSupport {N s : ℕ}
    (T : FixedCardFinset N s) (u : Fin s → F) (hu : ∀ i, u i ≠ 0) :
    orderedSupportValues (wordOnOrderedSupport T u)
      (by rw [support_wordOnOrderedSupport T u hu]; exact T.2) = u := by
  funext i
  have hsupport := support_wordOnOrderedSupport T u hu
  have horder : ∀ j,
      (support (wordOnOrderedSupport T u)).orderEmbOfFin
          (by rw [hsupport]; exact T.2) j =
        T.1.orderEmbOfFin T.2 j := by
    have heq := Finset.orderEmbOfFin_unique T.2
      (f := fun j =>
        (support (wordOnOrderedSupport T u)).orderEmbOfFin
          (by rw [hsupport]; exact T.2) j)
      (fun j => by
        let k : Fin N :=
          (support (wordOnOrderedSupport T u)).orderEmbOfFin
            (by rw [hsupport]; exact T.2) j
        have hk : k ∈ support (wordOnOrderedSupport T u) :=
          Finset.orderEmbOfFin_mem (support (wordOnOrderedSupport T u))
            (by rw [hsupport]; exact T.2) j
        have hmem : k ∈ support (wordOnOrderedSupport T u) ↔ k ∈ T.1 := by
          rw [hsupport]
        exact hmem.mp hk)
      ((support (wordOnOrderedSupport T u)).orderEmbOfFin
        (by rw [hsupport]; exact T.2)).strictMono
    exact fun j => congrFun heq j
  unfold orderedSupportValues
  rw [horder i]
  exact wordOnOrderedSupport_orderEmb_apply T u i

/-- Reconstruct a word from its support and its values in increasing support
order. -/
theorem wordOnOrderedSupport_orderedSupportValues {N s : ℕ}
    (x : Word F N) (hcard : (support x).card = s) :
    wordOnOrderedSupport (F := F) ⟨support x, hcard⟩
      (orderedSupportValues x hcard) = x := by
  funext k
  by_cases hk : k ∈ support x
  · let i : Fin s := ((support x).orderIsoOfFin hcard).symm ⟨k, hk⟩
    have hki : (support x).orderEmbOfFin hcard i = k := by
      exact congrArg Subtype.val
        (((support x).orderIsoOfFin hcard).apply_symm_apply ⟨k, hk⟩)
    rw [← hki, wordOnOrderedSupport_orderEmb_apply]
    rfl
  · have hxk : x k = 0 := not_ne_iff.mp (by
      simpa only [mem_support] using hk)
    simp [wordOnOrderedSupport, hk, hxk]

/-- Reduced sample space for one monomial-accumulator stage. -/
abbrev ReducedNonbinaryStageSample (F : Type*) [Monoid F] (N s : ℕ) :=
  FixedCardFinset N s × (Fin s → Fˣ)

/-- The concrete input represented by a reduced support/diagonal sample. -/
noncomputable def reducedStageInput {N s : ℕ} (v : Fin s → F)
    (z : ReducedNonbinaryStageSample F N s) : Word F N :=
  wordOnOrderedSupport z.1 (weightedActiveValues v z.2)

theorem support_reducedStageInput {N s : ℕ} (v : Fin s → F)
    (hv : ∀ i, v i ≠ 0) (z : ReducedNonbinaryStageSample F N s) :
    support (reducedStageInput v z) = z.1.1 := by
  apply support_wordOnOrderedSupport
  intro i
  exact mul_ne_zero (hv i) (Units.ne_zero (z.2 i))

theorem orderedSupportValues_reducedStageInput {N s : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0)
    (z : ReducedNonbinaryStageSample F N s) :
    orderedSupportValues (reducedStageInput v z)
      (by rw [support_reducedStageInput v hv z]; exact z.1.2) =
        weightedActiveValues v z.2 := by
  exact orderedSupportValues_wordOnOrderedSupport z.1 _
    (fun i => mul_ne_zero (hv i) (Units.ne_zero (z.2 i)))

/-- A reduced stage has the requested output weight and active return count. -/
def reducedStageStatEvent {N s : ℕ} (v : Fin s → F) (h a : ℕ)
    (z : ReducedNonbinaryStageSample F N s) : Prop :=
  hammingNorm (accumLinearEquiv F N (reducedStageInput v z)) = h ∧
    (accumulatorZeroReturnTrace (weightedActiveValues v z.2)).card = a

/-- Placement-indexed upper event.  Exact trace matching implies this event;
allowing additional returns makes its diagonal count monotone. -/
def reducedPlacementUpperEvent {N s h a : ℕ}
    (v : Fin s → F) (hh : h ≤ N)
    (p : NonbinaryAccumulatorPlacements N s h a)
    (z : ReducedNonbinaryStageSample F N s) : Prop :=
  z.1.1 = scheduleInputSupport p hh ∧
    PrescribesAccumulatorReturns v (decodedZeroReturnTrace p.1) z.2

/-- Every successful reduced sample is covered by a placement-indexed upper
event from the exact equation-(1) placement type. -/
theorem reducedStageStatEvent_exists_placement {N s h a : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0) (hs : 0 < s)
    (z : ReducedNonbinaryStageSample F N s)
    (hz : reducedStageStatEvent v h a z) :
    ∃ hh : h ≤ N, ∃ p : NonbinaryAccumulatorPlacements N s h a,
      reducedPlacementUpperEvent v hh p z := by
  have hsupport := support_reducedStageInput v hv z
  have hx : reducedStageInput v z ≠ 0 := by
    intro hxzero
    have : support (reducedStageInput v z) = ∅ := by
      ext k
      simp [hxzero, support]
    rw [hsupport] at this
    have hcard : s = 0 := by
      calc
        s = z.1.1.card := z.1.2.symm
        _ = 0 := by rw [this]; simp
    omega
  have hordered := orderedSupportValues_reducedStageInput v hv z
  obtain ⟨hh, p, htrace, -, hpSupport⟩ :=
    exists_placementOfWord_at_stats (s := s) (h := h) (a := a)
      (reducedStageInput v z) hx
      (by rw [hsupport]; exact z.1.2) hz.1
      (by rw [hordered]; exact hz.2)
  refine ⟨hh, p, hsupport.symm.trans hpSupport.symm, ?_⟩
  intro i hi
  unfold MatchesDecodedReturnTrace at htrace
  have hi' : i ∈ accumulatorZeroReturnTrace
      (orderedSupportValues (reducedStageInput v z)
        (by rw [hsupport]; exact z.1.2)) := by
    rw [htrace]
    exact hi
  unfold accumulatorZeroReturnTrace at hi'
  rw [Finset.mem_filter] at hi'
  simpa only [hordered] using hi'.2

/-- A placement upper event has at most `(q-1)^(s-a)` reduced samples.  Its
support coordinate is forced, and the remaining injection is the diagonal
restriction lemma. -/
theorem card_reducedPlacementUpperEvent_le {N s h a : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0) (hh : h ≤ N)
    (p : NonbinaryAccumulatorPlacements N s h a) :
    (Finset.univ.filter (reducedPlacementUpperEvent v hh p)).card ≤
      (Fintype.card F - 1) ^ (s - a) := by
  classical
  let E := {z : ReducedNonbinaryStageSample F N s //
    reducedPlacementUpperEvent v hh p z}
  let D := {d : Fin s → Fˣ //
    PrescribesAccumulatorReturns v (decodedZeroReturnTrace p.1) d}
  let f : E → D := fun z => ⟨z.1.2, z.2.2⟩
  have hf : Function.Injective f := by
    intro z w hd
    apply Subtype.ext
    apply Prod.ext
    · apply Subtype.ext
      exact z.2.1.trans w.2.1.symm
    · exact congrArg Subtype.val hd
  have hcard := Fintype.card_le_of_injective f hf
  have hdiag := card_prescribedAccumulatorReturns_le v hv
    (decodedZeroReturnTrace p.1)
  rw [card_decodedZeroReturnTrace] at hdiag
  change Fintype.card E ≤ Fintype.card D at hcard
  rw [Fintype.card_subtype] at hcard
  rw [Fintype.card_subtype] at hcard
  exact hcard.trans hdiag

/-- Exact probability bound for one placement fiber in the reduced sample
space. -/
theorem reducedPlacementUpperEvent_probability_le {N s h a : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0) (hh : h ≤ N)
    (p : NonbinaryAccumulatorPlacements N s h a) :
    uniformEventProbability (reducedPlacementUpperEvent v hh p) ≤
      (1 / (Nat.choose N s : ℚ)) *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
  classical
  have haS : a ≤ s := by
    have hcard := card_decodedZeroReturnTrace p.1
    have hle := Finset.card_le_univ (decodedZeroReturnTrace p.1)
    simp only [Fintype.card_fin, hcard] at hle
    exact hle
  have hsN : s ≤ N := by
    have hcard := card_scheduleInputSupport p hh
    have hle := Finset.card_le_univ (scheduleInputSupport p hh)
    simp only [Fintype.card_fin, hcard] at hle
    exact hle
  have hq : 0 < Fintype.card F - 1 := by
    have : 2 ≤ Fintype.card F := Fintype.one_lt_card
    omega
  have hchoose : 0 < Nat.choose N s := Nat.choose_pos hsN
  have hsplit : (Fintype.card F - 1) ^ s =
      (Fintype.card F - 1) ^ (s - a) *
        (Fintype.card F - 1) ^ a := by
    rw [← pow_add]
    congr 1
    exact (Nat.sub_add_cancel haS).symm
  unfold uniformEventProbability
  rw [Fintype.card_prod, card_fixedCardFinset, Fintype.card_fun,
    Fintype.card_fin, Fintype.card_units]
  have hcard := card_reducedPlacementUpperEvent_le v hv hh p
  calc
    ((Finset.univ.filter (reducedPlacementUpperEvent v hh p)).card : ℚ) /
        (((Nat.choose N s) * (Fintype.card F - 1) ^ s : ℕ) : ℚ) ≤
        (((Fintype.card F - 1) ^ (s - a) : ℕ) : ℚ) /
          (((Nat.choose N s) * (Fintype.card F - 1) ^ s : ℕ) : ℚ) := by
      apply div_le_div_of_nonneg_right
      · exact_mod_cast hcard
      · positivity
    _ = (1 / (Nat.choose N s : ℚ)) *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
      rw [hsplit]
      push_cast
      field_simp

/-- One reduced monomial-accumulator stage obeys equation (1), with the
`a` forced diagonal cancellations kept explicit for later projective
grouping. -/
theorem reducedStageStatEvent_probability_le {N s h a : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0)
    (hs : 0 < s) (hsN : s ≤ N) (hhs : a < s)
    (hh : 0 < h) (hhN : h ≤ N) :
    uniformEventProbability (reducedStageStatEvent (N := N) v h a) ≤
      nonbinaryAccumulatorKernel N s h a *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
  classical
  letI : Nonempty (FixedCardFinset N s) :=
    Fintype.card_pos_iff.mp (by rw [card_fixedCardFinset]; exact Nat.choose_pos hsN)
  let upper : ReducedNonbinaryStageSample F N s → Prop := fun z =>
    ∃ p : NonbinaryAccumulatorPlacements N s h a,
      reducedPlacementUpperEvent v hhN p z
  calc
    uniformEventProbability (reducedStageStatEvent (N := N) v h a) ≤
        uniformEventProbability upper := by
      apply uniformEventProbability_mono
      intro z hz
      obtain ⟨hh', p, hp⟩ :=
        reducedStageStatEvent_exists_placement v hv hs z hz
      have : hh' = hhN := Subsingleton.elim _ _
      exact ⟨p, by simpa only [this] using hp⟩
    _ ≤ ∑ p : NonbinaryAccumulatorPlacements N s h a,
        uniformEventProbability (reducedPlacementUpperEvent v hhN p) :=
      uniformEventProbability_exists_le_sum _
    _ ≤ ∑ _p : NonbinaryAccumulatorPlacements N s h a,
        (1 / (Nat.choose N s : ℚ)) *
          (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
      exact Finset.sum_le_sum fun p _ =>
        reducedPlacementUpperEvent_probability_le v hv hhN p
    _ = nonbinaryAccumulatorKernel N s h a *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
      rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ,
        nonbinaryAccumulatorKernel]
      push_cast
      ring

end

end BAA
