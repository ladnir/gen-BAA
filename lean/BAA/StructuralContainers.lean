import BAA.BlockParityCheck
import BAA.FiniteProbability
import BAA.StructuralExpectation
import BAA.UniformFinset
import Mathlib.Data.Fintype.Perm

/-!
# Concrete structural-container random variable

This is the left side whose exact endpoint-refined expectation is the corrected
replacement for G9.  It counts final supports once, independently of field
values and accumulator traces.
-/

namespace BAA

abbrev StructuralPermutationEnsemble (N : ℕ) :=
  Equiv.Perm (Fin N) × Equiv.Perm (Fin N)

/-- Number of size-`t` supports with a strict two-step local block-rank defect. -/
def structuralContainerCount (B m h t : ℕ)
    (omega : StructuralPermutationEnsemble (B * m)) : ℕ :=
  (Finset.univ.filter fun T : FixedCardFinset (B * m) t =>
    localBlockRank h
      (twoStepBackwardNeighborhood omega.1 omega.2 T.1) < t).card

/-- Exact finite expectation of the direct structural-container count. -/
noncomputable def concreteStructuralContainerExpectation
    (B m h t : ℕ) : ℚ :=
  uniformExpectationNat (structuralContainerCount B m h t)

theorem structuralContainerCount_le_choose (B m h t : ℕ)
    (omega : StructuralPermutationEnsemble (B * m)) :
    structuralContainerCount B m h t omega ≤ Nat.choose (B * m) t := by
  unfold structuralContainerCount
  exact (Finset.card_le_card (Finset.filter_subset _ _)).trans_eq
    (by simp [card_fixedCardFinset])

theorem concreteStructuralContainerExpectation_nonneg (B m h t : ℕ) :
    0 ≤ concreteStructuralContainerExpectation B m h t := by
  unfold concreteStructuralContainerExpectation uniformExpectationNat
  positivity

theorem concreteStructuralContainerExpectation_eq_sum_support_probabilities
    (B m h t : ℕ) :
    concreteStructuralContainerExpectation B m h t =
      ∑ T : FixedCardFinset (B * m) t,
        uniformEventProbability
          (fun omega : StructuralPermutationEnsemble (B * m) =>
            localBlockRank h
              (twoStepBackwardNeighborhood omega.1 omega.2 T.1) < t) := by
  unfold concreteStructuralContainerExpectation structuralContainerCount
  exact uniformExpectationNat_filter_card _

/-- Exact probability that a uniformly permuted fixed support has capped local
block rank at most `c`. -/
theorem uniform_permutation_localBlockRank_le_probability
    {B m : ℕ} (h c : ℕ) (S : Finset (Fin (B * m))) :
    uniformEventProbability (Omega := Equiv.Perm (Fin (B * m)))
      (fun sigma => localBlockRank h (permuteCoordinateSet sigma S) ≤ c) =
      (concreteCappedBlockRankPatternCount B m h S.card c : ℚ) /
        Nat.choose (B * m) S.card := by
  classical
  rw [uniform_permutation_finset_event_probability S
    (fun U => localBlockRank h U ≤ c)]
  unfold uniformEventProbability concreteCappedBlockRankPatternCount
  rw [card_fixedCardFinset]

/-- Strict-defect form used in G9. -/
theorem uniform_permutation_localBlockRank_lt_probability
    {B m t : ℕ} (ht : 0 < t) (h : ℕ)
    (S : Finset (Fin (B * m))) :
    uniformEventProbability (Omega := Equiv.Perm (Fin (B * m)))
      (fun sigma => localBlockRank h (permuteCoordinateSet sigma S) < t) =
      (concreteCappedBlockRankPatternCount B m h S.card (t - 1) : ℚ) /
        Nat.choose (B * m) S.card := by
  calc
    _ = uniformEventProbability (Omega := Equiv.Perm (Fin (B * m)))
        (fun sigma => localBlockRank h (permuteCoordinateSet sigma S) ≤ t - 1) := by
      apply uniformEventProbability_congr
      intro sigma
      omega
    _ = _ := uniform_permutation_localBlockRank_le_probability h (t - 1) S

/-- Exact two-permutation probability after the first forward boundary has
already produced the fixed support `S`. -/
theorem twoPermutationBlockDefectProbability_eq_concreteSecondStage
    {B m t : ℕ} (ht : 0 < t) (h : ℕ)
    (S : Finset (Fin (B * m))) (hS : S.Nonempty) :
    uniformEventProbability
      (fun omega : StructuralPermutationEnsemble (B * m) =>
        localBlockRank h
          (permuteCoordinateSet omega.1
            (forwardBoundary (permuteCoordinateSet omega.2 S))) < t) =
      concreteCorrectedSecondStage B m h (B * m) t S.card := by
  classical
  let P := Equiv.Perm (Fin (B * m))
  let q : P × P → Prop := fun omega =>
    localBlockRank h
      (permuteCoordinateSet omega.1
        (forwardBoundary (permuteCoordinateSet omega.2 S))) < t
  let f : FixedCardFinset (B * m) S.card → ℚ := fun U =>
    (concreteCappedBlockRankPatternCount B m h
      (forwardBoundary U.1).card (t - 1) : ℚ) /
        Nat.choose (B * m) (forwardBoundary U.1).card
  have hpos : 0 < S.card := Finset.card_pos.mpr hS
  have hcard : S.card ≤ B * m := by
    simpa using Finset.card_le_univ S
  calc
    uniformEventProbability q =
        uniformEventProbability (fun omega : P × P => q (omega.2, omega.1)) := by
      exact (uniformEventProbability_equiv (Equiv.prodComm P P) q).symm
    _ = uniformExpectationRat fun sigma₂ : P =>
        uniformEventProbability fun sigma₁ : P => q (sigma₁, sigma₂) :=
      uniformEventProbability_prod _
    _ = uniformExpectationRat fun sigma₂ : P =>
        f ⟨permuteCoordinateSet sigma₂ S, by
          rw [permuteCoordinateSet,
            Finset.card_image_of_injective _ sigma₂.injective]⟩ := by
      apply uniformExpectationRat_congr
      intro sigma₂
      exact uniform_permutation_localBlockRank_lt_probability ht h
        (forwardBoundary (permuteCoordinateSet sigma₂ S))
    _ = uniformExpectationRat f :=
      uniform_permutation_finset_expectation S f
    _ = concreteCorrectedSecondStage B m h (B * m) t S.card := by
      unfold uniformExpectationRat concreteCorrectedSecondStage
      rw [card_fixedCardFinset, sum_fixedCardFinset_by_endpoint hpos]
      have hsum :
          (∑ r : Fin (B * m + 1), ∑ e : Bool,
            ∑ U : EndpointRunSupportEvent (B * m) S.card r.val e, f U.1) =
          ∑ r : Fin (B * m + 1), ∑ e : Bool,
            (Fintype.card
              (EndpointRunEncoding (B * m) S.card r.val e) : ℚ) *
              ((concreteCappedBlockRankPatternCount B m h
                (runExpandedSize S.card r.val e) (t - 1) : ℕ) : ℚ) /
                Nat.choose (B * m) (runExpandedSize S.card r.val e) := by
        apply Finset.sum_congr rfl
        intro r _
        apply Finset.sum_congr rfl
        intro e _
        calc
          _ = ∑ _U : EndpointRunSupportEvent (B * m) S.card r.val e,
              ((concreteCappedBlockRankPatternCount B m h
                (runExpandedSize S.card r.val e) (t - 1) : ℕ) : ℚ) /
                Nat.choose (B * m) (runExpandedSize S.card r.val e) := by
            apply Fintype.sum_congr
            intro U
            have hr := endpointRunSupportEvent_run_pos hpos U
            simp only [f]
            rw [card_forwardBoundary_endpointRunSupportEvent hr U]
          _ = _ := by
            rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
            push_cast
            rw [card_endpointRunSupportEvent_eq_encoding hpos hcard]
            ring
      rw [hsum]
      simp_rw [div_eq_mul_inv]
      rw [Finset.sum_mul]
      simp_rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro r _
      apply Finset.sum_congr rfl
      intro e _
      ring

/-- Exact endpoint-corrected G9 for the direct finite block-pattern count.
Replacing that direct count by G8's polynomial coefficient sum is a separate
algebraic bridge. -/
theorem concreteStructuralContainerExpectation_eq_endpointCorrectedDirect
    {B m t : ℕ} (ht : 0 < t) (htN : t ≤ B * m) (h : ℕ) :
    concreteStructuralContainerExpectation B m h t =
      concreteCorrectedStructuralExpectation B m h (B * m) t := by
  classical
  rw [concreteStructuralContainerExpectation_eq_sum_support_probabilities,
    sum_fixedCardFinset_by_endpoint ht]
  unfold concreteCorrectedStructuralExpectation
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro e _
  calc
    (∑ U : EndpointRunSupportEvent (B * m) t r.val e,
      uniformEventProbability
        (fun omega : StructuralPermutationEnsemble (B * m) =>
          localBlockRank h
            (twoStepBackwardNeighborhood omega.1 omega.2 U.1.1) < t)) =
      ∑ _U : EndpointRunSupportEvent (B * m) t r.val e,
        concreteCorrectedSecondStage B m h (B * m) t
          (runExpandedSize t r.val e) := by
      apply Fintype.sum_congr
      intro U
      have hr := endpointRunSupportEvent_run_pos ht U
      let S := forwardBoundary U.1.1
      have hS : S.Nonempty := by
        have hUcard : 0 < U.1.1.card := by
          rw [U.1.2]
          exact ht
        obtain ⟨i, hi⟩ := Finset.card_pos.mp hUcard
        exact ⟨i, (mem_forwardBoundary i).mpr (Or.inl hi)⟩
      calc
        _ = uniformEventProbability
            (fun omega : StructuralPermutationEnsemble (B * m) =>
              localBlockRank h
                (permuteCoordinateSet omega.1
                  (forwardBoundary (permuteCoordinateSet omega.2 S))) < t) := by
          rfl
        _ = concreteCorrectedSecondStage B m h (B * m) t S.card :=
          twoPermutationBlockDefectProbability_eq_concreteSecondStage ht h S hS
        _ = _ := by
          rw [show S.card = runExpandedSize t r.val e from
            card_forwardBoundary_endpointRunSupportEvent hr U]
    _ = (Fintype.card
          (EndpointRunEncoding (B * m) t r.val e) : ℚ) *
        concreteCorrectedSecondStage B m h (B * m) t
          (runExpandedSize t r.val e) := by
      rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      push_cast
      rw [card_endpointRunSupportEvent_eq_encoding ht htN]

theorem concreteStructuralContainerExpectation_eq_corrected_of_G8
    {B m t : ℕ} (ht : 0 < t) (htN : t ≤ B * m) (h : ℕ)
    (hG8 : ∀ u,
      concreteCappedBlockRankPatternCount B m h u (t - 1) =
        cappedBlockRankPatternCount B m h u (t - 1)) :
    concreteStructuralContainerExpectation B m h t =
      correctedStructuralExpectation B m h (B * m) t := by
  rw [concreteStructuralContainerExpectation_eq_endpointCorrectedDirect ht htN h]
  exact concreteCorrectedStructuralExpectation_eq_corrected
    B m h (B * m) t hG8

/-- Corrected G9: the concrete structural-container expectation is exactly
the four-index endpoint-refined polynomial expression. -/
theorem concreteStructuralContainerExpectation_eq_corrected
    {B m t : ℕ} (ht : 0 < t) (htN : t ≤ B * m) (h : ℕ) :
    concreteStructuralContainerExpectation B m h t =
      correctedStructuralExpectation B m h (B * m) t := by
  rw [concreteStructuralContainerExpectation_eq_endpointCorrectedDirect ht htN h]
  exact concreteCorrectedStructuralExpectation_eq_corrected_unconditional
    B m h (B * m) t

/-- Every support counted by the structural random variable forces a genuine
nonzero BAA codeword, for every choice of diagonal coefficients. -/
theorem structuralContainer_forces_baaCodeword
    {F : Type*} [Field F] [DecidableEq F]
    {B m h t : ℕ} (C0 : Code F m)
    (H0 : Word F m →ₗ[F] Word F h) (hH0 : LinearMap.ker H0 = C0)
    (omega : StructuralPermutationEnsemble (B * m))
    (d₁ d₂ : Fin (B * m) → Fˣ) (T : FixedCardFinset (B * m) t)
    (hT : T ∈ Finset.univ.filter fun T : FixedCardFinset (B * m) t =>
      localBlockRank h
        (twoStepBackwardNeighborhood omega.1 omega.2 T.1) < t) :
    ∃ x : Word F (B * m),
      x ≠ 0 ∧ x ∈ supportedCode (F := F) T.1 ∧
        x ∈ baaCode F C0 omega.1.symm omega.2.symm d₁ d₂ := by
  have hdefect : localBlockRank h
      (twoStepBackwardNeighborhood omega.1 omega.2 T.1) < T.1.card := by
    have := (Finset.mem_filter.mp hT).2
    simpa [T.2] using this
  exact baaCodeword_of_localBlockRank_defect C0 H0 hH0
    omega.1.symm omega.2.symm d₁ d₂ T.1 (by simpa using hdefect)

end BAA
