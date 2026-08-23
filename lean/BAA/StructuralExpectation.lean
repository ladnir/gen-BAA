import BAA.BlockRankCount
import BAA.EndpointRuns

/-!
# Endpoint-corrected structural expression

This module records the exact four-index replacement for printed G9.  The
probabilistic equality with the concrete two-permutation container count is a
separate theorem obligation; defining the corrected expression here prevents
future work from accidentally reverting to the endpoint-free shorthand.
-/

namespace BAA

/-- One endpoint-refined summand of corrected G9, for `N = B*m`. -/
noncomputable def correctedStructuralSummand
    (B m h N t r₁ r₂ : ℕ) (e₁ e₂ : Bool) : ℚ :=
  let u₁ := runExpandedSize t r₁ e₁
  let u₂ := runExpandedSize u₁ r₂ e₂
  (Fintype.card (EndpointRunEncoding N t r₁ e₁) : ℚ) *
    ((Fintype.card (EndpointRunEncoding N u₁ r₂ e₂) : ℚ) /
      Nat.choose N u₁) *
    ((cappedBlockRankPatternCount B m h u₂ (t - 1) : ℕ) : ℚ) /
      Nat.choose N u₂

/-- The exact endpoint-refined replacement for G9.  Both run indices use
finite ranges large enough to include every feasible value; infeasible
composition types have cardinality zero. -/
noncomputable def correctedStructuralExpectation
    (B m h N t : ℕ) : ℚ :=
  ∑ r₁ : Fin (N + 1), ∑ e₁ : Bool,
    ∑ r₂ : Fin (N + 1), ∑ e₂ : Bool,
      correctedStructuralSummand B m h N t r₁.val r₂.val e₁ e₂

/-- The exact second-stage endpoint sum with the final block-rank count kept
as a direct finite-set cardinality. -/
noncomputable def concreteCorrectedSecondStage
    (B m h N t u₁ : ℕ) : ℚ :=
  ∑ r₂ : Fin (N + 1), ∑ e₂ : Bool,
    let u₂ := runExpandedSize u₁ r₂.val e₂
    ((Fintype.card (EndpointRunEncoding N u₁ r₂.val e₂) : ℚ) /
      Nat.choose N u₁) *
    ((concreteCappedBlockRankPatternCount B m h u₂ (t - 1) : ℕ) : ℚ) /
      Nat.choose N u₂

/-- Endpoint-corrected G9 with G8's last factor represented by its direct
finite-set cardinality. -/
noncomputable def concreteCorrectedStructuralExpectation
    (B m h N t : ℕ) : ℚ :=
  ∑ r₁ : Fin (N + 1), ∑ e₁ : Bool,
    (Fintype.card (EndpointRunEncoding N t r₁.val e₁) : ℚ) *
      concreteCorrectedSecondStage B m h N t
        (runExpandedSize t r₁.val e₁)

theorem correctedStructuralSummand_expanded
    (B m h N t r₁ r₂ : ℕ) (e₁ e₂ : Bool) :
    correctedStructuralSummand B m h N t r₁ r₂ e₁ e₂ =
      let u₁ := t + r₁ - endpointBit e₁
      let u₂ := u₁ + r₂ - endpointBit e₂
      (Fintype.card (EndpointRunEncoding N t r₁ e₁) : ℚ) *
        ((Fintype.card (EndpointRunEncoding N u₁ r₂ e₂) : ℚ) /
          Nat.choose N u₁) *
        ((cappedBlockRankPatternCount B m h u₂ (t - 1) : ℕ) : ℚ) /
          Nat.choose N u₂ := by
  rfl

/-- Once G8 identifies its polynomial coefficient sum with the direct block
pattern count, the direct endpoint formula is exactly the manuscript-facing
corrected expression. -/
theorem concreteCorrectedStructuralExpectation_eq_corrected
    (B m h N t : ℕ)
    (hG8 : ∀ u,
      concreteCappedBlockRankPatternCount B m h u (t - 1) =
        cappedBlockRankPatternCount B m h u (t - 1)) :
    concreteCorrectedStructuralExpectation B m h N t =
      correctedStructuralExpectation B m h N t := by
  unfold concreteCorrectedStructuralExpectation concreteCorrectedSecondStage
  unfold correctedStructuralExpectation correctedStructuralSummand
  apply Finset.sum_congr rfl
  intro r₁ _
  apply Finset.sum_congr rfl
  intro e₁ _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro r₂ _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro e₂ _
  simp only
  rw [hG8]
  ring

theorem concreteCorrectedStructuralExpectation_eq_corrected_unconditional
    (B m h N t : ℕ) :
    concreteCorrectedStructuralExpectation B m h N t =
      correctedStructuralExpectation B m h N t := by
  apply concreteCorrectedStructuralExpectation_eq_corrected
  intro u
  exact (cappedBlockRankPatternCount_eq_concrete B m h u (t - 1)).symm

end BAA
