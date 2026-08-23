import BAA.SupportPropagation

/-!
# Rank propagation through the backward BAA transform
-/

namespace BAA

section Transport

variable {F : Type*} [Field F] [DecidableEq F] {n r : ℕ}

/-- Restrict a linear map between two supported-coordinate subspaces when a
support-containment proof is available. -/
def supportedLinearMap
    (L : Word F n →ₗ[F] Word F n) (T U : Finset (Fin n))
    (hL : ∀ x, x ∈ supportedCode (F := F) T →
      L x ∈ supportedCode (F := F) U) :
    supportedCode (F := F) T →ₗ[F] supportedCode (F := F) U where
  toFun x := ⟨L x.1, hL x.1 x.2⟩
  map_add' _ _ := by ext; simp
  map_smul' _ _ := by ext; simp

/-- Pullback cannot increase selected-column rank beyond the rank available on
the containing backward support. -/
theorem columnRank_transportedParityCheck_le
    (H : Word F n →ₗ[F] Word F r) (E : Word F n ≃ₗ[F] Word F n)
    (T U : Finset (Fin n))
    (hE : ∀ x, x ∈ supportedCode (F := F) T →
      E.symm x ∈ supportedCode (F := F) U) :
    columnRank (transportedParityCheck H E) T ≤ columnRank H U := by
  unfold columnRank
  apply Submodule.finrank_mono
  intro y hy
  obtain ⟨x, rfl⟩ := hy
  let z : supportedCode (F := F) U := ⟨E.symm x.1, hE x.1 x.2⟩
  exact ⟨z, rfl⟩

end Transport

section BAA

variable {F : Type*} [Field F] [DecidableEq F] {B m r : ℕ}

theorem columnRank_baaParityCheck_le_backward
    (H : Word F (B * m) →ₗ[F] Word F r)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) (T : Finset (Fin (B * m))) :
    columnRank (baaParityCheck H sigma₁ sigma₂ d₁ d₂) T ≤
      columnRank H
        (twoStepBackwardNeighborhood sigma₁.symm sigma₂.symm T) := by
  apply columnRank_transportedParityCheck_le
  intro x hx
  rw [mem_supportedCode_iff_support_subset] at hx ⊢
  exact (support_baaTransform_symm_subset sigma₁ sigma₂ d₁ d₂ x).trans
    (by
      apply Finset.image_mono
      apply forwardBoundary_mono
      apply Finset.image_mono
      apply forwardBoundary_mono
      exact hx)

/-- An outer block-rank cap propagates through both backward difference and
permutation layers. -/
theorem baaParityCheck_satisfiesTwoStepBlockRankBound
    (H : Word F (B * m) →ₗ[F] Word F r)
    (h : ℕ)
    (hH : ∀ U, columnRank H U ≤ localBlockRank h U)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) :
    SatisfiesTwoStepBlockRankBound
      (baaParityCheck H sigma₁ sigma₂ d₁ d₂) sigma₁.symm sigma₂.symm h := by
  intro T
  exact (columnRank_baaParityCheck_le_backward H sigma₁ sigma₂ d₁ d₂ T).trans
    (hH _)

/-- Concrete coefficient-independent G5, conditional only on the local outer
parity-check rank cap.  Diagonal coefficients are arbitrary units. -/
theorem baaCoefficientIndependentObstruction
    (H : Word F (B * m) →ₗ[F] Word F r)
    (h : ℕ) (hH : ∀ U, columnRank H U ≤ localBlockRank h U)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) (T : Finset (Fin (B * m)))
    (hdefect : localBlockRank h
      (twoStepBackwardNeighborhood sigma₁.symm sigma₂.symm T) < T.card) :
    ∃ x : Word F (B * m),
      x ≠ 0 ∧ x ∈ supportedCode (F := F) T ∧
        baaParityCheck H sigma₁ sigma₂ d₁ d₂ x = 0 := by
  exact coefficientIndependentObstruction
    (baaParityCheck H sigma₁ sigma₂ d₁ d₂) sigma₁.symm sigma₂.symm h
    (baaParityCheck_satisfiesTwoStepBlockRankBound H h hH sigma₁ sigma₂ d₁ d₂)
    T hdefect

end BAA

end BAA
