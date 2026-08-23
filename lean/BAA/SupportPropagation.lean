import BAA.BlockObstruction

/-!
# Backward support propagation
-/

namespace BAA

section Difference

variable {F : Type*} [Field F] [DecidableEq F]

/-- A first difference can be nonzero only at an existing support coordinate
or immediately after one. -/
theorem support_diff_subset_forwardBoundary {N : ℕ} (x : Word F N) :
    support (diffLinearEquiv F N x) ⊆ forwardBoundary (support x) := by
  intro i hi
  cases N with
  | zero => exact Fin.elim0 i
  | succ n =>
      revert hi
      refine Fin.cases ?_ (fun j => ?_) i
      · intro hi
        have hx0 : x 0 ≠ 0 := by
          simpa only [diff_zero] using (mem_support _ 0).mp hi
        exact (mem_forwardBoundary 0).mpr (Or.inl ((mem_support x 0).mpr hx0))
      · intro hi
        have hdiff : x j.succ - x j.castSucc ≠ 0 := by
          simpa only [diff_succ] using (mem_support _ j.succ).mp hi
        have hside : x j.succ ≠ 0 ∨ x j.castSucc ≠ 0 := by
          by_contra h
          push_neg at h
          exact hdiff (by simp [h.1, h.2])
        apply (mem_forwardBoundary j.succ).mpr
        rcases hside with hnext | hprev
        · exact Or.inl ((mem_support x j.succ).mpr hnext)
        · exact Or.inr ⟨j.castSucc, (mem_support x j.castSucc).mpr hprev, rfl⟩

end Difference

section Monomial

variable {F : Type*} [Field F] [DecidableEq F] {N : ℕ}

theorem support_diagLinearEquiv (d : Fin N → Fˣ) (x : Word F N) :
    support (diagLinearEquiv F d x) = support x := by
  ext i
  simp [mem_support, diagLinearEquiv_apply]

theorem support_permLinearEquiv (sigma : Equiv.Perm (Fin N))
    (x : Word F N) :
    support (permLinearEquiv F sigma x) =
      permuteCoordinateSet sigma (support x) := by
  ext i
  rw [mem_support]
  simp only [permLinearEquiv_apply, permuteCoordinateSet, Finset.mem_image]
  constructor
  · intro hi
    exact ⟨sigma.symm i, (mem_support x _).mpr hi, sigma.apply_symm_apply i⟩
  · rintro ⟨j, hj, rfl⟩
    simpa using (mem_support x j).mp hj

theorem support_monomialLinearEquiv (sigma : Equiv.Perm (Fin N))
    (d : Fin N → Fˣ) (x : Word F N) :
    support (monomialLinearEquiv F sigma d x) =
      permuteCoordinateSet sigma (support x) := by
  rw [monomialLinearEquiv, LinearEquiv.trans_apply,
    support_permLinearEquiv, support_diagLinearEquiv]

theorem support_monomialLinearEquiv_symm
    (sigma : Equiv.Perm (Fin N)) (d : Fin N → Fˣ) (x : Word F N) :
    support ((monomialLinearEquiv F sigma d).symm x) =
      permuteCoordinateSet sigma.symm (support x) := by
  have h := support_monomialLinearEquiv sigma d
    ((monomialLinearEquiv F sigma d).symm x)
  rw [(monomialLinearEquiv F sigma d).apply_symm_apply] at h
  have himage := congrArg (permuteCoordinateSet sigma.symm) h
  simpa [permuteCoordinateSet, Finset.image_image] using himage.symm

end Monomial

section BAA

variable {F : Type*} [Field F] [DecidableEq F] {N : ℕ}

theorem forwardBoundary_mono {S T : Finset (Fin N)} (hST : S ⊆ T) :
    forwardBoundary S ⊆ forwardBoundary T := by
  intro j hj
  rw [mem_forwardBoundary] at hj ⊢
  rcases hj with hj | ⟨i, hi, hij⟩
  · exact Or.inl (hST hj)
  · exact Or.inr ⟨i, hST hi, hij⟩

/-- Pulling a final support backward through both BAA layers lands in the exact
two-step neighborhood, with inverse permutations dictated by the repository's
forward-transform convention. -/
theorem support_baaTransform_symm_subset
    (sigma₁ sigma₂ : Equiv.Perm (Fin N)) (d₁ d₂ : Fin N → Fˣ)
    (x : Word F N) :
    support ((baaTransform F sigma₁ sigma₂ d₁ d₂).symm x) ⊆
      twoStepBackwardNeighborhood sigma₁.symm sigma₂.symm (support x) := by
  change support ((monomialLinearEquiv F sigma₁ d₁).symm
    (diffLinearEquiv F N
      ((monomialLinearEquiv F sigma₂ d₂).symm
        (diffLinearEquiv F N x)))) ⊆ _
  rw [support_monomialLinearEquiv_symm]
  apply Finset.image_mono
  exact (support_diff_subset_forwardBoundary _).trans
    (forwardBoundary_mono (by
      rw [support_monomialLinearEquiv_symm]
      exact Finset.image_mono sigma₂.symm
        (support_diff_subset_forwardBoundary x)))

end BAA

end BAA
