import BAA.DifferenceNetwork

namespace BAA

def DifferenceLinkage {N r : ℕ}
    (rows cols : Fin r ↪ Fin N) : Prop :=
  ∃ σ : Equiv.Perm (Fin r), ∀ i, DifferenceStep (cols i) (rows (σ i))

theorem DifferenceStep.val_bounds {N : ℕ} {source target : Fin N}
    (h : DifferenceStep source target) :
    source.val ≤ target.val ∧ target.val ≤ source.val + 1 := by
  rcases h with h | h
  · subst target
    omega
  · omega

theorem differenceLinkage_unique {N r : ℕ}
    (rows cols : Fin r ↪ Fin N)
    {σ τ : Equiv.Perm (Fin r)}
    (hσ : ∀ i, DifferenceStep (cols i) (rows (σ i)))
    (hτ : ∀ i, DifferenceStep (cols i) (rows (τ i))) : σ = τ := by
  let e : Fin r ≃ Set.range cols := Equiv.ofInjective cols cols.injective
  let route : Equiv.Perm (Fin r) → Set.range cols → Fin N :=
    fun π s => rows (π (e.symm s))
  have route_strict (π : Equiv.Perm (Fin r))
      (hπ : ∀ i, DifferenceStep (cols i) (rows (π i))) :
      StrictMono (route π) := by
    intro a b hab
    have ha := (hπ (e.symm a)).val_bounds
    have hb := (hπ (e.symm b)).val_bounds
    have hea : cols (e.symm a) = a.1 := by
      exact congrArg Subtype.val (e.apply_symm_apply a)
    have heb : cols (e.symm b) = b.1 := by
      exact congrArg Subtype.val (e.apply_symm_apply b)
    have habval : a.1.val < b.1.val := by
      exact hab
    have hsrc : (cols (e.symm a)).val < (cols (e.symm b)).val := by
      simpa [hea, heb] using habval
    have hne : route π a ≠ route π b := by
      intro h
      have hp : π (e.symm a) = π (e.symm b) := rows.injective h
      have heq : a = b := by
        apply e.symm.injective
        exact π.injective hp
      exact (ne_of_lt hab) heq
    have hle : (rows (π (e.symm a))).val ≤
        (rows (π (e.symm b))).val :=
      ha.2.trans ((Nat.succ_le_iff.mpr hsrc).trans hb.1)
    have hvalne : (rows (π (e.symm a))).val ≠
        (rows (π (e.symm b))).val := by
      intro hv
      apply hne
      exact Fin.ext hv
    exact Fin.mk_lt_mk.mpr (lt_of_le_of_ne hle hvalne)
  have route_range (π : Equiv.Perm (Fin r)) :
      Set.range (route π) = Set.range rows := by
    ext y
    constructor
    · rintro ⟨s, rfl⟩
      exact ⟨π (e.symm s), rfl⟩
    · rintro ⟨i, rfl⟩
      obtain ⟨j, hj⟩ := π.surjective i
      refine ⟨e j, ?_⟩
      simp [route, hj]
  have hroutes : route σ = route τ :=
    ((route_strict σ hσ).range_inj (route_strict τ hτ)).mp
      ((route_range σ).trans (route_range τ).symm)
  apply Equiv.ext
  intro i
  apply rows.injective
  have h := congrFun hroutes (e i)
  simpa [route] using h

/-- A square minor of the first-difference matrix is nonzero exactly when the
selected sources and targets admit vertex-disjoint difference edges. -/
theorem orderedDifferenceMatrix_minor_ne_zero_iff_linkage
    {F : Type*} [Field F] {N r : ℕ}
    (rows cols : Fin r ↪ Fin N) :
    ((orderedDifferenceMatrix F N).submatrix rows cols).det ≠ 0 ↔
      DifferenceLinkage rows cols := by
  classical
  let M := (orderedDifferenceMatrix F N).submatrix rows cols
  constructor
  · intro hdet
    rw [Matrix.det_apply'] at hdet
    by_contra hlink
    apply hdet
    apply Finset.sum_eq_zero
    intro σ hσ
    have hfail : ∃ i, ¬ DifferenceStep (cols i) (rows (σ i)) := by
      rw [DifferenceLinkage] at hlink
      push Not at hlink
      exact hlink σ
    obtain ⟨i, hi⟩ := hfail
    have hentry :
        ((orderedDifferenceMatrix F N).submatrix rows cols) (σ i) i = 0 := by
      rw [Matrix.submatrix_apply]
      exact not_ne_iff.mp
        ((orderedDifferenceMatrix_ne_zero_iff (F := F) (rows (σ i)) (cols i)).not.mpr hi)
    have hprod :
        (∏ j, ((orderedDifferenceMatrix F N).submatrix rows cols) (σ j) j) = 0 := by
      exact Finset.prod_eq_zero (Finset.mem_univ i) hentry
    rw [hprod, mul_zero]
  · rintro ⟨σ, hσ⟩
    rw [Matrix.det_apply']
    rw [Finset.sum_eq_single σ]
    · apply mul_ne_zero
      · exact ((Equiv.Perm.sign σ).isUnit.map (Int.castRingHom F)).ne_zero
      · apply Finset.prod_ne_zero_iff.mpr
        intro i hi
        change orderedDifferenceMatrix F N (rows (σ i)) (cols i) ≠ 0
        exact (orderedDifferenceMatrix_ne_zero_iff
          (F := F) (rows (σ i)) (cols i)).mpr (hσ i)
    · intro τ hτ hne
      have hfail : ∃ i, ¬ DifferenceStep (cols i) (rows (τ i)) := by
        by_contra hall
        push Not at hall
        exact hne (differenceLinkage_unique rows cols hall hσ)
      obtain ⟨i, hi⟩ := hfail
      have hentry :
          ((orderedDifferenceMatrix F N).submatrix rows cols) (τ i) i = 0 := by
        rw [Matrix.submatrix_apply]
        exact not_ne_iff.mp
          ((orderedDifferenceMatrix_ne_zero_iff (F := F)
            (rows (τ i)) (cols i)).not.mpr hi)
      have hprod :
          (∏ j, ((orderedDifferenceMatrix F N).submatrix rows cols) (τ j) j) = 0 := by
        exact Finset.prod_eq_zero (Finset.mem_univ i) hentry
      rw [hprod, mul_zero]
    · simp

end BAA
