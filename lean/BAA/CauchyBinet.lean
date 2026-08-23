import Mathlib.LinearAlgebra.Matrix.Rank

open scoped BigOperators

namespace BAA
noncomputable def rectangularDetTerm
    {R I : Type*} [CommRing R] [Fintype I] {r : ℕ}
    (M : Matrix (Fin r) I R) (N : Matrix I (Fin r) R)
    (p : Fin r → I) : R :=
  ∑ σ : Equiv.Perm (Fin r), Equiv.Perm.sign σ *
    ∏ i, M (σ i) (p i) * N (p i) i

theorem det_rectangular_mul_eq_sum_functions
    {R I : Type*} [CommRing R] [Fintype I] {r : ℕ}
    (M : Matrix (Fin r) I R) (N : Matrix I (Fin r) R) :
    (M * N).det = ∑ p : Fin r → I, rectangularDetTerm M N p := by
  classical
  simp only [rectangularDetTerm, Matrix.det_apply', Matrix.mul_apply,
    Finset.prod_univ_sum, Finset.mul_sum, Fintype.piFinset_univ]
  rw [Finset.sum_comm]

theorem rectangularDetTerm_eq_zero_of_not_injective
    {R I : Type*} [CommRing R] [Fintype I] {r : ℕ}
    (M : Matrix (Fin r) I R) (N : Matrix I (Fin r) R)
    (p : Fin r → I) (hp : ¬ Function.Injective p) :
    rectangularDetTerm M N p = 0 := by
  classical
  obtain ⟨i, j, hpij, hij⟩ : ∃ i j, p i = p j ∧ i ≠ j := by
    rw [Function.Injective] at hp
    push Not at hp
    exact hp
  unfold rectangularDetTerm
  exact Finset.sum_involution (fun σ _ => σ * Equiv.swap i j)
      (fun σ _ => by
        have hprod : (∏ x, M (σ x) (p x)) =
            ∏ x, M ((σ * Equiv.swap i j) x) (p x) :=
          Fintype.prod_equiv (Equiv.swap i j) _ _
            (by simp [Equiv.apply_swap_eq_self hpij])
        simp [hprod, Equiv.Perm.sign_swap hij, -Equiv.Perm.sign_swap',
          Finset.prod_mul_distrib])
      (fun σ _ _ => (not_congr Equiv.mul_swap_eq_iff).mpr hij)
      (fun _ _ => Finset.mem_univ _) fun σ _ =>
        Equiv.mul_swap_involutive i j σ

theorem sum_permutations_rectangularDetTerm_eq_det_mul
    {R : Type*} [CommRing R] {r : ℕ}
    (M N : Matrix (Fin r) (Fin r) R) :
    (∑ τ : Equiv.Perm (Fin r), rectangularDetTerm M N τ) = (M * N).det := by
  classical
  rw [det_rectangular_mul_eq_sum_functions]
  symm
  calc
    (∑ p : Fin r → Fin r, rectangularDetTerm M N p) =
        ∑ p : Fin r → Fin r with Function.Bijective p,
          rectangularDetTerm M N p := by
      refine (Finset.sum_subset (Finset.filter_subset _ _) ?_).symm
      intro p _ hp
      apply rectangularDetTerm_eq_zero_of_not_injective
      simpa only [Finset.mem_filter, Finset.mem_univ, true_and,
        Finite.injective_iff_bijective] using hp
    _ = ∑ τ : Equiv.Perm (Fin r), rectangularDetTerm M N τ := by
      refine Finset.sum_bij (fun p hp => Equiv.ofBijective p (Finset.mem_filter.mp hp).2)
        (fun _ _ => Finset.mem_univ _) ?_ ?_ ?_
      · intro a₁ ha₁ a₂ ha₂ heq
        exact congrArg (fun e : Equiv.Perm (Fin r) => (e : Fin r → Fin r)) heq
      · intro τ _
        refine ⟨τ, Finset.mem_filter.mpr ⟨Finset.mem_univ _, τ.bijective⟩, ?_⟩
        apply Equiv.ext
        intro i
        rfl
      · intro p hp
        rfl

abbrev InjectiveFunction (r : ℕ) (I : Type*) :=
  {p : Fin r → I // Function.Injective p}

abbrev SizedSelection (I : Type*) (r : ℕ) :=
  {S : Finset I // S.card = r}

noncomputable def injectiveFunctionRange
    {I : Type*} [Fintype I] [DecidableEq I] {r : ℕ}
    (p : InjectiveFunction r I) : SizedSelection I r := by
  classical
  exact ⟨Finset.univ.image p.1, by
    rw [Finset.card_image_of_injective _ p.2]
    simp⟩

theorem sum_rectangularDetTerm_eq_sum_injective
    {R I : Type*} [CommRing R] [Fintype I] [DecidableEq I] {r : ℕ}
    (M : Matrix (Fin r) I R) (N : Matrix I (Fin r) R) :
    (∑ p : Fin r → I, rectangularDetTerm M N p) =
      ∑ p : InjectiveFunction r I, rectangularDetTerm M N p.1 := by
  classical
  calc
    (∑ p : Fin r → I, rectangularDetTerm M N p) =
        ∑ p : Fin r → I with Function.Injective p,
          rectangularDetTerm M N p := by
      refine (Finset.sum_subset (Finset.filter_subset _ _) ?_).symm
      intro p _ hp
      apply rectangularDetTerm_eq_zero_of_not_injective
      simpa only [Finset.mem_filter, Finset.mem_univ, true_and] using hp
    _ = _ := by
      apply Finset.sum_subtype
      intro p
      simp

noncomputable def selectionEquiv
    {I : Type*} [Fintype I] [DecidableEq I] {r : ℕ}
    (S : SizedSelection I r) : Fin r ≃ S.1 :=
  (S.1.equivFinOfCardEq S.2).symm

noncomputable def selectionEmbedding
    {I : Type*} [Fintype I] [DecidableEq I] {r : ℕ}
    (S : SizedSelection I r) : Fin r ↪ I :=
  (selectionEquiv S).toEmbedding.trans (Function.Embedding.subtype _)

noncomputable def injectionFiberEquivPerm
    {I : Type*} [Fintype I] [DecidableEq I] {r : ℕ}
    (S : SizedSelection I r) :
    {p : InjectiveFunction r I // injectiveFunctionRange p = S} ≃
      Equiv.Perm (Fin r) := by
  classical
  let e : Fin r ≃ S.1 := selectionEquiv S
  let toS : {p : InjectiveFunction r I // injectiveFunctionRange p = S} →
      Fin r ≃ S.1 := fun p =>
    Equiv.ofBijective
      (fun i => ⟨p.1.1 i, by
        have hi : p.1.1 i ∈ (injectiveFunctionRange p.1).1 := by
          exact Finset.mem_image.mpr ⟨i, Finset.mem_univ _, rfl⟩
        rw [p.2] at hi
        exact hi⟩)
      ⟨fun _ _ h => p.1.2 (congrArg Subtype.val h), fun y => by
        have hy : y.1 ∈ (injectiveFunctionRange p.1).1 := by
          rw [p.2]
          exact y.2
        obtain ⟨i, -, hi⟩ := Finset.mem_image.mp hy
        exact ⟨i, Subtype.ext hi⟩⟩
  exact
    { toFun := fun p => (toS p).trans e.symm
      invFun := fun q => ⟨⟨fun i => (e (q i)).1, fun _ _ h => by
          apply q.injective
          apply e.injective
          exact Subtype.ext h⟩, by
        apply Subtype.ext
        ext x
        constructor
        · intro hx
          obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hx
          exact (e (q i)).2
        · intro hx
          obtain ⟨i, hi⟩ := e.surjective ⟨x, hx⟩
          refine Finset.mem_image.mpr ⟨q.symm i, Finset.mem_univ _, ?_⟩
          simpa using congrArg Subtype.val hi⟩
      left_inv := fun p => by
        apply Subtype.ext
        apply Subtype.ext
        funext i
        simp [toS, e]
      right_inv := fun q => by
        apply Equiv.ext
        intro i
        simp [toS, e] }

theorem sum_injectionFiber_rectangularDetTerm
    {R I : Type*} [CommRing R] [Fintype I] [DecidableEq I] {r : ℕ}
    (M : Matrix (Fin r) I R) (N : Matrix I (Fin r) R)
    (S : SizedSelection I r) :
    (∑ p : {p : InjectiveFunction r I // injectiveFunctionRange p = S},
      rectangularDetTerm M N p.1.1) =
      (M.submatrix id (selectionEmbedding S)).det *
        (N.submatrix (selectionEmbedding S) id).det := by
  classical
  let E := injectionFiberEquivPerm S
  calc
    _ = ∑ q : Equiv.Perm (Fin r),
        rectangularDetTerm M N (fun i => selectionEmbedding S (q i)) := by
      apply Fintype.sum_equiv E
      intro p
      congr 1
      funext i
      simp [E, injectionFiberEquivPerm, selectionEmbedding, selectionEquiv]
    _ = ∑ q : Equiv.Perm (Fin r),
        rectangularDetTerm
          (M.submatrix id (selectionEmbedding S))
          (N.submatrix (selectionEmbedding S) id) q := by
      apply Fintype.sum_congr
      intro q
      rfl
    _ = ((M.submatrix id (selectionEmbedding S)) *
          (N.submatrix (selectionEmbedding S) id)).det :=
      sum_permutations_rectangularDetTerm_eq_det_mul _ _
    _ = _ := Matrix.det_mul _ _

/-- Rectangular Cauchy--Binet, indexed by cardinality-`r` coordinate
selections and an arbitrary but fixed enumeration of each selection. -/
theorem rectangular_cauchyBinet
    {R I : Type*} [CommRing R] [Fintype I] [DecidableEq I] {r : ℕ}
    (M : Matrix (Fin r) I R) (N : Matrix I (Fin r) R) :
    (M * N).det =
      ∑ S : SizedSelection I r,
        (M.submatrix id (selectionEmbedding S)).det *
          (N.submatrix (selectionEmbedding S) id).det := by
  classical
  rw [det_rectangular_mul_eq_sum_functions,
    sum_rectangularDetTerm_eq_sum_injective]
  rw [← Fintype.sum_fiberwise injectiveFunctionRange
    (fun p : InjectiveFunction r I => rectangularDetTerm M N p.1)]
  apply Fintype.sum_congr
  intro S
  exact sum_injectionFiber_rectangularDetTerm M N S

end BAA
