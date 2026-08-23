import BAA.Basic
import Mathlib.LinearAlgebra.Pi
import Mathlib.LinearAlgebra.Prod

/-!
# Ordered linear equivalences

The accumulator is defined recursively through the head/tail decomposition of
a finite vector.  This makes invertibility structural: the difference operator
is definitionally the inverse linear equivalence.
-/

namespace BAA

section Accumulator

variable (F : Type*) [CommRing F]

/-- Add the head coordinate to every coordinate in the tail. -/
private def addHeadEquiv (n : ℕ) :
    (F × (Fin n → F)) ≃ₗ[F] (F × (Fin n → F)) where
  toFun p := (p.1, fun i => p.1 + p.2 i)
  invFun p := (p.1, fun i => p.2 i - p.1)
  left_inv p := by
    apply Prod.ext
    · rfl
    · funext i
      simp
  right_inv p := by
    apply Prod.ext
    · rfl
    · funext i
      simp
  map_add' p q := by
    apply Prod.ext
    · rfl
    · funext i
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply]
      abel
  map_smul' c p := by
    apply Prod.ext
    · rfl
    · funext i
      exact (mul_add c p.1 (p.2 i)).symm

/-- Prefix summation on an ordered length-`n` word. -/
def accumLinearEquiv : (n : ℕ) → Word F n ≃ₗ[F] Word F n
  | 0 => LinearEquiv.refl F (Word F 0)
  | n + 1 =>
      (Fin.consLinearEquiv F (fun _ : Fin (n + 1) => F)).symm ≪≫ₗ
        (LinearEquiv.refl F F).prodCongr (accumLinearEquiv n) ≪≫ₗ
        addHeadEquiv F n ≪≫ₗ
        Fin.consLinearEquiv F (fun _ : Fin (n + 1) => F)

/-- First difference, defined as the inverse of prefix summation. -/
def diffLinearEquiv (n : ℕ) : Word F n ≃ₗ[F] Word F n :=
  (accumLinearEquiv F n).symm

@[simp]
theorem diff_accum (n : ℕ) (x : Word F n) :
    diffLinearEquiv F n (accumLinearEquiv F n x) = x :=
  (accumLinearEquiv F n).symm_apply_apply x

@[simp]
theorem accum_diff (n : ℕ) (x : Word F n) :
    accumLinearEquiv F n (diffLinearEquiv F n x) = x :=
  (accumLinearEquiv F n).apply_symm_apply x

@[simp]
theorem accum_zero (n : ℕ) (x : Word F (n + 1)) :
    accumLinearEquiv F (n + 1) x 0 = x 0 := by
  simp [accumLinearEquiv, addHeadEquiv]

@[simp]
theorem accum_succ (n : ℕ) (x : Word F (n + 1)) (i : Fin n) :
    accumLinearEquiv F (n + 1) x i.succ =
      x 0 + accumLinearEquiv F n (fun j => x j.succ) i := by
  simp [accumLinearEquiv, addHeadEquiv]
  congr 2

/-- The recursive accumulator is ordinary inclusive prefix summation. -/
theorem accum_apply_eq_sum_Iic (n : ℕ) (x : Word F n) (i : Fin n) :
    accumLinearEquiv F n x i = ∑ j ∈ Finset.Iic i, x j := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
      refine Fin.cases ?_ (fun i => ?_) i
      · rw [accum_zero]
        have hset : Finset.Iic (0 : Fin (n + 1)) = {0} := by ext j; simp
        rw [hset]
        simp
      · rw [accum_succ, ih]
        have hset : Finset.Iic i.succ =
            insert 0 ((Finset.Iic i).map (Fin.succEmb n)) := by
          ext j
          refine Fin.cases ?_ (fun j => ?_) j <;> simp
        rw [hset, Finset.sum_insert]
        · rw [Finset.sum_map]
          simp
        · simp

@[simp]
theorem diff_zero (n : ℕ) (y : Word F (n + 1)) :
    diffLinearEquiv F (n + 1) y 0 = y 0 := by
  simp [diffLinearEquiv, accumLinearEquiv, addHeadEquiv]

@[simp]
theorem diff_succ (n : ℕ) (y : Word F (n + 1)) (i : Fin n) :
    diffLinearEquiv F (n + 1) y i.succ = y i.succ - y i.castSucc := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
      change diffLinearEquiv F (n + 1) (fun j => y j.succ - y 0) i =
        y i.succ - y i.castSucc
      refine Fin.cases ?_ (fun j => ?_) i
      · simp
      · rw [ih]
        simp

end Accumulator

section Monomial

variable (F : Type*) [CommRing F]

/-- Reindex coordinates by a permutation. The output at `i` is the input at
`sigma.symm i`. -/
def permLinearEquiv {n : ℕ} (sigma : Equiv.Perm (Fin n)) :
    Word F n ≃ₗ[F] Word F n where
  toFun x i := x (sigma.symm i)
  invFun x i := x (sigma i)
  left_inv x := by
    funext i
    simp
  right_inv x := by
    funext i
    simp
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

@[simp]
theorem permLinearEquiv_apply {n : ℕ} (sigma : Equiv.Perm (Fin n))
    (x : Word F n) (i : Fin n) :
    permLinearEquiv F sigma x i = x (sigma.symm i) :=
  rfl

/-- Coordinatewise multiplication by nonzero field elements. -/
def diagLinearEquiv {n : ℕ} (d : Fin n → Fˣ) :
    Word F n ≃ₗ[F] Word F n where
  toFun x i := (d i : F) * x i
  invFun x i := ((d i)⁻¹ : Fˣ) * x i
  left_inv x := by
    funext i
    simp
  right_inv x := by
    funext i
    simp
  map_add' x y := by
    funext i
    simp [mul_add]
  map_smul' c x := by
    funext i
    simp [mul_assoc, mul_comm c]

@[simp]
theorem diagLinearEquiv_apply {n : ℕ} (d : Fin n → Fˣ)
    (x : Word F n) (i : Fin n) :
    diagLinearEquiv F d x i = (d i : F) * x i :=
  rfl

/-- The paper's convention `M = Pi D`: scale first, then permute. -/
def monomialLinearEquiv {n : ℕ} (sigma : Equiv.Perm (Fin n))
    (d : Fin n → Fˣ) : Word F n ≃ₗ[F] Word F n :=
  diagLinearEquiv F d ≪≫ₗ permLinearEquiv F sigma

end Monomial

end BAA
