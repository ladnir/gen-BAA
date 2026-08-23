import BAA.Shortening
import Mathlib.Algebra.Module.Submodule.Equiv
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Pi
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Repeated block outer codes
-/

namespace BAA

/-- The fixed convention that flattens `(block, local coordinate)` into a
single coordinate. -/
def blockCoordEquiv (B m : ℕ) : Fin B × Fin m ≃ Fin (B * m) :=
  finProdFinEquiv

section Reindex

variable (F : Type*) [Semiring F]

/-- Reindex a flat word as a block-indexed family of local words. -/
def blockLinearEquiv (B m : ℕ) :
    Word F (B * m) ≃ₗ[F] (Fin B → Word F m) where
  toFun x b j := x (blockCoordEquiv B m (b, j))
  invFun x i :=
    x ((blockCoordEquiv B m).symm i).1 ((blockCoordEquiv B m).symm i).2
  left_inv x := by
    funext i
    exact congrArg x ((blockCoordEquiv B m).apply_symm_apply i)
  right_inv x := by
    funext b j
    simp [blockCoordEquiv]
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

@[simp]
theorem blockLinearEquiv_apply (B m : ℕ) (x : Word F (B * m))
    (b : Fin B) (j : Fin m) :
    blockLinearEquiv F B m x b j = x (blockCoordEquiv B m (b, j)) :=
  rfl

end Reindex

section Outer

variable {F : Type*} [Semiring F]

/-- The blockwise product of copies of `C0`, before flattening coordinates. -/
def blockProductCode {m : ℕ} (C0 : Code F m) (B : ℕ) :
    Submodule F (Fin B → Word F m) :=
  Submodule.pi Set.univ fun _ => C0

@[simp]
theorem mem_blockProductCode {m B : ℕ} {C0 : Code F m}
    {x : Fin B → Word F m} :
    x ∈ blockProductCode C0 B ↔ ∀ b, x b ∈ C0 := by
  simp [blockProductCode]

/-- `B` repeated copies of the local code, in flat coordinate order. -/
def outerCode {m : ℕ} (C0 : Code F m) (B : ℕ) : Code F (B * m) :=
  (blockProductCode C0 B).comap (blockLinearEquiv F B m).toLinearMap

@[simp]
theorem mem_outerCode {m B : ℕ} {C0 : Code F m} {x : Word F (B * m)} :
    x ∈ outerCode C0 B ↔ ∀ b, (blockLinearEquiv F B m x) b ∈ C0 := by
  simp [outerCode]

/-- A block product code is linearly equivalent to a function into the local
code subtype. -/
def blockProductCodeEquiv {m : ℕ} (C0 : Code F m) (B : ℕ) :
    blockProductCode C0 B ≃ₗ[F] (Fin B → C0) where
  toFun x b := ⟨x.1 b, (mem_blockProductCode.mp x.2) b⟩
  invFun x := ⟨fun b => (x b : Word F m), mem_blockProductCode.mpr fun b => (x b).2⟩
  left_inv x := by
    apply Subtype.ext
    rfl
  right_inv x := by
    funext b
    apply Subtype.ext
    rfl
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

end Outer

section BlockShortening

variable {F : Type*} [Semiring F]

/-- Coordinates of `S` lying in block `b`, expressed in local indexing. -/
def localCoordinateSet {B m : ℕ} (S : Finset (Fin (B * m))) (b : Fin B) :
    Finset (Fin m) :=
  Finset.univ.filter fun j => blockCoordEquiv B m (b, j) ∈ S

@[simp]
theorem mem_localCoordinateSet {B m : ℕ} (S : Finset (Fin (B * m)))
    (b : Fin B) (j : Fin m) :
    j ∈ localCoordinateSet S b ↔ blockCoordEquiv B m (b, j) ∈ S := by
  simp [localCoordinateSet]

/-- Global support size is the sum of its local block support sizes. -/
theorem card_eq_sum_card_localCoordinateSet {B m : ℕ}
    (S : Finset (Fin (B * m))) :
    S.card = ∑ b : Fin B, (localCoordinateSet S b).card := by
  classical
  have hglobal : S.card =
      ∑ i : Fin (B * m), if i ∈ S then 1 else 0 := by
    simpa using (Finset.card_eq_sum_ite
      (s := S) (t := Finset.univ) (Finset.subset_univ S))
  rw [hglobal, ← (blockCoordEquiv B m).sum_comp, Fintype.sum_prod_type]
  apply Fintype.sum_congr
  intro b
  have hlocal : (localCoordinateSet S b).card =
      ∑ j : Fin m, if blockCoordEquiv B m (b, j) ∈ S then 1 else 0 := by
    simpa [localCoordinateSet] using (Finset.card_eq_sum_ite
      (s := localCoordinateSet S b) (t := Finset.univ)
      (Finset.subset_univ (localCoordinateSet S b)))
  exact hlocal.symm

/-- Blocks touched by the coordinate set `S`. -/
def activeBlockSet {B m : ℕ} (S : Finset (Fin (B * m))) : Finset (Fin B) :=
  Finset.univ.filter fun b => (localCoordinateSet S b).Nonempty

@[simp]
theorem mem_activeBlockSet {B m : ℕ} (S : Finset (Fin (B * m))) (b : Fin B) :
    b ∈ activeBlockSet S ↔ (localCoordinateSet S b).Nonempty := by
  simp [activeBlockSet]

/-- A global block-code shortening is the product of the corresponding local
shortenings. -/
def shorteningOuterCodeEquiv {B m : ℕ} (C0 : Code F m)
    (S : Finset (Fin (B * m))) :
    shortening (outerCode C0 B) S ≃ₗ[F]
      (b : Fin B) → shortening C0 (localCoordinateSet S b) where
  toFun x b :=
    ⟨blockLinearEquiv F B m x.1 b,
      ⟨(mem_outerCode.mp x.2.1) b, fun j hj =>
        x.2.2 (blockCoordEquiv B m (b, j)) (by simpa using hj)⟩⟩
  invFun x :=
    ⟨(blockLinearEquiv F B m).symm fun b => (x b).1,
      ⟨mem_outerCode.mpr fun b => by
        change blockLinearEquiv F B m
          ((blockLinearEquiv F B m).symm fun b => (x b).1) b ∈ C0
        rw [congrFun ((blockLinearEquiv F B m).apply_symm_apply
          (fun b => (x b).1)) b]
        exact (x b).2.1, fun i hi => by
        let p := (blockCoordEquiv B m).symm i
        have hp : blockCoordEquiv B m p = i :=
          (blockCoordEquiv B m).apply_symm_apply i
        have hlocal : p.2 ∉ localCoordinateSet S p.1 := by
          simpa [hp] using hi
        have hz := (x p.1).2.2 p.2 hlocal
        change (fun b => (x b).1) p.1 p.2 = 0
        exact hz⟩⟩
  left_inv x := by
    apply Subtype.ext
    exact (blockLinearEquiv F B m).symm_apply_apply x.1
  right_inv x := by
    funext b
    apply Subtype.ext
    exact congrFun ((blockLinearEquiv F B m).apply_symm_apply
      (fun b => (x b).1)) b
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

end BlockShortening

section Dimension

variable {F : Type*} [Field F]

theorem finrank_outerCode {m : ℕ} (C0 : Code F m) (B : ℕ) :
    Module.finrank F (outerCode C0 B) = B * Module.finrank F C0 := by
  change Module.finrank F
    ((blockProductCode C0 B).comap (blockLinearEquiv F B m).toLinearMap) = _
  rw [LinearEquiv.finrank_eq
    ((blockLinearEquiv F B m).ofSubmodule' (blockProductCode C0 B))]
  rw [LinearEquiv.finrank_eq (blockProductCodeEquiv C0 B)]
  rw [Module.finrank_pi_fintype]
  simp

/-- Shortening dimension of a block product is the sum of the local
shortening dimensions. -/
theorem shorteningDim_outerCode {B m : ℕ} (C0 : Code F m)
    (S : Finset (Fin (B * m))) :
    shorteningDim (outerCode C0 B) S =
      ∑ b : Fin B, shorteningDim C0 (localCoordinateSet S b) := by
  rw [shorteningDim,
    (shorteningOuterCodeEquiv C0 S).finrank_eq,
    Module.finrank_pi_fintype]
  rfl

end Dimension

end BAA
