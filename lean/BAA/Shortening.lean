import BAA.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Defs

/-!
# Coordinate support and shortening
-/

namespace BAA

section Support

variable {F : Type*} {n : ℕ} [Semiring F]

/-- The submodule of words that vanish outside `S`. -/
def supportedCode (S : Finset (Fin n)) : Code F n where
  carrier := {x | ∀ i, i ∉ S → x i = 0}
  zero_mem' := by
    intro i _
    rfl
  add_mem' := by
    intro x y hx hy i hi
    simp [hx i hi, hy i hi]
  smul_mem' := by
    intro c x hx i hi
    simp [hx i hi]

@[simp]
theorem mem_supportedCode {S : Finset (Fin n)} {x : Word F n} :
    x ∈ supportedCode (F := F) S ↔ ∀ i, i ∉ S → x i = 0 :=
  Iff.rfl

theorem mem_supportedCode_iff_support_subset [DecidableEq F]
    {S : Finset (Fin n)} {x : Word F n} :
    x ∈ supportedCode (F := F) S ↔ support x ⊆ S := by
  constructor
  · intro hx i hi
    by_contra hiS
    exact (mem_support x i).mp hi (hx i hiS)
  · intro hx i hiS
    by_contra hxi
    exact hiS (hx ((mem_support x i).mpr hxi))

/-- The shortening of `C` to a coordinate set `S`: codewords of `C` whose
support is contained in `S`. -/
def shortening (C : Code F n) (S : Finset (Fin n)) : Code F n :=
  C ⊓ supportedCode (F := F) S

@[simp]
theorem mem_shortening {C : Code F n} {S : Finset (Fin n)} {x : Word F n} :
    x ∈ shortening C S ↔ x ∈ C ∧ ∀ i, i ∉ S → x i = 0 :=
  Iff.rfl

theorem mem_shortening_iff [DecidableEq F]
    {C : Code F n} {S : Finset (Fin n)} {x : Word F n} :
    x ∈ shortening C S ↔ x ∈ C ∧ support x ⊆ S := by
  rw [mem_shortening, and_congr_right_iff]
  intro _
  exact mem_supportedCode_iff_support_subset

end Support

section Dimension

variable {F : Type*} {n : ℕ} [Field F]

/-- Dimension of the shortening supported inside `S`. -/
noncomputable def shorteningDim (C : Code F n) (S : Finset (Fin n)) : ℕ :=
  Module.finrank F (shortening C S)

end Dimension

end BAA
