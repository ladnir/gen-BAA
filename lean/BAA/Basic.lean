import Mathlib.InformationTheory.Hamming
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-!
# Basic objects for the BAA formalization

This module fixes the ambient representation of words and linear codes.  Later
modules build the ordered accumulator and the finite code ensembles on these
definitions.
-/

namespace BAA

/-- A length-`n` word over `F`, with coordinates ordered by `Fin n`. -/
abbrev Word (F : Type*) (n : ℕ) := Fin n → F

/-- A linear code is a submodule of the ambient word space. -/
abbrev Code (F : Type*) (n : ℕ) [Semiring F] := Submodule F (Word F n)

noncomputable instance instFintypeCode {F : Type*} [Semiring F] [Fintype F]
    {n : ℕ} (C : Code F n) : Fintype C := Fintype.ofFinite C

/-- The finite support of a word. -/
def support {F : Type*} {n : ℕ} [Zero F] [DecidableEq F]
    (x : Word F n) : Finset (Fin n) :=
  Finset.univ.filter fun i => x i ≠ 0

@[simp]
theorem mem_support {F : Type*} {n : ℕ} [Zero F] [DecidableEq F]
    (x : Word F n) (i : Fin n) :
    i ∈ support x ↔ x i ≠ 0 := by
  simp [support]

theorem card_support_eq_hammingNorm {F : Type*} {n : ℕ} [Zero F] [DecidableEq F]
    (x : Word F n) :
    (support x).card = hammingNorm x := by
  rfl

section MinimumDistance

variable {F : Type*} [Semiring F] [Fintype F] [DecidableEq F]

/-- The set of weights attained by nonzero codewords. -/
noncomputable def nonzeroWeights {n : ℕ} (C : Code F n) : Finset ℕ :=
  (Finset.univ.filter fun x : C => x ≠ 0).image fun x : C =>
    hammingNorm (x : Word F n)

/-- Minimum Hamming distance, with value zero for the zero code. -/
noncomputable def minimumDistance {n : ℕ} (C : Code F n) : ℕ :=
  if h : (nonzeroWeights C).Nonempty then (nonzeroWeights C).min' h else 0

theorem minimumDistance_zero_of_subsingleton {n : ℕ} (C : Code F n)
    [Subsingleton C] : minimumDistance C = 0 := by
  classical
  rw [minimumDistance, dif_neg]
  intro h
  obtain ⟨_, hw⟩ := h
  rw [nonzeroWeights] at hw
  obtain ⟨x, hx, _⟩ := Finset.mem_image.mp hw
  exact (Finset.mem_filter.mp hx).2 (Subsingleton.elim x 0)

theorem minimumDistance_le_weight {n : ℕ} (C : Code F n) (x : C)
    (hx : x ≠ 0) : minimumDistance C ≤ hammingNorm (x : Word F n) := by
  classical
  have hxmem : hammingNorm (x : Word F n) ∈ nonzeroWeights C := by
    rw [nonzeroWeights]
    exact Finset.mem_image.mpr
      ⟨x, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hx⟩, rfl⟩
  rw [minimumDistance, dif_pos ⟨_, hxmem⟩]
  exact Finset.min'_le _ _ hxmem

theorem minimumDistance_pos {n : ℕ} (C : Code F n) [Nontrivial C] :
    0 < minimumDistance C := by
  classical
  obtain ⟨x, hx⟩ : ∃ x : C, x ≠ 0 := exists_ne 0
  have hxmem : hammingNorm (x : Word F n) ∈ nonzeroWeights C := by
    rw [nonzeroWeights]
    exact Finset.mem_image.mpr
      ⟨x, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hx⟩, rfl⟩
  let S := nonzeroWeights C
  have hS : S.Nonempty := ⟨_, hxmem⟩
  rw [minimumDistance, dif_pos hS]
  apply Nat.pos_of_ne_zero
  intro hzero
  have hzeroMem : 0 ∈ S := by
    have hminMem := Finset.min'_mem S hS
    rw [hzero] at hminMem
    exact hminMem
  change 0 ∈ nonzeroWeights C at hzeroMem
  rw [nonzeroWeights] at hzeroMem
  obtain ⟨y, hy, hyweight⟩ := Finset.mem_image.mp hzeroMem
  have hyzero : (y : Word F n) = 0 :=
    hammingNorm_eq_zero.mp hyweight
  exact (Finset.mem_filter.mp hy).2 (Subtype.ext hyzero)

theorem minimumDistance_attained {n : ℕ} (C : Code F n) [Nontrivial C] :
    ∃ x : C, x ≠ 0 ∧ hammingNorm (x : Word F n) = minimumDistance C := by
  classical
  obtain ⟨x, hx⟩ : ∃ x : C, x ≠ 0 := exists_ne 0
  have hxmem : hammingNorm (x : Word F n) ∈ nonzeroWeights C := by
    rw [nonzeroWeights]
    exact Finset.mem_image.mpr
      ⟨x, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hx⟩, rfl⟩
  let S := nonzeroWeights C
  have hS : S.Nonempty := ⟨_, hxmem⟩
  have hminMem : S.min' hS ∈ S := Finset.min'_mem S hS
  change S.min' hS ∈ nonzeroWeights C at hminMem
  simp only [nonzeroWeights, Finset.mem_image] at hminMem
  obtain ⟨y, hy, hyweight⟩ := hminMem
  refine ⟨y, (Finset.mem_filter.mp hy).2, ?_⟩
  rw [minimumDistance, dif_pos hS]
  exact hyweight

end MinimumDistance

end BAA
