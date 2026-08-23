import BAA.OuterCode
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomial.Coeff

/-!
# Exact weight enumerators
-/

namespace BAA

open scoped Polynomial

section Weight

variable {F : Type*} [Zero F] [DecidableEq F]

theorem hammingNorm_eq_sum_indicator {n : ℕ} (x : Word F n) :
    hammingNorm x = ∑ i, if x i ≠ 0 then 1 else 0 := by
  simpa [hammingNorm] using
    (Finset.sum_boole (R := ℕ) (fun i : Fin n => x i ≠ 0) Finset.univ).symm

end Weight

section BlockWeight

variable {F : Type*} [Semiring F] [DecidableEq F]

theorem hammingNorm_blockLinearEquiv (B m : ℕ) (x : Word F (B * m)) :
    hammingNorm x = ∑ b, hammingNorm (blockLinearEquiv F B m x b) := by
  rw [hammingNorm_eq_sum_indicator]
  simp_rw [hammingNorm_eq_sum_indicator]
  rw [← Fintype.sum_prod_type']
  change (∑ i : Fin (B * m), if x i ≠ 0 then 1 else 0) =
    ∑ p : Fin B × Fin m, if x (blockCoordEquiv B m p) ≠ 0 then 1 else 0
  exact ((blockCoordEquiv B m).sum_comp
    (fun i => if x i ≠ 0 then 1 else 0)).symm

end BlockWeight

section Enumerator

variable {F : Type*} [Semiring F] [Fintype F] [DecidableEq F]

/-- The homogeneous Hamming-weight enumerator with natural coefficients. -/
noncomputable def weightEnumerator {n : ℕ} (C : Code F n) : Polynomial ℕ :=
  ∑ x : C, Polynomial.X ^ hammingNorm (x : Word F n)

theorem coeff_weightEnumerator {n w : ℕ} (C : Code F n) :
    (weightEnumerator C).coeff w =
      (Finset.univ.filter fun x : C => hammingNorm (x : Word F n) = w).card := by
  classical
  simp [weightEnumerator, Polynomial.coeff_X_pow, eq_comm]

/-- The subtype of an outer code, reindexed as one local-code word per block. -/
def outerCodeEquivBlocks {m : ℕ} (C0 : Code F m) (B : ℕ) :
    outerCode C0 B ≃ₗ[F] (Fin B → C0) :=
  (blockLinearEquiv F B m).ofSubmodule' (blockProductCode C0 B) ≪≫ₗ
    blockProductCodeEquiv C0 B

omit [Fintype F] in
theorem outerCodeEquivBlocks_weight {m : ℕ} (C0 : Code F m) (B : ℕ)
    (x : outerCode C0 B) :
    hammingNorm (x : Word F (B * m)) =
      ∑ b, hammingNorm ((outerCodeEquivBlocks C0 B x b : C0) : Word F m) := by
  rw [hammingNorm_blockLinearEquiv]
  rfl

/-- Equation B1: the repeated block code has the power of the local weight
enumerator. -/
theorem weightEnumerator_outerCode {m : ℕ} (C0 : Code F m) (B : ℕ) :
    weightEnumerator (outerCode C0 B) = (weightEnumerator C0) ^ B := by
  classical
  unfold weightEnumerator
  calc
    (∑ x : outerCode C0 B,
        Polynomial.X ^ hammingNorm (x : Word F (B * m))) =
        ∑ y : Fin B → C0,
          Polynomial.X ^ (∑ b, hammingNorm ((y b : C0) : Word F m)) := by
      apply Fintype.sum_equiv (outerCodeEquivBlocks C0 B).toEquiv
      intro x
      rw [outerCodeEquivBlocks_weight]
      congr 2
    _ = ∑ y : Fin B → C0,
          ∏ b, Polynomial.X ^ hammingNorm ((y b : C0) : Word F m) := by
      apply Fintype.sum_congr
      intro y
      simpa using
        (Finset.prod_pow_eq_pow_sum Finset.univ
          (fun b => hammingNorm ((y b : C0) : Word F m)) Polynomial.X).symm
    _ = ∏ _b : Fin B,
          ∑ c : C0, Polynomial.X ^ hammingNorm ((c : C0) : Word F m) := by
      rw [Fintype.prod_sum]
    _ = (∑ c : C0, Polynomial.X ^ hammingNorm ((c : C0) : Word F m)) ^ B := by
      simp

theorem coeff_weightEnumerator_outerCode {m : ℕ} (C0 : Code F m) (B w : ℕ) :
    (weightEnumerator (outerCode C0 B)).coeff w =
      ((weightEnumerator C0) ^ B).coeff w := by
  rw [weightEnumerator_outerCode]

end Enumerator

end BAA
