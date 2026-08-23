import BAA.OrderedLinear
import BAA.OuterCode

/-!
# The BAA construction
-/

namespace BAA

section

variable (F : Type*) [CommRing F]

/-- Two monomial interleavers and two prefix-sum accumulators, in forward
encoding order. -/
def baaTransform {n : ℕ}
    (sigma₁ sigma₂ : Equiv.Perm (Fin n)) (d₁ d₂ : Fin n → Fˣ) :
    Word F n ≃ₗ[F] Word F n :=
  monomialLinearEquiv F sigma₁ d₁ ≪≫ₗ
    accumLinearEquiv F n ≪≫ₗ
    monomialLinearEquiv F sigma₂ d₂ ≪≫ₗ
    accumLinearEquiv F n

/-- The chosen-block BAA code for fixed ensemble parameters. -/
def baaCode {m B : ℕ} (C0 : Code F m)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) : Code F (B * m) :=
  (outerCode C0 B).map (baaTransform F sigma₁ sigma₂ d₁ d₂).toLinearMap

end

section Dimension

variable {F : Type*} [Field F]

theorem finrank_baaCode {m B : ℕ} (C0 : Code F m)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) :
    Module.finrank F (baaCode F C0 sigma₁ sigma₂ d₁ d₂) =
      B * Module.finrank F C0 := by
  calc
    Module.finrank F (baaCode F C0 sigma₁ sigma₂ d₁ d₂) =
        Module.finrank F (outerCode C0 B) :=
      ((baaTransform F sigma₁ sigma₂ d₁ d₂).submoduleMap
        (outerCode C0 B)).finrank_eq.symm
    _ = B * Module.finrank F C0 := finrank_outerCode C0 B

end Dimension

end BAA
