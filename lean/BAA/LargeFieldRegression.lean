import BAA.NonbinaryTrace
import BAA.Projective
import Mathlib.Algebra.Field.ZMod
import Mathlib.FieldTheory.Finite.GaloisField

namespace BAA

abbrev nonbinaryPlacementFormulaHoldsBelow (limit : ℕ) : Prop :=
  ∀ N : Fin limit,
    ∀ s : Fin (N.val + 1),
      ∀ h : Fin (N.val + 1),
        ∀ a : Fin (s.val + 1),
          0 < s.val → a.val < s.val → 0 < h.val →
            Fintype.card (NonbinaryAccumulatorPlacements
              N.val s.val h.val a.val) =
              Nat.choose (s.val - a.val) a.val *
                Nat.choose (N.val - h.val) a.val *
                Nat.choose (h.val - 1) (s.val - a.val - 1)

/-- Kernel-reduced exhaustive equation-(1) regression through length seven. -/
theorem nonbinaryPlacementFormula_regression_N_le_7 :
    nonbinaryPlacementFormulaHoldsBelow 8 := by
  set_option maxRecDepth 100000 in
    decide

def cancellationSolutionCount (F : Type*) [Field F] [Fintype F] [DecidableEq F]
    (x y : F) : ℕ :=
  (Finset.univ.filter fun d : F => d ≠ 0 ∧ x * d + y = 0).card

local instance prime2 : Fact (Nat.Prime 2) := ⟨by decide⟩
local instance prime3 : Fact (Nat.Prime 3) := ⟨by decide⟩
local instance prime5 : Fact (Nat.Prime 5) := ⟨by decide⟩

theorem cancellationSolutionCount_zmod3 :
    ∀ x y : ZMod 3, x ≠ 0 → y ≠ 0 →
      cancellationSolutionCount (ZMod 3) x y = 1 := by
  decide

theorem cancellationSolutionCount_zmod5 :
    ∀ x y : ZMod 5, x ≠ 0 → y ≠ 0 →
      cancellationSolutionCount (ZMod 5) x y = 1 := by
  decide

/-- Small-extension-field instance of the exact fresh-multiplier cancellation
theorem. -/
theorem galoisField4_existsUnique_cancellation
    (x y : GaloisField 2 2) (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃! d : (GaloisField 2 2)ˣ, x * (d : GaloisField 2 2) + y = 0 :=
  by
    letI : Fintype (GaloisField 2 2) := Fintype.ofFinite _
    letI : DecidableEq (GaloisField 2 2) := Classical.decEq _
    exact existsUnique_unit_mul_add_eq_zero
      (F := GaloisField 2 2) x y hx hy

end BAA
