import BAA.ConcreteBinary
import Mathlib.Data.Finset.Powerset

/-!
# The binary Reed--Muller code `RM(2,5)`

This file realizes the code by evaluating an explicit basis of Boolean
polynomials of degree at most two.  The basis is chosen to be systematic on
the sixteen Boolean points of weight at most two, which makes injectivity a
small exact finite theorem rather than a 65,536-message search.
-/

namespace BAA

/-- Supports of Boolean monomials of degree at most two in five variables. -/
abbrev RM25Index := {S : Finset (Fin 5) // S.card ≤ 2}

theorem card_rm25Index : Fintype.card RM25Index = 16 := by
  decide

/-- A fixed ordering: empty support, the five singleton supports, then the ten
two-element supports in lexicographic order. -/
def rm25OrderedIndex (j : Fin 16) : RM25Index :=
  ⟨match j.val with
    | 0 => ∅
    | 1 => {0}
    | 2 => {1}
    | 3 => {2}
    | 4 => {3}
    | 5 => {4}
    | 6 => {0, 1}
    | 7 => {0, 2}
    | 8 => {0, 3}
    | 9 => {0, 4}
    | 10 => {1, 2}
    | 11 => {1, 3}
    | 12 => {1, 4}
    | 13 => {2, 3}
    | 14 => {2, 4}
    | _ => {3, 4}, by fin_cases j <;> decide⟩

theorem rm25OrderedIndex_bijective : Function.Bijective rm25OrderedIndex := by
  decide

noncomputable def rm25IndexEquiv : Fin 16 ≃ RM25Index :=
  Equiv.ofBijective rm25OrderedIndex rm25OrderedIndex_bijective

def rm25Monomial (S : Finset (Fin 5)) (x : BinaryWord 5) : ZMod 2 :=
  ∏ i ∈ S, x i

/-- A degree-at-most-two Newton basis.  On inputs of weight at most two it is
the Kronecker basis indexed by their supports. -/
def rm25Basis (S : RM25Index) (x : BinaryWord 5) : ZMod 2 :=
  if S.1.card = 0 then
    1 + (∑ i : Fin 5, x i) +
      ∑ P : {P : Finset (Fin 5) // P.card = 2}, rm25Monomial P.1 x
  else if S.1.card = 1 then
    rm25Monomial S.1 x *
      (1 + ∑ i ∈ (Finset.univ \ S.1), x i)
  else
    rm25Monomial S.1 x

/-- Characteristic Boolean point associated with a support. -/
def rm25Point (T : RM25Index) : BinaryWord 5 :=
  fun i => if i ∈ T.1 then 1 else 0

/-- The Newton basis evaluates to the identity matrix on the sixteen points
of weight at most two. -/
theorem rm25Basis_kronecker :
    ∀ S T : RM25Index,
      rm25Basis S (rm25Point T) = if S = T then 1 else 0 := by
  decide

/-- Canonical reindexing of the 32 Boolean points by `Fin 32`. -/
noncomputable def fin32BinaryWord5Equiv : Fin 32 ≃ BinaryWord 5 :=
  BitVec.equivFin.toEquiv.symm.trans (bitVecWordEquiv 5)

/-- Evaluation of a degree-at-most-two Boolean polynomial in the Newton basis. -/
noncomputable def rm25Evaluation :
    (RM25Index → ZMod 2) →ₗ[ZMod 2] BinaryWord 32 where
  toFun a i := ∑ S : RM25Index, a S * rm25Basis S (fin32BinaryWord5Equiv i)
  map_add' a b := by
    funext i
    simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]
  map_smul' c a := by
    funext i
    simp [Finset.mul_sum, mul_assoc]

/-- The same evaluation map with its coefficients in the fixed 16-position
ordering used by the concrete encoder. -/
noncomputable def rm25OrderedEvaluation :
    BinaryWord 16 →ₗ[ZMod 2] BinaryWord 32 where
  toFun a i := ∑ j : Fin 16,
    a j * rm25Basis (rm25OrderedIndex j) (fin32BinaryWord5Equiv i)
  map_add' a b := by
    funext i
    simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]
  map_smul' c a := by
    funext i
    simp [Finset.mul_sum, mul_assoc]

theorem rm25OrderedEvaluation_at_index (a : BinaryWord 16) (j : Fin 16) :
    rm25OrderedEvaluation a
      (fin32BinaryWord5Equiv.symm (rm25Point (rm25OrderedIndex j))) = a j := by
  simp [rm25OrderedEvaluation, rm25Basis_kronecker,
    rm25OrderedIndex_bijective.1.eq_iff]

theorem rm25OrderedEvaluation_injective :
    Function.Injective rm25OrderedEvaluation := by
  intro a b hab
  funext j
  have hcoord := congrFun hab
    (fin32BinaryWord5Equiv.symm (rm25Point (rm25OrderedIndex j)))
  simpa [rm25OrderedEvaluation_at_index] using hcoord

theorem rm25Evaluation_at_index (a : RM25Index → ZMod 2) (T : RM25Index) :
    rm25Evaluation a (fin32BinaryWord5Equiv.symm (rm25Point T)) = a T := by
  simp [rm25Evaluation, rm25Basis_kronecker]

theorem rm25Evaluation_injective : Function.Injective rm25Evaluation := by
  intro a b hab
  funext T
  have hcoord := congrFun hab (fin32BinaryWord5Equiv.symm (rm25Point T))
  simpa [rm25Evaluation_at_index] using hcoord

/-- The abstract binary `RM(2,5)` evaluation code. -/
noncomputable def rm25Code : Code (ZMod 2) 32 := rm25OrderedEvaluation.range

theorem finrank_rm25Code : Module.finrank (ZMod 2) rm25Code = 16 := by
  rw [rm25Code,
    rm25OrderedEvaluation.finrank_range_of_inj rm25OrderedEvaluation_injective]
  simpa [BinaryWord, Word] using
    (Module.finrank_fin_fun (R := ZMod 2) (n := 16))

end BAA
