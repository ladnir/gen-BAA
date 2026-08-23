import BAA.GenericRank
import BAA.OrderedLinear

/-!
# Difference-network matrix

This file gives the first-difference operator its concrete network matrix.
An edge runs from a coordinate to itself or to its immediate successor.  This
is the single-path relation used by the later vertex-disjoint linkage theorem.
-/

namespace BAA

/-- A directed edge in the ordered first-difference network. -/
def DifferenceStep {N : ℕ} (source target : Fin N) : Prop :=
  target = source ∨ source.val + 1 = target.val

/-- Lower-bidiagonal matrix of the first-difference operator. -/
def orderedDifferenceMatrix (F : Type*) [Field F] (N : ℕ) :
    Matrix (Fin N) (Fin N) F := fun target source =>
  if target = source then 1
  else if source.val + 1 = target.val then -1
  else 0

/-- Its nonzero entries are exactly the directed difference-network edges. -/
theorem orderedDifferenceMatrix_ne_zero_iff
    {F : Type*} [Field F] {N : ℕ} (target source : Fin N) :
    orderedDifferenceMatrix F N target source ≠ 0 ↔
      DifferenceStep source target := by
  unfold orderedDifferenceMatrix DifferenceStep
  by_cases hEq : target = source
  · simp [hEq]
  · simp [hEq, eq_comm]

/-- The concrete lower-bidiagonal matrix acts as the recursively defined
first-difference linear equivalence. -/
theorem orderedDifferenceMatrix_mulVec
    {F : Type*} [Field F] (N : ℕ) (x : Word F N) :
    (orderedDifferenceMatrix F N).mulVec x = diffLinearEquiv F N x := by
  funext i
  cases N with
  | zero => exact Fin.elim0 i
  | succ n =>
      refine Fin.cases ?_ (fun k => ?_) i
      · simp [Matrix.mulVec, dotProduct, orderedDifferenceMatrix, diff_zero]
      · rw [diff_succ]
        simp only [Matrix.mulVec, dotProduct, orderedDifferenceMatrix,
          ite_mul, one_mul, neg_mul, zero_mul]
        have hpoint (j : Fin (n + 1)) :
            (if k.succ = j then x j
              else if j.val + 1 = k.succ.val then -x j else 0) =
              (if j = k.succ then x j else 0) +
                (if j = k.castSucc then -x j else 0) := by
          by_cases hsucc : k.succ = j
          · subst j
            have hne : k.succ ≠ k.castSucc := Fin.castSucc_lt_succ.ne'
            simp [hne]
          · by_cases hpred : j.val + 1 = k.succ.val
            · have hj : j = k.castSucc := by
                apply Fin.ext
                have hpred' : j.val + 1 = k.val + 1 := by simpa using hpred
                exact Nat.add_right_cancel hpred'
              subst j
              have hne : k.succ ≠ k.castSucc := Fin.castSucc_lt_succ.ne'
              simp [hne, Ne.symm hne]
            · have hne : j ≠ k.castSucc := by
                intro hj
                subst j
                apply hpred
                simp
              have hval : ¬j.val = k.val := by
                intro hj
                apply hpred
                simpa [hj]
              simp [hsucc, Ne.symm hsucc, hne, hval]
        rw [Finset.sum_congr rfl (fun j _ => hpoint j)]
        rw [Finset.sum_add_distrib]
        simp [sub_eq_add_neg]

end BAA
