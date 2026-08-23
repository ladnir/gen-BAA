import BAA.BinaryRuns
import BAA.BlockObstruction

/-!
# Exact binary forward-boundary size

For a Boolean word read from a preceding zero, adjoining the successor of
every support coordinate adds exactly the falling transitions.  Rising and
falling transitions alternate, so the number of falling transitions is half
the total transition count, rounded down.
-/

namespace BAA

/-- Coordinatewise indicator of membership in a support or immediately after
that support, with `previous` supplying the bit before the list. -/
def boolListForwardBoundary (previous : Bool) : List Bool → List Bool
  | [] => []
  | b :: bits => (b || previous) :: boolListForwardBoundary b bits

@[simp]
theorem length_boolListForwardBoundary (previous : Bool) (bits : List Bool) :
    (boolListForwardBoundary previous bits).length = bits.length := by
  induction bits generalizing previous with
  | nil => rfl
  | cons b bits ih => simp [boolListForwardBoundary, ih]

/-- The Boolean counting identity behind the endpoint correction.  The two
statements are proved together because a suffix can be entered from either
state. -/
theorem count_true_boolListForwardBoundary (bits : List Bool) :
    (boolListForwardBoundary false bits).count true =
        bits.count true + (boolListDifferences false bits).count true / 2 ∧
      (boolListForwardBoundary true bits).count true =
        bits.count true + ((boolListDifferences true bits).count true + 1) / 2 := by
  induction bits with
  | nil => simp [boolListForwardBoundary, boolListDifferences]
  | cons b bits ih =>
      cases b <;>
        simp [boolListForwardBoundary, boolListDifferences, ih] <;> omega

/-- Boolean vector whose support is the forward boundary of the input
support. -/
def boolVectorForwardBoundary :
    (previous : Bool) → {N : ℕ} → List.Vector Bool N → List.Vector Bool N
  | _, 0, _ => List.Vector.nil
  | previous, _ + 1, v => List.Vector.ofFn fun i =>
      Fin.cases (v.get 0 || previous)
        (fun j => v.get j.succ || v.get j.castSucc) i

theorem boolVectorForwardBoundary_cons {N : ℕ} (previous b : Bool)
    (v : List.Vector Bool N) :
    boolVectorForwardBoundary previous (List.Vector.cons b v) =
      List.Vector.cons (b || previous) (boolVectorForwardBoundary b v) := by
  apply List.Vector.ext
  intro i
  refine Fin.cases ?_ (fun j => ?_) i
  · simp [boolVectorForwardBoundary]
  · cases N with
    | zero => exact Fin.elim0 j
    | succ n =>
        refine Fin.cases ?_ (fun k => ?_) j
        · simp only [boolVectorForwardBoundary, List.Vector.get_ofFn,
            List.Vector.get_cons_succ, List.Vector.get_cons_zero,
            Fin.cases_zero, Fin.cases_succ]
          rw [show Fin.castSucc (0 : Fin (n + 1)) =
              (0 : Fin (n + 2)) by rfl,
            List.Vector.get_cons_zero]
        · simp only [boolVectorForwardBoundary, List.Vector.get_ofFn,
            List.Vector.get_cons_succ, List.Vector.get_cons_zero,
            Fin.cases_zero, Fin.cases_succ]
          rw [show k.succ.castSucc = (k.castSucc).succ by rfl,
            List.Vector.get_cons_succ]

theorem toList_boolVectorForwardBoundary {N : ℕ} (previous : Bool)
    (v : List.Vector Bool N) :
    (boolVectorForwardBoundary previous v).toList =
      boolListForwardBoundary previous v.toList := by
  induction v generalizing previous with
  | nil => simp [boolVectorForwardBoundary, boolListForwardBoundary]
  | @cons n b v ih =>
      rw [boolVectorForwardBoundary_cons, List.Vector.toList_cons, ih]
      rfl

theorem support_boolVectorForwardBoundary {N : ℕ} (v : List.Vector Bool N) :
    support (boolVectorToBinaryWord (boolVectorForwardBoundary false v)) =
      forwardBoundary (support (boolVectorToBinaryWord v)) := by
  ext i
  cases N with
  | zero => exact Fin.elim0 i
  | succ n =>
      refine Fin.cases ?_ (fun j => ?_) i
      · simp [boolVectorForwardBoundary, boolVectorToBinaryWord,
          mem_support, mem_forwardBoundary]
      · have hex :
            (∃ k : Fin (n + 1), v.get k = true ∧ k.val = j.val) ↔
              v.get j.castSucc = true := by
          constructor
          · rintro ⟨k, hk, hval⟩
            have hkj : k = j.castSucc := by
              apply Fin.ext
              exact hval
            simpa [hkj] using hk
          · intro hj
            exact ⟨j.castSucc, hj, rfl⟩
        cases hnext : v.get j.succ <;>
          simp [mem_support, mem_forwardBoundary, boolVectorForwardBoundary,
            boolVectorToBinaryWord, hex, hnext]

/-- For every binary word, the exact forward-boundary size is its weight plus
half (rounded down) of its first-difference weight. -/
theorem card_forwardBoundary_support_binary {N : ℕ} (x : BinaryWord N) :
    (forwardBoundary (support x)).card =
      hammingNorm x + hammingNorm (diffLinearEquiv (ZMod 2) N x) / 2 := by
  let v := binaryWordToBoolVector x
  have hx : boolVectorToBinaryWord v = x :=
    boolVectorToBinaryWord_binaryWordToBoolVector x
  calc
    (forwardBoundary (support x)).card =
        hammingNorm (boolVectorToBinaryWord
          (boolVectorForwardBoundary false v)) := by
      rw [← card_support_eq_hammingNorm,
        support_boolVectorForwardBoundary, hx]
    _ = (boolVectorForwardBoundary false v).toList.count true :=
      hammingNorm_boolVectorToBinaryWord _
    _ = v.toList.count true +
        (boolListDifferences false v.toList).count true / 2 := by
      rw [toList_boolVectorForwardBoundary]
      exact (count_true_boolListForwardBoundary v.toList).1
    _ = hammingNorm x +
        hammingNorm (diffLinearEquiv (ZMod 2) N x) / 2 := by
      have hweight : v.toList.count true = hammingNorm x := by
        rw [← hammingNorm_boolVectorToBinaryWord v, hx]
      have hdiff : (boolListDifferences false v.toList).count true =
          hammingNorm (diffLinearEquiv (ZMod 2) N x) := by
        rw [← toList_boolVectorDifferences,
          ← hammingNorm_diff_boolVectorToBinaryWord v, hx]
      rw [hweight, hdiff]

end BAA
