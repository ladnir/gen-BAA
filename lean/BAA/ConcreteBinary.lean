import BAA.BinaryAccumulator
import BAA.WeightEnumerator
import Mathlib.Data.BitVec
import Mathlib.Data.Fintype.EquivFin

/-!
# Concrete binary words

Fixed local-code certificates use `BitVec` for their executable inner loops.
This module supplies the proved bridge to the abstract `ZMod 2` word model.
Bit index zero is coordinate zero, matching the appendix's least-significant-
bit-first convention.
-/

namespace BAA

noncomputable instance instFintypeBitVec (n : ℕ) : Fintype (BitVec n) :=
  Fintype.ofEquiv (Fin (2 ^ n)) BitVec.equivFin.toEquiv.symm

/-- Interpret a bit vector as an abstract binary word, least-significant bit
first. -/
def bitVecToWord {n : ℕ} (x : BitVec n) : BinaryWord n :=
  fun i => if x.getLsb i then 1 else 0

theorem bitVecToWord_injective {n : ℕ} :
    Function.Injective (@bitVecToWord n) := by
  intro x y hxy
  rw [BitVec.eq_of_getLsbD_eq_iff]
  intro i hi
  have hcoord := congrFun hxy ⟨i, hi⟩
  change (if x.getLsbD i then (1 : ZMod 2) else 0) =
    (if y.getLsbD i then (1 : ZMod 2) else 0) at hcoord
  revert hcoord
  cases x.getLsbD i <;> cases y.getLsbD i <;> simp

/-- The abstract/concrete representation equivalence.  Its forward direction
is executable; certificates use this direction in their hot loops. -/
noncomputable def bitVecWordEquiv (n : ℕ) : BitVec n ≃ BinaryWord n :=
  Equiv.ofBijective bitVecToWord <|
    (Fintype.bijective_iff_injective_and_card bitVecToWord).2 <| by
      constructor
      · exact bitVecToWord_injective
      · rw [Fintype.card_congr BitVec.equivFin.toEquiv]
        simp [BinaryWord, Word]

/-- Convert an abstract binary word back to a bit vector. -/
noncomputable def wordToBitVec {n : ℕ} : BinaryWord n → BitVec n :=
  (bitVecWordEquiv n).symm

@[simp]
theorem bitVecToWord_wordToBitVec {n : ℕ} (x : BinaryWord n) :
    bitVecToWord (wordToBitVec x) = x :=
  (bitVecWordEquiv n).apply_symm_apply x

@[simp]
theorem wordToBitVec_bitVecToWord {n : ℕ} (x : BitVec n) :
    wordToBitVec (bitVecToWord x) = x :=
  (bitVecWordEquiv n).symm_apply_apply x

/-- Machine-level population count, returned as a natural. -/
def bitVecWeight {n : ℕ} (x : BitVec n) : ℕ := x.cpop.toNat

private theorem sum_range_getLsbD_toNat {n : ℕ} (x : BitVec n) (k : ℕ) :
    (∑ i ∈ Finset.range k, (x.getLsbD i).toNat) = x.cpopNatRec k 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [Finset.sum_range_succ, BitVec.cpopNatRec_succ, ih]
      rw [BitVec.cpopNatRec_eq (x := x) (n := k)
        (acc := 0 + (x.getLsbD k).toNat)]
      omega

/-- The executable population count agrees with abstract Hamming weight. -/
theorem hammingNorm_bitVecToWord {n : ℕ} (x : BitVec n) :
    hammingNorm (bitVecToWord x) = bitVecWeight x := by
  rw [hammingNorm_eq_sum_indicator]
  calc
    (∑ i : Fin n, if bitVecToWord x i ≠ 0 then 1 else 0) =
        ∑ i : Fin n, (x.getLsb i).toNat := by
      apply Fintype.sum_congr
      intro i
      change (if (if x.getLsb i then (1 : ZMod 2) else 0) ≠ 0 then 1 else 0) =
        (x.getLsb i).toNat
      cases x.getLsb i <;> simp
    _ = ∑ i ∈ Finset.range n, (x.getLsbD i).toNat := by
      rw [Finset.sum_fin_eq_sum_range]
      apply Finset.sum_congr rfl
      intro i hi
      have hin : i < n := Finset.mem_range.mp hi
      simp only [hin, dite_true]
      change (x.toNat.testBit i).toNat = (x.toNat.testBit i).toNat
      rfl
    _ = bitVecWeight x := by
      rw [sum_range_getLsbD_toNat, bitVecWeight, BitVec.toNat_cpop]

end BAA
