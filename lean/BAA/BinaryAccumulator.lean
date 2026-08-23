import BAA.OrderedLinear
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Rat.Defs
import Mathlib.Logic.Equiv.Fintype

/-!
# Binary accumulator input/output enumerator

This module fixes executable versions of the two sides of B2.  The general
counting theorem will be proved from a run decomposition; the small exhaustive
certificate guards the statement and all endpoint conventions while that proof
is developed.
-/

namespace BAA

abbrev BinaryWord (n : ℕ) := Word (ZMod 2) n

/-- A binary word is equivalent to its support. -/
def binaryWordSupportEquiv (N : ℕ) : BinaryWord N ≃ Finset (Fin N) where
  toFun := support
  invFun S i := if i ∈ S then 1 else 0
  left_inv x := by
    funext i
    by_cases hi : x i = 0
    · simp [support, hi]
    · have hval0 : (x i).val ≠ 0 := (ZMod.val_eq_zero (x i)).not.mpr hi
      have hval1 : (x i).val = 1 := by
        have hlt := (x i).val_lt
        omega
      have hi1 : x i = 1 := (ZMod.val_eq_one (by omega) (x i)).mp hval1
      simp [support, hi1]
  right_inv S := by
    ext i
    simp [support]

abbrev BinaryWeightSlice (N w : ℕ) :=
  {x : BinaryWord N // hammingNorm x = w}

/-- Restrict the support equivalence to a Hamming slice. -/
def binaryWeightSliceSupportEquiv (N w : ℕ) :
    BinaryWeightSlice N w ≃ {S : Finset (Fin N) // S.card = w} where
  toFun x := ⟨support x.1, by simpa [card_support_eq_hammingNorm] using x.2⟩
  invFun S := ⟨(binaryWordSupportEquiv N).symm S.1, by
    rw [← card_support_eq_hammingNorm]
    change ((binaryWordSupportEquiv N) ((binaryWordSupportEquiv N).symm S.1)).card = w
    rw [(binaryWordSupportEquiv N).apply_symm_apply S.1]
    exact S.2⟩
  left_inv x := by
    apply Subtype.ext
    exact (binaryWordSupportEquiv N).symm_apply_apply x.1
  right_inv S := by
    apply Subtype.ext
    exact (binaryWordSupportEquiv N).apply_symm_apply S.1

theorem card_binaryWeightSlice (N w : ℕ) :
    Fintype.card (BinaryWeightSlice N w) = Nat.choose N w := by
  rw [Fintype.card_congr (binaryWeightSliceSupportEquiv N w)]
  simp

theorem support_permBinaryWord {N : ℕ} (sigma : Equiv.Perm (Fin N))
    (x : BinaryWord N) :
    support (permLinearEquiv (ZMod 2) sigma x) =
      (support x).map sigma.toEmbedding := by
  ext i
  simp [support, permLinearEquiv_apply]

theorem hammingNorm_permBinaryWord {N : ℕ} (sigma : Equiv.Perm (Fin N))
    (x : BinaryWord N) :
    hammingNorm (permLinearEquiv (ZMod 2) sigma x) = hammingNorm x := by
  rw [← card_support_eq_hammingNorm, support_permBinaryWord, Finset.card_map,
    card_support_eq_hammingNorm]

/-- The permutation action is transitive on every binary Hamming slice. -/
theorem exists_permBinaryWord_of_weight_eq {N : ℕ} {x y : BinaryWord N}
    (hxy : hammingNorm x = hammingNorm y) :
    ∃ sigma : Equiv.Perm (Fin N), permLinearEquiv (ZMod 2) sigma x = y := by
  have hcard : (support x).card = (support y).card := by
    simpa [card_support_eq_hammingNorm] using hxy
  obtain ⟨sigma, hsigma⟩ :=
    Equiv.Perm.exists_map_finset_eq (support x) (support y) hcard
  refine ⟨sigma, (binaryWordSupportEquiv N).injective ?_⟩
  change support (permLinearEquiv (ZMod 2) sigma x) = support y
  rw [support_permBinaryWord, hsigma]

/-- The literal number of binary inputs of weight `w` whose accumulated output
has weight `h`. -/
def binaryAccumulatorCount (N w h : ℕ) : ℕ :=
  (Finset.univ.filter fun x : BinaryWord N =>
    hammingNorm x = w ∧
      hammingNorm (accumLinearEquiv (ZMod 2) N x) = h).card

/-- The equivalent output-side count: output weight is `h` and the first
difference has weight `w`. -/
def binaryAccumulatorOutputCount (N w h : ℕ) : ℕ :=
  (Finset.univ.filter fun y : BinaryWord N =>
    hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧ hammingNorm y = h).card

def binaryAccumulatorEventEquiv (N w h : ℕ) :
    {x : BinaryWord N //
      hammingNorm x = w ∧ hammingNorm (accumLinearEquiv (ZMod 2) N x) = h} ≃
    {y : BinaryWord N //
      hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧ hammingNorm y = h} where
  toFun x := ⟨accumLinearEquiv (ZMod 2) N x.1, by
    rw [diff_accum]
    exact x.2⟩
  invFun y := ⟨diffLinearEquiv (ZMod 2) N y.1, by
    rw [accum_diff]
    exact y.2⟩
  left_inv x := by
    apply Subtype.ext
    exact diff_accum (ZMod 2) N x.1
  right_inv y := by
    apply Subtype.ext
    exact accum_diff (ZMod 2) N y.1

theorem binaryAccumulatorCount_eq_outputCount (N w h : ℕ) :
    binaryAccumulatorCount N w h = binaryAccumulatorOutputCount N w h := by
  classical
  unfold binaryAccumulatorCount binaryAccumulatorOutputCount
  rw [← Fintype.card_of_subtype
      (p := fun x : BinaryWord N =>
        hammingNorm x = w ∧ hammingNorm (accumLinearEquiv (ZMod 2) N x) = h)
      (Finset.univ.filter fun x : BinaryWord N =>
        hammingNorm x = w ∧ hammingNorm (accumLinearEquiv (ZMod 2) N x) = h)
      (by simp),
    ← Fintype.card_of_subtype
      (p := fun y : BinaryWord N =>
        hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧ hammingNorm y = h)
      (Finset.univ.filter fun y : BinaryWord N =>
        hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧ hammingNorm y = h)
      (by simp)]
  exact Fintype.card_congr (binaryAccumulatorEventEquiv N w h)

/-- The closed form on the right side of B2. Natural subtraction makes the
expression total; the corrected B2 statement assumes `1 <= w` and `1 <= h`. -/
def binaryAccumulatorFormula (N w h : ℕ) : ℕ :=
  Nat.choose (h - 1) ((w + 1) / 2 - 1) *
    Nat.choose (N - h) (w / 2)

/-- B3's exact conditional probability, expressed as a finite cardinality
ratio. The denominator is the size of the weight-`w` slice by
`card_binaryWeightSlice`. -/
def binaryAccumulatorProbability (N w h : ℕ) : ℚ :=
  (binaryAccumulatorCount N w h : ℚ) / Nat.choose N w

theorem binaryAccumulatorProbability_eq_formula {N w h : ℕ}
    (hB2 : binaryAccumulatorCount N w h = binaryAccumulatorFormula N w h) :
    binaryAccumulatorProbability N w h =
      (binaryAccumulatorFormula N w h : ℚ) / Nat.choose N w := by
  simp [binaryAccumulatorProbability, hB2]

theorem binaryAccumulatorCount_output_zero {N w : ℕ} (hw : 0 < w) :
    binaryAccumulatorCount N w 0 = 0 := by
  rw [binaryAccumulatorCount, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  intro x _ hx
  have hacc : accumLinearEquiv (ZMod 2) N x = 0 :=
    hammingNorm_eq_zero.mp hx.2
  have hx0 : x = 0 := (accumLinearEquiv (ZMod 2) N).injective <| by
    simpa using hacc
  have : w = 0 := by
    rw [← hx.1, hx0, hammingNorm_zero]
  exact hw.ne' this

/-- Exhaustively compare B2 for all positive input weights and all output
weights at lengths below `limit`. -/
abbrev binaryAccumulatorFormulaHoldsUpTo (limit : ℕ) : Prop :=
  ∀ N : Fin limit,
    ∀ w : Fin N.val,
      ∀ h : Fin N.val,
        binaryAccumulatorCount N.val (w.val + 1) (h.val + 1) =
          binaryAccumulatorFormula N.val (w.val + 1) (h.val + 1)

/-- Executable regression certificate for every positive input and output
weight at lengths through eight. -/
theorem binaryAccumulatorFormula_small :
    binaryAccumulatorFormulaHoldsUpTo 9 := by
  set_option maxRecDepth 100000 in
    decide

end BAA
