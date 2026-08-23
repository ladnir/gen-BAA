import BAA.BinaryAccumulator
import BAA.FiniteProbability
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Tactic.NormNum

/-!
# Uniform binary Hamming slices

This module proves the finite counting fact used at both interleavers: a
uniform coordinate permutation sends a fixed binary word uniformly over its
Hamming slice.  The proof is by explicit bijections between permutation
fibers, followed by finite double counting.
-/

namespace BAA

def permutationFiber {N : ℕ} (x y : BinaryWord N) :=
  {sigma : Equiv.Perm (Fin N) // permLinearEquiv (ZMod 2) sigma x = y}

noncomputable instance {N : ℕ} (x y : BinaryWord N) :
    Fintype (permutationFiber x y) := by
  classical
  refine Fintype.ofFinset
    (p := {sigma : Equiv.Perm (Fin N) |
      permLinearEquiv (ZMod 2) sigma x = y})
    (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
      permLinearEquiv (ZMod 2) sigma x = y) ?_
  simp

private theorem permBinaryWord_comp {N : ℕ} (sigma tau : Equiv.Perm (Fin N))
    (x : BinaryWord N) :
    permLinearEquiv (ZMod 2) tau (permLinearEquiv (ZMod 2) sigma x) =
      permLinearEquiv (ZMod 2) (sigma.trans tau) x := by
  rfl

/-- Right composition by a permutation taking `y₁` to `y₂` gives an explicit
bijection between the corresponding permutation fibers. -/
def permutationFiberEquiv {N : ℕ} (x y₁ y₂ : BinaryWord N)
    (tau : Equiv.Perm (Fin N))
    (htau : permLinearEquiv (ZMod 2) tau y₁ = y₂) :
    permutationFiber x y₁ ≃ permutationFiber x y₂ where
  toFun sigma := ⟨sigma.1.trans tau, by
    rw [← permBinaryWord_comp, sigma.2, htau]⟩
  invFun sigma := ⟨sigma.1.trans tau.symm, by
    rw [← permBinaryWord_comp, sigma.2]
    rw [← htau]
    funext i
    simp [permLinearEquiv_apply]⟩
  left_inv sigma := by
    apply Subtype.ext
    ext i
    simp
  right_inv sigma := by
    apply Subtype.ext
    ext i
    simp

theorem card_permutationFiber_eq_of_weight_eq {N : ℕ} (x : BinaryWord N)
    {y₁ y₂ : BinaryWord N} (hy : hammingNorm y₁ = hammingNorm y₂) :
    Fintype.card (permutationFiber x y₁) =
      Fintype.card (permutationFiber x y₂) := by
  classical
  obtain ⟨tau, htau⟩ := exists_permBinaryWord_of_weight_eq hy
  exact Fintype.card_congr (permutationFiberEquiv x y₁ y₂ tau htau)

private def permImageInSlice {N : ℕ} (x : BinaryWord N) :
    Equiv.Perm (Fin N) → BinaryWeightSlice N (hammingNorm x) :=
  fun sigma => ⟨permLinearEquiv (ZMod 2) sigma x,
    hammingNorm_permBinaryWord sigma x⟩

private theorem permImageInSlice_surjective {N : ℕ} (x : BinaryWord N) :
    Function.Surjective (permImageInSlice x) := by
  intro y
  obtain ⟨sigma, hsigma⟩ := exists_permBinaryWord_of_weight_eq y.2.symm
  exact ⟨sigma, Subtype.ext hsigma⟩

/-- All fibers of the map from coordinate permutations to the Hamming slice
of `x` have the same cardinality. -/
theorem card_permImageInSlice_fiber_eq {N : ℕ} (x : BinaryWord N)
    (y₁ y₂ : BinaryWeightSlice N (hammingNorm x)) :
    (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
      permImageInSlice x sigma = y₁).card =
    (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
      permImageInSlice x sigma = y₂).card := by
  classical
  rw [← Fintype.card_of_subtype
      (p := fun sigma : Equiv.Perm (Fin N) => permImageInSlice x sigma = y₁)
      _ (by simp),
    ← Fintype.card_of_subtype
      (p := fun sigma : Equiv.Perm (Fin N) => permImageInSlice x sigma = y₂)
      _ (by simp)]
  let e₁ : {sigma : Equiv.Perm (Fin N) // permImageInSlice x sigma = y₁} ≃
      permutationFiber x y₁.1 := Equiv.subtypeEquiv (Equiv.refl _) (by
        intro sigma
        simp only [permImageInSlice, Equiv.refl_apply]
        exact Subtype.ext_iff)
  let e₂ : {sigma : Equiv.Perm (Fin N) // permImageInSlice x sigma = y₂} ≃
      permutationFiber x y₂.1 := Equiv.subtypeEquiv (Equiv.refl _) (by
        intro sigma
        simp only [permImageInSlice, Equiv.refl_apply]
        exact Subtype.ext_iff)
  rw [Fintype.card_congr e₁, Fintype.card_congr e₂]
  exact card_permutationFiber_eq_of_weight_eq x (y₁.2.trans y₂.2.symm)

/-- Double counting: the permutation group is the size of the Hamming slice
times any one of its (equipotent) image fibers. -/
theorem card_perm_eq_slice_mul_fiber {N : ℕ} (x : BinaryWord N)
    (y : BinaryWeightSlice N (hammingNorm x)) :
    Fintype.card (Equiv.Perm (Fin N)) =
      Fintype.card (BinaryWeightSlice N (hammingNorm x)) *
        (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
          permImageInSlice x sigma = y).card := by
  classical
  rw [Fintype.card, Finset.card_eq_sum_card_fiberwise
    (s := Finset.univ) (t := Finset.univ) (f := permImageInSlice x) (by simp)]
  have hfiber := fun z : BinaryWeightSlice N (hammingNorm x) =>
    card_permImageInSlice_fiber_eq x z y
  simp_rw [hfiber]
  simp

/-- Exact finite probability that a uniform coordinate permutation sends `x`
to a specified `y` in the same Hamming slice. -/
theorem uniform_permutation_probability {N : ℕ} (x y : BinaryWord N)
    (hy : hammingNorm y = hammingNorm x) :
    ((Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
        permLinearEquiv (ZMod 2) sigma x = y).card : ℚ) /
      Fintype.card (Equiv.Perm (Fin N)) =
        1 / Nat.choose N (hammingNorm x) := by
  classical
  let ys : BinaryWeightSlice N (hammingNorm x) := ⟨y, hy⟩
  have hcount := card_perm_eq_slice_mul_fiber x ys
  rw [card_binaryWeightSlice] at hcount
  have hchoose : Nat.choose N (hammingNorm x) ≠ 0 := by
    have hxle : hammingNorm x ≤ N := by
      simpa using (hammingNorm_le_card_fintype (x := x))
    exact (Nat.choose_pos hxle).ne'
  have hperm : Fintype.card (Equiv.Perm (Fin N)) ≠ 0 := Fintype.card_ne_zero
  have hfiber :
      (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
        permImageInSlice x sigma = ys).card =
      (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
        permLinearEquiv (ZMod 2) sigma x = y).card := by
    congr 1
    ext sigma
    simp [permImageInSlice, ys]
  rw [hfiber] at hcount
  have hpermQ : (Fintype.card (Equiv.Perm (Fin N)) : ℚ) ≠ 0 := by
    exact Nat.cast_ne_zero.mpr hperm
  have hchooseQ : (Nat.choose N (hammingNorm x) : ℚ) ≠ 0 := by
    exact Nat.cast_ne_zero.mpr hchoose
  apply div_eq_div_iff hpermQ hchooseQ |>.2
  norm_num [hcount, mul_comm]

/-- A uniform coordinate permutation of a fixed binary word is the uniform
distribution on its Hamming slice, tested against an arbitrary predicate. -/
theorem uniform_permutation_event_probability {N : ℕ} (x : BinaryWord N)
    (p : BinaryWord N → Prop) [DecidablePred p] :
    uniformEventProbability (Omega := Equiv.Perm (Fin N))
      (fun sigma => p (permLinearEquiv (ZMod 2) sigma x)) =
    uniformEventProbability (Omega := BinaryWeightSlice N (hammingNorm x))
      (fun y => p y.1) := by
  classical
  let xs : BinaryWeightSlice N (hammingNorm x) := ⟨x, rfl⟩
  let k := (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
    permImageInSlice x sigma = xs).card
  have hgroup := card_perm_eq_slice_mul_fiber x xs
  change Fintype.card (Equiv.Perm (Fin N)) =
    Fintype.card (BinaryWeightSlice N (hammingNorm x)) * k at hgroup
  have hevent :
      (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
        p (permLinearEquiv (ZMod 2) sigma x)).card =
      (Finset.univ.filter fun y : BinaryWeightSlice N (hammingNorm x) =>
        p y.1).card * k := by
    rw [Finset.card_eq_sum_card_fiberwise
      (s := Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
        p (permLinearEquiv (ZMod 2) sigma x))
      (t := Finset.univ.filter fun y : BinaryWeightSlice N (hammingNorm x) => p y.1)
      (f := permImageInSlice x)]
    · have hfiber : ∀ y ∈ (Finset.univ.filter fun y :
          BinaryWeightSlice N (hammingNorm x) => p y.1),
          (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
            p (permLinearEquiv (ZMod 2) sigma x) ∧ permImageInSlice x sigma = y).card = k := by
        intro y hy
        have hcard := card_permImageInSlice_fiber_eq x y xs
        change (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
          p (permLinearEquiv (ZMod 2) sigma x) ∧ permImageInSlice x sigma = y).card =
          (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
            permImageInSlice x sigma = xs).card
        calc
          _ = (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
              permImageInSlice x sigma = y).card := by
            congr 1
            ext sigma
            simp only [Finset.mem_filter, Finset.mem_univ, true_and]
            constructor
            · exact And.right
            · intro hsigma
              refine ⟨?_, hsigma⟩
              have hyp : p y.1 := (Finset.mem_filter.mp hy).2
              have hval := congrArg Subtype.val hsigma
              change permLinearEquiv (ZMod 2) sigma x = y.1 at hval
              exact hval.symm ▸ hyp
          _ = _ := hcard
      calc
        _ = ∑ y ∈ (Finset.univ.filter fun y :
            BinaryWeightSlice N (hammingNorm x) => p y.1), k := by
          apply Finset.sum_congr rfl
          intro y hy
          simpa [Finset.filter_filter, and_assoc, and_comm, and_left_comm] using
            hfiber y hy
        _ = _ := by simp
    · intro sigma hsigma
      have hp := (Finset.mem_filter.mp hsigma).2
      apply Finset.mem_filter.mpr
      exact ⟨Finset.mem_univ _, by simpa [permImageInSlice] using hp⟩
  have hk : k ≠ 0 := by
    change (Finset.univ.filter fun sigma : Equiv.Perm (Fin N) =>
      permImageInSlice x sigma = xs).card ≠ 0
    rw [Finset.card_ne_zero]
    refine ⟨Equiv.refl _, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩⟩
    apply Subtype.ext
    rfl
  unfold uniformEventProbability
  rw [hevent, hgroup]
  push_cast
  have hkQ : (k : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hk
  field_simp

end BAA
