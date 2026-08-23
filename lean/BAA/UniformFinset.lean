import BAA.UniformPermutation
import BAA.BlockObstruction
import BAA.BinaryRuns

/-!
# Uniform coordinate permutations of finite supports
-/

namespace BAA

/-- A uniform permutation sends a fixed support uniformly over all supports of
the same cardinality. -/
theorem uniform_permutation_finset_event_probability {N : ℕ}
    (S : Finset (Fin N))
    (p : Finset (Fin N) → Prop) [DecidablePred p] :
    uniformEventProbability (Omega := Equiv.Perm (Fin N))
      (fun sigma => p (permuteCoordinateSet sigma S)) =
    uniformEventProbability (Omega := FixedCardFinset N S.card)
      (fun T => p T.1) := by
  classical
  let x : BinaryWord N := (binaryWordSupportEquiv N).symm S
  calc
    _ = uniformEventProbability (Omega := Equiv.Perm (Fin N))
        (fun sigma => p (support
          (permLinearEquiv (ZMod 2) sigma x))) := by
      apply uniformEventProbability_congr
      intro sigma
      have hsupp := support_permBinaryWord sigma x
      have hx : support x = S := (binaryWordSupportEquiv N).apply_symm_apply S
      rw [hsupp, hx]
      have hmap : permuteCoordinateSet sigma S =
          S.map sigma.toEmbedding := by
        simp [permuteCoordinateSet, Finset.map_eq_image]
      rw [hmap]
    _ = uniformEventProbability
        (Omega := BinaryWeightSlice N (hammingNorm x))
        (fun y => p (support y.1)) :=
      uniform_permutation_event_probability x (fun y => p (support y))
    _ = uniformEventProbability (Omega := FixedCardFinset N S.card)
        (fun T => p T.1) := by
      have hxweight : hammingNorm x = S.card := by
        rw [← card_support_eq_hammingNorm]
        exact congrArg Finset.card
          ((binaryWordSupportEquiv N).apply_symm_apply S)
      let e : BinaryWeightSlice N (hammingNorm x) ≃
          FixedCardFinset N S.card :=
        Equiv.subtypeEquiv (binaryWordSupportEquiv N) (by
          intro y
          change hammingNorm y = hammingNorm x ↔ (support y).card = S.card
          rw [card_support_eq_hammingNorm, hxweight])
      calc
        _ = uniformEventProbability
            (fun y : BinaryWeightSlice N (hammingNorm x) => p (e y).1) := by
          apply uniformEventProbability_congr
          intro y
          change p (support y.1) ↔ p (support y.1)
          rfl
        _ = _ := uniformEventProbability_equiv e (fun T => p T.1)

/-- Exact probability of hitting a particular same-size support. -/
theorem uniform_permutation_finset_probability {N : ℕ}
    (S T : Finset (Fin N)) (hcard : T.card = S.card) :
    uniformEventProbability (Omega := Equiv.Perm (Fin N))
      (fun sigma => permuteCoordinateSet sigma S = T) =
      1 / Nat.choose N S.card := by
  calc
    _ = uniformEventProbability (Omega := FixedCardFinset N S.card)
        (fun U => U.1 = T) :=
      uniform_permutation_finset_event_probability S (fun U => U = T)
    _ = 1 / Nat.choose N S.card := by
      unfold uniformEventProbability
      rw [card_fixedCardFinset]
      let target : FixedCardFinset N S.card := ⟨T, hcard⟩
      have hone :
          (Finset.univ.filter fun U : FixedCardFinset N S.card => U.1 = T).card = 1 := by
        have hset :
            Finset.univ.filter (fun U : FixedCardFinset N S.card => U.1 = T) =
              {target} := by
          ext U
          simp [target, Subtype.ext_iff]
        rw [hset]
        simp
      rw [hone]
      norm_num

/-- A fixed support acted on by a uniform permutation has the same exact
expectation as a uniformly sampled support of the same size. -/
theorem uniform_permutation_finset_expectation {N : ℕ}
    (S : Finset (Fin N)) (f : FixedCardFinset N S.card → ℚ) :
    uniformExpectationRat (Omega := Equiv.Perm (Fin N))
      (fun sigma => f ⟨permuteCoordinateSet sigma S, by
        rw [permuteCoordinateSet,
          Finset.card_image_of_injective _ sigma.injective]⟩) =
    uniformExpectationRat f := by
  classical
  let stat : Equiv.Perm (Fin N) → FixedCardFinset N S.card := fun sigma =>
    ⟨permuteCoordinateSet sigma S, by
      rw [permuteCoordinateSet,
        Finset.card_image_of_injective _ sigma.injective]⟩
  rw [uniformExpectationRat_groupBy stat f]
  have hprob : ∀ T : FixedCardFinset N S.card,
      uniformEventProbability (fun sigma => stat sigma = T) =
        1 / (Nat.choose N S.card : ℚ) := by
    intro T
    calc
      _ = uniformEventProbability
          (fun sigma : Equiv.Perm (Fin N) =>
            permuteCoordinateSet sigma S = T.1) := by
        apply uniformEventProbability_congr
        intro sigma
        simp [stat, Subtype.ext_iff]
      _ = _ := uniform_permutation_finset_probability S T.1 T.2
  simp_rw [hprob]
  unfold uniformExpectationRat
  rw [card_fixedCardFinset]
  simp_rw [div_eq_mul_inv]
  rw [Finset.sum_mul]
  ring

end BAA
