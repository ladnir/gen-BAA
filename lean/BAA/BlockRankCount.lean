import BAA.BlockRankPolynomial
import BAA.BinaryRuns
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Finite support interpretation of the block-rank polynomial

This module proves the combinatorial bridge required by G8.  A global support
is first reindexed as one local support per block.  The resulting product
enumerator is then matched to the bivariate polynomial coefficients.
-/

namespace BAA

/-- A global coordinate set is equivalent to one local coordinate set for
each block. -/
def blockFinsetEquiv (B m : ℕ) :
    Finset (Fin (B * m)) ≃ (Fin B → Finset (Fin m)) where
  toFun S := localCoordinateSet S
  invFun f := Finset.univ.filter fun i =>
    ((blockCoordEquiv B m).symm i).2 ∈ f ((blockCoordEquiv B m).symm i).1
  left_inv S := by
    ext i
    let p := (blockCoordEquiv B m).symm i
    have hp : blockCoordEquiv B m p = i :=
      (blockCoordEquiv B m).apply_symm_apply i
    simp [localCoordinateSet, p, hp]
  right_inv f := by
    funext b
    ext j
    simp [localCoordinateSet]

@[simp]
theorem blockFinsetEquiv_apply (B m : ℕ) (S : Finset (Fin (B * m)))
    (b : Fin B) : blockFinsetEquiv B m S b = localCoordinateSet S b :=
  rfl

theorem card_blockFinsetEquiv (B m : ℕ) (S : Finset (Fin (B * m))) :
    S.card = ∑ b : Fin B, (blockFinsetEquiv B m S b).card := by
  exact card_eq_sum_card_localCoordinateSet S

theorem localBlockRank_blockFinsetEquiv (B m h : ℕ)
    (S : Finset (Fin (B * m))) :
    localBlockRank h S =
      ∑ b : Fin B, min (blockFinsetEquiv B m S b).card h := by
  rfl

abbrev FixedCardFinsetPartition (n : ℕ) :=
  Σ k : Fin (n + 1), FixedCardFinset n k.val

def fixedCardFinsetPartitionToFinset {n : ℕ} :
    FixedCardFinsetPartition n → Finset (Fin n)
  | ⟨_, S⟩ => S.1

theorem fixedCardFinsetPartitionToFinset_bijective {n : ℕ} :
    Function.Bijective (@fixedCardFinsetPartitionToFinset n) := by
  constructor
  · rintro ⟨k, S⟩ ⟨l, T⟩ h
    have hkl : k = l := by
      apply Fin.ext
      have hcards : S.1.card = T.1.card := by
        apply congrArg Finset.card
        exact h
      rw [S.2, T.2] at hcards
      exact hcards
    subst l
    have hST : S = T := Subtype.ext h
    subst T
    rfl
  · intro S
    let k : Fin (n + 1) := ⟨S.card, by
      have := Finset.card_le_univ S
      simpa using Nat.lt_succ_of_le this⟩
    exact ⟨⟨k, ⟨S, rfl⟩⟩, rfl⟩

noncomputable def fixedCardFinsetPartitionEquiv (n : ℕ) :
    FixedCardFinsetPartition n ≃ Finset (Fin n) :=
  Equiv.ofBijective fixedCardFinsetPartitionToFinset
    fixedCardFinsetPartitionToFinset_bijective

theorem sum_finset_by_card {n : ℕ} {R : Type*} [AddCommMonoid R]
    (f : Finset (Fin n) → R) :
    (∑ S : Finset (Fin n), f S) =
      ∑ k : Fin (n + 1), ∑ S : FixedCardFinset n k.val, f S.1 := by
  rw [← (fixedCardFinsetPartitionEquiv n).sum_comp f]
  simp only [FixedCardFinsetPartition, Fintype.sum_sigma]
  rfl

/-- Direct one-block support/capped-rank enumerator. -/
noncomputable def concreteBlockRankPolynomial (m h : ℕ) :
    MvPolynomial (Fin 2) ℕ :=
  ∑ S : Finset (Fin m),
    MvPolynomial.monomial (blockRankExponent S.card (min S.card h)) 1

/-- G7 equals the direct finite-support enumerator for one block. -/
theorem concreteBlockRankPolynomial_eq (m h : ℕ) :
    concreteBlockRankPolynomial m h = blockRankPolynomial m h := by
  unfold concreteBlockRankPolynomial blockRankPolynomial
  rw [sum_finset_by_card]
  have hnormalize :
      (∑ k : Fin (m + 1),
        ∑ S : FixedCardFinset m k.val,
          MvPolynomial.monomial
            (blockRankExponent S.1.card (min S.1.card h)) 1) =
      ∑ k : Fin (m + 1),
        ∑ _S : FixedCardFinset m k.val,
          MvPolynomial.monomial
            (blockRankExponent k.val (min k.val h)) 1 := by
    apply Fintype.sum_congr
    intro k
    apply Fintype.sum_congr
    intro S
    rw [S.2]
  rw [hnormalize]
  apply Fintype.sum_congr
  intro l
  rw [show (∑ _S : FixedCardFinset m l.val,
      MvPolynomial.monomial
        (blockRankExponent l.val (min l.val h)) 1) =
      MvPolynomial.monomial
        (blockRankExponent l.val (min l.val h)) (Nat.choose m l.val) by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      card_fixedCardFinset]
    change MvPolynomial.C (Nat.choose m l.val) *
        MvPolynomial.monomial
          (blockRankExponent l.val (min l.val h)) 1 = _
    rw [MvPolynomial.C_mul_monomial, mul_one]]

theorem sum_blockRankExponent {B m h : ℕ}
    (f : Fin B → Finset (Fin m)) :
    (∑ b : Fin B, blockRankExponent (f b).card (min (f b).card h)) =
      blockRankExponent (∑ b : Fin B, (f b).card)
        (∑ b : Fin B, min (f b).card h) := by
  ext i
  fin_cases i <;> simp [blockRankExponent]

theorem prod_blockRankMonomial {B m h : ℕ}
    (f : Fin B → Finset (Fin m)) :
    (∏ b : Fin B,
      MvPolynomial.monomial
        (blockRankExponent (f b).card (min (f b).card h)) 1) =
      MvPolynomial.monomial
        (blockRankExponent (∑ b : Fin B, (f b).card)
          (∑ b : Fin B, min (f b).card h)) 1 := by
  calc
    _ = MvPolynomial.monomial
        (∑ b : Fin B,
          blockRankExponent (f b).card (min (f b).card h)) 1 := by
      induction (Finset.univ : Finset (Fin B)) using Finset.induction_on with
      | empty => simp
      | @insert b s hb ih =>
          rw [Finset.prod_insert hb, Finset.sum_insert hb, ih,
            MvPolynomial.monomial_mul]
          simp
    _ = _ := by rw [sum_blockRankExponent f]

/-- Direct enumerator of global supports by size and capped block rank. -/
noncomputable def concreteGlobalBlockRankPolynomial (B m h : ℕ) :
    MvPolynomial (Fin 2) ℕ :=
  ∑ S : Finset (Fin (B * m)),
    MvPolynomial.monomial
      (blockRankExponent S.card (localBlockRank h S)) 1

theorem concreteGlobalBlockRankPolynomial_eq (B m h : ℕ) :
    concreteGlobalBlockRankPolynomial B m h = blockRankPolynomial m h ^ B := by
  rw [← concreteBlockRankPolynomial_eq]
  unfold concreteGlobalBlockRankPolynomial concreteBlockRankPolynomial
  rw [← (blockFinsetEquiv B m).symm.sum_comp]
  rw [Fintype.sum_pow]
  apply Fintype.sum_congr
  intro f
  let S := (blockFinsetEquiv B m).symm f
  have hf : blockFinsetEquiv B m S = f :=
    (blockFinsetEquiv B m).apply_symm_apply f
  have hcard : S.card = ∑ b : Fin B, (f b).card := by
    rw [card_blockFinsetEquiv]
    simp_rw [hf]
  have hrank : localBlockRank h S =
      ∑ b : Fin B, min (f b).card h := by
    rw [localBlockRank_blockFinsetEquiv]
    simp_rw [hf]
  change MvPolynomial.monomial
      (blockRankExponent S.card (localBlockRank h S)) 1 = _
  rw [hcard, hrank, prod_blockRankMonomial]

theorem blockRankExponent_injective :
    Function.Injective fun p : ℕ × ℕ => blockRankExponent p.1 p.2 := by
  rintro ⟨u, z⟩ ⟨u', z'⟩ hep
  have hzero := congrArg (fun e : Fin 2 →₀ ℕ => e 0) hep
  have hone := congrArg (fun e : Fin 2 →₀ ℕ => e 1) hep
  simp [blockRankExponent] at hzero hone
  exact Prod.ext hzero hone

def concreteExactBlockRankPatternCount
    (B m h u z : ℕ) : ℕ :=
  (Finset.univ.filter fun S : Finset (Fin (B * m)) =>
    S.card = u ∧ localBlockRank h S = z).card

theorem coeff_concreteGlobalBlockRankPolynomial
    (B m h u z : ℕ) :
    MvPolynomial.coeff (blockRankExponent u z)
        (concreteGlobalBlockRankPolynomial B m h) =
      concreteExactBlockRankPatternCount B m h u z := by
  classical
  unfold concreteGlobalBlockRankPolynomial concreteExactBlockRankPatternCount
  simp only [MvPolynomial.coeff_sum]
  simp_rw [MvPolynomial.coeff_monomial]
  have hexp : ∀ S : Finset (Fin (B * m)),
      blockRankExponent u z =
          blockRankExponent S.card (localBlockRank h S) ↔
        S.card = u ∧ localBlockRank h S = z := by
    intro S
    constructor
    · intro he
      have hp : (u, z) = (S.card, localBlockRank h S) :=
        blockRankExponent_injective he
      exact ⟨(congrArg Prod.fst hp).symm, (congrArg Prod.snd hp).symm⟩
    · rintro ⟨hs, hr⟩
      rw [hs, hr]
  have hexp' : ∀ S : Finset (Fin (B * m)),
      blockRankExponent S.card (localBlockRank h S) =
          blockRankExponent u z ↔
        S.card = u ∧ localBlockRank h S = z := by
    intro S
    rw [eq_comm]
    exact hexp S
  simp_rw [hexp']
  rw [show (∑ S : Finset (Fin (B * m)),
      if S.card = u ∧ localBlockRank h S = z then 1 else 0) =
      (Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S = z).card by
    simpa using (Finset.card_eq_sum_ite
      (s := Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S = z)
      (t := Finset.univ) (Finset.subset_univ _))]

theorem exactBlockRankPatternCount_eq_concrete
    (B m h u z : ℕ) :
    exactBlockRankPatternCount B m h u z =
      concreteExactBlockRankPatternCount B m h u z := by
  unfold exactBlockRankPatternCount
  rw [← concreteGlobalBlockRankPolynomial_eq,
    coeff_concreteGlobalBlockRankPolynomial]

def fixedCardRankLeEquiv (B m h u c : ℕ) :
    {U : {S : Finset (Fin (B * m)) // S.card = u} //
      localBlockRank h (U.1 : Finset (Fin (B * m))) ≤ c} ≃
      {S : Finset (Fin (B * m)) //
        S.card = u ∧ localBlockRank h S ≤ c} where
  toFun U := ⟨(U.1 : Finset (Fin (B * m))), U.1.property, U.property⟩
  invFun S := ⟨⟨S.1, S.2.1⟩, S.2.2⟩
  left_inv U := by cases U with | mk U hU => cases U; rfl
  right_inv S := by cases S; rfl

theorem concreteCappedBlockRankPatternCount_eq_globalFilter
    (B m h u c : ℕ) :
    concreteCappedBlockRankPatternCount B m h u c =
      (Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S ≤ c).card := by
  classical
  unfold concreteCappedBlockRankPatternCount
  rw [← Fintype.card_of_subtype
      (p := fun U : {S : Finset (Fin (B * m)) // S.card = u} =>
        localBlockRank h U.1 ≤ c) _ (by simp),
    ← Fintype.card_of_subtype
      (p := fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S ≤ c) _ (by simp)]
  exact Fintype.card_congr (fixedCardRankLeEquiv B m h u c)

theorem card_globalBlockRank_le_eq_sum_exact
    (B m h u c : ℕ) :
    (Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S ≤ c).card =
      ∑ z ∈ Finset.range (c + 1),
        concreteExactBlockRankPatternCount B m h u z := by
  classical
  rw [show (Finset.univ.filter fun S : Finset (Fin (B * m)) =>
      S.card = u ∧ localBlockRank h S ≤ c).card =
      ∑ S : Finset (Fin (B * m)),
        if S.card = u ∧ localBlockRank h S ≤ c then 1 else 0 by
    simpa using (Finset.card_eq_sum_ite
      (s := Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S ≤ c)
      (t := Finset.univ) (Finset.subset_univ _))]
  simp_rw [concreteExactBlockRankPatternCount]
  simp_rw [show ∀ z,
      (Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S = z).card =
      ∑ S : Finset (Fin (B * m)),
        if S.card = u ∧ localBlockRank h S = z then 1 else 0 by
    intro z
    simpa using (Finset.card_eq_sum_ite
      (s := Finset.univ.filter fun S : Finset (Fin (B * m)) =>
        S.card = u ∧ localBlockRank h S = z)
      (t := Finset.univ) (Finset.subset_univ _))]
  rw [Finset.sum_comm]
  apply Fintype.sum_congr
  intro S
  by_cases hs : S.card = u
  · by_cases hr : localBlockRank h S ≤ c
    · rw [if_pos ⟨hs, hr⟩]
      rw [Finset.sum_eq_single (localBlockRank h S)]
      · simp [hs]
      · intro z hz hne
        simp [hs, Ne.symm hne]
      · simp [hr]
    · rw [if_neg (by simp [hs, hr])]
      symm
      apply Finset.sum_eq_zero
      intro z hz
      have hzle : z ≤ c := by simpa using hz
      simp only [hs, true_and, ite_eq_right_iff]
      intro heq
      omega
  · simp [hs]

/-- G8: the polynomial coefficient sum is exactly the direct count of support
patterns with size `u` and capped block rank at most `c`. -/
theorem cappedBlockRankPatternCount_eq_concrete
    (B m h u c : ℕ) :
    cappedBlockRankPatternCount B m h u c =
      concreteCappedBlockRankPatternCount B m h u c := by
  unfold cappedBlockRankPatternCount
  simp_rw [exactBlockRankPatternCount_eq_concrete]
  rw [← card_globalBlockRank_le_eq_sum_exact,
    concreteCappedBlockRankPatternCount_eq_globalFilter]

end BAA
