import BAA.BlockRankPolynomial
import BAA.BinaryRuns
import BAA.MDS

/-!
# Exact finite MDS support/dimension profiles
-/

namespace BAA

/-- L1: the local support/positive-shortening-dimension polynomial of an
`[m,k,m-k+1]` MDS code.  Variable `0` marks support and variable `1` marks
shortening dimension. -/
noncomputable def mdsSupportDimensionPolynomial (m k : ℕ) :
    MvPolynomial (Fin 2) ℕ :=
  1 + ∑ l ∈ Finset.Icc (m - k + 1) m,
    MvPolynomial.monomial
      (blockRankExponent l (l - (m - k + 1) + 1)) (Nat.choose m l)

/-- The exponent appearing in L1 is the exact shortening dimension. -/
theorem IsMDS.shorteningDim_eq_supportExponent
    {F : Type*} [Field F] {m k l : ℕ} {C : Code F m}
    (hC : IsMDS C k) (S : Finset (Fin m))
    (hcard : S.card = l) :
    BAA.shorteningDim C S = l + k - m := by
  rw [hC.shorteningDim_positivePart, hcard]

/-- There are exactly `choose m l` coordinate supports of local size `l`. -/
theorem card_localSupports (m l : ℕ) :
    Fintype.card (FixedCardFinset m l) = Nat.choose m l :=
  card_fixedCardFinset m l

/-- Unnormalized L2: total positive shortening dimension equals total support
minus `d-1` for every active block. -/
theorem supportDimension_balance
    {I : Type*} [Fintype I] (d : ℕ) (l : I → ℕ)
    (hl : ∀ i, d ≤ l i) :
    (∑ i, (l i - d + 1 : ℤ)) =
      (∑ i, (l i : ℤ)) - (d - 1 : ℤ) * Fintype.card I := by
  have hpoint : ∀ i, (l i - d + 1 : ℤ) = (l i : ℤ) - (d - 1 : ℤ) := by
    intro i
    push_cast [hl i]
    omega
  simp_rw [hpoint]
  rw [Finset.sum_sub_distrib]
  simp
  ring

/-- L2 in rate notation, after substituting `d-1=m-k`. -/
theorem supportDimension_balance_mds
    {I : Type*} [Fintype I] (m k : ℕ) (l : I → ℕ)
    (hk : k ≤ m) (hl : ∀ i, m - k + 1 ≤ l i) :
    (∑ i, (l i - (m - k + 1) + 1 : ℤ)) =
      (∑ i, (l i : ℤ)) - (m - k : ℤ) * Fintype.card I := by
  have hpoint : ∀ i,
      (l i - (m - k + 1) + 1 : ℤ) = (l i : ℤ) - (m - k : ℤ) := by
    intro i
    push_cast [hk, hl i]
    omega
  simp_rw [hpoint]
  rw [Finset.sum_sub_distrib]
  simp
  ring

/-- R3: the exact `[16,4,13]` local polynomial. -/
theorem mdsSupportDimensionPolynomial_16_4 :
    mdsSupportDimensionPolynomial 16 4 =
      1 +
        MvPolynomial.monomial (blockRankExponent 13 1) 560 +
        MvPolynomial.monomial (blockRankExponent 14 2) 120 +
        MvPolynomial.monomial (blockRankExponent 15 3) 16 +
        MvPolynomial.monomial (blockRankExponent 16 4) 1 := by
  have hIcc : Finset.Icc 13 16 = {13, 14, 15, 16} := by decide
  rw [mdsSupportDimensionPolynomial, hIcc]
  norm_num [Nat.choose]
  ring

/-- R4 before normalization: for `[16,4,13]`, each active block subtracts
`12`, i.e. fraction `3/4` after division by the block length. -/
theorem supportDimension_balance_16_4
    {I : Type*} [Fintype I] (l : I → ℕ) (hl : ∀ i, 13 ≤ l i) :
    (∑ i, (l i - 13 + 1 : ℤ)) =
      (∑ i, (l i : ℤ)) - 12 * Fintype.card I := by
  simpa using supportDimension_balance_mds 16 4 l (by norm_num) hl

end BAA
