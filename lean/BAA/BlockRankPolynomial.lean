import BAA.BlockObstruction
import Mathlib.RingTheory.MvPolynomial.Basic

/-!
# Block-rank generating functions

These are the exact finite objects in G7 and G8.  Variable `0` marks support
size and variable `1` marks capped block rank.
-/

namespace BAA

/-- Exponent vector for `x^u y^z`. -/
noncomputable def blockRankExponent (u z : ℕ) : Fin 2 →₀ ℕ :=
  Finsupp.single 0 u + Finsupp.single 1 z

/-- G7: one-block support/capped-rank polynomial. -/
noncomputable def blockRankPolynomial (m h : ℕ) : MvPolynomial (Fin 2) ℕ :=
  ∑ l : Fin (m + 1),
    MvPolynomial.monomial
      (blockRankExponent l.val (min l.val h)) (Nat.choose m l.val)

/-- Coefficient `[x^u y^z] B_{m,h}(x,y)^B`. -/
noncomputable def exactBlockRankPatternCount
    (B m h u z : ℕ) : ℕ :=
  MvPolynomial.coeff (blockRankExponent u z) (blockRankPolynomial m h ^ B)

/-- G8 for `N = B*m`: count all support patterns of size `u` whose capped
block rank is at most `c`. -/
noncomputable def cappedBlockRankPatternCount
    (B m h u c : ℕ) : ℕ :=
  ∑ z ∈ Finset.range (c + 1), exactBlockRankPatternCount B m h u z

/-- Direct finite-set count represented by G8.  The polynomial/count bridge is
the remaining algebraic coefficient theorem. -/
def concreteCappedBlockRankPatternCount
  (B m h u c : ℕ) : ℕ :=
  (Finset.univ.filter fun U :
      {S : Finset (Fin (B * m)) // S.card = u} =>
    localBlockRank h U.1 ≤ c).card

theorem cappedBlockRankPatternCount_eq_sum_le
    (B m h u c : ℕ) :
    cappedBlockRankPatternCount B m h u c =
      ∑ z ∈ Finset.range (c + 1),
        MvPolynomial.coeff (blockRankExponent u z)
          (blockRankPolynomial m h ^ B) :=
  rfl

end BAA
