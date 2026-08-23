import BAA.Basic
import BAA.FiniteProbability
import Mathlib.Data.Nat.Cast.Field
import Mathlib.LinearAlgebra.Projectivization.Cardinality

/-!
# Finite projective representatives

For coordinate vectors we choose the unique representative whose first
nonzero coordinate is one.  The projective cardinality theorem then gives the
exact number of scalar classes used in the large-field union bound.
-/

namespace BAA

open scoped LinearAlgebra.Projectivization

section Normalize

variable {F : Type*} [Field F] [DecidableEq F] {r : ℕ}

theorem support_nonempty_of_ne_zero (x : Fin r → F) (hx : x ≠ 0) :
    (support x).Nonempty := by
  by_contra hempty
  rw [Finset.not_nonempty_iff_eq_empty] at hempty
  apply hx
  funext i
  by_contra hi
  have himem : i ∈ support x := (mem_support x i).mpr hi
  rw [hempty] at himem
  exact (Finset.notMem_empty i) himem

/-- The least nonzero coordinate of a nonzero finite vector. -/
noncomputable def firstNonzero (x : Fin r → F) (hx : x ≠ 0) : Fin r :=
  (support x).min' (support_nonempty_of_ne_zero x hx)

theorem firstNonzero_ne_zero (x : Fin r → F) (hx : x ≠ 0) :
    x (firstNonzero x hx) ≠ 0 := by
  exact (mem_support x _).mp
    (Finset.min'_mem (support x) (support_nonempty_of_ne_zero x hx))

/-- Normalize a nonzero vector so that its first nonzero coordinate is one. -/
noncomputable def normalizeProjective (x : Fin r → F) (hx : x ≠ 0) : Fin r → F :=
  (x (firstNonzero x hx))⁻¹ • x

theorem normalizeProjective_first (x : Fin r → F) (hx : x ≠ 0) :
    normalizeProjective x hx (firstNonzero x hx) = 1 := by
  simp [normalizeProjective, firstNonzero_ne_zero x hx]

theorem normalizeProjective_ne_zero (x : Fin r → F) (hx : x ≠ 0) :
    normalizeProjective x hx ≠ 0 := by
  intro hzero
  have := congrFun hzero (firstNonzero x hx)
  simp [normalizeProjective_first x hx] at this

theorem support_smul_ne_zero (a : F) (ha : a ≠ 0) (x : Fin r → F) :
    support (a • x) = support x := by
  ext i
  simp [support, ha]

theorem firstNonzero_smul (a : F) (ha : a ≠ 0) (x : Fin r → F) (hx : x ≠ 0) :
    firstNonzero (a • x) (smul_ne_zero ha hx) = firstNonzero x hx := by
  unfold firstNonzero
  have hs := support_smul_ne_zero a ha x
  apply (Finset.min'_eq_iff _ _ _).2
  constructor
  · rw [hs]
    exact Finset.min'_mem _ _
  · intro b hb
    apply Finset.min'_le
    rwa [hs] at hb

/-- First-coordinate normalization is invariant under a nonzero scalar. -/
theorem normalizeProjective_smul (a : F) (ha : a ≠ 0)
    (x : Fin r → F) (hx : x ≠ 0) :
    normalizeProjective (a • x) (smul_ne_zero ha hx) =
      normalizeProjective x hx := by
  funext i
  rw [normalizeProjective, normalizeProjective,
    firstNonzero_smul a ha x hx]
  simp only [Pi.smul_apply, smul_eq_mul]
  rw [mul_inv_rev, mul_assoc, ← mul_assoc a⁻¹ a (x i),
    inv_mul_cancel₀ ha, one_mul]

end Normalize

section Cardinality

variable (F : Type*) [Field F] [Finite F]

/-- Exact number of projective classes in an `r`-dimensional vector space. -/
theorem card_projectivization_fin (r : ℕ) :
    Nat.card (ℙ F (Fin r → F)) =
      (Nat.card F ^ r - 1) / (Nat.card F - 1) := by
  rw [Projectivization.card'']
  simp [Nat.card_fun]

/-- The finite geometric-series form of the same projective count. -/
theorem card_projectivization_fin_geom (r : ℕ) :
    Nat.card (ℙ F (Fin r → F)) =
      ∑ i ∈ Finset.range r, Nat.card F ^ i := by
  apply Projectivization.card_of_finrank F (Fin r → F)
  exact Module.finrank_fin_fun F

end Cardinality

section CancellationBound

variable (F : Type*) [Field F] [Fintype F] [DecidableEq F]

/-- Exact rational form of the paper's projective anti-cancellation factor. -/
noncomputable def projectiveCancellationBound (r c : ℕ) : ℚ :=
  min 1
    (((Nat.card (ℙ F (Fin r → F)) : ℕ) : ℚ) /
      ((Fintype.card F - 1 : ℕ) : ℚ) ^ c)

theorem card_diagonal_choices (c : ℕ) :
    Fintype.card (Fin c → Fˣ) = (Fintype.card F - 1) ^ c := by
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_units]

/-- A nondegenerate cancellation equation determines one and only one unit
multiplier. -/
theorem existsUnique_unit_mul_add_eq_zero (x y : F) (hx : x ≠ 0) (hy : y ≠ 0) :
    ∃! d : Fˣ, x * (d : F) + y = 0 := by
  let z : F := -y / x
  have hz : z ≠ 0 := div_ne_zero (neg_ne_zero.mpr hy) hx
  let d : Fˣ := Units.mk0 z hz
  have hcancel : x * z + y = 0 := by
    dsimp [z]
    rw [mul_comm x, div_mul_cancel₀ _ hx, neg_add_cancel]
  refine ⟨d, ?_, ?_⟩
  · exact hcancel
  · intro e he
    apply Units.ext
    apply (mul_left_cancel₀ hx)
    calc
      x * (e : F) = -y := eq_neg_of_add_eq_zero_left he
      _ = x * (d : F) := (eq_neg_of_add_eq_zero_left hcancel).symm

/-- `c` prescribed independent unit multipliers have exact probability
`(q-1)^{-c}` under the finite uniform ensemble. -/
theorem prescribed_diagonal_probability (c : ℕ) (target : Fin c → Fˣ) :
    uniformEventProbability (fun d : Fin c → Fˣ => d = target) =
      1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ c := by
  classical
  unfold uniformEventProbability
  rw [card_diagonal_choices]
  simp only [Finset.filter_eq', Finset.mem_univ, if_true,
    Finset.card_singleton, Nat.cast_one, Nat.cast_pow]

theorem uniformEventProbability_le_one {Omega : Type*}
    [Fintype Omega] [Nonempty Omega] (p : Omega → Prop) [DecidablePred p] :
    uniformEventProbability p ≤ 1 := by
  unfold uniformEventProbability
  rw [div_le_one]
  · exact_mod_cast Finset.card_le_card (Finset.filter_subset _ _)
  · exact_mod_cast Fintype.card_pos

/-- If at most `P` multiplier tuples satisfy an event, its exact uniform
probability is at most `P / (q-1)^c`. -/
theorem uniform_diagonal_event_le_of_card {c P : ℕ}
    (p : (Fin c → Fˣ) → Prop) [DecidablePred p]
    (hp : (Finset.univ.filter p).card ≤ P) :
    uniformEventProbability p ≤
      (P : ℚ) / ((Fintype.card F - 1 : ℕ) : ℚ) ^ c := by
  unfold uniformEventProbability
  rw [card_diagonal_choices]
  simp only [Nat.cast_pow]
  apply div_le_div_of_nonneg_right
  · exact_mod_cast hp
  · positivity

/-- Abstract finite union bound behind (2): after grouping by projective
classes, it is enough to prove that at most one multiplier tuple succeeds per
class. -/
theorem projective_cancellation_union_bound {r c : ℕ}
    (p : (Fin c → Fˣ) → Prop) [DecidablePred p]
    (hp : (Finset.univ.filter p).card ≤ Nat.card (ℙ F (Fin r → F))) :
    uniformEventProbability p ≤ projectiveCancellationBound F r c := by
  rw [projectiveCancellationBound, le_min_iff]
  constructor
  · exact uniformEventProbability_le_one p
  · exact uniform_diagonal_event_le_of_card F p hp

theorem projectiveCancellationBound_formula (r c : ℕ) :
    projectiveCancellationBound F r c =
      min 1
        (((Fintype.card F ^ r - 1 : ℕ) : ℚ) /
          ((Fintype.card F - 1 : ℕ) : ℚ) /
          ((Fintype.card F - 1 : ℕ) : ℚ) ^ c) := by
  rw [projectiveCancellationBound, card_projectivization_fin]
  rw [Nat.card_eq_fintype_card]
  congr 2
  exact Nat.cast_div_charZero ((Fintype.card F).sub_one_dvd_pow_sub_one r)

end CancellationBound

end BAA
