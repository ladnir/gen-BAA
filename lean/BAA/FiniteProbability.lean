import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Rat.Lemmas
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Exact uniform probability on finite types

The paper's finite ensembles are represented by exact cardinality ratios in
`Rat`.  This avoids measure-theoretic side conditions for finite uniform
sample spaces.
-/

namespace BAA

section

variable {Omega : Type*} [Fintype Omega]

/-- Exact probability of a finite event under the uniform distribution. -/
noncomputable def uniformEventProbability (p : Omega → Prop) [DecidablePred p] : ℚ :=
  ((Finset.univ.filter p).card : ℚ) / Fintype.card Omega

/-- Exact expectation of a natural-valued random variable under the uniform
distribution. -/
noncomputable def uniformExpectationNat (X : Omega → ℕ) : ℚ :=
  ((∑ omega, X omega : ℕ) : ℚ) / Fintype.card Omega

/-- Exact expectation of a rational-valued random variable under the uniform
distribution. -/
noncomputable def uniformExpectationRat (X : Omega → ℚ) : ℚ :=
  (∑ omega, X omega) / Fintype.card Omega

theorem uniformExpectationNat_eq_rat (X : Omega → ℕ) :
    uniformExpectationNat X = uniformExpectationRat fun omega => X omega := by
  unfold uniformExpectationNat uniformExpectationRat
  rw [Nat.cast_sum]

theorem uniformEventProbability_congr {p q : Omega → Prop}
    [DecidablePred p] [DecidablePred q] (h : ∀ omega, p omega ↔ q omega) :
    uniformEventProbability p = uniformEventProbability q := by
  unfold uniformEventProbability
  have hfin : Finset.univ.filter p = Finset.univ.filter q := by
    ext omega
    simp [h omega]
  rw [hfin]

/-- Monotonicity of exact uniform probability under event inclusion. -/
theorem uniformEventProbability_mono [Nonempty Omega]
    {p q : Omega → Prop} [DecidablePred p] [DecidablePred q]
    (hpq : ∀ omega, p omega → q omega) :
    uniformEventProbability p ≤ uniformEventProbability q := by
  unfold uniformEventProbability
  apply div_le_div_of_nonneg_right
  · exact_mod_cast Finset.card_le_card (by
      intro omega
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      exact hpq omega)
  · positivity

/-- Uniform finite probabilities are invariant under a bijective change of
sample-space coordinates. -/
theorem uniformEventProbability_equiv
    {Lambda : Type*} [Fintype Lambda] (e : Omega ≃ Lambda)
    (p : Lambda → Prop) [DecidablePred p] :
    uniformEventProbability (fun omega => p (e omega)) =
      uniformEventProbability p := by
  classical
  unfold uniformEventProbability
  have hevent :
      Fintype.card {omega : Omega // p (e omega)} =
        Fintype.card {lambda : Lambda // p lambda} := by
    exact Fintype.card_congr
      (Equiv.subtypeEquiv e (fun _ => Iff.rfl))
  rw [← Fintype.card_of_subtype
      (p := fun omega : Omega => p (e omega)) _ (by simp),
    ← Fintype.card_of_subtype (p := p) _ (by simp),
    hevent, Fintype.card_congr e]

theorem uniformExpectationRat_congr {X Y : Omega → ℚ}
    (h : ∀ omega, X omega = Y omega) :
    uniformExpectationRat X = uniformExpectationRat Y := by
  unfold uniformExpectationRat
  rw [show X = Y from funext h]

/-- Uniform rational expectations are invariant under a bijective change of
sample-space coordinates. -/
theorem uniformExpectationRat_equiv
    {Lambda : Type*} [Fintype Lambda] (e : Omega ≃ Lambda)
    (X : Lambda → ℚ) :
    uniformExpectationRat (fun omega => X (e omega)) =
      uniformExpectationRat X := by
  unfold uniformExpectationRat
  rw [e.sum_comp, Fintype.card_congr e]

theorem uniformExpectationRat_sum {I : Type*} [Fintype I]
    (X : I → Omega → ℚ) :
    uniformExpectationRat (fun omega => ∑ i, X i omega) =
      ∑ i, uniformExpectationRat fun omega => X i omega := by
  unfold uniformExpectationRat
  rw [Finset.sum_comm]
  simp_rw [div_eq_mul_inv]
  rw [Finset.sum_mul]

theorem uniformExpectationRat_mono [Nonempty Omega] {X Y : Omega → ℚ}
    (hXY : ∀ omega, X omega ≤ Y omega) :
    uniformExpectationRat X ≤ uniformExpectationRat Y := by
  unfold uniformExpectationRat
  apply div_le_div_of_nonneg_right
  · exact Finset.sum_le_sum fun omega _ => hXY omega
  · positivity

theorem uniformEventProbability_eq_expectationRat_indicator
    (p : Omega → Prop) [DecidablePred p] :
    uniformEventProbability p =
      uniformExpectationRat fun omega => if p omega then 1 else 0 := by
  classical
  unfold uniformEventProbability uniformExpectationRat
  have hcard : (Finset.univ.filter p).card =
      ∑ omega, if p omega then 1 else 0 := by
    simpa using (Finset.card_eq_sum_ite
      (s := Finset.univ.filter p) (t := Finset.univ) (Finset.subset_univ _))
  rw [hcard, Nat.cast_sum]
  simp

/-- Swap a finite expected filtered-cardinality count with the sum of the
individual event probabilities. -/
theorem uniformExpectationNat_filter_card
    {I : Type*} [Fintype I] (p : Omega → I → Prop)
    [∀ omega, DecidablePred (p omega)] [∀ i, DecidablePred fun omega => p omega i] :
    uniformExpectationNat
      (fun omega => (Finset.univ.filter fun i => p omega i).card) =
      ∑ i, uniformEventProbability fun omega => p omega i := by
  classical
  rw [uniformExpectationNat_eq_rat]
  calc
    uniformExpectationRat
        (fun omega => ((Finset.univ.filter fun i => p omega i).card : ℚ)) =
        uniformExpectationRat
          (fun omega => ∑ i, if p omega i then (1 : ℚ) else 0) := by
      apply uniformExpectationRat_congr
      intro omega
      rw [show (Finset.univ.filter fun i => p omega i).card =
          ∑ i, if p omega i then 1 else 0 by
        simpa using (Finset.card_eq_sum_ite
          (s := Finset.univ.filter fun i => p omega i)
          (t := Finset.univ) (Finset.subset_univ _))]
      push_cast
      simp
    _ = ∑ i, uniformExpectationRat
          (fun omega => if p omega i then 1 else 0) :=
      uniformExpectationRat_sum _
    _ = ∑ i, uniformEventProbability (fun omega => p omega i) := by
      apply Finset.sum_congr rfl
      intro i _
      exact (uniformEventProbability_eq_expectationRat_indicator
        (fun omega => p omega i)).symm

/-- Union bound for an arbitrary finite family of events, stated entirely in
the exact finite probability interface. -/
theorem uniformEventProbability_exists_le_sum [Nonempty Omega]
    {I : Type*} [Fintype I] (p : I → Omega → Prop)
    [∀ i, DecidablePred (p i)]
    [DecidablePred fun omega => ∃ i, p i omega] :
    uniformEventProbability (fun omega => ∃ i, p i omega) ≤
      ∑ i, uniformEventProbability (p i) := by
  classical
  rw [uniformEventProbability_eq_expectationRat_indicator]
  calc
    uniformExpectationRat
        (fun omega => if (∃ i, p i omega) then 1 else 0) ≤
        uniformExpectationRat
          (fun omega => ∑ i, if p i omega then 1 else 0) := by
      apply uniformExpectationRat_mono
      intro omega
      by_cases h : ∃ i, p i omega
      · obtain ⟨i, hi⟩ := h
        have hex : ∃ j : I, p j omega := ⟨i, hi⟩
        simp only [if_pos hex]
        calc
          (1 : ℚ) = if p i omega then 1 else 0 := by simp [hi]
          _ ≤ ∑ j, if p j omega then 1 else 0 := by
            rw [← Finset.add_sum_erase Finset.univ
              (fun j => if p j omega then (1 : ℚ) else 0)
              (Finset.mem_univ i)]
            exact le_add_of_nonneg_right (by positivity)
      · simp [h]
    _ = ∑ i, uniformExpectationRat
          (fun omega => if p i omega then 1 else 0) :=
      uniformExpectationRat_sum _
    _ = ∑ i, uniformEventProbability (p i) := by
      apply Finset.sum_congr rfl
      intro i _
      exact (uniformEventProbability_eq_expectationRat_indicator (p i)).symm

theorem uniformEventProbability_prod [Nonempty Omega]
    {Lambda : Type*} [Fintype Lambda] [Nonempty Lambda]
    (p : Omega × Lambda → Prop) [DecidablePred p] :
    uniformEventProbability p =
      uniformExpectationRat fun omega =>
        uniformEventProbability (Omega := Lambda) fun lambda => p (omega, lambda) := by
  classical
  rw [uniformEventProbability_eq_expectationRat_indicator]
  unfold uniformExpectationRat uniformEventProbability
  rw [Fintype.card_prod, Fintype.sum_prod_type]
  simp_rw [show ∀ omega,
      ((Finset.univ.filter fun lambda : Lambda => p (omega, lambda)).card : ℚ) =
        ∑ lambda, if p (omega, lambda) then 1 else 0 by
    intro omega
    rw [show (Finset.univ.filter fun lambda : Lambda => p (omega, lambda)).card =
        ∑ lambda, if p (omega, lambda) then 1 else 0 by
      simpa using (Finset.card_eq_sum_ite
        (s := Finset.univ.filter fun lambda : Lambda => p (omega, lambda))
        (t := Finset.univ) (Finset.subset_univ _))]
    simp]
  have hOmega : (Fintype.card Omega : ℚ) ≠ 0 := by
    exact Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hLambda : (Fintype.card Lambda : ℚ) ≠ 0 := by
    exact Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  simp_rw [div_eq_mul_inv]
  rw [← Finset.sum_mul]
  push_cast
  field_simp

/-- An event depending only on the left coordinate has the same probability
in a finite product space. -/
theorem uniformEventProbability_prod_left [Nonempty Omega]
    {Lambda : Type*} [Fintype Lambda] [Nonempty Lambda]
    (p : Omega → Prop) [DecidablePred p] :
    uniformEventProbability (fun z : Omega × Lambda => p z.1) =
      uniformEventProbability p := by
  classical
  rw [uniformEventProbability_prod]
  calc
    uniformExpectationRat (fun omega =>
        uniformEventProbability (fun _lambda : Lambda => p omega)) =
        uniformExpectationRat (fun omega => if p omega then 1 else 0) := by
      apply uniformExpectationRat_congr
      intro omega
      by_cases hp : p omega
      · unfold uniformEventProbability
        simp [hp, Fintype.card_ne_zero]
      · unfold uniformEventProbability
        simp [hp]
    _ = uniformEventProbability p :=
      (uniformEventProbability_eq_expectationRat_indicator p).symm

/-- Conditional product bound for two finite independent coordinates. -/
theorem uniformEventProbability_and_prod_le [Nonempty Omega]
    {Lambda : Type*} [Fintype Lambda] [Nonempty Lambda]
    (p : Omega → Prop) [DecidablePred p]
    (q : Omega → Lambda → Prop) [∀ omega, DecidablePred (q omega)]
    (c : ℚ) (hc : 0 ≤ c)
    (hq : ∀ omega, p omega →
      uniformEventProbability (q omega) ≤ c) :
    uniformEventProbability (fun z : Omega × Lambda => p z.1 ∧ q z.1 z.2) ≤
      uniformEventProbability p * c := by
  classical
  rw [uniformEventProbability_prod]
  calc
    uniformExpectationRat (fun omega =>
        uniformEventProbability (fun lambda => p omega ∧ q omega lambda)) ≤
        uniformExpectationRat (fun omega => if p omega then c else 0) := by
      apply uniformExpectationRat_mono
      intro omega
      by_cases hp : p omega
      · simpa only [hp, true_and, if_true] using hq omega hp
      · have hempty : uniformEventProbability
            (fun lambda => p omega ∧ q omega lambda) = 0 := by
          unfold uniformEventProbability
          simp [hp]
        rw [if_neg hp, hempty]
    _ = uniformEventProbability p * c := by
      unfold uniformExpectationRat uniformEventProbability
      have hcard : (Finset.univ.filter p).card =
          ∑ omega, if p omega then 1 else 0 := by
        simpa using (Finset.card_eq_sum_ite
          (s := Finset.univ.filter p) (t := Finset.univ)
          (Finset.subset_univ _))
      rw [hcard, Nat.cast_sum]
      simp only [Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
      have hsum : (∑ omega, if p omega then c else 0) =
          (∑ omega, if p omega then (1 : ℚ) else 0) * c := by
        rw [Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro omega _
        by_cases hp : p omega <;> simp [hp]
      rw [hsum]
      ring

/-- Law of total expectation for a finite-valued statistic. -/
theorem uniformExpectationRat_groupBy [Nonempty Omega]
    {I : Type*} [Fintype I] [DecidableEq I] (stat : Omega → I) (f : I → ℚ) :
    uniformExpectationRat (fun omega => f (stat omega)) =
      ∑ i, uniformEventProbability (fun omega => stat omega = i) * f i := by
  classical
  unfold uniformExpectationRat uniformEventProbability
  rw [← Finset.sum_fiberwise_of_maps_to'
    (s := Finset.univ) (t := Finset.univ) (g := stat) (by simp) f]
  simp_rw [Finset.sum_eq_card_nsmul, nsmul_eq_mul]
  push_cast
  conv_lhs => rw [div_eq_mul_inv, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem card_positive_event_le_sum (X : Omega → ℕ) :
    (Finset.univ.filter fun omega => 0 < X omega).card ≤ ∑ omega, X omega := by
  classical
  calc
    (Finset.univ.filter fun omega => 0 < X omega).card =
        ∑ omega ∈ Finset.univ, if 0 < X omega then 1 else 0 :=
      by
        convert (Finset.card_eq_sum_ite
          (s := Finset.univ.filter fun omega => 0 < X omega)
          (t := Finset.univ) (Finset.subset_univ _)) using 1 <;> simp
    _ ≤ ∑ omega ∈ Finset.univ, X omega := by
      apply Finset.sum_le_sum
      intro omega _
      split_ifs with h
      · omega
      · omega
    _ = ∑ omega, X omega := by simp

/-- Finite first-moment/Markov bound at threshold one. -/
theorem uniform_probability_positive_le_expectation (X : Omega → ℕ) :
    uniformEventProbability (fun omega => 0 < X omega) ≤
      uniformExpectationNat X := by
  classical
  unfold uniformEventProbability uniformExpectationNat
  apply div_le_div_of_nonneg_right
  · exact_mod_cast card_positive_event_le_sum X
  · positivity

end

end BAA
