import BAA.NonbinaryTrace
import BAA.Projective

/-!
# Finite diagonal counts for prescribed accumulator returns

For a nowhere-zero active value sequence, every prescribed return coordinate
forces its fresh unit multiplier once all earlier multipliers are known.  Thus
restriction to the complement of the prescribed return set is injective.
-/

namespace BAA

section

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- Active values after coordinatewise multiplication by units. -/
def weightedActiveValues {s : ℕ} (v : Fin s → F) (d : Fin s → Fˣ) :
    Fin s → F :=
  fun i => v i * (d i : F)

/-- Every index in `A` is a zero return.  Additional zero returns are allowed;
this monotone event is the one needed for an upper bound. -/
def PrescribesAccumulatorReturns {s : ℕ} (v : Fin s → F)
    (A : Finset (Fin s)) (d : Fin s → Fˣ) : Prop :=
  ∀ i ∈ A, accumulatorActiveState (weightedActiveValues v d) i = 0

noncomputable local instance prescribesAccumulatorReturnsDecidable {s : ℕ}
    (v : Fin s → F) (A : Finset (Fin s)) (d : Fin s → Fˣ) :
    Decidable (PrescribesAccumulatorReturns v A d) :=
  Classical.propDecidable _

/-- Retain only diagonal multipliers outside the prescribed return set. -/
def diagonalOutsideReturns {s : ℕ} (A : Finset (Fin s))
    (d : Fin s → Fˣ) : {i : Fin s // i ∉ A} → Fˣ :=
  fun i => d i.1

theorem diagonalOutsideReturns_injective_on_prescribed {s : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0) (A : Finset (Fin s)) :
    Function.Injective (fun d : {d : Fin s → Fˣ //
      PrescribesAccumulatorReturns v A d} => diagonalOutsideReturns A d.1) := by
  intro d e hde
  apply Subtype.ext
  funext i
  apply Units.ext
  have hall : ∀ n (hn : n < s),
      ((d.1 ⟨n, hn⟩ : Fˣ) : F) = ((e.1 ⟨n, hn⟩ : Fˣ) : F) := by
    intro n hn
    induction n using Nat.strong_induction_on with
    | h n ih =>
        let i : Fin s := ⟨n, hn⟩
        by_cases hiA : i ∈ A
        · have hd0 := d.2 i hiA
          have he0 := e.2 i hiA
          have hset : Finset.Iic i = insert i (Finset.Iio i) := by
            ext j
            simp only [Finset.mem_Iic, Finset.mem_insert, Finset.mem_Iio]
            omega
          have hinot : i ∉ Finset.Iio i := by simp
          unfold accumulatorActiveState at hd0 he0
          rw [hset, Finset.sum_insert hinot] at hd0 he0
          have hprevious :
              (∑ j ∈ Finset.Iio i, weightedActiveValues v d.1 j) =
                ∑ j ∈ Finset.Iio i, weightedActiveValues v e.1 j := by
            apply Finset.sum_congr rfl
            intro j hj
            have hj' : j < i := Finset.mem_Iio.mp hj
            have hjn : j.val < i.val := hj'
            change j.val < n at hjn
            simp only [weightedActiveValues]
            rw [ih j.val hjn j.isLt]
          rw [hprevious] at hd0
          simp only [weightedActiveValues] at hd0 he0
          have hmul : v i * (d.1 i : F) = v i * (e.1 i : F) := by
            exact add_right_cancel (hd0.trans he0.symm)
          exact (mul_left_cancel₀ (hv i)) hmul
        · have hout := congrFun hde (⟨i, hiA⟩ : {j : Fin s // j ∉ A})
          exact congrArg (fun z : Fˣ => (z : F)) hout
  exact hall i.val i.isLt

/-- At most `(q-1)^(s-|A|)` full diagonal assignments realize all prescribed
returns in `A`. -/
theorem card_prescribedAccumulatorReturns_le {s : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0) (A : Finset (Fin s)) :
    (Finset.univ.filter (PrescribesAccumulatorReturns v A)).card ≤
      (Fintype.card F - 1) ^ (s - A.card) := by
  classical
  let E := {d : Fin s → Fˣ // PrescribesAccumulatorReturns v A d}
  have hinj : Function.Injective (fun d : E => diagonalOutsideReturns A d.1) :=
    diagonalOutsideReturns_injective_on_prescribed v hv A
  have hcard := Fintype.card_le_of_injective _ hinj
  have hcomp : Fintype.card {i : Fin s // i ∉ A} = s - A.card := by
    let e : {i : Fin s // i ∉ A} ≃ ↥(Aᶜ) :=
      { toFun i := ⟨i.1, Finset.mem_compl.mpr i.2⟩
        invFun i := ⟨i.1, Finset.mem_compl.mp i.2⟩
        left_inv _ := rfl
        right_inv _ := rfl }
    rw [Fintype.card_congr e, Fintype.card_coe, Finset.card_compl,
      Fintype.card_fin]
  change Fintype.card E ≤ _ at hcard
  rw [Fintype.card_subtype] at hcard
  simpa only [E, Fintype.card_fun, hcomp, Fintype.card_units] using hcard

/-- Uniform-probability form: prescribing `a` fresh returns costs at most
`(q-1)^{-a}`. -/
theorem prescribedAccumulatorReturns_probability_le {s : ℕ}
    (v : Fin s → F) (hv : ∀ i, v i ≠ 0) (A : Finset (Fin s)) :
    uniformEventProbability (PrescribesAccumulatorReturns v A) ≤
      1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ A.card := by
  classical
  unfold uniformEventProbability
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_units]
  have hcard := card_prescribedAccumulatorReturns_le v hv A
  have hq : 0 < Fintype.card F - 1 := by
    have htwo : 2 ≤ Fintype.card F := Fintype.one_lt_card
    omega
  have hsplit : (Fintype.card F - 1) ^ s =
      (Fintype.card F - 1) ^ (s - A.card) *
        (Fintype.card F - 1) ^ A.card := by
    rw [← pow_add]
    congr 1
    exact (Nat.sub_add_cancel (by simpa using Finset.card_le_univ A)).symm
  rw [hsplit]
  push_cast
  rw [div_mul_eq_div_mul_one_div]
  calc
    ((Finset.univ.filter (PrescribesAccumulatorReturns v A)).card : ℚ) /
          ((Fintype.card F - 1 : ℕ) : ℚ) ^ (s - A.card) *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ A.card) ≤
        1 * (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ A.card) := by
      apply mul_le_mul_of_nonneg_right
      · rw [div_le_one (by positivity)]
        exact_mod_cast hcard
      · positivity
    _ = 1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ A.card := one_mul _

end

end BAA
