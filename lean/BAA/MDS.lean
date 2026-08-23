import BAA.ReedSolomon
import BAA.OuterCode
import BAA.Shortening
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

/-!
# MDS restriction interface and shortening profile

An `[n,k]` code is MDS when restriction to every `k` coordinates is a linear
isomorphism.  This interface makes the shortening calculation independent of
any particular generator matrix.
-/

namespace BAA

section Restriction

variable {F : Type*} {n : ℕ} [Field F]

/-- Restrict a codeword to the coordinate set `S`. -/
def coordinateRestriction (C : Code F n) (S : Finset (Fin n)) :
    C →ₗ[F] (S → F) where
  toFun c i := c.1 i.1
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- A finite code is MDS when every restriction to `k` coordinates is an
isomorphism.  The dimension and length bounds are included explicitly so the
definition remains meaningful at edge cases. -/
def IsMDS (C : Code F n) (k : ℕ) : Prop :=
  k ≤ n ∧ Module.finrank F C = k ∧
    ∀ S : Finset (Fin n), S.card = k →
      Function.Bijective (coordinateRestriction C S)

theorem IsMDS.restriction_injective_of_le_card {C : Code F n} {k : ℕ}
    (hC : IsMDS C k) (T : Finset (Fin n)) (hkT : k ≤ T.card) :
    Function.Injective (coordinateRestriction C T) := by
  classical
  obtain ⟨K, hKT, hKcard⟩ := Finset.exists_subset_card_eq hkT
  intro c d hcd
  apply (hC.2.2 K hKcard).1
  funext i
  let iT : T := ⟨i.1, hKT i.2⟩
  exact congrFun hcd iT

theorem IsMDS.restriction_surjective_of_card_le {C : Code F n} {k : ℕ}
    (hC : IsMDS C k) (T : Finset (Fin n)) (hTk : T.card ≤ k) :
    Function.Surjective (coordinateRestriction C T) := by
  classical
  obtain ⟨K, hTK, hKcard⟩ :=
    Finset.exists_superset_card_eq hTk (by simpa using hC.1)
  intro y
  let yK : K → F := fun i =>
    if hi : i.1 ∈ T then y ⟨i.1, hi⟩ else 0
  obtain ⟨c, hc⟩ := (hC.2.2 K hKcard).2 yK
  refine ⟨c, ?_⟩
  funext i
  have hiK : i.1 ∈ K := hTK i.2
  have hi := congrFun hc (⟨i.1, hiK⟩ : K)
  simpa [coordinateRestriction, yK, i.2] using hi

theorem IsMDS.finrank_restriction_range {C : Code F n} {k : ℕ}
    (hC : IsMDS C k) (T : Finset (Fin n)) :
    Module.finrank F (LinearMap.range (coordinateRestriction C T)) =
      min k T.card := by
  classical
  by_cases hTk : T.card ≤ k
  · have hsurj := hC.restriction_surjective_of_card_le T hTk
    rw [LinearMap.range_eq_top.mpr hsurj]
    simp [hTk, Module.finrank_pi_fintype, hC.2.1]
  · have hkT : k ≤ T.card := Nat.le_of_lt (Nat.lt_of_not_ge hTk)
    rw [(coordinateRestriction C T).finrank_range_of_inj
      (hC.restriction_injective_of_le_card T hkT), hC.2.1]
    exact (min_eq_left hkT).symm

/-- Codewords supported inside `S` are the kernel of restriction to its
coordinate complement. -/
noncomputable def shorteningEquivRestrictionKer (C : Code F n)
    (S : Finset (Fin n)) :
    shortening C S ≃ₗ[F] LinearMap.ker (coordinateRestriction C Sᶜ) where
  toFun x :=
    ⟨⟨x.1, x.2.1⟩, by
      change coordinateRestriction C Sᶜ ⟨x.1, x.2.1⟩ = 0
      funext i
      exact x.2.2 i.1 (Finset.mem_compl.mp i.2)⟩
  invFun x :=
    ⟨x.1.1, ⟨x.1.2, fun i hi => by
      have hxker : coordinateRestriction C Sᶜ x.1 = 0 := x.2
      have hzero := congrFun hxker (⟨i, Finset.mem_compl.mpr hi⟩ : ↥(Sᶜ))
      exact hzero⟩⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem IsMDS.shorteningDim {C : Code F n} {k : ℕ} (hC : IsMDS C k)
    (S : Finset (Fin n)) :
    BAA.shorteningDim C S = k - (n - S.card) := by
  classical
  rw [BAA.shorteningDim, (shorteningEquivRestrictionKer C S).finrank_eq]
  have hrank := (coordinateRestriction C Sᶜ).finrank_range_add_finrank_ker
  rw [hC.finrank_restriction_range, hC.2.1] at hrank
  have hcard : Sᶜ.card = n - S.card := by
    rw [Finset.card_compl, Fintype.card_fin]
  rw [hcard] at hrank
  omega

theorem IsMDS.shorteningDim_positivePart {C : Code F n} {k : ℕ}
    (hC : IsMDS C k) (S : Finset (Fin n)) :
    BAA.shorteningDim C S = S.card + k - n := by
  rw [IsMDS.shorteningDim hC]
  have hSn : S.card ≤ n := by
    simpa using Finset.card_le_univ S
  omega

end Restriction

section BlockMDS

variable {F : Type*} [Field F]

/-- Exact blockwise shortening profile for repeated MDS local codes. -/
theorem IsMDS.shorteningDim_outerCode {B m k : ℕ} {C0 : Code F m}
    (hC0 : IsMDS C0 k) (S : Finset (Fin (B * m))) :
    BAA.shorteningDim (outerCode C0 B) S =
      ∑ b : Fin B, ((localCoordinateSet S b).card + k - m) := by
  rw [BAA.shorteningDim_outerCode]
  apply Fintype.sum_congr
  intro b
  exact hC0.shorteningDim_positivePart (localCoordinateSet S b)

/-- For an admissible MDS support (every touched block is large enough to
support a nonzero local word), the global support excess is exactly one
`m-k` contribution per active block.  This is the equality case underlying
the paper's bound (4). -/
theorem IsMDS.card_eq_shorteningDim_add_activeBlocks_mul {B m k : ℕ}
    {C0 : Code F m} (hC0 : IsMDS C0 k) (S : Finset (Fin (B * m)))
    (hadmissible : ∀ b : Fin B, (localCoordinateSet S b).Nonempty →
      m - k + 1 ≤ (localCoordinateSet S b).card) :
    S.card = BAA.shorteningDim (outerCode C0 B) S +
      (activeBlockSet S).card * (m - k) := by
  classical
  have hkm : k ≤ m := hC0.1
  rw [card_eq_sum_card_localCoordinateSet,
    hC0.shorteningDim_outerCode]
  calc
    (∑ b : Fin B, (localCoordinateSet S b).card) =
        ∑ b : Fin B,
          (((localCoordinateSet S b).card + k - m) +
            if (localCoordinateSet S b).Nonempty then m - k else 0) := by
      apply Fintype.sum_congr
      intro b
      by_cases hb : (localCoordinateSet S b).Nonempty
      · have hcardm : (localCoordinateSet S b).card ≤ m := by
          simpa using Finset.card_le_univ (localCoordinateSet S b)
        have hadm := hadmissible b hb
        simp only [if_pos hb]
        omega
      · have hcard0 : (localCoordinateSet S b).card = 0 := by
          simpa [Finset.not_nonempty_iff_eq_empty] using hb
        have hsub : k - m = 0 := Nat.sub_eq_zero_of_le hkm
        simp [hb, hcard0, hsub]
    _ = (∑ b : Fin B, ((localCoordinateSet S b).card + k - m)) +
        ∑ b : Fin B,
          if (localCoordinateSet S b).Nonempty then m - k else 0 := by
      change Finset.univ.sum (fun b : Fin B =>
        ((localCoordinateSet S b).card + k - m) +
          (if (localCoordinateSet S b).Nonempty then m - k else 0)) = _
      rw [Finset.sum_add_distrib]
    _ = (∑ b : Fin B, ((localCoordinateSet S b).card + k - m)) +
        (activeBlockSet S).card * (m - k) := by
      congr 1
      change Finset.univ.sum (fun b : Fin B =>
        if (localCoordinateSet S b).Nonempty then m - k else 0) = _
      have hone : (activeBlockSet S).card =
          ∑ b : Fin B,
            if (localCoordinateSet S b).Nonempty then 1 else 0 := by
        simpa [activeBlockSet] using
          (Finset.card_filter
            (fun b : Fin B => (localCoordinateSet S b).Nonempty)
            Finset.univ)
      calc
        Finset.univ.sum (fun b : Fin B =>
            if (localCoordinateSet S b).Nonempty then m - k else 0) =
            Finset.univ.sum (fun b : Fin B =>
              (if (localCoordinateSet S b).Nonempty then 1 else 0) * (m - k)) := by
          apply Finset.sum_congr rfl
          intro b _
          by_cases hb : (localCoordinateSet S b).Nonempty <;> simp [hb]
        _ = (∑ b : Fin B,
              if (localCoordinateSet S b).Nonempty then 1 else 0) * (m - k) := by
          rw [Finset.sum_mul]
        _ = (activeBlockSet S).card * (m - k) := by rw [hone]

end BlockMDS

section ReedSolomon

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

theorem isMDS_reedSolomonCode {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hkm : k ≤ m) :
    IsMDS (reedSolomonCode k points) k := by
  classical
  refine ⟨hkm, finrank_reedSolomonCode points hpoints hkm, ?_⟩
  intro S hScard
  have hinj : Function.Injective
      (coordinateRestriction (reedSolomonCode k points) S) := by
    intro c d hcd
    by_cases hk0 : k = 0
    · subst k
      apply Subtype.ext
      obtain ⟨a, ha⟩ := c.2
      obtain ⟨b, hb⟩ := d.2
      have hab : a = b := by
        funext j
        exact Fin.elim0 (Fin.cast hk0 j)
      rw [← ha, ← hb]
      exact congrArg (reedSolomonEvaluation S.card points) hab
    obtain ⟨a, ha⟩ := c.2
    obtain ⟨b, hb⟩ := d.2
    apply Subtype.ext
    rw [← ha, ← hb]
    apply congrArg (reedSolomonEvaluation k points)
    apply Polynomial.injective_ofFn k
    apply Polynomial.eq_of_natDegree_lt_card_of_eval_eq
      (Polynomial.ofFn k a) (Polynomial.ofFn k b)
      (ι := S) (f := fun i : S => points i.1)
      (hpoints.comp Subtype.val_injective)
    · intro i
      have hi := congrFun hcd i
      change c.1 i.1 = d.1 i.1 at hi
      rw [← ha, ← hb] at hi
      simpa [reedSolomonEvaluation] using hi
    · have haDeg := Polynomial.ofFn_natDegree_lt
        (Nat.one_le_iff_ne_zero.mpr hk0) a
      have hbDeg := Polynomial.ofFn_natDegree_lt
        (Nat.one_le_iff_ne_zero.mpr hk0) b
      simpa only [Fintype.card_coe, hScard] using (max_lt haDeg hbDeg)
  have hdim : Module.finrank F (reedSolomonCode k points) =
      Module.finrank F (S → F) := by
    rw [finrank_reedSolomonCode points hpoints hkm,
      Module.finrank_pi_fintype]
    simp [hScard]
  exact ⟨hinj,
    (LinearMap.injective_iff_surjective_of_finrank_eq_finrank hdim).mp hinj⟩

theorem shorteningDim_reedSolomonCode {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hkm : k ≤ m)
    (S : Finset (Fin m)) :
    BAA.shorteningDim (reedSolomonCode k points) S = k - (m - S.card) :=
  IsMDS.shorteningDim (isMDS_reedSolomonCode points hpoints hkm) S

theorem shorteningDim_reedSolomonCode_positivePart {m k : ℕ}
    (points : Fin m → F) (hpoints : Function.Injective points)
    (hkm : k ≤ m) (S : Finset (Fin m)) :
    BAA.shorteningDim (reedSolomonCode k points) S =
      S.card + k - m := by
  exact IsMDS.shorteningDim_positivePart
    (isMDS_reedSolomonCode points hpoints hkm) S

theorem shorteningDim_outerReedSolomonCode {B m k : ℕ}
    (points : Fin m → F) (hpoints : Function.Injective points)
    (hkm : k ≤ m) (S : Finset (Fin (B * m))) :
    BAA.shorteningDim (outerCode (reedSolomonCode k points) B) S =
      ∑ b : Fin B, ((localCoordinateSet S b).card + k - m) :=
  IsMDS.shorteningDim_outerCode
    (isMDS_reedSolomonCode points hpoints hkm) S

end ReedSolomon

end BAA
