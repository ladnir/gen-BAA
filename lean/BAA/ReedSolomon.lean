import BAA.Basic
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Algebra.Polynomial.Eval.SMul
import Mathlib.Algebra.Polynomial.OfFn
import Mathlib.Algebra.Polynomial.Roots

/-!
# Reed--Solomon evaluation codes

Messages are coefficient vectors of length `k`.  `Polynomial.ofFn` turns them
into polynomials of degree below `k`, which are evaluated at `m` distinct field
points.
-/

namespace BAA

open Polynomial

section

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

noncomputable def reedSolomonEvaluation (k : ℕ) {m : ℕ} (points : Fin m → F) :
    (Fin k → F) →ₗ[F] Word F m where
  toFun a i := (Polynomial.ofFn k a).eval (points i)
  map_add' a b := by
    funext i
    simp
  map_smul' c a := by
    funext i
    simp [smul_eq_mul]

theorem reedSolomonEvaluation_injective {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hkm : k ≤ m) :
    Function.Injective (reedSolomonEvaluation k points) := by
  intro a b hab
  by_cases hk : k = 0
  · subst k
    exact Subsingleton.elim a b
  apply Polynomial.injective_ofFn k
  apply Polynomial.eq_of_natDegree_lt_card_of_eval_eq
    (Polynomial.ofFn k a) (Polynomial.ofFn k b) hpoints
  · intro i
    exact congrFun hab i
  · have ha := Polynomial.ofFn_natDegree_lt (Nat.one_le_iff_ne_zero.mpr hk) a
    have hb := Polynomial.ofFn_natDegree_lt (Nat.one_le_iff_ne_zero.mpr hk) b
    simp only [Fintype.card_fin]
    omega

noncomputable def reedSolomonCode (k : ℕ) {m : ℕ} (points : Fin m → F) : Code F m :=
  (reedSolomonEvaluation k points).range

theorem finrank_reedSolomonCode {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hkm : k ≤ m) :
    Module.finrank F (reedSolomonCode k points) = k := by
  rw [reedSolomonCode,
    (reedSolomonEvaluation k points).finrank_range_of_inj
      (reedSolomonEvaluation_injective points hpoints hkm)]
  exact Module.finrank_fin_fun F

/-- Evaluation of a nonzero polynomial of degree below `k` has at most `k-1`
zero coordinates. -/
theorem reedSolomon_weight_lower {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hk : 0 < k) (hkm : k ≤ m)
    (a : Fin k → F) (ha : a ≠ 0) :
    m - k + 1 ≤ hammingNorm (reedSolomonEvaluation k points a) := by
  classical
  let p := Polynomial.ofFn k a
  have hp0 : p ≠ 0 := fun hp => ha <| Polynomial.injective_ofFn k <| by
    simpa [p] using hp
  let Z : Finset (Fin m) := Finset.univ.filter fun i => p.eval (points i) = 0
  have hZroots : Z.card ≤ p.roots.toFinset.card := by
    apply Finset.card_le_card_of_injOn points
    · intro i hi
      have hi0 : p.eval (points i) = 0 := (Finset.mem_filter.mp hi).2
      simpa [Polynomial.mem_roots hp0, IsRoot.def] using hi0
    · exact hpoints.injOn
  have hZk : Z.card < k :=
    lt_of_le_of_lt (hZroots.trans (Multiset.toFinset_card_le p.roots)) <|
      (Polynomial.card_roots' p).trans_lt <|
        Polynomial.ofFn_natDegree_lt (Nat.one_le_iff_ne_zero.mpr hk.ne') a
  have hpartition :
      Z.card + hammingNorm (reedSolomonEvaluation k points a) = m := by
    rw [hammingNorm]
    simpa [Z, p, reedSolomonEvaluation] using
      (Finset.card_filter_add_card_filter_not
        (s := (Finset.univ : Finset (Fin m)))
        (p := fun i => p.eval (points i) = 0))
  omega

theorem minimumDistance_reedSolomonCode_lower {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hk : 0 < k) (hkm : k ≤ m) :
    m - k + 1 ≤ minimumDistance (reedSolomonCode k points) := by
  letI : Nontrivial (reedSolomonCode k points) := by
    refine ⟨⟨⟨reedSolomonEvaluation k points (fun _ => 1), ⟨_, rfl⟩⟩, 0, ?_⟩⟩
    intro hzero
    have hmap : reedSolomonEvaluation k points (fun _ => 1) = 0 :=
      congrArg Subtype.val hzero
    have hinput : (fun _ : Fin k => (1 : F)) = 0 :=
      reedSolomonEvaluation_injective points hpoints hkm (by simpa using hmap)
    have hcoord := congrFun hinput ⟨0, hk⟩
    simp at hcoord
  obtain ⟨c, hc0, hcweight⟩ := minimumDistance_attained (reedSolomonCode k points)
  rw [← hcweight]
  obtain ⟨a, ha⟩ := c.2
  have ha0 : a ≠ 0 := by
    intro hzero
    apply hc0
    apply Subtype.ext
    rw [← ha, hzero]
    simp
  rw [← ha]
  exact reedSolomon_weight_lower points hpoints hk hkm a ha0

/-- Embed the first `k - 1` evaluation positions into the full block. -/
def reedSolomonRootIndex {m k : ℕ} (hkm : k ≤ m) : Fin (k - 1) → Fin m :=
  fun i => ⟨i, i.isLt.trans_le ((Nat.sub_le k 1).trans hkm)⟩

theorem reedSolomonRootIndex_injective {m k : ℕ} (hkm : k ≤ m) :
    Function.Injective (reedSolomonRootIndex hkm) := by
  intro i j hij
  exact Fin.ext (Fin.mk.inj hij)

/-- The standard degree-`k-1` word witnessing the Reed--Solomon distance. -/
noncomputable def reedSolomonDistancePolynomial {m k : ℕ} (points : Fin m → F)
    (hkm : k ≤ m) : F[X] :=
  ∏ j : Fin (k - 1), (X - C (points (reedSolomonRootIndex hkm j)))

theorem reedSolomonDistancePolynomial_monic {m k : ℕ} (points : Fin m → F)
    (hkm : k ≤ m) : (reedSolomonDistancePolynomial points hkm).Monic := by
  classical
  apply Polynomial.monic_prod_of_monic
  intro i _
  exact monic_X_sub_C _

theorem reedSolomonDistancePolynomial_natDegree {m k : ℕ} (points : Fin m → F)
    (hkm : k ≤ m) :
    (reedSolomonDistancePolynomial points hkm).natDegree = k - 1 := by
  classical
  simp [reedSolomonDistancePolynomial,
    Polynomial.natDegree_finsetProd_X_sub_C_eq_card]

theorem minimumDistance_reedSolomonCode_upper {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hk : 0 < k) (hkm : k ≤ m) :
    minimumDistance (reedSolomonCode k points) ≤ m - k + 1 := by
  classical
  let p := reedSolomonDistancePolynomial points hkm
  let a : Fin k → F := Polynomial.toFn k p
  have hpdeg : p.natDegree < k := by
    have hpdegree : p.natDegree = k - 1 := by
      simpa [p] using reedSolomonDistancePolynomial_natDegree points hkm
    rw [hpdegree]
    omega
  have hof : Polynomial.ofFn k a = p :=
    Polynomial.ofFn_comp_toFn_eq_id_of_natDegree_lt hpdeg
  have hp0 : p ≠ 0 := (reedSolomonDistancePolynomial_monic points hkm).ne_zero
  have ha0 : a ≠ 0 := by
    intro ha
    apply hp0
    rw [← hof, ha]
    exact Polynomial.ofFn_zero k
  let c : reedSolomonCode k points :=
    ⟨reedSolomonEvaluation k points a, ⟨a, rfl⟩⟩
  have hc0 : c ≠ 0 := by
    intro hc
    have heval : reedSolomonEvaluation k points a = 0 := congrArg Subtype.val hc
    exact ha0 <| reedSolomonEvaluation_injective points hpoints hkm (by simpa using heval)
  apply (minimumDistance_le_weight (reedSolomonCode k points) c hc0).trans
  let Z : Finset (Fin m) := Finset.univ.filter fun i => p.eval (points i) = 0
  have hroots : k - 1 ≤ Z.card := by
    simpa using Finset.card_le_card_of_injOn
      (s := (Finset.univ : Finset (Fin (k - 1)))) (t := Z)
      (reedSolomonRootIndex hkm)
      (fun j _ => by
        simp only [Z, Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and]
        change (reedSolomonDistancePolynomial points hkm).eval
          (points (reedSolomonRootIndex hkm j)) = 0
        rw [reedSolomonDistancePolynomial, Polynomial.eval_prod]
        apply Finset.prod_eq_zero (Finset.mem_univ j)
        simp)
      (reedSolomonRootIndex_injective hkm).injOn
  have hpartition : Z.card + hammingNorm (c : Word F m) = m := by
    rw [hammingNorm]
    simpa [Z, c, reedSolomonEvaluation, hof] using
      (Finset.card_filter_add_card_filter_not
        (s := (Finset.univ : Finset (Fin m)))
        (p := fun i => p.eval (points i) = 0))
  omega

theorem minimumDistance_reedSolomonCode {m k : ℕ} (points : Fin m → F)
    (hpoints : Function.Injective points) (hk : 0 < k) (hkm : k ≤ m) :
    minimumDistance (reedSolomonCode k points) = m - k + 1 := by
  apply Nat.le_antisymm
  · exact minimumDistance_reedSolomonCode_upper points hpoints hk hkm
  · exact minimumDistance_reedSolomonCode_lower points hpoints hk hkm

end

end BAA
