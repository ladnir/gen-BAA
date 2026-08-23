import BAA.RankPropagation
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# Block-diagonal outer parity checks
-/

namespace BAA

section Definition

variable {F : Type*} [Field F] {B m h : ℕ}

/-- Repeat a local parity checker independently on every outer block. -/
def blockParityCheck (H0 : Word F m →ₗ[F] Word F h) :
    Word F (B * m) →ₗ[F] Word F (B * h) where
  toFun x := (blockLinearEquiv F B h).symm fun b =>
    H0 (blockLinearEquiv F B m x b)
  map_add' x y := by
    apply (blockLinearEquiv F B h).injective
    funext b
    simp
  map_smul' c x := by
    apply (blockLinearEquiv F B h).injective
    funext b
    simp

theorem ker_blockParityCheck (C0 : Code F m)
    (H0 : Word F m →ₗ[F] Word F h) (hH0 : LinearMap.ker H0 = C0) :
    LinearMap.ker (blockParityCheck (B := B) H0) = outerCode C0 B := by
  ext x
  rw [LinearMap.mem_ker, mem_outerCode]
  constructor
  · intro hx b
    rw [← hH0, LinearMap.mem_ker]
    have hb := congrFun (congrArg (blockLinearEquiv F B h) hx) b
    simpa [blockParityCheck] using hb
  · intro hx
    have hall : (fun b => H0 (blockLinearEquiv F B m x b)) = 0 := by
      funext b
      change H0 (blockLinearEquiv F B m x b) = 0
      rw [← LinearMap.mem_ker, hH0]
      exact hx b
    change (blockLinearEquiv F B h).symm
      (fun b => H0 (blockLinearEquiv F B m x b)) = 0
    rw [hall, LinearEquiv.map_zero]

end Definition

section RankContainer

variable {F : Type*} [Field F] {B m h : ℕ}

def blockRangeSubmodule (H0 : Word F m →ₗ[F] Word F h)
    (U : Finset (Fin (B * m))) : Submodule F (Fin B → Word F h) :=
  Submodule.pi Set.univ fun b =>
    LinearMap.range (columnRestriction H0 (localCoordinateSet U b))

def piSubmoduleLinearEquiv
    (C : Fin B → Submodule F (Word F h)) :
    Submodule.pi Set.univ C ≃ₗ[F] ((b : Fin B) → C b) where
  toFun x b := ⟨x.1 b, by
    have hx := x.2 b (Set.mem_univ b)
    exact hx⟩
  invFun x := ⟨fun b => (x b).1, by
    intro b _
    exact (x b).2⟩
  left_inv x := by ext; rfl
  right_inv x := by funext b; apply Subtype.ext; rfl
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem finrank_blockRangeSubmodule
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    Module.finrank F (blockRangeSubmodule H0 U) =
      ∑ b : Fin B,
        columnRank H0 (localCoordinateSet U b) := by
  unfold blockRangeSubmodule
  rw [LinearEquiv.finrank_eq (piSubmoduleLinearEquiv fun b =>
    LinearMap.range (columnRestriction H0 (localCoordinateSet U b)))]
  rw [Module.finrank_pi_fintype]
  rfl

def blockRangeFlat (H0 : Word F m →ₗ[F] Word F h)
    (U : Finset (Fin (B * m))) : Code F (B * h) :=
  (blockRangeSubmodule H0 U).map (blockLinearEquiv F B h).symm.toLinearMap

theorem finrank_blockRangeFlat
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    Module.finrank F (blockRangeFlat H0 U) =
      ∑ b : Fin B, columnRank H0 (localCoordinateSet U b) := by
  rw [blockRangeFlat]
  rw [← (Submodule.equivMapOfInjective
    (blockLinearEquiv F B h).symm.toLinearMap
    (blockLinearEquiv F B h).symm.injective
    (blockRangeSubmodule H0 U)).finrank_eq]
  exact finrank_blockRangeSubmodule H0 U

theorem range_columnRestriction_blockParityCheck_le
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    LinearMap.range (columnRestriction (blockParityCheck (B := B) H0) U) ≤
      blockRangeFlat H0 U := by
  intro y hy
  obtain ⟨x, rfl⟩ := hy
  let v : Fin B → Word F h := fun b =>
    H0 (blockLinearEquiv F B m x.1 b)
  have hv : v ∈ blockRangeSubmodule H0 U := by
    intro b _
    let xb : supportedCode (F := F) (localCoordinateSet U b) :=
      ⟨blockLinearEquiv F B m x.1 b, by
        intro j hj
        exact x.2 (blockCoordEquiv B m (b, j)) (by simpa using hj)⟩
    exact ⟨xb, rfl⟩
  refine ⟨v, hv, ?_⟩
  rfl

theorem columnRank_blockParityCheck_le_sum
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    columnRank (blockParityCheck (B := B) H0) U ≤
      ∑ b : Fin B, columnRank H0 (localCoordinateSet U b) := by
  change Module.finrank F
      (LinearMap.range (columnRestriction (blockParityCheck H0) U)) ≤ _
  calc
    _ ≤ Module.finrank F (blockRangeFlat H0 U) :=
      Submodule.finrank_mono
        (range_columnRestriction_blockParityCheck_le H0 U)
    _ = _ := finrank_blockRangeFlat H0 U

theorem columnRank_le_min_card_length
    (H0 : Word F m →ₗ[F] Word F h) (S : Finset (Fin m)) :
    columnRank H0 S ≤ min S.card h := by
  apply Nat.le_min.mpr
  constructor
  · have hrankNullity := LinearMap.finrank_range_add_finrank_ker
      (columnRestriction H0 S)
    rw [finrank_supportedCode F S] at hrankNullity
    unfold columnRank
    omega
  · unfold columnRank
    exact (Submodule.finrank_le _).trans_eq (by simp)

/-- The repeated local parity checker satisfies the exact G4 block-rank cap. -/
theorem blockParityCheck_columnRank_le_localBlockRank
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    columnRank (blockParityCheck (B := B) H0) U ≤ localBlockRank h U := by
  exact (columnRank_blockParityCheck_le_sum H0 U).trans
    (Finset.sum_le_sum fun b _ => columnRank_le_min_card_length H0 _)

/-- G5 for a concrete repeated local parity checker, with no remaining rank-cap
hypothesis. -/
theorem baaBlockCoefficientIndependentObstruction
    [DecidableEq F]
    (H0 : Word F m →ₗ[F] Word F h)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) (T : Finset (Fin (B * m)))
    (hdefect : localBlockRank h
      (twoStepBackwardNeighborhood sigma₁.symm sigma₂.symm T) < T.card) :
    ∃ x : Word F (B * m),
      x ≠ 0 ∧ x ∈ supportedCode (F := F) T ∧
        baaParityCheck (blockParityCheck (B := B) H0)
          sigma₁ sigma₂ d₁ d₂ x = 0 := by
  exact baaCoefficientIndependentObstruction
    (blockParityCheck (B := B) H0) h
    (blockParityCheck_columnRank_le_localBlockRank H0)
    sigma₁ sigma₂ d₁ d₂ T hdefect

/-- Code-level form of G5: if the local checker has kernel `C0`, the forced
kernel word is a nonzero BAA codeword supported in `T`. -/
theorem baaCodeword_of_localBlockRank_defect
    [DecidableEq F] (C0 : Code F m)
    (H0 : Word F m →ₗ[F] Word F h) (hH0 : LinearMap.ker H0 = C0)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) (T : Finset (Fin (B * m)))
    (hdefect : localBlockRank h
      (twoStepBackwardNeighborhood sigma₁.symm sigma₂.symm T) < T.card) :
    ∃ x : Word F (B * m),
      x ≠ 0 ∧ x ∈ supportedCode (F := F) T ∧
        x ∈ baaCode F C0 sigma₁ sigma₂ d₁ d₂ := by
  obtain ⟨x, hx0, hxT, hxq⟩ :=
    baaBlockCoefficientIndependentObstruction H0 sigma₁ sigma₂ d₁ d₂ T hdefect
  refine ⟨x, hx0, hxT, ?_⟩
  have hxker : x ∈ LinearMap.ker
      (baaParityCheck (blockParityCheck (B := B) H0)
        sigma₁ sigma₂ d₁ d₂) := LinearMap.mem_ker.mpr hxq
  rw [ker_baaParityCheck C0 (blockParityCheck (B := B) H0)
    (ker_blockParityCheck C0 H0 hH0) sigma₁ sigma₂ d₁ d₂] at hxker
  exact hxker

end RankContainer

end BAA
