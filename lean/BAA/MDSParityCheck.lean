import BAA.BlockParityCheck
import BAA.MDS

namespace BAA

/-- The kernel of a selected parity checker is the corresponding shortening. -/
noncomputable def columnRestrictionKerEquivShortening
    {F : Type*} [Field F] {m h : ℕ}
    (C : Code F m) (H : Word F m →ₗ[F] Word F h)
    (hker : LinearMap.ker H = C) (S : Finset (Fin m)) :
    LinearMap.ker (columnRestriction H S) ≃ₗ[F] shortening C S where
  toFun x := ⟨x.1.1, by
    constructor
    · rw [← hker]
      exact LinearMap.mem_ker.mpr x.2
    · exact x.1.2⟩
  invFun x := ⟨⟨x.1, x.2.2⟩, by
    change H x.1 = 0
    rw [← LinearMap.mem_ker, hker]
    exact x.2.1⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

theorem columnRank_eq_card_sub_shorteningDim
    {F : Type*} [Field F] {m h : ℕ}
    (C : Code F m) (H : Word F m →ₗ[F] Word F h)
    (hker : LinearMap.ker H = C) (S : Finset (Fin m)) :
    columnRank H S = S.card - shorteningDim C S := by
  have hrank := LinearMap.finrank_range_add_finrank_ker (columnRestriction H S)
  rw [(columnRestrictionKerEquivShortening C H hker S).finrank_eq,
    finrank_supportedCode] at hrank
  unfold columnRank shorteningDim
  omega

theorem IsMDS.columnRank_parityCheck
    {F : Type*} [Field F] {m k h : ℕ}
    {C : Code F m} (hC : IsMDS C k)
    (H : Word F m →ₗ[F] Word F h) (hker : LinearMap.ker H = C)
    (hh : h = m - k) (S : Finset (Fin m)) :
    columnRank H S = min S.card h := by
  rw [columnRank_eq_card_sub_shorteningDim C H hker S,
    hC.shorteningDim_positivePart, hh]
  omega

theorem range_columnRestriction_blockParityCheck_eq
    {F : Type*} [Field F] {B m h : ℕ}
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    LinearMap.range (columnRestriction (blockParityCheck (B := B) H0) U) =
      blockRangeFlat H0 U := by
  apply le_antisymm
  · exact range_columnRestriction_blockParityCheck_le H0 U
  · intro y hy
    obtain ⟨v, hv, rfl⟩ := hy
    have hvlocal (b : Fin B) :
        v b ∈ LinearMap.range
          (columnRestriction H0 (localCoordinateSet U b)) :=
      hv b (Set.mem_univ b)
    choose xb hxb using hvlocal
    let xword : Word F (B * m) :=
      (blockLinearEquiv F B m).symm (fun b => (xb b).1)
    have hxword : xword ∈ supportedCode (F := F) U := by
      intro i hi
      let p := (blockCoordEquiv B m).symm i
      have hp : blockCoordEquiv B m p = i :=
        (blockCoordEquiv B m).apply_symm_apply i
      have hj : p.2 ∉ localCoordinateSet U p.1 := by
        intro hj
        apply hi
        rw [← hp]
        exact (mem_localCoordinateSet U p.1 p.2).mp hj
      change (xb p.1).1 p.2 = 0
      exact (xb p.1).2 p.2 hj
    refine ⟨⟨xword, hxword⟩, ?_⟩
    apply (blockLinearEquiv F B h).injective
    funext b
    simpa [columnRestriction, blockParityCheck, xword] using hxb b

theorem columnRank_blockParityCheck_eq_sum
    {F : Type*} [Field F] {B m h : ℕ}
    (H0 : Word F m →ₗ[F] Word F h) (U : Finset (Fin (B * m))) :
    columnRank (blockParityCheck (B := B) H0) U =
      ∑ b : Fin B, columnRank H0 (localCoordinateSet U b) := by
  simp only [columnRank]
  rw [range_columnRestriction_blockParityCheck_eq H0 U,
    finrank_blockRangeFlat]
  apply Finset.sum_congr rfl
  intro b hb
  rfl

theorem IsMDS.columnRank_blockParityCheck
    {F : Type*} [Field F] {B m k h : ℕ}
    {C : Code F m} (hC : IsMDS C k)
    (H0 : Word F m →ₗ[F] Word F h) (hker : LinearMap.ker H0 = C)
    (hh : h = m - k) (U : Finset (Fin (B * m))) :
    columnRank (blockParityCheck (B := B) H0) U = localBlockRank h U := by
  rw [columnRank_blockParityCheck_eq_sum]
  unfold localBlockRank
  apply Finset.sum_congr rfl
  intro b hb
  exact hC.columnRank_parityCheck H0 hker hh _

end BAA
