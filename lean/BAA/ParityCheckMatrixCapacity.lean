import BAA.MDSParityCheck
import BAA.LayeredBlockCapacity

namespace BAA

/-- Matrix of a parity-check linear map in the standard coordinate bases. -/
noncomputable def parityCheckMatrix
    {F : Type*} [Field F] {n q : ℕ}
    (H : Word F n →ₗ[F] Word F q) : Matrix (Fin q) (Fin n) F :=
  LinearMap.toMatrix (Pi.basisFun F (Fin n)) (Pi.basisFun F (Fin q)) H

/-- Matrix and linear-map conventions agree in the standard coordinate
bases. -/
theorem parityCheckMatrix_mulVec
    {F : Type*} [Field F] {n q : ℕ}
    (H : Word F n →ₗ[F] Word F q) (x : Word F n) :
    (parityCheckMatrix H).mulVec x = H x := by
  ext i
  have hi := congrFun
    (LinearMap.toMatrix_mulVec_repr (Pi.basisFun F (Fin n))
      (Pi.basisFun F (Fin q)) H x) i
  have hxrepr : ((Pi.basisFun F (Fin n)).repr x : Fin n → F) = x := by
    funext j
    exact Pi.basisFun_repr F (Fin n) x j
  rw [hxrepr] at hi
  simpa only [parityCheckMatrix, Pi.basisFun_repr] using hi

noncomputable def supportedCodeBasis
    {F : Type*} [Field F] {n : ℕ} (S : Finset (Fin n)) :
    Module.Basis S F (supportedCode (F := F) S) :=
  (Pi.basisFun F S).map (supportedCodeLinearEquiv F S).symm

theorem toMatrix_columnRestriction
    {F : Type*} [Field F] {n q : ℕ}
    (H : Word F n →ₗ[F] Word F q) (S : Finset (Fin n)) :
    LinearMap.toMatrix (supportedCodeBasis S) (Pi.basisFun F (Fin q))
        (columnRestriction H S) =
      (parityCheckMatrix H).submatrix id Subtype.val := by
  ext i j
  rw [LinearMap.toMatrix_apply]
  simp [supportedCodeBasis, supportedCodeLinearEquiv, supportedCodeExtension,
    columnRestriction, parityCheckMatrix, LinearMap.toMatrix_apply]
  apply congrFun
  apply congrArg H
  funext x
  by_cases hx : x = j.1
  · subst x
    simp [j.2]
  · by_cases hxS : x ∈ S
    · have hsub : (⟨x, hxS⟩ : S) ≠ j := by
        intro heq
        exact hx (congrArg Subtype.val heq)
      simp [hxS, hx, hsub]
    · simp [hxS, hx]

theorem rank_parityCheckMatrix_subtype
    {F : Type*} [Field F] {n q : ℕ}
    (H : Word F n →ₗ[F] Word F q) (S : Finset (Fin n)) :
    ((parityCheckMatrix H).submatrix id (Subtype.val : S → Fin n)).rank =
      columnRank H S := by
  rw [← toMatrix_columnRestriction H S]
  rw [Matrix.rank_eq_finrank_range_toLin _
    (Pi.basisFun F (Fin q)) (supportedCodeBasis S)]
  rw [Matrix.toLin_toMatrix]
  rfl

theorem rank_parityCheckMatrix_selection
    {F : Type*} [Field F] {n q r : ℕ}
    (H : Word F n →ₗ[F] Word F q)
    (S : Finset (Fin n)) (hS : S.card = r) :
    ((parityCheckMatrix H).submatrix id (selectionEmbedding ⟨S, hS⟩)).rank =
      columnRank H S := by
  rw [show (parityCheckMatrix H).submatrix id (selectionEmbedding ⟨S, hS⟩) =
      ((parityCheckMatrix H).submatrix id (Subtype.val : S → Fin n)).submatrix
        (Equiv.refl _) (selectionEquiv ⟨S, hS⟩) by rfl]
  rw [Matrix.rank_submatrix]
  exact rank_parityCheckMatrix_subtype H S

theorem exists_rows_det_submatrix_ne_zero_iff_rank_eq
    {F Rows Cols : Type*} [Field F] [Fintype Rows] {r : ℕ}
    (M : Matrix Rows Cols F) (cols : Fin r ↪ Cols) :
    (∃ rows : Fin r → Rows, (M.submatrix rows cols).det ≠ 0) ↔
      (M.submatrix id cols).rank = r := by
  classical
  let N := M.submatrix id cols
  constructor
  · rintro ⟨rows, hdet⟩
    have hrankSquare : (M.submatrix rows cols).rank = r := by
      change (Matrix.of fun i j => M (rows i) (cols j)).rank = r
      simpa using
        (Matrix.linearIndependent_rows_of_det_ne_zero hdet).rank_matrix
    apply Nat.le_antisymm
    · simpa [N] using N.rank_le_card_width
    · calc
        r = (M.submatrix rows cols).rank := hrankSquare.symm
        _ ≤ N.rank := Matrix.rank_submatrix_le N rows id
  · intro hrank
    obtain ⟨g₀, hg₀, -, hg₀ind⟩ :=
      Submodule.exists_fun_fin_finrank_span_eq F (Set.range N.row)
    have hdim : r = Module.finrank F
        (Submodule.span F (Set.range N.row)) :=
      hrank.symm.trans N.rank_eq_finrank_span_row
    let e : Fin r ≃ Fin (Module.finrank F
        (Submodule.span F (Set.range N.row))) := finCongr hdim
    let g : Fin r → (Fin r → F) := g₀ ∘ e
    have hg (i : Fin r) : g i ∈ Set.range N.row := hg₀ (e i)
    have hgind : LinearIndependent F g := hg₀ind.comp e e.injective
    choose rows hrows using hg
    refine ⟨rows, ?_⟩
    let Q : Matrix (Fin r) (Fin r) F := M.submatrix rows cols
    have hQrow (i : Fin r) : Q.row i = g i := by
      ext j
      exact congrFun (hrows i) j
    have hQind : LinearIndependent F Q.row := by
      rw [show Q.row = g from funext hQrow]
      exact hgind
    have hQunit : IsUnit Q := Matrix.linearIndependent_rows_iff_isUnit.mp hQind
    exact isUnit_iff_ne_zero.mp ((Matrix.isUnit_iff_isUnit_det Q).mp hQunit)

theorem IsMDS.parityCheckMatrix_hasLocalBlockMinorCapacity
    {F : Type*} [Field F] {B m k h : ℕ}
    {C : Code F m} (hC : IsMDS C k)
    (H0 : Word F m →ₗ[F] Word F h) (hker : LinearMap.ker H0 = C)
    (hh : h = m - k) :
    HasLocalBlockMinorCapacity
      (parityCheckMatrix (blockParityCheck (B := B) H0)) h := by
  intro r S hS
  rw [exists_rows_det_submatrix_ne_zero_iff_rank_eq]
  rw [rank_parityCheckMatrix_selection]
  rw [hC.columnRank_blockParityCheck H0 hker hh]
  have hle : localBlockRank h S ≤ S.card := by
    unfold localBlockRank
    calc
      _ ≤ ∑ b : Fin B, (localCoordinateSet S b).card :=
        Finset.sum_le_sum fun b hb => min_le_left _ _
      _ = S.card := (card_eq_sum_card_localCoordinateSet S).symm
  omega

/-- The paper's concrete MDS-block statement: generic rank of the backward
two-difference product is exactly the maximum number of disjoint paths that
obey the `h`-per-block outer capacities. -/
theorem IsMDS.genericBackwardRank_is_maximalBlockCapacityPacking
    {F : Type*} [Field F] {B m k h : ℕ}
    {C : Code F m} (hC : IsMDS C k)
    (H0 : Word F m →ₗ[F] Word F h) (hker : LinearMap.ker H0 = C)
    (hh : h = m - k) :
    let H := parityCheckMatrix (blockParityCheck (B := B) H0)
    (∃ (cols : Fin (genericLayeredMatrixRank H
          (orderedDifferenceMatrix F (B * m))
          (orderedDifferenceMatrix F (B * m))) → Fin (B * m)),
      HasTwoDifferenceCapacityPacking h cols) ∧
    (∀ {r : ℕ} (cols : Fin r → Fin (B * m)),
      HasTwoDifferenceCapacityPacking h cols →
        r ≤ genericLayeredMatrixRank H
          (orderedDifferenceMatrix F (B * m))
          (orderedDifferenceMatrix F (B * m))) := by
  dsimp only
  exact genericTwoDifferenceRank_is_maximalCapacityPacking _
    (hC.parityCheckMatrix_hasLocalBlockMinorCapacity H0 hker hh)

end BAA
