import BAA.LayeredCauchyBinet

namespace BAA

theorem exists_fullRank_minor
    {K Rows Cols : Type*} [Field K] [Fintype Rows] [Fintype Cols]
    (M : Matrix Rows Cols K) :
    ∃ (rows : Fin M.rank → Rows) (cols : Fin M.rank → Cols),
      (M.submatrix rows cols).det ≠ 0 := by
  classical
  obtain ⟨f₀, hf₀, -, hf₀ind⟩ :=
    Submodule.exists_fun_fin_finrank_span_eq K (Set.range M.col)
  let ec : Fin M.rank ≃
      Fin (Module.finrank K (Submodule.span K (Set.range M.col))) :=
    finCongr M.rank_eq_finrank_span_cols
  let f : Fin M.rank → (Rows → K) := f₀ ∘ ec
  have hf (i : Fin M.rank) : f i ∈ Set.range M.col := hf₀ (ec i)
  have hfind : LinearIndependent K f := hf₀ind.comp ec ec.injective
  choose cols hcols using hf
  have hcols_inj : Function.Injective cols := by
    intro i j hij
    apply hfind.injective
    rw [← hcols i, ← hcols j, hij]
  let N : Matrix Rows (Fin M.rank) K := M.submatrix id cols
  have hNcol (i : Fin M.rank) : N.col i = f i := by
    ext row
    exact congrFun (hcols i) row
  have hNind : LinearIndependent K N.col := by
    rw [show N.col = f from funext hNcol]
    exact hfind
  have hNrank : N.rank = M.rank := by
    rw [N.rank_eq_finrank_span_cols]
    have h := (linearIndependent_iff_card_eq_finrank_span.mp hNind).symm
    rw [Set.finrank, Fintype.card_fin] at h
    exact h
  obtain ⟨g₀, hg₀, -, hg₀ind⟩ :=
    Submodule.exists_fun_fin_finrank_span_eq K (Set.range N.row)
  have hrowdim : M.rank =
      Module.finrank K (Submodule.span K (Set.range N.row)) :=
    hNrank.symm.trans N.rank_eq_finrank_span_row
  let er : Fin M.rank ≃
      Fin (Module.finrank K (Submodule.span K (Set.range N.row))) :=
    finCongr hrowdim
  let g : Fin M.rank → (Fin M.rank → K) := g₀ ∘ er
  have hg (i : Fin M.rank) : g i ∈ Set.range N.row := hg₀ (er i)
  have hgind : LinearIndependent K g := hg₀ind.comp er er.injective
  choose rows hrows using hg
  refine ⟨rows, cols, ?_⟩
  let Q : Matrix (Fin M.rank) (Fin M.rank) K := M.submatrix rows cols
  have hQrow (i : Fin M.rank) : Q.row i = g i := by
    ext j
    exact congrFun (hrows i) j
  have hQind : LinearIndependent K Q.row := by
    rw [show Q.row = g from funext hQrow]
    exact hgind
  have hQunit : IsUnit Q := Matrix.linearIndependent_rows_iff_isUnit.mp hQind
  have hdetunit : IsUnit Q.det := (Matrix.isUnit_iff_isUnit_det Q).mp hQunit
  exact isUnit_iff_ne_zero.mp hdetunit

/-- The generic rank is attained by a layered minor packing. -/
theorem exists_layeredPacking_of_card_genericLayeredMatrixRank
    {F Rows I J Cols : Type*} [Field F]
    [Fintype Rows] [Fintype I] [Fintype J] [Fintype Cols]
    [DecidableEq I] [DecidableEq J]
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F) :
    ∃ (rows : Fin (genericLayeredMatrixRank A B C) → Rows)
        (cols : Fin (genericLayeredMatrixRank A B C) → Cols),
      HasLayeredMinorPacking A B C rows cols := by
  classical
  let P := genericLayeredMatrix A B C
  change ∃ (rows : Fin (genericMatrix P).rank → Rows)
      (cols : Fin (genericMatrix P).rank → Cols),
    HasLayeredMinorPacking A B C rows cols
  obtain ⟨rows, cols, hminorK⟩ := exists_fullRank_minor (genericMatrix P)
  refine ⟨rows, cols,
    (genericLayeredMinor_ne_zero_iff_layeredPacking A B C rows cols).mp ?_⟩
  intro hminor
  apply hminorK
  change ((P.map (algebraMap (GenericCoefficientRing (Sum I J) F)
    (GenericCoefficientField (Sum I J) F))).submatrix rows cols).det = 0
  rw [Matrix.submatrix_map]
  change ((algebraMap (GenericCoefficientRing (Sum I J) F)
    (GenericCoefficientField (Sum I J) F)).mapMatrix
      (P.submatrix rows cols)).det = 0
  rw [← (algebraMap (GenericCoefficientRing (Sum I J) F)
    (GenericCoefficientField (Sum I J) F)).map_det, hminor, map_zero]

/-- Exact algebraic packing characterization of generic layered rank: a
packing of rank size exists, and every packing has at most that size. -/
theorem genericLayeredMatrixRank_is_maximalPacking
    {F Rows I J Cols : Type*} [Field F]
    [Fintype Rows] [Fintype I] [Fintype J] [Fintype Cols]
    [DecidableEq I] [DecidableEq J]
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F) :
    (∃ (rows : Fin (genericLayeredMatrixRank A B C) → Rows)
        (cols : Fin (genericLayeredMatrixRank A B C) → Cols),
      HasLayeredMinorPacking A B C rows cols) ∧
    (∀ {r : ℕ} (rows : Fin r → Rows) (cols : Fin r → Cols),
      HasLayeredMinorPacking A B C rows cols →
        r ≤ genericLayeredMatrixRank A B C) := by
  refine ⟨exists_layeredPacking_of_card_genericLayeredMatrixRank A B C, ?_⟩
  intro r rows cols hpack
  exact hpack.card_le_genericLayeredMatrixRank A B C rows cols

end BAA
