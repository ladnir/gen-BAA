import BAA.DifferenceLinkage
import BAA.GenericRankExact

namespace BAA

def HasTwoDifferencePathPacking
    {F Rows : Type*} [Field F] {N r : ℕ}
    (H : Matrix Rows (Fin N) F)
    (rows : Fin r → Rows) (cols : Fin r → Fin N) : Prop :=
  ∃ hcols : Function.Injective cols,
    ∃ (S T : Finset (Fin N)) (hS : S.card = r) (hT : T.card = r),
      (H.submatrix rows (selectionEmbedding ⟨S, hS⟩)).det ≠ 0 ∧
      DifferenceLinkage (selectionEmbedding ⟨S, hS⟩)
        (selectionEmbedding ⟨T, hT⟩) ∧
      DifferenceLinkage (selectionEmbedding ⟨T, hT⟩) ⟨cols, hcols⟩

theorem layeredPacking_twoDifference_iff
    {F Rows : Type*} [Field F] {N r : ℕ}
    (H : Matrix Rows (Fin N) F)
    (rows : Fin r → Rows) (cols : Fin r → Fin N) :
    HasLayeredMinorPacking H (orderedDifferenceMatrix F N)
        (orderedDifferenceMatrix F N) rows cols ↔
      HasTwoDifferencePathPacking H rows cols := by
  classical
  constructor
  · rintro ⟨S, T, hS, hT, hcoef⟩
    let s : Fin r ↪ Fin N := selectionEmbedding ⟨S, hS⟩
    let t : Fin r ↪ Fin N := selectionEmbedding ⟨T, hT⟩
    have hprod :
        (H.submatrix rows s).det *
          ((orderedDifferenceMatrix F N).submatrix s t).det *
          ((orderedDifferenceMatrix F N).submatrix t cols).det ≠ 0 := by
      exact hcoef
    have hleft : (H.submatrix rows s).det *
        ((orderedDifferenceMatrix F N).submatrix s t).det ≠ 0 :=
      left_ne_zero_of_mul hprod
    have hH : (H.submatrix rows s).det ≠ 0 :=
      left_ne_zero_of_mul hleft
    have hmid : ((orderedDifferenceMatrix F N).submatrix s t).det ≠ 0 :=
      right_ne_zero_of_mul hleft
    have hlast : ((orderedDifferenceMatrix F N).submatrix t cols).det ≠ 0 :=
      right_ne_zero_of_mul hprod
    have hcols : Function.Injective cols := by
      have hli := Matrix.linearIndependent_cols_of_det_ne_zero hlast
      intro i j hij
      apply hli.injective
      ext k
      simp [Matrix.submatrix_apply, hij]
    refine ⟨hcols, S, T, hS, hT, hH,
      (orderedDifferenceMatrix_minor_ne_zero_iff_linkage s t).mp hmid, ?_⟩
    exact (orderedDifferenceMatrix_minor_ne_zero_iff_linkage t ⟨cols, hcols⟩).mp hlast
  · rintro ⟨hcols, S, T, hS, hT, hH, hmid, hlast⟩
    refine ⟨S, T, hS, hT, ?_⟩
    change (H.submatrix rows (selectionEmbedding ⟨S, hS⟩)).det *
        ((orderedDifferenceMatrix F N).submatrix
          (selectionEmbedding ⟨S, hS⟩) (selectionEmbedding ⟨T, hT⟩)).det *
        ((orderedDifferenceMatrix F N).submatrix
          (selectionEmbedding ⟨T, hT⟩) cols).det ≠ 0
    apply mul_ne_zero (mul_ne_zero hH
      ((orderedDifferenceMatrix_minor_ne_zero_iff_linkage _ _).mpr hmid))
    exact (orderedDifferenceMatrix_minor_ne_zero_iff_linkage _ ⟨cols, hcols⟩).mpr hlast

/-- Exact path-packing characterization of the generic rank for two consecutive
first-difference layers.  The first minor is the outer acceptance condition;
the next two clauses are vertex-disjoint path linkages. -/
theorem genericTwoDifferenceRank_is_maximalPathPacking
    {F Rows : Type*} [Field F] [Fintype Rows] {N : ℕ}
    (H : Matrix Rows (Fin N) F) :
    (∃ (rows : Fin (genericLayeredMatrixRank H
          (orderedDifferenceMatrix F N) (orderedDifferenceMatrix F N)) → Rows)
        (cols : Fin (genericLayeredMatrixRank H
          (orderedDifferenceMatrix F N) (orderedDifferenceMatrix F N)) → Fin N),
      HasTwoDifferencePathPacking H rows cols) ∧
    (∀ {r : ℕ} (rows : Fin r → Rows) (cols : Fin r → Fin N),
      HasTwoDifferencePathPacking H rows cols →
        r ≤ genericLayeredMatrixRank H
          (orderedDifferenceMatrix F N) (orderedDifferenceMatrix F N)) := by
  constructor
  · obtain ⟨rows, cols, hpack⟩ :=
      exists_layeredPacking_of_card_genericLayeredMatrixRank H
        (orderedDifferenceMatrix F N) (orderedDifferenceMatrix F N)
    exact ⟨rows, cols, (layeredPacking_twoDifference_iff H rows cols).mp hpack⟩
  · intro r rows cols hpaths
    have hpack := (layeredPacking_twoDifference_iff H rows cols).mpr hpaths
    exact hpack.card_le_genericLayeredMatrixRank H
      (orderedDifferenceMatrix F N) (orderedDifferenceMatrix F N) rows cols

end BAA
