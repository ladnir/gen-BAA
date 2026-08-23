import BAA.DifferenceLinkage

namespace BAA

instance differenceStepDecidable {N : ℕ} (source target : Fin N) :
    Decidable (DifferenceStep source target) := by
  unfold DifferenceStep
  infer_instance

instance differenceLinkageDecidable {N r : ℕ} (rows cols : Fin r ↪ Fin N) :
    Decidable (DifferenceLinkage rows cols) := by
  unfold DifferenceLinkage
  infer_instance

set_option maxHeartbeats 1000000 in
/-- Executable regression for every ordered subset pair at lengths below eight. -/
theorem differenceMinorLinkage_finset_regression_below8 :
    ∀ (N : Fin 8) (S T : Finset (Fin N.val)) (hT : T.card = S.card),
      let rows : Fin S.card ↪ Fin N.val := (S.orderEmbOfFin rfl).toEmbedding
      let cols : Fin S.card ↪ Fin N.val := (T.orderEmbOfFin hT).toEmbedding
      (((orderedDifferenceMatrix ℚ N.val).submatrix rows cols).det ≠ 0 ↔
        DifferenceLinkage rows cols) := by
  native_decide

set_option maxHeartbeats 2000000 in
/-- Executable regression for every ordered subset pair at length eight. -/
theorem differenceMinorLinkage_finset_regression_N8 :
    ∀ (S T : Finset (Fin 8)) (hT : T.card = S.card),
      let rows : Fin S.card ↪ Fin 8 := (S.orderEmbOfFin rfl).toEmbedding
      let cols : Fin S.card ↪ Fin 8 := (T.orderEmbOfFin hT).toEmbedding
      (((orderedDifferenceMatrix ℚ 8).submatrix rows cols).det ≠ 0 ↔
        DifferenceLinkage rows cols) := by
  native_decide

end BAA
