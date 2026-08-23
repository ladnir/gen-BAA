import BAA.LayeredDifferencePaths
import BAA.BlockObstruction

namespace BAA

/-- The outer matrix realizes exactly the sum of its local block capacities. -/
def HasLocalBlockMinorCapacity
    {F Rows : Type*} [Field F] {B m : ℕ}
    (H : Matrix Rows (Fin (B * m)) F) (h : ℕ) : Prop :=
  ∀ {r : ℕ} (S : Finset (Fin (B * m))) (hS : S.card = r),
    (∃ rows : Fin r → Rows,
      (H.submatrix rows (selectionEmbedding ⟨S, hS⟩)).det ≠ 0) ↔
      r ≤ localBlockRank h S

/-- Two disjoint difference linkages ending at admissible local capacities. -/
def HasTwoDifferenceCapacityPacking
    {B m r : ℕ} (h : ℕ)
    (cols : Fin r → Fin (B * m)) : Prop :=
  ∃ hcols : Function.Injective cols,
    ∃ (S T : Finset (Fin (B * m))) (hS : S.card = r) (hT : T.card = r),
      r ≤ localBlockRank h S ∧
      DifferenceLinkage (selectionEmbedding ⟨S, hS⟩)
        (selectionEmbedding ⟨T, hT⟩) ∧
      DifferenceLinkage (selectionEmbedding ⟨T, hT⟩) ⟨cols, hcols⟩

theorem genericTwoDifferenceRank_is_maximalCapacityPacking
    {F Rows : Type*} [Field F] [Fintype Rows] {B m h : ℕ}
    (H : Matrix Rows (Fin (B * m)) F)
    (hcap : HasLocalBlockMinorCapacity H h) :
    (∃ (cols : Fin (genericLayeredMatrixRank H
          (orderedDifferenceMatrix F (B * m))
          (orderedDifferenceMatrix F (B * m))) → Fin (B * m)),
      HasTwoDifferenceCapacityPacking h cols) ∧
    (∀ {r : ℕ} (cols : Fin r → Fin (B * m)),
      HasTwoDifferenceCapacityPacking h cols →
        r ≤ genericLayeredMatrixRank H
          (orderedDifferenceMatrix F (B * m))
          (orderedDifferenceMatrix F (B * m))) := by
  let D := orderedDifferenceMatrix F (B * m)
  constructor
  · obtain ⟨rows, cols, hpaths⟩ :=
      (genericTwoDifferenceRank_is_maximalPathPacking H).1
    obtain ⟨hcols, S, T, hS, hT, hH, hmid, hlast⟩ := hpaths
    exact ⟨cols, hcols, S, T, hS, hT,
      (hcap S hS).mp ⟨rows, hH⟩, hmid, hlast⟩
  · intro r cols hpaths
    obtain ⟨hcols, S, T, hS, hT, hcapacity, hmid, hlast⟩ := hpaths
    obtain ⟨rows, hH⟩ := (hcap S hS).mpr hcapacity
    apply (genericTwoDifferenceRank_is_maximalPathPacking H).2 rows cols
    exact ⟨hcols, S, T, hS, hT, hH, hmid, hlast⟩

end BAA
