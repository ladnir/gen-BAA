import BAA.RM25Spectrum

namespace BAA

set_option maxHeartbeats 2000000 in
theorem rm25ChunkWeightSpectrum_04 :
    ∀ offset : Fin 16,
      rm25ChunkProfile (rm25ChunkIndex 4 offset) =
        rm25ExpectedChunkProfile (rm25ChunkIndex 4 offset) := by
  intro offset
  fin_cases offset <;> set_option maxRecDepth 100000 in decide

end BAA
