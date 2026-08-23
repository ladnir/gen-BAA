import BAA.RM25Cert00
import BAA.RM25Cert01
import BAA.RM25Cert02
import BAA.RM25Cert03
import BAA.RM25Cert04
import BAA.RM25Cert05
import BAA.RM25Cert06
import BAA.RM25Cert07
import BAA.RM25Cert08
import BAA.RM25Cert09
import BAA.RM25Cert0A
import BAA.RM25Cert0B
import BAA.RM25Cert0C
import BAA.RM25Cert0D
import BAA.RM25Cert0E
import BAA.RM25Cert0F

namespace BAA

set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
/-- Aggregation of the sixteen cached 4,096-message RM spectrum blocks. -/
theorem rm25ChunkWeightSpectrum (hi : Fin 256) :
    rm25ChunkProfile hi = rm25ExpectedChunkProfile hi := by
  let block : Fin 16 := ⟨hi.val / 16, by omega⟩
  let offset : Fin 16 := ⟨hi.val % 16, Nat.mod_lt _ (by omega)⟩
  have hcert :
      rm25ChunkProfile (rm25ChunkIndex block offset) =
        rm25ExpectedChunkProfile (rm25ChunkIndex block offset) := by
    have hb : block.val = 0 ∨ block.val = 1 ∨ block.val = 2 ∨
        block.val = 3 ∨ block.val = 4 ∨ block.val = 5 ∨
        block.val = 6 ∨ block.val = 7 ∨ block.val = 8 ∨
        block.val = 9 ∨ block.val = 10 ∨ block.val = 11 ∨
        block.val = 12 ∨ block.val = 13 ∨ block.val = 14 ∨
        block.val = 15 := by omega
    rcases hb with h | h | h | h | h | h | h | h |
      h | h | h | h | h | h | h | h
    · have hk : block = (0 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_00 offset
    · have hk : block = (1 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_01 offset
    · have hk : block = (2 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_02 offset
    · have hk : block = (3 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_03 offset
    · have hk : block = (4 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_04 offset
    · have hk : block = (5 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_05 offset
    · have hk : block = (6 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_06 offset
    · have hk : block = (7 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_07 offset
    · have hk : block = (8 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_08 offset
    · have hk : block = (9 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_09 offset
    · have hk : block = (10 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_0A offset
    · have hk : block = (11 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_0B offset
    · have hk : block = (12 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_0C offset
    · have hk : block = (13 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_0D offset
    · have hk : block = (14 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_0E offset
    · have hk : block = (15 : Fin 16) := by
        apply Fin.ext
        simpa using h
      simpa [hk] using rm25ChunkWeightSpectrum_0F offset
  have hidx : rm25ChunkIndex block offset = hi := by
    apply Fin.ext
    simp [rm25ChunkIndex, block, offset]
    have hdecomp := Nat.mod_add_div hi.val 16
    omega
  simpa [hidx] using hcert

end BAA
