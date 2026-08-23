import BAA.GolayAbstract

/-!
# Kernel certificates for the extended Golay spectrum

The 4,096 messages are partitioned by their high four bits into sixteen
256-message chunks.  This keeps each kernel reduction small and predictable.
-/

namespace BAA

/-- Select a parity-extended cyclic generator row without allocation. -/
def golayRowSelect (b : Bool) (row : BitVec 24) : BitVec 24 :=
  if b then row else BitVec.ofNat 24 0

/-- Cached parity-extended shifts of the cyclic generator `0xAE3`. -/
def golayLiteralRow (j : Fin 12) : BitVec 24 :=
  match j.val with
  | 0 => BitVec.ofNat 24 0x800AE3
  | 1 => BitVec.ofNat 24 0x8015C6
  | 2 => BitVec.ofNat 24 0x802B8C
  | 3 => BitVec.ofNat 24 0x805718
  | 4 => BitVec.ofNat 24 0x80AE30
  | 5 => BitVec.ofNat 24 0x815C60
  | 6 => BitVec.ofNat 24 0x82B8C0
  | 7 => BitVec.ofNat 24 0x857180
  | 8 => BitVec.ofNat 24 0x8AE300
  | 9 => BitVec.ofNat 24 0x95C600
  | 10 => BitVec.ofNat 24 0xAB8C00
  | _ => BitVec.ofNat 24 0xD71800

/-- The cached rows are exactly the parity extensions of the twelve shifted
cyclic-generator rows. -/
@[simp]
theorem golayLiteralRow_eq (j : Fin 12) :
    golayLiteralRow j = golayEncode (golayMessageRow j) := by
  fin_cases j <;> decide

/-- Executable encoder for the span of the twelve parity-extended cyclic
generator rows. -/
def golayRowEncode (m : BitVec 12) : BitVec 24 :=
  BitVec.xor (golayRowSelect (m.getLsbD 0) (golayLiteralRow 0)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 1) (golayLiteralRow 1)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 2) (golayLiteralRow 2)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 3) (golayLiteralRow 3)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 4) (golayLiteralRow 4)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 5) (golayLiteralRow 5)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 6) (golayLiteralRow 6)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 7) (golayLiteralRow 7)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 8) (golayLiteralRow 8)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 9) (golayLiteralRow 9)) <|
  BitVec.xor (golayRowSelect (m.getLsbD 10) (golayLiteralRow 10)) <|
    golayRowSelect (m.getLsbD 11) (golayLiteralRow 11)

def golayChunkMessage (hi : Fin 16) (lo : Fin 256) : BitVec 12 :=
  BitVec.ofNat 12 (lo.val + 256 * hi.val)

def golayChunkWeightCount (hi : Fin 16) (w : ℕ) : ℕ :=
  (Finset.univ.filter fun lo : Fin 256 =>
    bitVecWeight (golayRowEncode (golayChunkMessage hi lo)) = w).card

private def golayWeightProfileA (w : ℕ) : ℕ :=
  if w = 0 then 1 else if w = 8 then 130 else if w = 12 then 120
  else if w = 16 then 5 else 0

private def golayWeightProfileB (w : ℕ) : ℕ :=
  if w = 8 then 80 else if w = 12 then 160 else if w = 16 then 16 else 0

private def golayWeightProfileC (w : ℕ) : ℕ :=
  if w = 8 then 40 else if w = 12 then 176 else if w = 16 then 40 else 0

private def golayWeightProfileD (w : ℕ) : ℕ :=
  if w = 8 then 16 else if w = 12 then 160 else if w = 16 then 80 else 0

private def golayWeightProfileE (w : ℕ) : ℕ :=
  if w = 8 then 5 else if w = 12 then 120 else if w = 16 then 130
  else if w = 24 then 1 else 0

def golayExpectedChunkWeightCount (hi : Fin 16) (w : ℕ) : ℕ :=
  if hi.val = 0 then golayWeightProfileA w
  else if hi.val = 12 then golayWeightProfileE w
  else if hi.val = 1 ∨ hi.val = 2 ∨ hi.val = 5 ∨ hi.val = 10 then
    golayWeightProfileB w
  else if hi.val = 6 ∨ hi.val = 9 ∨ hi.val = 13 ∨ hi.val = 14 then
    golayWeightProfileD w
  else golayWeightProfileC w

set_option maxHeartbeats 4000000 in
theorem golayChunkWeightSpectrum :
    ∀ hi : Fin 16, ∀ w : Fin 25,
      golayChunkWeightCount hi w.val = golayExpectedChunkWeightCount hi w.val := by
  intro hi
  fin_cases hi <;>
    set_option maxRecDepth 100000 in decide

def golayWeightCount (w : ℕ) : ℕ :=
  ∑ hi : Fin 16, golayChunkWeightCount hi w

def golayExpectedWeightCount (w : ℕ) : ℕ :=
  if w = 0 then 1
  else if w = 8 then 759
  else if w = 12 then 2576
  else if w = 16 then 759
  else if w = 24 then 1
  else 0

theorem golayWeightSpectrum :
    ∀ w : Fin 25, golayWeightCount w.val = golayExpectedWeightCount w.val := by
  intro w
  rw [golayWeightCount]
  simp_rw [golayChunkWeightSpectrum]
  fin_cases w <;> decide

set_option maxHeartbeats 2000000 in
theorem golayMinimumWeight :
    (∀ hi : Fin 16, ∀ lo : Fin 256,
      hi ≠ 0 ∨ lo ≠ 0 →
        8 ≤ bitVecWeight (golayRowEncode (golayChunkMessage hi lo))) ∧
    (∃ hi : Fin 16, ∃ lo : Fin 256,
      bitVecWeight (golayRowEncode (golayChunkMessage hi lo)) = 8) := by
  constructor
  · intro hi
    fin_cases hi <;>
      set_option maxRecDepth 100000 in decide
  · exact ⟨0, 1, by decide⟩

/-! ## Bridge to the abstract linear code -/

def golayChunkHigh (m : Fin 4096) : Fin 16 :=
  ⟨m.val / 256, by omega⟩

def golayChunkLow (m : Fin 4096) : Fin 256 :=
  ⟨m.val % 256, Nat.mod_lt _ (by omega)⟩

theorem golayChunkMessage_high_low (m : Fin 4096) :
    golayChunkMessage (golayChunkHigh m) (golayChunkLow m) =
      BitVec.ofNat 12 m.val := by
  apply BitVec.toNat_injective
  simp [golayChunkMessage, golayChunkHigh, golayChunkLow,
    BitVec.toNat_ofNat, Nat.mod_add_div]

theorem golayChunkHigh_ne_zero_or_low_ne_zero {m : Fin 4096} (hm : m ≠ 0) :
    golayChunkHigh m ≠ 0 ∨ golayChunkLow m ≠ 0 := by
  by_contra h
  push_neg at h
  apply hm
  apply Fin.ext
  have hhigh := congrArg Fin.val h.1
  have hlow := congrArg Fin.val h.2
  have hdecomp := Nat.mod_add_div m.val 256
  simp [golayChunkHigh] at hhigh
  simp [golayChunkLow] at hlow
  omega

theorem golayIndicator_xor (a b : Bool) :
    (if Bool.xor a b then (1 : ZMod 2) else 0) =
      (if a then 1 else 0) + (if b then 1 else 0) := by
  cases a <;> cases b <;> decide

theorem bitVecToWord_golayXor {n : ℕ} (x y : BitVec n) :
    bitVecToWord (BitVec.xor x y) = bitVecToWord x + bitVecToWord y := by
  funext i
  simp only [bitVecToWord, Pi.add_apply]
  rw [show (BitVec.xor x y).getLsb i =
      (BitVec.xor x y).getLsbD i.val from by
        exact (BitVec.getLsbD_eq_getElem i.isLt).symm,
    show x.getLsb i = x.getLsbD i.val from
      (BitVec.getLsbD_eq_getElem i.isLt).symm,
    show y.getLsb i = y.getLsbD i.val from
      (BitVec.getLsbD_eq_getElem i.isLt).symm]
  have hxor : (BitVec.xor x y).getLsbD i.val =
      Bool.xor (x.getLsbD i.val) (y.getLsbD i.val) := by
    exact BitVec.getLsbD_xor
  rw [hxor]
  exact golayIndicator_xor _ _

theorem bitVecToWord_golayRowSelect (b : Bool) (row : BitVec 24) :
    bitVecToWord (golayRowSelect b row) =
      if b then bitVecToWord row else 0 := by
  cases b with
  | false =>
      funext i
      simp [golayRowSelect, bitVecToWord, ← BitVec.getLsbD_eq_getElem]
  | true => simp [golayRowSelect]

theorem golayIndicator_smul_word {n : ℕ} (b : Bool) (x : BinaryWord n) :
    (if b then (1 : ZMod 2) else 0) • x = if b then x else 0 := by
  cases b <;> simp

theorem golayLinearMap_bridge (m : BitVec 12) :
    golayLinearMap (bitVecToWord m) = bitVecToWord (golayRowEncode m) := by
  unfold golayRowEncode
  simp only [bitVecToWord_golayXor, bitVecToWord_golayRowSelect]
  have hsum : golayLinearMap (bitVecToWord m) =
      ∑ j : Fin 12, (bitVecToWord m j) • golayRowWord j := by
    funext i
    simp [golayLinearMap]
  rw [hsum]
  simp [golayRowWord, Fin.sum_univ_succ, bitVecToWord,
    golayIndicator_smul_word]

theorem golayBitVec_weight_lower (m : BitVec 12) (hm : m ≠ 0) :
    8 ≤ bitVecWeight (golayRowEncode m) := by
  let mf : Fin 4096 := ⟨m.toNat, by simpa using m.isLt⟩
  have hmf : mf ≠ 0 := by
    intro hzero
    apply hm
    apply BitVec.toNat_injective
    have hv := congrArg Fin.val hzero
    simpa [mf] using hv
  have hcert := golayMinimumWeight.1 (golayChunkHigh mf) (golayChunkLow mf)
    (golayChunkHigh_ne_zero_or_low_ne_zero hmf)
  rw [golayChunkMessage_high_low] at hcert
  simpa [mf] using hcert

theorem golayCode_weight_lower (c : golayCode) (hc : c ≠ 0) :
    8 ≤ hammingNorm (c : BinaryWord 24) := by
  obtain ⟨m, hm⟩ := c.2
  have hm0 : m ≠ 0 := by
    intro hmzero
    apply hc
    apply Subtype.ext
    rw [← hm]
    simp [hmzero]
  let b := wordToBitVec m
  have hb0 : b ≠ 0 := by
    intro hbzero
    apply hm0
    calc
      m = bitVecToWord b := (bitVecToWord_wordToBitVec m).symm
      _ = 0 := by
        rw [hbzero]
        funext i
        simp [bitVecToWord]
  have henc : golayLinearMap m = bitVecToWord (golayRowEncode b) := by
    rw [← bitVecToWord_wordToBitVec m]
    exact golayLinearMap_bridge b
  calc
    8 ≤ bitVecWeight (golayRowEncode b) := golayBitVec_weight_lower b hb0
    _ = hammingNorm (bitVecToWord (golayRowEncode b)) :=
      (hammingNorm_bitVecToWord _).symm
    _ = hammingNorm (golayLinearMap m) := congrArg hammingNorm henc.symm
    _ = hammingNorm (c : BinaryWord 24) := congrArg hammingNorm hm

theorem minimumDistance_golayCode : minimumDistance golayCode = 8 := by
  obtain ⟨hi, lo, hweight⟩ := golayMinimumWeight.2
  let b := golayChunkMessage hi lo
  let m := bitVecToWord b
  let c : golayCode := ⟨golayLinearMap m, ⟨m, rfl⟩⟩
  have hcweight : hammingNorm (c : BinaryWord 24) = 8 := by
    change hammingNorm (golayLinearMap m) = 8
    rw [golayLinearMap_bridge b, hammingNorm_bitVecToWord]
    exact hweight
  have hc0 : c ≠ 0 := by
    intro hc
    have hz' := congrArg
      (fun x : golayCode => hammingNorm (x : BinaryWord 24)) hc
    have hz : hammingNorm (c : BinaryWord 24) = 0 := by simpa using hz'
    omega
  letI : Nontrivial golayCode := ⟨⟨c, 0, hc0⟩⟩
  apply Nat.le_antisymm
  · simpa [hcweight] using minimumDistance_le_weight golayCode c hc0
  · obtain ⟨d, hd0, hdweight⟩ := minimumDistance_attained golayCode
    rw [← hdweight]
    exact golayCode_weight_lower d hd0

end BAA
