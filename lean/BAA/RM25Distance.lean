import BAA.RM25Certificate

namespace BAA

theorem rm25Indicator_xor (a b : Bool) :
    (if Bool.xor a b then (1 : ZMod 2) else 0) =
      (if a then 1 else 0) + (if b then 1 else 0) := by
  cases a <;> cases b <;> decide

theorem bitVecToWord_rm25Xor {n : ℕ} (x y : BitVec n) :
    bitVecToWord (BitVec.xor x y) = bitVecToWord x + bitVecToWord y := by
  funext i
  simp only [bitVecToWord, Pi.add_apply]
  rw [show (BitVec.xor x y).getLsb i =
      (BitVec.xor x y).getLsbD i.val from
        (BitVec.getLsbD_eq_getElem i.isLt).symm,
    show x.getLsb i = x.getLsbD i.val from
      (BitVec.getLsbD_eq_getElem i.isLt).symm,
    show y.getLsb i = y.getLsbD i.val from
      (BitVec.getLsbD_eq_getElem i.isLt).symm]
  have hxor : (BitVec.xor x y).getLsbD i.val =
      Bool.xor (x.getLsbD i.val) (y.getLsbD i.val) := BitVec.getLsbD_xor
  rw [hxor]
  exact rm25Indicator_xor _ _

theorem bitVecToWord_rm25Select (b : Bool) (row : BitVec 32) :
    bitVecToWord (rm25Select b row) = if b then bitVecToWord row else 0 := by
  cases b with
  | false =>
      funext i
      simp [rm25Select, bitVecToWord, ← BitVec.getLsbD_eq_getElem]
  | true => simp [rm25Select]

theorem rm25Indicator_smul_word {n : ℕ} (b : Bool) (x : BinaryWord n) :
    (if b then (1 : ZMod 2) else 0) • x = if b then x else 0 := by
  cases b <;> simp

theorem rm25OrderedEvaluation_bridge (m : BitVec 16) :
    rm25OrderedEvaluation (bitVecToWord m) = bitVecToWord (rm25Encode m) := by
  unfold rm25Encode
  simp only [bitVecToWord_rm25Xor, bitVecToWord_rm25Select]
  have hsum : rm25OrderedEvaluation (bitVecToWord m) =
      ∑ j : Fin 16, (bitVecToWord m j) • bitVecToWord (rm25LiteralRow j) := by
    funext i
    simp [rm25OrderedEvaluation, ← rm25LiteralRow_correct]
  rw [hsum]
  simp [Fin.sum_univ_succ, bitVecToWord, rm25Indicator_smul_word]
  rfl

theorem rm25Encode_injective : Function.Injective rm25Encode := by
  intro x y hxy
  apply bitVecToWord_injective
  apply rm25OrderedEvaluation_injective
  rw [rm25OrderedEvaluation_bridge, rm25OrderedEvaluation_bridge, hxy]

def rm25RecognizedWeight (w : ℕ) : Prop :=
  w = 0 ∨ w = 8 ∨ w = 12 ∨ w = 16 ∨ w = 20 ∨ w = 24 ∨ w = 32

instance instDecidableRM25RecognizedWeight (w : ℕ) :
    Decidable (rm25RecognizedWeight w) := by
  unfold rm25RecognizedWeight
  infer_instance

def rm25ProfileTotal (p : RM25SparseProfile) : ℕ :=
  p.1 + p.2.1 + p.2.2.1 + p.2.2.2.1 + p.2.2.2.2.1 +
    p.2.2.2.2.2.1 + p.2.2.2.2.2.2

theorem rm25ProfileTotal_increment (p : RM25SparseProfile) (w : ℕ) :
    rm25ProfileTotal (rm25ProfileIncrement p w) =
      rm25ProfileTotal p + if rm25RecognizedWeight w then 1 else 0 := by
  rcases p with ⟨c0, c8, c12, c16, c20, c24, c32⟩
  simp only [rm25ProfileTotal, rm25ProfileIncrement, rm25RecognizedWeight]
  split_ifs <;> simp_all <;> omega

theorem rm25ProfileTotal_foldl (hi : Fin 256) (l : List (Fin 256))
    (p : RM25SparseProfile) :
    rm25ProfileTotal
        (l.foldl (fun counts lo => rm25ProfileIncrement counts
          (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)))) p) =
      rm25ProfileTotal p +
        (l.filter fun lo => decide (rm25RecognizedWeight
          (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))))).length := by
  induction l generalizing p with
  | nil => simp
  | cons lo l ih =>
      simp only [List.foldl_cons]
      rw [ih, rm25ProfileTotal_increment]
      by_cases h : rm25RecognizedWeight
          (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)))
      · simp [h]
        omega
      · simp [h]

theorem rm25ChunkProfile_total (hi : Fin 256) :
    rm25ProfileTotal (rm25ChunkProfile hi) =
      ((List.finRange 256).filter fun lo => decide (rm25RecognizedWeight
        (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))))).length := by
  unfold rm25ChunkProfile
  simpa [rm25ProfileTotal] using
    rm25ProfileTotal_foldl hi (List.finRange 256) (0, 0, 0, 0, 0, 0, 0)

theorem rm25ExpectedChunkProfile_total :
    ∀ hi : Fin 256, rm25ProfileTotal (rm25ExpectedChunkProfile hi) = 256 := by
  intro hi
  fin_cases hi <;> decide

theorem rm25EveryWeightRecognized (hi lo : Fin 256) :
    rm25RecognizedWeight
      (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))) := by
  have htotal : rm25ProfileTotal (rm25ChunkProfile hi) = 256 := by
    rw [rm25ChunkWeightSpectrum, rm25ExpectedChunkProfile_total]
  have hfilter := rm25ChunkProfile_total hi
  rw [htotal] at hfilter
  have hlength :
      ((List.finRange 256).filter fun x => decide (rm25RecognizedWeight
        (bitVecWeight (rm25Encode (rm25ChunkMessage hi x))))).length =
        (List.finRange 256).length := by
    simpa using hfilter.symm
  have hall := List.length_filter_eq_length_iff.mp hlength
  exact of_decide_eq_true (hall lo (List.mem_finRange lo))

theorem rm25BitVec_weight_lower (m : BitVec 16) (hm : m ≠ 0) :
    8 ≤ bitVecWeight (rm25Encode m) := by
  let hi : Fin 256 := ⟨m.toNat / 256, by have := m.isLt; omega⟩
  let lo : Fin 256 := ⟨m.toNat % 256, Nat.mod_lt _ (by omega)⟩
  have hchunk : rm25ChunkMessage hi lo = m := by
    apply BitVec.toNat_injective
    simp [rm25ChunkMessage, hi, lo]
    have hdecomp := Nat.mod_add_div m.toNat 256
    omega
  have hrecognized := rm25EveryWeightRecognized hi lo
  rw [hchunk] at hrecognized
  rcases hrecognized with hzero | h8 | h12 | h16 | h20 | h24 | h32
  · exfalso
    apply (rm25Encode_injective.ne hm)
    have henczero : rm25Encode (0 : BitVec 16) = 0 := by decide
    rw [henczero]
    apply bitVecToWord_injective
    calc
      bitVecToWord (rm25Encode m) = 0 := by
        apply hammingNorm_eq_zero.mp
        rw [hammingNorm_bitVecToWord, hzero]
      _ = bitVecToWord 0 := by
        funext i
        simp [bitVecToWord]
  all_goals omega

theorem rm25ChunkMessage_ne_zero {hi lo : Fin 256}
    (h : hi ≠ 0 ∨ lo ≠ 0) : rm25ChunkMessage hi lo ≠ 0 := by
  intro hz
  have hv := congrArg BitVec.toNat hz
  simp [rm25ChunkMessage] at hv
  rcases h with hhi | hlo
  · apply hhi
    apply Fin.ext
    omega
  · apply hlo
    apply Fin.ext
    omega

theorem rm25MinimumWeight :
    (∀ hi lo : Fin 256, hi ≠ 0 ∨ lo ≠ 0 →
      8 ≤ bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))) ∧
    (∃ hi lo : Fin 256,
      bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)) = 8) := by
  constructor
  · intro hi lo h
    exact rm25BitVec_weight_lower _ (rm25ChunkMessage_ne_zero h)
  · exact ⟨0, 2, by decide⟩

theorem rm25Code_weight_lower (c : rm25Code) (hc : c ≠ 0) :
    8 ≤ hammingNorm (c : BinaryWord 32) := by
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
  let hi : Fin 256 := ⟨b.toNat / 256, by have := b.isLt; omega⟩
  let lo : Fin 256 := ⟨b.toNat % 256, Nat.mod_lt _ (by omega)⟩
  have hchunk : rm25ChunkMessage hi lo = b := by
    apply BitVec.toNat_injective
    simp [rm25ChunkMessage, hi, lo]
    have hdecomp := Nat.mod_add_div b.toNat 256
    omega
  have hp : hi ≠ 0 ∨ lo ≠ 0 := by
    by_contra hp
    push Not at hp
    apply hb0
    rw [← hchunk]
    rw [hp.1, hp.2]
    rfl
  have hcert := rm25MinimumWeight.1 hi lo hp
  rw [hchunk] at hcert
  have henc : rm25OrderedEvaluation m = bitVecToWord (rm25Encode b) := by
    rw [← bitVecToWord_wordToBitVec m]
    exact rm25OrderedEvaluation_bridge b
  calc
    8 ≤ bitVecWeight (rm25Encode b) := hcert
    _ = hammingNorm (bitVecToWord (rm25Encode b)) :=
      (hammingNorm_bitVecToWord _).symm
    _ = hammingNorm (rm25OrderedEvaluation m) := congrArg hammingNorm henc.symm
    _ = hammingNorm (c : BinaryWord 32) := congrArg hammingNorm hm

theorem minimumDistance_rm25Code : minimumDistance rm25Code = 8 := by
  obtain ⟨hi, lo, hweight⟩ := rm25MinimumWeight.2
  let b := rm25ChunkMessage hi lo
  let m := bitVecToWord b
  let c : rm25Code := ⟨rm25OrderedEvaluation m, ⟨m, rfl⟩⟩
  have hcweight : hammingNorm (c : BinaryWord 32) = 8 := by
    change hammingNorm (rm25OrderedEvaluation m) = 8
    rw [rm25OrderedEvaluation_bridge b, hammingNorm_bitVecToWord]
    exact hweight
  have hc0 : c ≠ 0 := by
    intro hc
    have hz' := congrArg
      (fun x : rm25Code => hammingNorm (x : BinaryWord 32)) hc
    have hz : hammingNorm (c : BinaryWord 32) = 0 := by simpa using hz'
    omega
  letI : Nontrivial rm25Code := ⟨⟨c, 0, hc0⟩⟩
  apply Nat.le_antisymm
  · simpa [hcweight] using minimumDistance_le_weight rm25Code c hc0
  · obtain ⟨d, hd0, hdweight⟩ := minimumDistance_attained rm25Code
    rw [← hdweight]
    exact rm25Code_weight_lower d hd0

end BAA
