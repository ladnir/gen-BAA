import BAA.AppendixCode
import Mathlib.LinearAlgebra.Matrix.ToLin

/-!
# The concrete extended primitive binary BCH code of length 32

Field elements are represented by five polynomial-basis bits.  Multiplication
reduces modulo `X^5 + X^2 + 1`; hence `0b00010` is the distinguished element
`alpha`.  The root checks use coordinates zero through thirty as coefficients
in least-significant-bit order and coordinate thirty-one as the parity bit.
-/

namespace BAA

abbrev GF32Bits := BitVec 5

def gf32Zero : GF32Bits := BitVec.ofNat 5 0
def gf32One : GF32Bits := BitVec.ofNat 5 1
def gf32Alpha : GF32Bits := BitVec.ofNat 5 2

def gf32Add (a b : GF32Bits) : GF32Bits := BitVec.xor a b

/-- Multiplication by `X`, reduced by `X^5 = X^2 + 1`. -/
def gf32XTime (a : GF32Bits) : GF32Bits :=
  let shifted := BitVec.ofNat 5 (2 * a.toNat)
  if a.getLsbD 4 then BitVec.xor shifted (BitVec.ofNat 5 0x05) else shifted

private def gf32Select (b : Bool) (a : GF32Bits) : GF32Bits :=
  if b then a else gf32Zero

/-- Carryless polynomial multiplication followed by reduction modulo
`X^5 + X^2 + 1`. -/
def gf32Mul (a b : GF32Bits) : GF32Bits :=
  gf32Add (gf32Select (b.getLsbD 0) a) <|
  gf32Add (gf32Select (b.getLsbD 1) (gf32XTime a)) <|
  gf32Add (gf32Select (b.getLsbD 2) (gf32XTime (gf32XTime a))) <|
  gf32Add (gf32Select (b.getLsbD 3) (gf32XTime (gf32XTime (gf32XTime a)))) <|
    gf32Select (b.getLsbD 4)
      (gf32XTime (gf32XTime (gf32XTime (gf32XTime a))))

def gf32Pow (a : GF32Bits) : ℕ → GF32Bits
  | 0 => gf32One
  | n + 1 => gf32Mul (gf32Pow a n) a

theorem gf32_inverse_law :
    ∀ a : GF32Bits, a ≠ gf32Zero →
      ∃ b : GF32Bits, gf32Mul a b = gf32One := by
  set_option maxRecDepth 100000 in
    decide

/-- `alpha` has multiplicative order 31, certifying the primitive-element
convention used by the BCH root checks. -/
theorem gf32Alpha_order_certificate :
    gf32Pow gf32Alpha 31 = gf32One ∧
      ∀ e : Fin 30, gf32Pow gf32Alpha (e.val + 1) ≠ gf32One := by
  set_option maxRecDepth 100000 in
    decide

theorem gf32Alpha_powers_injective :
    Function.Injective (fun e : Fin 31 => gf32Pow gf32Alpha e.val) := by
  set_option maxRecDepth 100000 in
    decide

theorem gf32Alpha_powers_nonzero :
    ∀ e : Fin 31, gf32Pow gf32Alpha e.val ≠ gf32Zero := by
  set_option maxRecDepth 100000 in
    decide

/-- One output bit of the root evaluation at `alpha^(j+1)`. -/
def bchRootSyndromeBit (x : BinaryWord 32) (j : Fin 10) (k : Fin 5) : ZMod 2 :=
  ∑ i : Fin 31,
    if (gf32Pow gf32Alpha ((j.val + 1) * i.val)).getLsbD k.val
      then x (Fin.castLE (by omega) i) else 0

def bchParitySyndrome (x : BinaryWord 32) : ZMod 2 :=
  ∑ i : Fin 32, x i

/-- The ten consecutive primitive root checks, together with even parity. -/
def extendedPrimitiveBCHCheck :
    BinaryWord 32 →ₗ[ZMod 2] ((Fin 10 → Fin 5 → ZMod 2) × ZMod 2) where
  toFun x := (fun j k => bchRootSyndromeBit x j k, bchParitySyndrome x)
  map_add' x y := by
    apply Prod.ext
    · funext j k
      change bchRootSyndromeBit (x + y) j k =
        bchRootSyndromeBit x j k + bchRootSyndromeBit y j k
      simp only [bchRootSyndromeBit, Pi.add_apply]
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      split_ifs <;> simp
    · simp [bchParitySyndrome, Finset.sum_add_distrib]
  map_smul' c x := by
    apply Prod.ext
    · funext j k
      change bchRootSyndromeBit (c • x) j k =
        c • bchRootSyndromeBit x j k
      simp only [bchRootSyndromeBit, Pi.smul_apply, smul_eq_mul]
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      split_ifs <;> simp
    · simp [bchParitySyndrome, Finset.mul_sum]

def extendedPrimitiveBCHCode : Code (ZMod 2) 32 :=
  LinearMap.ker extendedPrimitiveBCHCheck

/-! ## An invertible 21-coordinate parity-check minor -/

def bchIndependentRootIndex (r : Fin 20) : Fin 10 :=
  ⟨2 * (r.val / 5), by omega⟩

def bchIndependentBitIndex (r : Fin 20) : Fin 5 :=
  ⟨r.val % 5, Nat.mod_lt _ (by omega)⟩

/-- Select all five coordinate bits of the root checks at exponents
`1, 3, 5, 7`, followed by the parity check. -/
def bchIndependentProjection :
    ((Fin 10 → Fin 5 → ZMod 2) × ZMod 2) →ₗ[ZMod 2] BinaryWord 21 where
  toFun s r := if hr : r.val < 20 then
    s.1 (bchIndependentRootIndex ⟨r.val, hr⟩)
      (bchIndependentBitIndex ⟨r.val, hr⟩)
  else s.2
  map_add' x y := by
    funext r
    by_cases hr : r.val < 20 <;> simp [hr]
  map_smul' c x := by
    funext r
    by_cases hr : r.val < 20 <;> simp [hr]

/-- Embed the first 21 coordinates and set the final 11 coordinates to zero. -/
def bchLowEmbedding : BinaryWord 21 →ₗ[ZMod 2] BinaryWord 32 where
  toFun x i := if hi : i.val < 21 then x ⟨i.val, hi⟩ else 0
  map_add' x y := by
    funext i
    by_cases hi : i.val < 21 <;> simp [hi]
  map_smul' c x := by
    funext i
    by_cases hi : i.val < 21 <;> simp [hi]

def bchLowCheck : BinaryWord 21 →ₗ[ZMod 2] BinaryWord 21 :=
  bchIndependentProjection.comp (extendedPrimitiveBCHCheck.comp bchLowEmbedding)

def bchLowCheckMatrix : Matrix (Fin 21) (Fin 21) (ZMod 2) :=
  LinearMap.toMatrix' bchLowCheck

def bchLowInverseRow : Fin 21 → BitVec 21 :=
  ![BitVec.ofNat 21 0x1129B1, BitVec.ofNat 21 0x0FDC92,
    BitVec.ofNat 21 0x1C1995, BitVec.ofNat 21 0x0F61DA,
    BitVec.ofNat 21 0x1B5765, BitVec.ofNat 21 0x0EC6BB,
    BitVec.ofNat 21 0x12D067, BitVec.ofNat 21 0x1DC3EF,
    BitVec.ofNat 21 0x06FE8E, BitVec.ofNat 21 0x19312C,
    BitVec.ofNat 21 0x153D18, BitVec.ofNat 21 0x0A65E0,
    BitVec.ofNat 21 0x015380, BitVec.ofNat 21 0x1FE971,
    BitVec.ofNat 21 0x0C0A12, BitVec.ofNat 21 0x0E5CA4,
    BitVec.ofNat 21 0x04B168, BitVec.ofNat 21 0x139601,
    BitVec.ofNat 21 0x1338A2, BitVec.ofNat 21 0x0548D4,
    BitVec.ofNat 21 0x1A2338]

def bchLowInverseMatrix : Matrix (Fin 21) (Fin 21) (ZMod 2) :=
  fun i j => if (bchLowInverseRow i).getLsb j then 1 else 0

set_option maxHeartbeats 1000000 in
theorem bchLowInverseMatrix_mul :
    bchLowInverseMatrix * bchLowCheckMatrix = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    set_option maxRecDepth 100000 in decide

theorem bchLowCheck_injective : Function.Injective bchLowCheck := by
  intro x y hxy
  have hmatrix : bchLowCheckMatrix.mulVec x = bchLowCheckMatrix.mulVec y := by
    simpa [bchLowCheckMatrix] using hxy
  have hinverse := congrArg bchLowInverseMatrix.mulVec hmatrix
  calc
    x = (1 : Matrix (Fin 21) (Fin 21) (ZMod 2)).mulVec x :=
      (Matrix.one_mulVec x).symm
    _ = (bchLowInverseMatrix * bchLowCheckMatrix).mulVec x := by
      rw [bchLowInverseMatrix_mul]
    _ = bchLowInverseMatrix.mulVec (bchLowCheckMatrix.mulVec x) :=
      (Matrix.mulVec_mulVec x bchLowInverseMatrix bchLowCheckMatrix).symm
    _ = bchLowInverseMatrix.mulVec (bchLowCheckMatrix.mulVec y) := hinverse
    _ = (bchLowInverseMatrix * bchLowCheckMatrix).mulVec y :=
      Matrix.mulVec_mulVec y bchLowInverseMatrix bchLowCheckMatrix
    _ = (1 : Matrix (Fin 21) (Fin 21) (ZMod 2)).mulVec y := by
      rw [bchLowInverseMatrix_mul]
    _ = y := Matrix.one_mulVec y

/-! ## A systematic 11-row encoder for the root-check kernel -/

def bchRow0 : BitVec 32 := BitVec.ofNat 32 0x003A6B7F
def bchRow1 : BitVec 32 := BitVec.ofNat 32 0x004EBD81
def bchRow2 : BitVec 32 := BitVec.ofNat 32 0x009D7B02
def bchRow3 : BitVec 32 := BitVec.ofNat 32 0x01009D7B
def bchRow4 : BitVec 32 := BitVec.ofNat 32 0x02013AF6
def bchRow5 : BitVec 32 := BitVec.ofNat 32 0x040275EC
def bchRow6 : BitVec 32 := BitVec.ofNat 32 0x0804EBD8
def bchRow7 : BitVec 32 := BitVec.ofNat 32 0x1009D7B0
def bchRow8 : BitVec 32 := BitVec.ofNat 32 0x2013AF60
def bchRow9 : BitVec 32 := BitVec.ofNat 32 0x401D35BF
def bchRow10 : BitVec 32 := BitVec.ofNat 32 0x801626D5

private def bchSelect (b : Bool) (row : BitVec 32) : BitVec 32 :=
  if b then row else BitVec.ofNat 32 0

/-- The rows are systematic in coordinates 21 through 31. -/
def bchEncode (m : BitVec 11) : BitVec 32 :=
  BitVec.xor (bchSelect (m.getLsbD 0) bchRow0) <|
  BitVec.xor (bchSelect (m.getLsbD 1) bchRow1) <|
  BitVec.xor (bchSelect (m.getLsbD 2) bchRow2) <|
  BitVec.xor (bchSelect (m.getLsbD 3) bchRow3) <|
  BitVec.xor (bchSelect (m.getLsbD 4) bchRow4) <|
  BitVec.xor (bchSelect (m.getLsbD 5) bchRow5) <|
  BitVec.xor (bchSelect (m.getLsbD 6) bchRow6) <|
  BitVec.xor (bchSelect (m.getLsbD 7) bchRow7) <|
  BitVec.xor (bchSelect (m.getLsbD 8) bchRow8) <|
  BitVec.xor (bchSelect (m.getLsbD 9) bchRow9) <|
    bchSelect (m.getLsbD 10) bchRow10

def bchDecode (c : BitVec 32) : BitVec 11 :=
  BitVec.ofNat 11 (c.toNat / 2 ^ 21)

set_option maxHeartbeats 1000000 in
theorem bchDecode_encode_fin :
    ∀ m : Fin 2048,
      bchDecode (bchEncode (BitVec.ofNat 11 m.val)) = BitVec.ofNat 11 m.val := by
  set_option maxRecDepth 100000 in
    decide

theorem bchDecode_encode (m : BitVec 11) : bchDecode (bchEncode m) = m := by
  let mf : Fin 2048 := ⟨m.toNat, by simpa using m.isLt⟩
  simpa [mf] using bchDecode_encode_fin mf

theorem bchEncode_injective : Function.Injective bchEncode :=
  Function.LeftInverse.injective bchDecode_encode

def bchWeightCount (w : ℕ) : ℕ :=
  (Finset.univ.filter fun m : Fin 2048 =>
    bitVecWeight (bchEncode (BitVec.ofNat 11 m.val)) = w).card

def bchExpectedWeightCount (w : ℕ) : ℕ :=
  if w = 0 then 1
  else if w = 12 then 496
  else if w = 16 then 1054
  else if w = 20 then 496
  else if w = 32 then 1
  else 0

set_option maxHeartbeats 2000000 in
theorem bchWeightSpectrum :
    ∀ w : Fin 33, bchWeightCount w.val = bchExpectedWeightCount w.val := by
  set_option maxRecDepth 100000 in
    decide

theorem bchMinimumWeight :
    (∀ m : Fin 2048, m ≠ 0 →
      12 ≤ bitVecWeight (bchEncode (BitVec.ofNat 11 m.val))) ∧
    (∃ m : Fin 2048,
      bitVecWeight (bchEncode (BitVec.ofNat 11 m.val)) = 12) := by
  set_option maxRecDepth 100000 in
    decide

def bchRowWord : Fin 11 → BinaryWord 32 :=
  ![bitVecToWord bchRow0, bitVecToWord bchRow1, bitVecToWord bchRow2,
    bitVecToWord bchRow3, bitVecToWord bchRow4, bitVecToWord bchRow5,
    bitVecToWord bchRow6, bitVecToWord bchRow7, bitVecToWord bchRow8,
    bitVecToWord bchRow9, bitVecToWord bchRow10]

def bchLinearMap : BinaryWord 11 →ₗ[ZMod 2] BinaryWord 32 where
  toFun m i := ∑ j : Fin 11, m j * bchRowWord j i
  map_add' x y := by
    funext i
    simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]
  map_smul' c x := by
    funext i
    simp [Finset.mul_sum, mul_assoc]

theorem bchRowWord_systematic :
    ∀ i j : Fin 11,
      bchRowWord j ⟨i.val + 21, by omega⟩ = if i = j then 1 else 0 := by
  decide

theorem bchLinearMap_systematic (m : BinaryWord 11) (i : Fin 11) :
    bchLinearMap m ⟨i.val + 21, by omega⟩ = m i := by
  simp [bchLinearMap, bchRowWord_systematic, mul_ite]

theorem bchLinearMap_injective : Function.Injective bchLinearMap := by
  intro x y hxy
  funext i
  rw [← bchLinearMap_systematic x i, ← bchLinearMap_systematic y i, hxy]

theorem zmod2Indicator_xor (a b : Bool) :
    (if a ^^ b then (1 : ZMod 2) else 0) =
      (if a then 1 else 0) + (if b then 1 else 0) := by
  cases a <;> cases b <;> decide

theorem zmod2Indicator_swap (a b : Bool) :
    (if a then (if b then (1 : ZMod 2) else 0) else 0) =
      if b then (if a then 1 else 0) else 0 := by
  cases a <;> cases b <;> rfl

theorem bitVecToWord_xor {n : ℕ} (x y : BitVec n) :
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
      Bool.xor (x.getLsbD i.val) (y.getLsbD i.val) := by
    exact BitVec.getLsbD_xor
  rw [hxor]
  exact zmod2Indicator_xor _ _

theorem bitVecToWord_bchSelect (b : Bool) (row : BitVec 32) :
    bitVecToWord (bchSelect b row) =
      if b then bitVecToWord row else 0 := by
  cases b with
  | false =>
      funext i
      simp [bchSelect, bitVecToWord, ← BitVec.getLsbD_eq_getElem]
  | true => simp [bchSelect]

theorem zmod2Indicator_smul_word {n : ℕ} (b : Bool) (x : BinaryWord n) :
    (if b then (1 : ZMod 2) else 0) • x = if b then x else 0 := by
  cases b <;> simp

theorem bchLinearMap_bridge (m : BitVec 11) :
    bchLinearMap (bitVecToWord m) = bitVecToWord (bchEncode m) := by
  unfold bchEncode
  simp only [bitVecToWord_xor, bitVecToWord_bchSelect]
  have hsum : bchLinearMap (bitVecToWord m) =
      ∑ j : Fin 11, (bitVecToWord m j) • bchRowWord j := by
    funext i
    simp [bchLinearMap]
  rw [hsum]
  simp [bchRowWord, Fin.sum_univ_succ, bitVecToWord,
    zmod2Indicator_smul_word]
  rfl

def bchGeneratorCode : Code (ZMod 2) 32 := bchLinearMap.range

theorem finrank_bchGeneratorCode :
    Module.finrank (ZMod 2) bchGeneratorCode = 11 := by
  rw [bchGeneratorCode, bchLinearMap.finrank_range_of_inj bchLinearMap_injective]
  simpa [BinaryWord, Word] using
    (Module.finrank_fin_fun (R := ZMod 2) (n := 11))

theorem bchGeneratorCode_weight_lower (c : bchGeneratorCode) (hc : c ≠ 0) :
    12 ≤ hammingNorm (c : BinaryWord 32) := by
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
  let mf : Fin 2048 := ⟨b.toNat, by simpa using b.isLt⟩
  have hmf0 : mf ≠ 0 := by
    intro hzero
    apply hb0
    apply BitVec.toNat_injective
    have hv := congrArg Fin.val hzero
    simpa [mf] using hv
  have hcert := bchMinimumWeight.1 mf hmf0
  have henc : bchLinearMap m = bitVecToWord (bchEncode b) := by
    rw [← bitVecToWord_wordToBitVec m]
    exact bchLinearMap_bridge b
  calc
    12 ≤ bitVecWeight (bchEncode (BitVec.ofNat 11 mf.val)) := hcert
    _ = bitVecWeight (bchEncode b) := by simp [mf]
    _ = hammingNorm (bitVecToWord (bchEncode b)) :=
      (hammingNorm_bitVecToWord _).symm
    _ = hammingNorm (bchLinearMap m) := congrArg hammingNorm henc.symm
    _ = hammingNorm (c : BinaryWord 32) := congrArg hammingNorm hm

theorem minimumDistance_bchGeneratorCode :
    minimumDistance bchGeneratorCode = 12 := by
  obtain ⟨mf, hweight⟩ := bchMinimumWeight.2
  let b := BitVec.ofNat 11 mf.val
  let m := bitVecToWord b
  let c : bchGeneratorCode := ⟨bchLinearMap m, ⟨m, rfl⟩⟩
  have hcweight : hammingNorm (c : BinaryWord 32) = 12 := by
    change hammingNorm (bchLinearMap m) = 12
    rw [bchLinearMap_bridge b, hammingNorm_bitVecToWord]
    exact hweight
  have hc0 : c ≠ 0 := by
    intro hc
    have hz' := congrArg
      (fun x : bchGeneratorCode => hammingNorm (x : BinaryWord 32)) hc
    have hz : hammingNorm (c : BinaryWord 32) = 0 := by simpa using hz'
    omega
  letI : Nontrivial bchGeneratorCode := ⟨⟨c, 0, hc0⟩⟩
  apply Nat.le_antisymm
  · simpa [hcweight] using minimumDistance_le_weight bchGeneratorCode c hc0
  · obtain ⟨d, hd0, hdweight⟩ := minimumDistance_attained bchGeneratorCode
    rw [← hdweight]
    exact bchGeneratorCode_weight_lower d hd0

theorem bchRowWord_extendedPrimitiveBCHCheck :
    ∀ j : Fin 11, extendedPrimitiveBCHCheck (bchRowWord j) = 0 := by
  set_option maxRecDepth 100000 in
    decide

theorem bchGeneratorCode_le_extendedPrimitiveBCHCode :
    bchGeneratorCode ≤ extendedPrimitiveBCHCode := by
  rintro c ⟨m, rfl⟩
  change extendedPrimitiveBCHCheck (bchLinearMap m) = 0
  have hrows : bchLinearMap m = ∑ j : Fin 11, m j • bchRowWord j := by
    funext i
    simp [bchLinearMap]
  rw [hrows, map_sum]
  apply Finset.sum_eq_zero
  intro j _
  rw [map_smul, bchRowWord_extendedPrimitiveBCHCheck]
  simp

def bchHighRestriction :
    extendedPrimitiveBCHCode →ₗ[ZMod 2] BinaryWord 11 where
  toFun c i := c.1 ⟨i.val + 21, by omega⟩
  map_add' x y := rfl
  map_smul' c x := rfl

theorem bchHighRestriction_injective : Function.Injective bchHighRestriction := by
  intro c d hcd
  apply Subtype.ext
  let residual : BinaryWord 32 := c.1 - d.1
  have hhigh : ∀ i : Fin 11, residual ⟨i.val + 21, by omega⟩ = 0 := by
    intro i
    apply sub_eq_zero.mpr
    exact congrFun hcd i
  have hresMem : residual ∈ extendedPrimitiveBCHCode :=
    Submodule.sub_mem _ c.2 d.2
  let low : BinaryWord 21 := fun i => residual (Fin.castAdd 11 i)
  have hpad : bchLowEmbedding low = residual := by
    funext i
    obtain ⟨j | j, rfl⟩ :=
      (finSumFinEquiv : Fin 21 ⊕ Fin 11 ≃ Fin 32).surjective i
    · rw [finSumFinEquiv_apply_left]
      simp [bchLowEmbedding, low]
    · rw [finSumFinEquiv_apply_right]
      have hindex : Fin.natAdd 21 j = ⟨j.val + 21, by omega⟩ := by
        apply Fin.ext
        simp
        omega
      rw [hindex]
      simpa [bchLowEmbedding, low] using (hhigh j).symm
  have hfull : extendedPrimitiveBCHCheck residual = 0 := by
    exact hresMem
  have hlow : bchLowCheck low = 0 := by
    change bchIndependentProjection
      (extendedPrimitiveBCHCheck (bchLowEmbedding low)) = 0
    rw [hpad, hfull]
    simp
  have hlow0 : low = 0 := by
    apply bchLowCheck_injective
    simpa using hlow
  have hresidual0 : residual = 0 := by
    rw [← hpad, hlow0]
    simp
  exact sub_eq_zero.mp hresidual0

theorem finrank_extendedPrimitiveBCHCode_le :
    Module.finrank (ZMod 2) extendedPrimitiveBCHCode ≤ 11 := by
  calc
    Module.finrank (ZMod 2) extendedPrimitiveBCHCode ≤
        Module.finrank (ZMod 2) (BinaryWord 11) :=
      bchHighRestriction.finrank_le_finrank_of_injective
        bchHighRestriction_injective
    _ = 11 := by
      simpa [BinaryWord, Word] using
        (Module.finrank_fin_fun (R := ZMod 2) (n := 11))

/-- The systematic 11-row encoder spans the entire concrete root-check
kernel. -/
theorem bchGeneratorCode_eq_extendedPrimitiveBCHCode :
    bchGeneratorCode = extendedPrimitiveBCHCode := by
  apply Submodule.eq_of_le_of_finrank_le
    bchGeneratorCode_le_extendedPrimitiveBCHCode
  rw [finrank_bchGeneratorCode]
  exact finrank_extendedPrimitiveBCHCode_le

theorem finrank_extendedPrimitiveBCHCode :
    Module.finrank (ZMod 2) extendedPrimitiveBCHCode = 11 := by
  rw [← bchGeneratorCode_eq_extendedPrimitiveBCHCode]
  exact finrank_bchGeneratorCode

theorem minimumDistance_extendedPrimitiveBCHCode :
    minimumDistance extendedPrimitiveBCHCode = 12 := by
  rw [← bchGeneratorCode_eq_extendedPrimitiveBCHCode]
  exact minimumDistance_bchGeneratorCode

/-- Each hexadecimal appendix row passes all ten concrete `GF(32)` root
checks and the parity check. -/
theorem appendixRowWord_extendedPrimitiveBCHCheck :
    ∀ j : Fin 8, extendedPrimitiveBCHCheck (appendixRowWord j) = 0 := by
  set_option maxRecDepth 100000 in
    decide

theorem appendixLinearMap_extendedPrimitiveBCHCheck (m : BinaryWord 8) :
    extendedPrimitiveBCHCheck (appendixLinearMap m) = 0 := by
  have hrows : appendixLinearMap m =
      ∑ j : Fin 8, m j • appendixRowWord j := by
    funext i
    simp [appendixLinearMap]
  rw [hrows, map_sum]
  apply Finset.sum_eq_zero
  intro j _
  rw [map_smul, appendixRowWord_extendedPrimitiveBCHCheck]
  simp

/-- The appendix `[32,8,12]` code is a subcode of the concrete extended
primitive BCH code. -/
theorem appendixCode_le_extendedPrimitiveBCHCode :
    appendixCode ≤ extendedPrimitiveBCHCode := by
  rintro c ⟨m, rfl⟩
  exact appendixLinearMap_extendedPrimitiveBCHCheck m

theorem appendixCode_le_bchGeneratorCode : appendixCode ≤ bchGeneratorCode := by
  rw [bchGeneratorCode_eq_extendedPrimitiveBCHCode]
  exact appendixCode_le_extendedPrimitiveBCHCode

end BAA
