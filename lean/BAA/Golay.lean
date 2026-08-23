import BAA.ConcreteBinary

/-!
# Extended binary Golay code

The length-23 cyclic encoder uses the paper's generator
`X^11 + X^9 + X^7 + X^6 + X^5 + X + 1` (`0xAE3`, least-significant
coefficient first).  The executable path is deliberately unrolled.
-/

namespace BAA

def golayGenerator : BitVec 23 := BitVec.ofNat 23 0xAE3

private def golaySelect {n : Nat} (b : Bool) (row : BitVec n) : BitVec n :=
  if b then row else BitVec.ofNat n 0

private def golayShiftedRow (j : Nat) : BitVec 23 :=
  BitVec.ofNat 23 (0xAE3 * 2 ^ j)

/-- Multiplication by the cyclic generator, unrolled over the twelve message
bits.  No reduction modulo `X^23 - 1` is needed because the largest degree is
22. -/
def golayCyclicEncode (m : BitVec 12) : BitVec 23 :=
  BitVec.xor (golaySelect (m.getLsbD 0) (golayShiftedRow 0)) <|
  BitVec.xor (golaySelect (m.getLsbD 1) (golayShiftedRow 1)) <|
  BitVec.xor (golaySelect (m.getLsbD 2) (golayShiftedRow 2)) <|
  BitVec.xor (golaySelect (m.getLsbD 3) (golayShiftedRow 3)) <|
  BitVec.xor (golaySelect (m.getLsbD 4) (golayShiftedRow 4)) <|
  BitVec.xor (golaySelect (m.getLsbD 5) (golayShiftedRow 5)) <|
  BitVec.xor (golaySelect (m.getLsbD 6) (golayShiftedRow 6)) <|
  BitVec.xor (golaySelect (m.getLsbD 7) (golayShiftedRow 7)) <|
  BitVec.xor (golaySelect (m.getLsbD 8) (golayShiftedRow 8)) <|
  BitVec.xor (golaySelect (m.getLsbD 9) (golayShiftedRow 9)) <|
  BitVec.xor (golaySelect (m.getLsbD 10) (golayShiftedRow 10)) <|
    golaySelect (m.getLsbD 11) (golayShiftedRow 11)

/-- Add coordinate 23 so that every encoded word has even weight. -/
def golayParityExtend (x : BitVec 23) : BitVec 24 :=
  BitVec.ofNat 24 (x.toNat + (bitVecWeight x % 2) * 2 ^ 23)

def golayEncode (m : BitVec 12) : BitVec 24 :=
  golayParityExtend (golayCyclicEncode m)

private def golayInverseShiftedRow (j : Nat) : BitVec 12 :=
  BitVec.ofNat 12 (0x49F * 2 ^ j)

/-- Multiply the low twelve cyclic-code coordinates by the inverse series
`0x49F`, which satisfies `0xAE3 * 0x49F = 1 mod X^12` over `GF(2)`. -/
def golayDecode (c : BitVec 23) : BitVec 12 :=
  BitVec.xor (golaySelect (c.getLsbD 0) (golayInverseShiftedRow 0)) <|
  BitVec.xor (golaySelect (c.getLsbD 1) (golayInverseShiftedRow 1)) <|
  BitVec.xor (golaySelect (c.getLsbD 2) (golayInverseShiftedRow 2)) <|
  BitVec.xor (golaySelect (c.getLsbD 3) (golayInverseShiftedRow 3)) <|
  BitVec.xor (golaySelect (c.getLsbD 4) (golayInverseShiftedRow 4)) <|
  BitVec.xor (golaySelect (c.getLsbD 5) (golayInverseShiftedRow 5)) <|
  BitVec.xor (golaySelect (c.getLsbD 6) (golayInverseShiftedRow 6)) <|
  BitVec.xor (golaySelect (c.getLsbD 7) (golayInverseShiftedRow 7)) <|
  BitVec.xor (golaySelect (c.getLsbD 8) (golayInverseShiftedRow 8)) <|
  BitVec.xor (golaySelect (c.getLsbD 9) (golayInverseShiftedRow 9)) <|
  BitVec.xor (golaySelect (c.getLsbD 10) (golayInverseShiftedRow 10)) <|
    golaySelect (c.getLsbD 11) (golayInverseShiftedRow 11)

set_option maxHeartbeats 1000000 in
theorem golayDecode_encode_coord :
    ∀ m : Fin 4096, ∀ i : Fin 12,
      (golayDecode (golayCyclicEncode (BitVec.ofNat 12 m.val))).getLsb i =
        (BitVec.ofNat 12 m.val).getLsb i := by
  set_option maxRecDepth 100000 in
  decide

theorem golayDecode_encode (m : BitVec 12) :
    golayDecode (golayCyclicEncode m) = m := by
  rw [BitVec.eq_of_getLsbD_eq_iff]
  intro i hi
  let mf : Fin 4096 := ⟨m.toNat, by simpa using m.isLt⟩
  simpa [mf, hi] using golayDecode_encode_coord mf ⟨i, hi⟩

theorem golayCyclicEncode_injective : Function.Injective golayCyclicEncode := by
  exact Function.LeftInverse.injective golayDecode_encode

end BAA
