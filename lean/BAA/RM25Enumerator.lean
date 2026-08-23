import BAA.RM25Distance

namespace BAA

open scoped Polynomial

noncomputable def rm25ProfilePolynomial (p : RM25SparseProfile) : Polynomial ℕ :=
  Polynomial.C p.1 +
  Polynomial.C p.2.1 * Polynomial.X ^ 8 +
  Polynomial.C p.2.2.1 * Polynomial.X ^ 12 +
  Polynomial.C p.2.2.2.1 * Polynomial.X ^ 16 +
  Polynomial.C p.2.2.2.2.1 * Polynomial.X ^ 20 +
  Polynomial.C p.2.2.2.2.2.1 * Polynomial.X ^ 24 +
  Polynomial.C p.2.2.2.2.2.2 * Polynomial.X ^ 32

theorem rm25ProfilePolynomial_increment {p : RM25SparseProfile} {w : ℕ}
    (hw : rm25RecognizedWeight w) :
    rm25ProfilePolynomial (rm25ProfileIncrement p w) =
      rm25ProfilePolynomial p + Polynomial.X ^ w := by
  rcases p with ⟨c0, c8, c12, c16, c20, c24, c32⟩
  rcases hw with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [rm25ProfilePolynomial, rm25ProfileIncrement] <;> ring

theorem rm25ProfilePolynomial_add (p q : RM25SparseProfile) :
    rm25ProfilePolynomial (rm25ProfileAdd p q) =
      rm25ProfilePolynomial p + rm25ProfilePolynomial q := by
  rcases p with ⟨a0, a8, a12, a16, a20, a24, a32⟩
  rcases q with ⟨b0, b8, b12, b16, b20, b24, b32⟩
  simp [rm25ProfilePolynomial, rm25ProfileAdd, map_add]
  ring

theorem rm25ProfilePolynomial_foldEncode (hi : Fin 256)
    (l : List (Fin 256)) (p : RM25SparseProfile)
    (hall : ∀ lo ∈ l, rm25RecognizedWeight
      (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)))) :
    rm25ProfilePolynomial
        (l.foldl (fun counts lo => rm25ProfileIncrement counts
          (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)))) p) =
      rm25ProfilePolynomial p +
        (l.map fun lo => Polynomial.X ^
          bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))).sum := by
  induction l generalizing p with
  | nil => simp
  | cons lo l ih =>
      simp only [List.foldl_cons]
      rw [ih (rm25ProfileIncrement p
        (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))))]
      · rw [rm25ProfilePolynomial_increment (hall lo (by simp))]
        simp [add_assoc]
      · intro x hx
        exact hall x (by simp [hx])

theorem rm25ChunkPolynomial (hi : Fin 256) :
    rm25ProfilePolynomial (rm25ChunkProfile hi) =
      ∑ lo : Fin 256,
        Polynomial.X ^ bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)) := by
  have hfold := rm25ProfilePolynomial_foldEncode hi (List.finRange 256)
    (0, 0, 0, 0, 0, 0, 0) (fun lo _ => rm25EveryWeightRecognized hi lo)
  unfold rm25ChunkProfile at hfold
  rw [show rm25ProfilePolynomial (0, 0, 0, 0, 0, 0, 0) = 0 by
    simp [rm25ProfilePolynomial]] at hfold
  simp only [zero_add] at hfold
  change rm25ProfilePolynomial
      ((List.finRange 256).foldl (fun counts lo => rm25ProfileIncrement counts
        (bitVecWeight (rm25Encode (rm25ChunkMessage hi lo))))
        (0, 0, 0, 0, 0, 0, 0)) = _
  rw [hfold]
  simp only [← List.sum_toFinset _ (List.nodup_finRange 256),
    List.toFinset_finRange]

theorem rm25ProfilePolynomial_foldAdd (l : List (Fin 256))
    (f : Fin 256 → RM25SparseProfile) (p : RM25SparseProfile) :
    rm25ProfilePolynomial (l.foldl (fun acc i => rm25ProfileAdd acc (f i)) p) =
      rm25ProfilePolynomial p + (l.map fun i => rm25ProfilePolynomial (f i)).sum := by
  induction l generalizing p with
  | nil => simp
  | cons i l ih =>
      simp only [List.foldl_cons]
      rw [ih, rm25ProfilePolynomial_add]
      simp [add_assoc]

theorem rm25SumExpectedChunkPolynomials :
    (∑ hi : Fin 256, rm25ProfilePolynomial (rm25ExpectedChunkProfile hi)) =
      rm25ProfilePolynomial rm25SummedExpectedProfile := by
  have hfold := rm25ProfilePolynomial_foldAdd (List.finRange 256)
    rm25ExpectedChunkProfile (0, 0, 0, 0, 0, 0, 0)
  unfold rm25SummedExpectedProfile at hfold
  rw [show rm25ProfilePolynomial (0, 0, 0, 0, 0, 0, 0) = 0 by
    simp [rm25ProfilePolynomial]] at hfold
  simp only [zero_add] at hfold
  change _ = rm25ProfilePolynomial
    ((List.finRange 256).foldl
      (fun acc i => rm25ProfileAdd acc (rm25ExpectedChunkProfile i))
      (0, 0, 0, 0, 0, 0, 0))
  rw [hfold]
  simp only [← List.sum_toFinset _ (List.nodup_finRange 256),
    List.toFinset_finRange]

def rm25ChunkEquiv : Fin 256 × Fin 256 ≃ BitVec 16 where
  toFun p := rm25ChunkMessage p.1 p.2
  invFun b :=
    (⟨b.toNat / 256, by have := b.isLt; omega⟩,
      ⟨b.toNat % 256, Nat.mod_lt _ (by omega)⟩)
  left_inv p := by
    rcases p with ⟨hi, lo⟩
    apply Prod.ext
    · apply Fin.ext
      simp [rm25ChunkMessage]
      omega
    · apply Fin.ext
      simp [rm25ChunkMessage]
  right_inv b := by
    apply BitVec.toNat_injective
    simp [rm25ChunkMessage]
    have hdecomp := Nat.mod_add_div b.toNat 256
    omega

theorem rm25ChunkEquiv_apply (hi lo : Fin 256) :
    rm25ChunkEquiv (hi, lo) = rm25ChunkMessage hi lo := by
  rfl

noncomputable def rm25MessageCodeEquiv : BinaryWord 16 ≃ rm25Code :=
  (LinearEquiv.ofInjective rm25OrderedEvaluation
    rm25OrderedEvaluation_injective).toEquiv

noncomputable def rm25BitVecCodeEquiv : BitVec 16 ≃ rm25Code :=
  (bitVecWordEquiv 16).trans rm25MessageCodeEquiv

theorem rm25BitVecCodeEquiv_weight (m : BitVec 16) :
    hammingNorm ((rm25BitVecCodeEquiv m : rm25Code) : BinaryWord 32) =
      bitVecWeight (rm25Encode m) := by
  change hammingNorm (rm25OrderedEvaluation (bitVecToWord m)) = _
  rw [rm25OrderedEvaluation_bridge, hammingNorm_bitVecToWord]

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 100000 in
theorem weightEnumerator_rm25Code :
    weightEnumerator rm25Code =
      1 + 620 * Polynomial.X ^ 8 + 13888 * Polynomial.X ^ 12 +
      36518 * Polynomial.X ^ 16 + 13888 * Polynomial.X ^ 20 +
      620 * Polynomial.X ^ 24 + Polynomial.X ^ 32 := by
  unfold weightEnumerator
  calc
    (∑ c : rm25Code,
        Polynomial.X ^ hammingNorm (c : BinaryWord 32)) =
        ∑ m : BitVec 16,
          Polynomial.X ^ hammingNorm
            ((rm25BitVecCodeEquiv m : rm25Code) : BinaryWord 32) := by
      symm
      exact Fintype.sum_equiv rm25BitVecCodeEquiv _ _ (fun _ => rfl)
    _ = ∑ m : BitVec 16, Polynomial.X ^ bitVecWeight (rm25Encode m) := by
      apply Fintype.sum_congr
      intro m
      rw [rm25BitVecCodeEquiv_weight]
    _ = ∑ p : Fin 256 × Fin 256,
        Polynomial.X ^ bitVecWeight (rm25Encode (rm25ChunkEquiv p)) := by
      symm
      exact Fintype.sum_equiv rm25ChunkEquiv _ _ (fun _ => rfl)
    _ = ∑ hi : Fin 256, ∑ lo : Fin 256,
        Polynomial.X ^ bitVecWeight (rm25Encode (rm25ChunkMessage hi lo)) := by
      rw [Fintype.sum_prod_type]
      apply Fintype.sum_congr
      intro hi
      apply Fintype.sum_congr
      intro lo
      rw [rm25ChunkEquiv_apply]
    _ = ∑ hi : Fin 256, rm25ProfilePolynomial (rm25ChunkProfile hi) := by
      apply Fintype.sum_congr
      intro hi
      exact (rm25ChunkPolynomial hi).symm
    _ = ∑ hi : Fin 256,
        rm25ProfilePolynomial (rm25ExpectedChunkProfile hi) := by
      apply Fintype.sum_congr
      intro hi
      rw [rm25ChunkWeightSpectrum]
    _ = rm25ProfilePolynomial rm25SummedExpectedProfile :=
      rm25SumExpectedChunkPolynomials
    _ = rm25ProfilePolynomial (1, 620, 13888, 36518, 13888, 620, 1) := by
      rw [rm25WeightSpectrum]
    _ = 1 + 620 * Polynomial.X ^ 8 + 13888 * Polynomial.X ^ 12 +
        36518 * Polynomial.X ^ 16 + 13888 * Polynomial.X ^ 20 +
        620 * Polynomial.X ^ 24 + Polynomial.X ^ 32 := by
      simp [rm25ProfilePolynomial]

end BAA
