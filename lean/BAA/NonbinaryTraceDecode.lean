import BAA.NonbinaryTrace
import Mathlib.Data.Finset.Sort

/-!
# Decoding compressed zero-return traces
-/

namespace BAA

/-- Decode the `i`-th compressed coordinate by inserting one mandatory
separator before every zero return. -/
def decodedReturnIndex {s a : ℕ} (t : CompressedZeroReturnTrace s a)
    (i : Fin a) : Fin s := by
  let b := t.1.orderEmbOfFin t.2 i
  have hcard : a ≤ s - a := by
    have hle : t.1.card ≤ s - a := by
      simpa using Finset.card_le_univ t.1
    simpa [t.2] using hle
  exact ⟨b.val + i.val + 1, by
    have hb := b.isLt
    have hi := i.isLt
    omega⟩

theorem decodedReturnIndex_strictMono {s a : ℕ}
    (t : CompressedZeroReturnTrace s a) :
    StrictMono (decodedReturnIndex t) := by
  intro i j hij
  have hb := (t.1.orderEmbOfFin t.2).strictMono hij
  change (t.1.orderEmbOfFin t.2 i).val + i.val + 1 <
    (t.1.orderEmbOfFin t.2 j).val + j.val + 1
  omega

/-- The actual active-symbol indices at which the state returns to zero. -/
def decodedZeroReturnTrace {s a : ℕ}
    (t : CompressedZeroReturnTrace s a) : Finset (Fin s) :=
  Finset.univ.image (decodedReturnIndex t)

@[simp]
theorem card_decodedZeroReturnTrace {s a : ℕ}
    (t : CompressedZeroReturnTrace s a) :
    (decodedZeroReturnTrace t).card = a := by
  rw [decodedZeroReturnTrace, Finset.card_image_of_injective _
    (decodedReturnIndex_strictMono t).injective]
  simp

theorem zero_not_mem_decodedZeroReturnTrace {s a : ℕ}
    (t : CompressedZeroReturnTrace s a) (hs : 0 < s) :
    (⟨0, hs⟩ : Fin s) ∉ decodedZeroReturnTrace t := by
  intro h
  rw [decodedZeroReturnTrace, Finset.mem_image] at h
  obtain ⟨i, _, hi⟩ := h
  have hpos : 0 < (decodedReturnIndex t i).val := by
    change 0 < (t.1.orderEmbOfFin t.2 i).val + i.val + 1
    omega
  have := congrArg Fin.val hi
  simp only at this
  omega

theorem decodedZeroReturnTrace_nonadjacent {s a : ℕ}
    (t : CompressedZeroReturnTrace s a) (i : Fin (s - 1)) :
    ¬ (⟨i.val, by omega⟩ ∈ decodedZeroReturnTrace t ∧
      ⟨i.val + 1, by omega⟩ ∈ decodedZeroReturnTrace t) := by
  rintro ⟨hi, hj⟩
  rw [decodedZeroReturnTrace, Finset.mem_image] at hi hj
  obtain ⟨p, _, hp⟩ := hi
  obtain ⟨q, _, hq⟩ := hj
  have hpval := congrArg Fin.val hp
  have hqval := congrArg Fin.val hq
  simp only at hpval hqval
  have hpq : p < q := by
    by_contra h
    have hqp : q ≤ p := le_of_not_gt h
    have hmono := (decodedReturnIndex_strictMono t).monotone hqp
    change (decodedReturnIndex t q).val ≤ (decodedReturnIndex t p).val at hmono
    omega
  have hsep : (decodedReturnIndex t p).val + 1 <
      (decodedReturnIndex t q).val := by
    have hb := (t.1.orderEmbOfFin t.2).strictMono hpq
    change (t.1.orderEmbOfFin t.2 p).val + p.val + 1 + 1 <
      (t.1.orderEmbOfFin t.2 q).val + q.val + 1
    omega
  omega

theorem decodedZeroReturnTrace_injective {s a : ℕ} :
    Function.Injective
      (@decodedZeroReturnTrace s a) := by
  intro t u htu
  apply Subtype.ext
  have hucard : (decodedZeroReturnTrace u).card = a :=
    card_decodedZeroReturnTrace u
  have huorder : decodedReturnIndex u =
      (decodedZeroReturnTrace u).orderEmbOfFin hucard :=
    Finset.orderEmbOfFin_unique hucard
      (fun i => Finset.mem_image.mpr ⟨i, Finset.mem_univ _, rfl⟩)
      (decodedReturnIndex_strictMono u)
  have htorderU : decodedReturnIndex t =
      (decodedZeroReturnTrace u).orderEmbOfFin hucard :=
    Finset.orderEmbOfFin_unique hucard
      (fun i => by
        rw [← htu]
        exact Finset.mem_image.mpr ⟨i, Finset.mem_univ _, rfl⟩)
      (decodedReturnIndex_strictMono t)
  have hdecoded : decodedReturnIndex t = decodedReturnIndex u := by
    exact htorderU.trans huorder.symm
  have hemb : t.1.orderEmbOfFin t.2 = u.1.orderEmbOfFin u.2 := by
    ext i
    have hi := congrFun hdecoded i
    change (t.1.orderEmbOfFin t.2 i).val =
      (u.1.orderEmbOfFin u.2 i).val
    have := congrArg Fin.val hi
    change (t.1.orderEmbOfFin t.2 i).val + i.val + 1 =
      (u.1.orderEmbOfFin u.2 i).val + i.val + 1 at this
    omega
  rw [← t.1.image_orderEmbOfFin_univ t.2,
    ← u.1.image_orderEmbOfFin_univ u.2, hemb]

/-! ## Encoding every valid trace -/

/-- Consecutive elements of a nonadjacent ordered trace differ by at least two. -/
theorem validTrace_orderEmb_succ_gap {s a : ℕ}
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A))
    (i : Fin (a - 1)) :
    (A.orderEmbOfFin hcard ⟨i.val, by omega⟩).val + 2 ≤
      (A.orderEmbOfFin hcard ⟨i.val + 1, by omega⟩).val := by
  have hlt := (A.orderEmbOfFin hcard).strictMono
    (show (⟨i.val, by omega⟩ : Fin a) < ⟨i.val + 1, by omega⟩ by simp)
  have hne :
      (A.orderEmbOfFin hcard ⟨i.val, by omega⟩).val + 1 ≠
        (A.orderEmbOfFin hcard ⟨i.val + 1, by omega⟩).val := by
    intro heq
    let q : Fin (s - 1) :=
      ⟨(A.orderEmbOfFin hcard ⟨i.val, by omega⟩).val, by
        have hnext := (A.orderEmbOfFin hcard ⟨i.val + 1, by omega⟩).isLt
        omega⟩
    apply hnonadj q
    constructor
    · simpa [q] using A.orderEmbOfFin_mem hcard ⟨i.val, by omega⟩
    · have hmem := A.orderEmbOfFin_mem hcard ⟨i.val + 1, by omega⟩
      simpa [q, heq] using hmem
  omega

/-- Along a valid trace, ordered return indices gain at least twice the
difference of their ranks. -/
theorem validTrace_orderEmb_two_mul_add_sub_le {s a : ℕ}
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A))
    {i j : Fin a} (hij : i ≤ j) :
    (A.orderEmbOfFin hcard i).val + 2 * (j.val - i.val) ≤
      (A.orderEmbOfFin hcard j).val := by
  generalize hd : j.val - i.val = d
  induction d generalizing j with
  | zero =>
      have hijEq : i = j := Fin.ext (by omega)
      subst j
      simp
  | succ d ih =>
      have hjpos : 0 < j.val := by omega
      let previous : Fin a := ⟨j.val - 1, by omega⟩
      have hiprevious : i ≤ previous := by
        change i.val ≤ j.val - 1
        omega
      have hdiff : previous.val - i.val = d := by
        dsimp [previous]
        omega
      have hind := ih hiprevious hdiff
      let q : Fin (a - 1) := ⟨j.val - 1, by omega⟩
      have hstep := validTrace_orderEmb_succ_gap A hcard hnonadj q
      have hqprev : (⟨q.val, by omega⟩ : Fin a) = previous := by
        apply Fin.ext
        simp [q, previous]
      have hqnext : (⟨q.val + 1, by omega⟩ : Fin a) = j := by
        apply Fin.ext
        simp [q]
        omega
      rw [hqprev, hqnext] at hstep
      omega

/-- Delete the mandatory separator preceding each ordered return. -/
def compressedReturnIndexOfValid {s a : ℕ}
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs : 0 < s, (⟨0, hs⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A))
    (i : Fin a) : Fin (s - a) := by
  have ha : 0 < a := lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have hfirstPos : 0 < (A.orderEmbOfFin hcard ⟨0, ha⟩).val := by
    have hmem := A.orderEmbOfFin_mem hcard ⟨0, ha⟩
    by_contra hz
    have hzval : (A.orderEmbOfFin hcard ⟨0, ha⟩).val = 0 := by omega
    have hs : 0 < s := lt_of_le_of_lt (Nat.zero_le _)
      (A.orderEmbOfFin hcard ⟨0, ha⟩).isLt
    have heq : A.orderEmbOfFin hcard ⟨0, ha⟩ = (⟨0, hs⟩ : Fin s) :=
      Fin.ext hzval
    exact hzero hs (heq ▸ hmem)
  have hlower := validTrace_orderEmb_two_mul_add_sub_le A hcard hnonadj
    (i := ⟨0, ha⟩) (j := i) (show 0 ≤ i.val by omega)
  have hlower' : i.val + 1 ≤ (A.orderEmbOfFin hcard i).val := by
    change (A.orderEmbOfFin hcard ⟨0, ha⟩).val + 2 * (i.val - 0) ≤
      (A.orderEmbOfFin hcard i).val at hlower
    omega
  have hlast := validTrace_orderEmb_two_mul_add_sub_le A hcard hnonadj
    (i := i) (j := ⟨a - 1, by omega⟩) (show i.val ≤ a - 1 by omega)
  have hlastBound := (A.orderEmbOfFin hcard ⟨a - 1, by omega⟩).isLt
  have hlast' : (A.orderEmbOfFin hcard i).val +
      2 * (a - 1 - i.val) < s := by
    change (A.orderEmbOfFin hcard i).val + 2 * ((a - 1) - i.val) ≤
      (A.orderEmbOfFin hcard ⟨a - 1, by omega⟩).val at hlast
    omega
  exact ⟨(A.orderEmbOfFin hcard i).val - i.val - 1, by
    change (A.orderEmbOfFin hcard i).val - i.val - 1 < s - a
    omega⟩

theorem compressedReturnIndexOfValid_strictMono {s a : ℕ}
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs : 0 < s, (⟨0, hs⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    StrictMono (compressedReturnIndexOfValid A hcard hzero hnonadj) := by
  intro i j hij
  have hgap := validTrace_orderEmb_two_mul_add_sub_le A hcard hnonadj
    (i := i) (j := j) hij.le
  change (A.orderEmbOfFin hcard i).val + 2 * (j.val - i.val) ≤
    (A.orderEmbOfFin hcard j).val at hgap
  simp only [compressedReturnIndexOfValid]
  change (A.orderEmbOfFin hcard i).val - i.val - 1 <
    (A.orderEmbOfFin hcard j).val - j.val - 1
  have ha : 0 < a := by omega
  have hfirstPos : 0 < (A.orderEmbOfFin hcard ⟨0, ha⟩).val := by
    have hmem := A.orderEmbOfFin_mem hcard ⟨0, ha⟩
    by_contra hz
    have hzval : (A.orderEmbOfFin hcard ⟨0, ha⟩).val = 0 := by omega
    have hs : 0 < s := by
      have := (A.orderEmbOfFin hcard ⟨0, ha⟩).isLt
      omega
    have heq : A.orderEmbOfFin hcard ⟨0, ha⟩ = (⟨0, hs⟩ : Fin s) :=
      Fin.ext hzval
    exact hzero hs (heq ▸ hmem)
  have hlower := validTrace_orderEmb_two_mul_add_sub_le A hcard hnonadj
    (i := ⟨0, ha⟩) (j := i) (show 0 ≤ i.val by omega)
  have hiLower : i.val + 1 ≤ (A.orderEmbOfFin hcard i).val := by
    change (A.orderEmbOfFin hcard ⟨0, ha⟩).val +
      2 * (i.val - 0) ≤ (A.orderEmbOfFin hcard i).val at hlower
    omega
  have hjLower : j.val + 1 ≤ (A.orderEmbOfFin hcard j).val := by
    have hlowerJ := validTrace_orderEmb_two_mul_add_sub_le A hcard hnonadj
      (i := ⟨0, ha⟩) (j := j) (show 0 ≤ j.val by omega)
    change (A.orderEmbOfFin hcard ⟨0, ha⟩).val +
      2 * (j.val - 0) ≤ (A.orderEmbOfFin hcard j).val at hlowerJ
    omega
  omega

/-- Canonically compress a valid zero-return trace. -/
def compressedTraceOfValid {s a : ℕ}
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs : 0 < s, (⟨0, hs⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    CompressedZeroReturnTrace s a :=
  ⟨Finset.univ.image (compressedReturnIndexOfValid A hcard hzero hnonadj), by
    rw [Finset.card_image_of_injective _
      (compressedReturnIndexOfValid_strictMono A hcard hzero hnonadj).injective]
    simp⟩

/-- Compression followed by decoding recovers every valid trace. -/
theorem decodedZeroReturnTrace_compressedTraceOfValid {s a : ℕ}
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs : 0 < s, (⟨0, hs⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    decodedZeroReturnTrace (compressedTraceOfValid A hcard hzero hnonadj) = A := by
  by_cases ha : a = 0
  · have hA : A = ∅ := Finset.card_eq_zero.mp (hcard.trans ha)
    calc
      decodedZeroReturnTrace (compressedTraceOfValid A hcard hzero hnonadj) = ∅ :=
        Finset.card_eq_zero.mp (by
          simpa [ha] using card_decodedZeroReturnTrace
            (compressedTraceOfValid A hcard hzero hnonadj))
      _ = A := hA.symm
  · have haPos : 0 < a := Nat.pos_of_ne_zero ha
    let t := compressedTraceOfValid A hcard hzero hnonadj
    have hcompressedOrder : t.1.orderEmbOfFin t.2 =
        compressedReturnIndexOfValid A hcard hzero hnonadj := by
      exact (Finset.orderEmbOfFin_unique t.2
        (fun i => Finset.mem_image.mpr ⟨i, Finset.mem_univ _, rfl⟩)
        (compressedReturnIndexOfValid_strictMono A hcard hzero hnonadj)).symm
    have hdecoded : decodedReturnIndex t = A.orderEmbOfFin hcard := by
      funext i
      apply Fin.ext
      change (t.1.orderEmbOfFin t.2 i).val + i.val + 1 =
        (A.orderEmbOfFin hcard i).val
      rw [hcompressedOrder]
      simp only [compressedReturnIndexOfValid]
      change (A.orderEmbOfFin hcard i).val - i.val - 1 + i.val + 1 =
        (A.orderEmbOfFin hcard i).val
      have hfirstPos : 0 <
          (A.orderEmbOfFin hcard ⟨0, haPos⟩).val := by
        have hmem := A.orderEmbOfFin_mem hcard ⟨0, haPos⟩
        by_contra hz
        have hzval : (A.orderEmbOfFin hcard ⟨0, haPos⟩).val = 0 := by omega
        have hs : 0 < s := by
          have := (A.orderEmbOfFin hcard ⟨0, haPos⟩).isLt
          omega
        have heq : A.orderEmbOfFin hcard ⟨0, haPos⟩ = (⟨0, hs⟩ : Fin s) :=
          Fin.ext hzval
        exact hzero hs (heq ▸ hmem)
      have hlower := validTrace_orderEmb_two_mul_add_sub_le A hcard hnonadj
        (i := ⟨0, haPos⟩) (j := i) (show 0 ≤ i.val by omega)
      have hiLower : i.val + 1 ≤ (A.orderEmbOfFin hcard i).val := by
        change (A.orderEmbOfFin hcard ⟨0, haPos⟩).val +
          2 * (i.val - 0) ≤ (A.orderEmbOfFin hcard i).val at hlower
        omega
      omega
    rw [decodedZeroReturnTrace, hdecoded]
    exact A.image_orderEmbOfFin_univ hcard

/-- The zero-return trace of every nowhere-zero active-symbol sequence is in
the range of the compressed decoder. -/
theorem exists_compressedTrace_of_activeSequence
    {F : Type*} [Field F] [DecidableEq F] {s : ℕ}
    (u : Fin s → F) (hu : ∀ i, u i ≠ 0) :
    ∃ t : CompressedZeroReturnTrace s (accumulatorZeroReturnTrace u).card,
      decodedZeroReturnTrace t = accumulatorZeroReturnTrace u := by
  let A := accumulatorZeroReturnTrace u
  let t := compressedTraceOfValid A rfl
    (fun hs => first_not_mem_zeroReturnTrace u hs hu)
    (zeroReturnTrace_nonadjacent u hu)
  exact ⟨t, decodedZeroReturnTrace_compressedTraceOfValid A rfl
    (fun hs => first_not_mem_zeroReturnTrace u hs hu)
    (zeroReturnTrace_nonadjacent u hu)⟩

/-- Actual decoded traces, represented as finite sets rather than compressed
coordinates. -/
abbrev DecodedZeroReturnTrace (s a : ℕ) :=
  Set.range (@decodedZeroReturnTrace s a)

noncomputable def compressedTraceEquivDecoded (s a : ℕ) :
    CompressedZeroReturnTrace s a ≃ DecodedZeroReturnTrace s a :=
  Equiv.ofInjective decodedZeroReturnTrace decodedZeroReturnTrace_injective

theorem card_decodedZeroReturnTraceType (s a : ℕ) :
    Fintype.card (DecodedZeroReturnTrace s a) = Nat.choose (s - a) a := by
  rw [← Fintype.card_congr (compressedTraceEquivDecoded s a)]
  exact card_compressedZeroReturnTrace s a

end BAA
