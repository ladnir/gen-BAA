import BAA.BinaryAccumulator
import Mathlib.Combinatorics.Enumerative.Composition
import Mathlib.Data.Fintype.Fin
import Mathlib.Data.List.SplitBy

/-!
# Run-cut parameter space for the binary accumulator

The two binomial factors in B2 count cuts between one-runs and zero-return
positions.  This module defines that finite target space and proves its
cardinality, constructs the explicit equivalence from accumulator events to
this space, and derives the exact B2 formula.
-/

namespace BAA

abbrev FixedCardFinset (n k : ℕ) :=
  {S : Finset (Fin n) // S.card = k}

theorem card_fixedCardFinset (n k : ℕ) :
    Fintype.card (FixedCardFinset n k) = Nat.choose n k := by
  simp

/-- Run cuts for an output of length `N`, input transition weight `w`, and
output weight `h`.  The first component selects the internal cuts between
positive one-runs; the second selects the zero-return cuts after accounting
for the possibly empty leading zero gap. -/
abbrev BinaryRunCuts (N w h : ℕ) :=
  FixedCardFinset (h - 1) ((w + 1) / 2 - 1) ×
    FixedCardFinset (N - h) (w / 2)

theorem card_binaryRunCuts (N w h : ℕ) :
    Fintype.card (BinaryRunCuts N w h) = binaryAccumulatorFormula N w h := by
  rw [Fintype.card_prod, card_fixedCardFinset, card_fixedCardFinset]
  rfl

/-! ## Fixed-length compositions -/

/-- Shift an internal cut one place to the right, into the boundary coordinates
`1, ..., n - 1` of a composition of `n`. -/
def internalBoundaryEmbedding (n : ℕ) : Fin (n - 1) ↪ Fin (n + 1) where
  toFun j := ⟨j + 1, by omega⟩
  inj' := by
    intro a b hab
    apply Fin.ext
    exact Nat.add_right_cancel (congrArg Fin.val hab)

theorem compositionAsSetEquiv_symm_boundaries {n : ℕ}
    (S : Finset (Fin (n - 1))) :
    ((compositionAsSetEquiv n).symm S).boundaries =
      insert 0 (insert (Fin.last n) (S.map (internalBoundaryEmbedding n))) := by
  ext i
  simp only [compositionAsSetEquiv, Equiv.coe_fn_symm_mk,
    Set.mem_toFinset, Set.mem_setOf_eq, Finset.mem_insert, Finset.mem_map]
  constructor
  · rintro (hi | hi | ⟨j, hj, hv⟩)
    · exact Or.inl hi
    · exact Or.inr (Or.inl hi)
    · exact Or.inr (Or.inr ⟨j, hj, Fin.ext hv.symm⟩)
  · rintro (hi | hi | ⟨j, hj, hv⟩)
    · exact Or.inl hi
    · exact Or.inr (Or.inl hi)
    · exact Or.inr (Or.inr ⟨j, hj, congrArg Fin.val hv.symm⟩)

/-- A cut set of cardinality `k` decodes to a composition with `k + 1`
positive blocks.  The positivity hypothesis excludes the degenerate
composition of zero. -/
theorem compositionAsSetEquiv_symm_length {n : ℕ} (hn : 0 < n)
    (S : Finset (Fin (n - 1))) :
    ((compositionAsSetEquiv n).symm S).length = S.card + 1 := by
  rw [CompositionAsSet.length, compositionAsSetEquiv_symm_boundaries]
  have hlast : Fin.last n ∉ S.map (internalBoundaryEmbedding n) := by
    simp [internalBoundaryEmbedding, Fin.ext_iff]
    omega
  have hzero :
      (0 : Fin (n + 1)) ∉
        insert (Fin.last n) (S.map (internalBoundaryEmbedding n)) := by
    simp only [Finset.mem_insert, Finset.mem_map, not_or]
    constructor
    · intro h
      have hv := congrArg Fin.val h
      exact hn.ne' hv.symm
    · rintro ⟨j, -, h⟩
      have hv := congrArg Fin.val h
      simp [internalBoundaryEmbedding] at hv
  rw [Finset.card_insert_of_notMem hzero,
    Finset.card_insert_of_notMem hlast, Finset.card_map]
  omega

/-- Compositions of `n` into exactly `k` positive blocks. -/
abbrev FixedLengthComposition (n k : ℕ) :=
  {c : Composition n // c.length = k}

/-- The standard cuts-between-blocks bijection, restricted to a fixed number
of positive blocks. -/
def fixedLengthCompositionEquiv {n k : ℕ} (hn : 0 < n) (hk : 0 < k) :
    FixedLengthComposition n k ≃ FixedCardFinset (n - 1) (k - 1) :=
  Equiv.subtypeEquiv ((compositionEquiv n).trans (compositionAsSetEquiv n)) <| by
    intro c
    constructor
    · intro hc
      have hlen := compositionAsSetEquiv_symm_length hn
        (((compositionEquiv n).trans (compositionAsSetEquiv n)) c)
      change ((compositionAsSetEquiv n).symm
        ((compositionAsSetEquiv n) ((compositionEquiv n) c))).length = _ at hlen
      rw [Equiv.symm_apply_apply] at hlen
      change c.toCompositionAsSet.length =
        (((compositionEquiv n).trans (compositionAsSetEquiv n)) c).card + 1 at hlen
      have hlen' : c.length =
          (((compositionEquiv n).trans (compositionAsSetEquiv n)) c).card + 1 := by
        rw [Composition.toCompositionAsSet_length] at hlen
        exact hlen
      change (((compositionEquiv n).trans (compositionAsSetEquiv n)) c).card = k - 1
      omega
    · intro hcard
      have hlen := compositionAsSetEquiv_symm_length hn
        (((compositionEquiv n).trans (compositionAsSetEquiv n)) c)
      change ((compositionAsSetEquiv n).symm
        ((compositionAsSetEquiv n) ((compositionEquiv n) c))).length = _ at hlen
      rw [Equiv.symm_apply_apply] at hlen
      change c.toCompositionAsSet.length =
        (((compositionEquiv n).trans (compositionAsSetEquiv n)) c).card + 1 at hlen
      have hlen' : c.length =
          (((compositionEquiv n).trans (compositionAsSetEquiv n)) c).card + 1 := by
        rw [Composition.toCompositionAsSet_length] at hlen
        exact hlen
      omega

theorem card_fixedLengthComposition {n k : ℕ} (hn : 0 < n) (hk : 0 < k) :
    Fintype.card (FixedLengthComposition n k) = Nat.choose (n - 1) (k - 1) := by
  rw [Fintype.card_congr (fixedLengthCompositionEquiv hn hk)]
  exact card_fixedCardFinset (n - 1) (k - 1)

/-- The run data behind B2: positive one-run lengths, followed by a
composition whose first block is one plus the optional leading zero gap and
whose remaining blocks are the positive zero-return gaps. -/
abbrev BinaryRunCompositions (N w h : ℕ) :=
  FixedLengthComposition h ((w + 1) / 2) ×
    FixedLengthComposition (N - h + 1) (w / 2 + 1)

def binaryRunCompositionsEquiv (N w h : ℕ) (hw : 0 < w) (hh : 0 < h) :
    BinaryRunCompositions N w h ≃ BinaryRunCuts N w h :=
  (fixedLengthCompositionEquiv hh (by omega)).prodCongr
    (fixedLengthCompositionEquiv (by omega) (by omega))

theorem card_binaryRunCompositions (N w h : ℕ) (hw : 0 < w) (hh : 0 < h) :
    Fintype.card (BinaryRunCompositions N w h) = binaryAccumulatorFormula N w h := by
  rw [Fintype.card_congr (binaryRunCompositionsEquiv N w h hw hh)]
  exact card_binaryRunCuts N w h

/-! ## Executable alternating-run decoder -/

/-- Concatenate constant Boolean blocks with the supplied lengths. -/
def constantBlocks (b : Bool) : List ℕ → List Bool
  | [] => []
  | n :: ns => List.replicate n b ++ constantBlocks b ns

@[simp]
theorem length_constantBlocks (b : Bool) (ns : List ℕ) :
    (constantBlocks b ns).length = ns.sum := by
  induction ns with
  | nil => rfl
  | cons n ns ih => simp [constantBlocks, ih]

@[simp]
theorem count_true_replicate_false (n : ℕ) :
    (List.replicate n false).count true = 0 := by
  rw [List.count_replicate]
  rfl

@[simp]
theorem count_true_constantBlocks_false (ns : List ℕ) :
    (constantBlocks false ns).count true = 0 := by
  induction ns with
  | nil => rfl
  | cons n ns ih => simp [constantBlocks, ih]

@[simp]
theorem count_true_constantBlocks_true (ns : List ℕ) :
    (constantBlocks true ns).count true = ns.sum := by
  induction ns with
  | nil => rfl
  | cons n ns ih => simp [constantBlocks, ih]

/-- Interleave zero blocks and one blocks.  Valid B2 run data has either the
same number of blocks or one extra zero block; the total definition also gives
a useful executable normal form for malformed intermediate data. -/
def alternatingBits : List ℕ → List ℕ → List Bool
  | [], ones => constantBlocks true ones
  | gaps, [] => constantBlocks false gaps
  | g :: gaps, a :: ones =>
      List.replicate g false ++ List.replicate a true ++
        alternatingBits gaps ones

@[simp]
theorem length_alternatingBits (gaps ones : List ℕ) :
    (alternatingBits gaps ones).length = gaps.sum + ones.sum := by
  induction gaps generalizing ones with
  | nil => simp [alternatingBits]
  | cons g gaps ih =>
      cases ones with
      | nil => simp [alternatingBits]
      | cons a ones => simp [alternatingBits, ih, Nat.add_assoc, Nat.add_left_comm]

@[simp]
theorem count_true_alternatingBits (gaps ones : List ℕ) :
    (alternatingBits gaps ones).count true = ones.sum := by
  induction gaps generalizing ones with
  | nil => simp [alternatingBits]
  | cons g gaps ih =>
      cases ones with
      | nil => simp [alternatingBits]
      | cons a ones => simp [alternatingBits, ih, List.count_replicate]

/-- Every Boolean list has a normalized alternating-run decomposition.  The
first zero gap may be empty; all one runs and all later zero gaps are
positive. -/
theorem exists_alternatingBits_decomposition (bits : List Bool) :
    ∃ gaps ones : List ℕ,
      ones.Forall (fun a => 0 < a) ∧
      gaps.tail.Forall (fun g => 0 < g) ∧
      (gaps.length = ones.length ∨ gaps.length = ones.length + 1) ∧
      alternatingBits gaps ones = bits := by
  induction bits with
  | nil => exact ⟨[], [], by simp [alternatingBits, constantBlocks]⟩
  | cons b bits ih =>
      obtain ⟨gaps, ones, hones, hgaps, hlen, hdecode⟩ := ih
      cases b with
      | false =>
          cases gaps with
          | nil =>
              have honesNil : ones = [] := by
                apply List.eq_nil_of_length_eq_zero
                simp only [List.length_nil] at hlen
                omega
              subst ones
              have hbitsNil : bits = [] := by
                simpa [alternatingBits, constantBlocks] using hdecode.symm
              subst bits
              exact ⟨[1], [], by simp [alternatingBits, constantBlocks]⟩
          | cons g gs =>
              refine ⟨(g + 1) :: gs, ones, hones, ?_, ?_, ?_⟩
              · simpa using hgaps
              · simpa using hlen
              · cases ones <;>
                  simpa [alternatingBits, constantBlocks, List.replicate_succ,
                    List.append_assoc] using congrArg (false :: ·) hdecode
      | true =>
          cases gaps with
          | nil =>
              have honesNil : ones = [] := by
                apply List.eq_nil_of_length_eq_zero
                simp only [List.length_nil] at hlen
                omega
              subst ones
              have hbitsNil : bits = [] := by
                simpa [alternatingBits, constantBlocks] using hdecode.symm
              subst bits
              exact ⟨[0], [1], by simp [alternatingBits, constantBlocks]⟩
          | cons g gs =>
              cases g with
              | zero =>
                  cases ones with
                  | nil =>
                      have hgsNil : gs = [] := by
                        apply List.eq_nil_of_length_eq_zero
                        simp only [List.length_nil, List.length_cons] at hlen
                        omega
                      subst gs
                      have hbitsNil : bits = [] := by
                        simpa [alternatingBits, constantBlocks] using hdecode.symm
                      subst bits
                      exact ⟨[0], [1], by simp [alternatingBits, constantBlocks]⟩
                  | cons a as =>
                      refine ⟨0 :: gs, (a + 1) :: as, ?_, ?_, ?_, ?_⟩
                      · have hparts : 0 < a ∧ as.Forall (fun x => 0 < x) := by
                          simpa using hones
                        simpa using And.intro (by omega : 0 < a + 1) hparts.2
                      · simpa using hgaps
                      · simpa using hlen
                      · simpa [alternatingBits, List.replicate_succ,
                          List.append_assoc] using congrArg (true :: ·) hdecode
              | succ g =>
                  refine ⟨0 :: (g + 1) :: gs, 1 :: ones, ?_, ?_, ?_, ?_⟩
                  · simpa using hones
                  · have hgs : gs.Forall fun x => 0 < x := by simpa using hgaps
                    simpa using hgs
                  · simp only [List.length_cons] at hlen ⊢
                    omega
                  · simpa [alternatingBits, List.replicate_succ,
                      List.append_assoc] using congrArg (true :: ·) hdecode

/-- Remove the one artificial zero from the first block of the positive
composition used to encode an optional leading zero gap. -/
def zeroGapBlocks {z : ℕ} (c : Composition (z + 1)) : List ℕ :=
  match c.blocks with
  | [] => []
  | g :: gaps => (g - 1) :: gaps

@[simp]
theorem length_zeroGapBlocks {z : ℕ} (c : Composition (z + 1)) :
    (zeroGapBlocks c).length = c.length := by
  change (zeroGapBlocks c).length = c.blocks.length
  cases hblocks : c.blocks with
  | nil =>
      have hsum := c.blocks_sum
      rw [hblocks] at hsum
      simp at hsum
  | cons g gaps => simp [zeroGapBlocks, hblocks]

@[simp]
theorem sum_zeroGapBlocks {z : ℕ} (c : Composition (z + 1)) :
    (zeroGapBlocks c).sum = z := by
  cases hblocks : c.blocks with
  | nil =>
      have hsum := c.blocks_sum
      rw [hblocks] at hsum
      simp at hsum
  | cons g gaps =>
      have hg : 0 < g := c.blocks_pos (by simp [hblocks])
      have hsum := c.blocks_sum
      rw [hblocks] at hsum
      simp only [List.sum_cons] at hsum
      simp [zeroGapBlocks, hblocks]
      omega

@[simp]
theorem tail_zeroGapBlocks {z : ℕ} (c : Composition (z + 1)) :
    (zeroGapBlocks c).tail = c.blocks.tail := by
  cases hblocks : c.blocks with
  | nil => simp [zeroGapBlocks, hblocks]
  | cons g gaps => simp [zeroGapBlocks, hblocks]

theorem forall_pos_zeroGapBlocks_tail {z : ℕ} (c : Composition (z + 1)) :
    (zeroGapBlocks c).tail.Forall fun g => 0 < g := by
  rw [tail_zeroGapBlocks, List.forall_iff_forall_mem]
  intro g hg
  exact c.blocks_pos (List.mem_of_mem_tail hg)

theorem zeroGapBlocks_injective {z : ℕ} :
    Function.Injective (@zeroGapBlocks z) := by
  intro c d heq
  apply Composition.ext
  cases hc : c.blocks with
  | nil =>
      have hsum := c.blocks_sum
      rw [hc] at hsum
      simp at hsum
  | cons g gs =>
      cases hd : d.blocks with
      | nil =>
          have hsum := d.blocks_sum
          rw [hd] at hsum
          simp at hsum
      | cons q qs =>
          have hdecoded : (g - 1) :: gs = (q - 1) :: qs := by
            simpa [zeroGapBlocks, hc, hd] using heq
          have hg : 0 < g := c.blocks_pos (by simp [hc])
          have hq : 0 < q := d.blocks_pos (by simp [hd])
          have hgq : g = q := by
            have := (List.cons.inj hdecoded).1
            omega
          have hgsqs : gs = qs := (List.cons.inj hdecoded).2
          simpa [hc, hd, hgq, hgsqs]

/-- Decode the two positive compositions into the alternating Boolean output
word used by B2. -/
def binaryRunCompositionBits {N w h : ℕ} (d : BinaryRunCompositions N w h) :
    List Bool :=
  alternatingBits (zeroGapBlocks d.2.1) d.1.1.blocks

theorem length_binaryRunCompositionBits {N w h : ℕ}
    (hhN : h ≤ N) (d : BinaryRunCompositions N w h) :
    (binaryRunCompositionBits d).length = N := by
  simp [binaryRunCompositionBits, d.1.1.blocks_sum]
  omega

theorem count_true_binaryRunCompositionBits {N w h : ℕ}
    (d : BinaryRunCompositions N w h) :
    (binaryRunCompositionBits d).count true = h := by
  simp [binaryRunCompositionBits, d.1.1.blocks_sum]

/-- Interpret a fixed-length Boolean vector as a binary field-valued word. -/
def boolVectorToBinaryWord {N : ℕ} (v : List.Vector Bool N) : BinaryWord N :=
  fun i => if v.get i then 1 else 0

/-- Read a binary field-valued word as a Boolean vector. -/
def binaryWordToBoolVector {N : ℕ} (x : BinaryWord N) : List.Vector Bool N :=
  List.Vector.ofFn fun i => x i ≠ 0

theorem boolVectorToBinaryWord_binaryWordToBoolVector {N : ℕ}
    (x : BinaryWord N) :
    boolVectorToBinaryWord (binaryWordToBoolVector x) = x := by
  funext i
  by_cases hi : x i = 0
  · simp [binaryWordToBoolVector, boolVectorToBinaryWord, hi]
  · have hval0 : (x i).val ≠ 0 := (ZMod.val_eq_zero (x i)).not.mpr hi
    have hval1 : (x i).val = 1 := by
      have hlt := (x i).val_lt
      omega
    have hi1 : x i = 1 := (ZMod.val_eq_one (by omega) (x i)).mp hval1
    simp [binaryWordToBoolVector, boolVectorToBinaryWord, hi, hi1]

theorem binaryWordToBoolVector_boolVectorToBinaryWord {N : ℕ}
    (v : List.Vector Bool N) :
    binaryWordToBoolVector (boolVectorToBinaryWord v) = v := by
  apply List.Vector.ext
  intro i
  cases hv : v.get i <;>
    simp [binaryWordToBoolVector, boolVectorToBinaryWord, hv]

/-- Binary words and fixed-length Boolean lists are definitionally separate
representations of the same finite data. -/
def binaryWordBoolVectorEquiv (N : ℕ) :
    BinaryWord N ≃ List.Vector Bool N where
  toFun := binaryWordToBoolVector
  invFun := boolVectorToBinaryWord
  left_inv := boolVectorToBinaryWord_binaryWordToBoolVector
  right_inv := binaryWordToBoolVector_boolVectorToBinaryWord

theorem boolVectorToBinaryWord_injective {N : ℕ} :
    Function.Injective (@boolVectorToBinaryWord N) :=
  (binaryWordBoolVectorEquiv N).symm.injective

/-! ## Canonical run parser -/

/-- Maximal constant Boolean runs. -/
def boolRuns (bits : List Bool) : List (List Bool) :=
  bits.splitBy fun a b => a == b

@[simp]
theorem flatten_boolRuns (bits : List Bool) : (boolRuns bits).flatten = bits := by
  exact List.flatten_splitBy _ _

theorem nil_notMem_boolRuns (bits : List Bool) : [] ∉ boolRuns bits := by
  exact List.nil_notMem_splitBy _ _

def boolRunLengths (bits : List Bool) : List ℕ :=
  (boolRuns bits).map List.length

theorem forall_pos_boolRunLengths (bits : List Bool) :
    (boolRunLengths bits).Forall fun n => 0 < n := by
  rw [List.forall_iff_forall_mem]
  intro n hn
  obtain ⟨run, hrun, rfl⟩ := List.mem_map.mp hn
  exact List.length_pos_iff_ne_nil.mpr (List.ne_nil_of_mem_splitBy hrun)

@[simp]
theorem sum_boolRunLengths (bits : List Bool) :
    (boolRunLengths bits).sum = bits.length := by
  rw [boolRunLengths, ← List.length_flatten, flatten_boolRuns]

theorem isChain_eq_of_mem_boolRuns (bits run : List Bool)
    (hrun : run ∈ boolRuns bits) :
    run.IsChain (· = ·) := by
  have hchain := List.isChain_of_mem_splitBy hrun
  exact hchain.imp fun _ _ h => by simpa using h

/-- Length of the maximal prefix equal to `b`. -/
def leadingCount (b : Bool) (bits : List Bool) : ℕ :=
  (bits.takeWhile fun x => x == b).length

theorem leadingCount_replicate_cons_ne (b c : Bool) (hbc : c ≠ b)
    (n : ℕ) (tail : List Bool) :
    leadingCount b (List.replicate n b ++ c :: tail) = n := by
  induction n with
  | zero => cases b <;> cases c <;> simp_all [leadingCount]
  | succ n ih =>
      cases b <;> cases c <;>
        simp_all [List.replicate_succ, leadingCount]

theorem leadingCount_replicate_append_of_boundary (b : Bool)
    (n : ℕ) (tail : List Bool)
    (htail : tail = [] ∨ ∃ c rest, tail = c :: rest ∧ c ≠ b) :
    leadingCount b (List.replicate n b ++ tail) = n := by
  rcases htail with rfl | ⟨c, rest, rfl, hcb⟩
  · simp [leadingCount]
  · exact leadingCount_replicate_cons_ne b c hcb n rest

theorem alternatingBits_eq_nil_or_cons_false (gaps ones : List ℕ)
    (hgaps : gaps.Forall fun g => 0 < g)
    (hlen : gaps.length = ones.length ∨ gaps.length = ones.length + 1) :
    alternatingBits gaps ones = [] ∨
      ∃ tail, alternatingBits gaps ones = false :: tail := by
  cases gaps with
  | nil =>
      have hones : ones = [] := by
        apply List.eq_nil_of_length_eq_zero
        simp only [List.length_nil] at hlen
        omega
      subst ones
      simp [alternatingBits, constantBlocks]
  | cons g gaps =>
      have hg : 0 < g := by
        rw [List.forall_iff_forall_mem] at hgaps
        exact hgaps g (by simp)
      obtain ⟨g', rfl⟩ := Nat.exists_eq_succ_of_ne_zero hg.ne'
      right
      cases ones with
      | nil =>
          refine ⟨List.replicate g' false ++ constantBlocks false gaps, ?_⟩
          simp [alternatingBits, constantBlocks, List.replicate_succ]
      | cons a ones =>
          refine ⟨List.replicate g' false ++ List.replicate a true ++
            alternatingBits gaps ones, ?_⟩
          simp [alternatingBits, List.replicate_succ]

theorem leadingCount_false_alternatingBits_cons (g : ℕ) (gaps : List ℕ)
    {a : ℕ} (ha : 0 < a) (ones : List ℕ) :
    leadingCount false (alternatingBits (g :: gaps) (a :: ones)) = g := by
  obtain ⟨a', rfl⟩ := Nat.exists_eq_succ_of_ne_zero ha.ne'
  rw [alternatingBits, List.replicate_succ]
  simpa [List.append_assoc] using
    (leadingCount_replicate_cons_ne false true (by decide) g
      (List.replicate a' true ++ alternatingBits gaps ones))

theorem leadingCount_true_after_firstGap (g : ℕ) (gaps : List ℕ)
    {a : ℕ} (ha : 0 < a) (ones : List ℕ)
    (hgaps : gaps.Forall fun z => 0 < z)
    (hlen : gaps.length = ones.length ∨ gaps.length = ones.length + 1) :
    leadingCount true
      ((alternatingBits (g :: gaps) (a :: ones)).drop g) = a := by
  rw [alternatingBits, List.append_assoc]
  rw [List.drop_append_of_le_length (by simp)]
  simp only [List.drop_replicate, Nat.sub_self, List.replicate_zero,
    List.nil_append]
  apply leadingCount_replicate_append_of_boundary
  rcases alternatingBits_eq_nil_or_cons_false gaps ones hgaps hlen with h | ⟨tail, h⟩
  · exact Or.inl h
  · exact Or.inr ⟨false, tail, h, by decide⟩

/-- Positive alternating runs, with a possibly empty leading zero run, have
unique run lengths once the numbers of zero and one runs are fixed. -/
theorem alternatingBits_injective_of_valid
    {gaps₁ gaps₂ ones₁ ones₂ : List ℕ}
    (hones₁ : ones₁.Forall fun a => 0 < a)
    (hones₂ : ones₂.Forall fun a => 0 < a)
    (hgaps₁ : gaps₁.tail.Forall fun g => 0 < g)
    (hgaps₂ : gaps₂.tail.Forall fun g => 0 < g)
    (hlen₁ : gaps₁.length = ones₁.length ∨
      gaps₁.length = ones₁.length + 1)
    (hlen₂ : gaps₂.length = ones₂.length ∨
      gaps₂.length = ones₂.length + 1)
    (hgapsLength : gaps₁.length = gaps₂.length)
    (honesLength : ones₁.length = ones₂.length)
    (heq : alternatingBits gaps₁ ones₁ = alternatingBits gaps₂ ones₂) :
    gaps₁ = gaps₂ ∧ ones₁ = ones₂ := by
  induction ones₁ generalizing gaps₁ gaps₂ ones₂ with
  | nil =>
      have hones₂nil : ones₂ = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using honesLength.symm
      subst ones₂
      cases gaps₁ with
      | nil =>
          have hgaps₂nil : gaps₂ = [] := by
            apply List.eq_nil_of_length_eq_zero
            simpa using hgapsLength.symm
          subst gaps₂
          exact ⟨rfl, rfl⟩
      | cons g gs =>
          have hgs : gs = [] := by
            apply List.eq_nil_of_length_eq_zero
            simp only [List.length_nil, List.length_cons] at hlen₁
            omega
          subst gs
          cases gaps₂ with
          | nil => simp at hgapsLength
          | cons q qs =>
              have hqs : qs = [] := by
                apply List.eq_nil_of_length_eq_zero
                simp only [List.length_nil, List.length_cons] at hlen₂
                omega
              subst qs
              have hgq : g = q := by
                have hlength := congrArg List.length heq
                simpa [alternatingBits, constantBlocks] using hlength
              subst q
              exact ⟨rfl, rfl⟩
  | cons a as ih =>
      cases ones₂ with
      | nil => simp at honesLength
      | cons b bs =>
          cases gaps₁ with
          | nil => simp at hlen₁
          | cons g gs =>
              cases gaps₂ with
              | nil => simp at hgapsLength
              | cons q qs =>
                  have ha : 0 < a := by
                    rw [List.forall_iff_forall_mem] at hones₁
                    exact hones₁ a (by simp)
                  have hb : 0 < b := by
                    rw [List.forall_iff_forall_mem] at hones₂
                    exact hones₂ b (by simp)
                  have hgq : g = q := by
                    rw [← leadingCount_false_alternatingBits_cons g gs ha as,
                      ← leadingCount_false_alternatingBits_cons q qs hb bs]
                    exact congrArg (leadingCount false) heq
                  subst q
                  have hab : a = b := by
                    rw [← leadingCount_true_after_firstGap g gs ha as
                        (by simpa using hgaps₁) (by simpa using hlen₁),
                      ← leadingCount_true_after_firstGap g qs hb bs
                        (by simpa using hgaps₂) (by simpa using hlen₂)]
                    exact congrArg (fun bits => leadingCount true (bits.drop g)) heq
                  subst b
                  have htailEq : alternatingBits gs as = alternatingBits qs bs := by
                    simpa [alternatingBits] using heq
                  have hones₁' : as.Forall fun x => 0 < x := by
                    have hparts : 0 < a ∧ as.Forall (fun x => 0 < x) := by
                      simpa using hones₁
                    exact hparts.2
                  have hones₂' : bs.Forall fun x => 0 < x := by
                    have hparts : 0 < a ∧ bs.Forall (fun x => 0 < x) := by
                      simpa using hones₂
                    exact hparts.2
                  have hgaps₁' : gs.tail.Forall fun x => 0 < x := by
                    rw [List.forall_iff_forall_mem] at hgaps₁ ⊢
                    intro x hx
                    exact hgaps₁ x (List.mem_of_mem_tail hx)
                  have hgaps₂' : qs.tail.Forall fun x => 0 < x := by
                    rw [List.forall_iff_forall_mem] at hgaps₂ ⊢
                    intro x hx
                    exact hgaps₂ x (List.mem_of_mem_tail hx)
                  have hlen₁' : gs.length = as.length ∨
                      gs.length = as.length + 1 := by
                    simpa using hlen₁
                  have hlen₂' : qs.length = bs.length ∨
                      qs.length = bs.length + 1 := by
                    simpa using hlen₂
                  have hgapsLength' : gs.length = qs.length := by
                    simpa using hgapsLength
                  have honesLength' : as.length = bs.length := by
                    simpa using honesLength
                  obtain ⟨rfl, rfl⟩ := ih hones₁' hones₂' hgaps₁' hgaps₂'
                    hlen₁' hlen₂' hgapsLength' honesLength' htailEq
                  exact ⟨rfl, rfl⟩

theorem binaryRunCompositionBits_injective {N w h : ℕ} :
    Function.Injective (@binaryRunCompositionBits N w h) := by
  intro d e heq
  have honesD : d.1.1.blocks.Forall fun a => 0 < a := by
    rw [List.forall_iff_forall_mem]
    exact fun _ ha => d.1.1.blocks_pos ha
  have honesE : e.1.1.blocks.Forall fun a => 0 < a := by
    rw [List.forall_iff_forall_mem]
    exact fun _ ha => e.1.1.blocks_pos ha
  have hlenD :
      (zeroGapBlocks d.2.1).length = d.1.1.blocks.length ∨
        (zeroGapBlocks d.2.1).length = d.1.1.blocks.length + 1 := by
    rw [length_zeroGapBlocks]
    change d.2.1.length = d.1.1.length ∨
      d.2.1.length = d.1.1.length + 1
    rw [d.1.2, d.2.2]
    omega
  have hlenE :
      (zeroGapBlocks e.2.1).length = e.1.1.blocks.length ∨
        (zeroGapBlocks e.2.1).length = e.1.1.blocks.length + 1 := by
    rw [length_zeroGapBlocks]
    change e.2.1.length = e.1.1.length ∨
      e.2.1.length = e.1.1.length + 1
    rw [e.1.2, e.2.2]
    omega
  have hgapsLength :
      (zeroGapBlocks d.2.1).length = (zeroGapBlocks e.2.1).length := by
    rw [length_zeroGapBlocks, length_zeroGapBlocks, d.2.2, e.2.2]
  have honesLength : d.1.1.blocks.length = e.1.1.blocks.length := by
    change d.1.1.length = e.1.1.length
    rw [d.1.2, e.1.2]
  have hruns := alternatingBits_injective_of_valid honesD honesE
    (forall_pos_zeroGapBlocks_tail d.2.1)
    (forall_pos_zeroGapBlocks_tail e.2.1) hlenD hlenE
    hgapsLength honesLength (by simpa [binaryRunCompositionBits] using heq)
  apply Prod.ext
  · apply Subtype.ext
    apply Composition.ext
    exact hruns.2
  · apply Subtype.ext
    exact zeroGapBlocks_injective hruns.1

theorem hammingNorm_boolVectorToBinaryWord {N : ℕ} (v : List.Vector Bool N) :
    hammingNorm (boolVectorToBinaryWord v) = v.toList.count true := by
  rw [← Fin.card_filter_univ_eq_vector_get_eq_count true v]
  apply congrArg Finset.card
  ext i
  cases hv : v.get i <;> simp [boolVectorToBinaryWord, hv]

def binaryRunCompositionVector {N w h : ℕ} (hhN : h ≤ N)
    (d : BinaryRunCompositions N w h) : List.Vector Bool N :=
  ⟨binaryRunCompositionBits d, length_binaryRunCompositionBits hhN d⟩

/-- The executable output word decoded from B2 run data. -/
def binaryRunCompositionWord {N w h : ℕ} (hhN : h ≤ N)
    (d : BinaryRunCompositions N w h) : BinaryWord N :=
  boolVectorToBinaryWord (binaryRunCompositionVector hhN d)

theorem hammingNorm_binaryRunCompositionWord {N w h : ℕ} (hhN : h ≤ N)
    (d : BinaryRunCompositions N w h) :
    hammingNorm (binaryRunCompositionWord hhN d) = h := by
  rw [binaryRunCompositionWord, hammingNorm_boolVectorToBinaryWord]
  exact count_true_binaryRunCompositionBits d

/-! ## Boolean first differences -/

def boolVectorDifferences : {N : ℕ} → Bool → List.Vector Bool N → List.Vector Bool N
  | 0, _, _ => List.Vector.nil
  | _ + 1, previous, v => List.Vector.ofFn fun i =>
      Fin.cases (v.get 0 != previous)
        (fun j => v.get j.succ != v.get j.castSucc) i

theorem binaryValue_bne (a b : Bool) :
    (if a != b then (1 : ZMod 2) else 0) =
      (if a then 1 else 0) - (if b then 1 else 0) := by
  cases a <;> cases b <;> decide

/-- Boolean adjacent differences agree coordinatewise with the abstract linear
first-difference operator. -/
theorem diff_boolVectorToBinaryWord {N : ℕ} (v : List.Vector Bool N) :
    diffLinearEquiv (ZMod 2) N (boolVectorToBinaryWord v) =
      boolVectorToBinaryWord (boolVectorDifferences false v) := by
  cases N with
  | zero =>
      funext i
      exact Fin.elim0 i
  | succ n =>
      funext i
      refine Fin.cases ?_ (fun j => ?_) i
      · simp [boolVectorDifferences, boolVectorToBinaryWord]
      · simp [boolVectorDifferences, boolVectorToBinaryWord]
        generalize v.get j.succ = a
        generalize v.get j.castSucc = b
        cases a <;> cases b <;> decide

theorem hammingNorm_diff_boolVectorToBinaryWord {N : ℕ}
    (v : List.Vector Bool N) :
    hammingNorm (diffLinearEquiv (ZMod 2) N (boolVectorToBinaryWord v)) =
      (boolVectorDifferences false v).toList.count true := by
  rw [diff_boolVectorToBinaryWord, hammingNorm_boolVectorToBinaryWord]

def boolListDifferences (previous : Bool) : List Bool → List Bool
  | [] => []
  | b :: bits => (b != previous) :: boolListDifferences b bits

theorem boolVectorDifferences_cons {N : ℕ} (previous b : Bool)
    (v : List.Vector Bool N) :
    boolVectorDifferences previous (List.Vector.cons b v) =
      List.Vector.cons (b != previous) (boolVectorDifferences b v) := by
  apply List.Vector.ext
  intro i
  refine Fin.cases ?_ (fun j => ?_) i
  · simp [boolVectorDifferences]
  · cases N with
    | zero => exact Fin.elim0 j
    | succ n =>
        refine Fin.cases ?_ (fun k => ?_) j
        · simp only [boolVectorDifferences, List.Vector.get_ofFn,
            List.Vector.get_cons_succ, Fin.cases_succ]
          rw [show Fin.castSucc (0 : Fin (n + 1)) = (0 : Fin (n + 2)) by rfl,
            List.Vector.get_cons_zero, Fin.cases_zero, List.Vector.get_zero]
        · simp [boolVectorDifferences]

theorem toList_boolVectorDifferences {N : ℕ} (previous : Bool)
    (v : List.Vector Bool N) :
    (boolVectorDifferences previous v).toList =
      boolListDifferences previous v.toList := by
  induction v generalizing previous with
  | nil => rfl
  | @cons n b v ih =>
      rw [boolVectorDifferences_cons, List.Vector.toList_cons, ih]
      rfl

@[simp]
theorem length_boolListDifferences (previous : Bool) (bits : List Bool) :
    (boolListDifferences previous bits).length = bits.length := by
  induction bits generalizing previous with
  | nil => rfl
  | cons b bits ih => simp [boolListDifferences, ih]

theorem boolListDifferences_same_replicate_append (b : Bool) (n : ℕ)
    (tail : List Bool) :
    boolListDifferences b (List.replicate n b ++ tail) =
      List.replicate n false ++ boolListDifferences b tail := by
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate_succ, boolListDifferences, ih]

theorem count_boolListDifferences_replicate_append (previous b : Bool)
    {n : ℕ} (hn : 0 < n) (tail : List Bool) :
    (boolListDifferences previous (List.replicate n b ++ tail)).count true =
      (if b != previous then 1 else 0) +
        (boolListDifferences b tail).count true := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn.ne'
  rw [List.replicate_succ, List.cons_append]
  simp only [boolListDifferences, List.count_cons]
  rw [boolListDifferences_same_replicate_append]
  simp [List.count_replicate]
  omega

/-- An alternating list with positive one blocks and positive zero blocks after
the optional leading gap has one transition per one block and one per later
zero block. -/
theorem count_boolListDifferences_alternatingBits (gaps ones : List ℕ)
    (hones : ones.Forall fun a => 0 < a)
    (hgaps : gaps.tail.Forall fun g => 0 < g)
    (hlen : gaps.length = ones.length ∨ gaps.length = ones.length + 1) :
    (boolListDifferences false (alternatingBits gaps ones)).count true =
      ones.length + (gaps.length - 1) := by
  induction ones generalizing gaps with
  | nil =>
      cases gaps with
      | nil => rfl
      | cons g gaps =>
          cases gaps with
          | nil =>
              rw [show alternatingBits [g] [] =
                List.replicate g false by simp [alternatingBits, constantBlocks]]
              rw [show List.replicate g false =
                List.replicate g false ++ [] by simp,
                boolListDifferences_same_replicate_append]
              simp [boolListDifferences]
          | cons z zs => simp at hlen
  | cons a ones ih =>
      have honesMem := List.forall_iff_forall_mem.mp hones
      have ha : 0 < a := honesMem a (by simp)
      have hones' : ones.Forall fun a => 0 < a :=
        List.forall_iff_forall_mem.mpr fun x hx => honesMem x (by simp [hx])
      cases gaps with
      | nil => simp at hlen
      | cons g gaps =>
          have hgapsAll : gaps.Forall fun z => 0 < z := by simpa using hgaps
          have hlen' :
              gaps.length = ones.length ∨ gaps.length = ones.length + 1 := by
            simpa using hlen
          have hgapsMem := List.forall_iff_forall_mem.mp hgapsAll
          have hgapsTail : gaps.tail.Forall fun z => 0 < z :=
            List.forall_iff_forall_mem.mpr fun x hx =>
              hgapsMem x (List.mem_of_mem_tail hx)
          have ih' := ih gaps hones' hgapsTail hlen'
          rw [alternatingBits, List.append_assoc,
            boolListDifferences_same_replicate_append]
          simp only [List.count_append, count_true_replicate_false, zero_add]
          rw [count_boolListDifferences_replicate_append false true ha]
          simp
          cases ones with
          | nil =>
              cases gaps with
              | nil => rfl
              | cons z zs =>
                  cases zs with
                  | nil =>
                      have hz : 0 < z := hgapsMem z (by simp)
                      have hzcount :=
                        count_boolListDifferences_replicate_append true false hz []
                      norm_num at hzcount
                      rw [show alternatingBits [z] [] =
                        List.replicate z false by simp [alternatingBits, constantBlocks]]
                      rw [hzcount]
                      simp [boolListDifferences]
                  | cons u us => simp at hlen'
          | cons a' ones' =>
              cases gaps with
              | nil => simp at hlen'
              | cons z zs =>
                  have hz : 0 < z := hgapsMem z (by simp)
                  have hswitch :
                      (boolListDifferences true
                        (alternatingBits (z :: zs) (a' :: ones'))).count true =
                      1 + (boolListDifferences false
                        (alternatingBits (z :: zs) (a' :: ones'))).count true := by
                    rw [alternatingBits, List.append_assoc,
                      count_boolListDifferences_replicate_append true false hz,
                      count_boolListDifferences_replicate_append false false hz]
                    simp
                  rw [hswitch, ih']
                  simp only [List.length_cons] at hlen' ⊢
                  omega

theorem hammingNorm_diff_binaryRunCompositionWord {N w h : ℕ}
    (hw : 0 < w) (hhN : h ≤ N) (d : BinaryRunCompositions N w h) :
    hammingNorm
      (diffLinearEquiv (ZMod 2) N (binaryRunCompositionWord hhN d)) = w := by
  rw [binaryRunCompositionWord, hammingNorm_diff_boolVectorToBinaryWord,
    toList_boolVectorDifferences]
  change (boolListDifferences false
    (binaryRunCompositionBits d)).count true = w
  rw [binaryRunCompositionBits]
  have hones : d.1.1.blocks.Forall fun a => 0 < a := by
    rw [List.forall_iff_forall_mem]
    exact fun _ ha => d.1.1.blocks_pos ha
  have hgaps := forall_pos_zeroGapBlocks_tail d.2.1
  have hlen :
      (zeroGapBlocks d.2.1).length = d.1.1.blocks.length ∨
        (zeroGapBlocks d.2.1).length = d.1.1.blocks.length + 1 := by
    rw [length_zeroGapBlocks]
    change d.2.1.length = d.1.1.length ∨
      d.2.1.length = d.1.1.length + 1
    rw [d.1.2, d.2.2]
    omega
  rw [count_boolListDifferences_alternatingBits _ _ hones hgaps hlen,
    length_zeroGapBlocks]
  change d.1.1.length + (d.2.1.length - 1) = w
  rw [d.1.2, d.2.2]
  omega

/-- The forward half of the symbolic B2 bijection. -/
def binaryRunCompositionToOutputEvent {N w h : ℕ}
    (hw : 0 < w) (hhN : h ≤ N) :
    BinaryRunCompositions N w h →
      {y : BinaryWord N //
        hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧
          hammingNorm y = h} :=
  fun d => ⟨binaryRunCompositionWord hhN d,
    hammingNorm_diff_binaryRunCompositionWord hw hhN d,
    hammingNorm_binaryRunCompositionWord hhN d⟩

theorem binaryRunCompositionWord_injective {N w h : ℕ} (hhN : h ≤ N) :
    Function.Injective (@binaryRunCompositionWord N w h hhN) := by
  intro d e heq
  apply binaryRunCompositionBits_injective
  have hvectors : binaryRunCompositionVector hhN d =
      binaryRunCompositionVector hhN e :=
    boolVectorToBinaryWord_injective (by
      simpa [binaryRunCompositionWord] using heq)
  exact congrArg List.Vector.toList hvectors

theorem binaryRunCompositionToOutputEvent_injective {N w h : ℕ}
    (hw : 0 < w) (hhN : h ≤ N) :
    Function.Injective (binaryRunCompositionToOutputEvent hw hhN) := by
  intro d e heq
  apply binaryRunCompositionWord_injective hhN
  exact congrArg Subtype.val heq

theorem binaryRunCompositionToOutputEvent_surjective {N w h : ℕ}
    (hw : 0 < w) (hh : 0 < h) (hhN : h ≤ N) :
    Function.Surjective (binaryRunCompositionToOutputEvent hw hhN) := by
  intro y
  let bits := (binaryWordToBoolVector y.1).toList
  have hbitsLength : bits.length = N := by
    simp [bits]
  have hbitsWeight : bits.count true = h := by
    rw [← hammingNorm_boolVectorToBinaryWord]
    rw [boolVectorToBinaryWord_binaryWordToBoolVector]
    exact y.2.2
  have hbitsDiff : (boolListDifferences false bits).count true = w := by
    rw [← toList_boolVectorDifferences]
    rw [← hammingNorm_diff_boolVectorToBinaryWord]
    rw [boolVectorToBinaryWord_binaryWordToBoolVector]
    exact y.2.1
  obtain ⟨gaps, ones, hones, hgaps, hlen, hdecode⟩ :=
    exists_alternatingBits_decomposition bits
  have honesSum : ones.sum = h := by
    rw [← count_true_alternatingBits, hdecode]
    exact hbitsWeight
  have hgapsSum : gaps.sum = N - h := by
    have htotal := length_alternatingBits gaps ones
    rw [hdecode, hbitsLength, honesSum] at htotal
    omega
  have hrunCount : ones.length + (gaps.length - 1) = w := by
    rw [← count_boolListDifferences_alternatingBits gaps ones hones hgaps hlen,
      hdecode]
    exact hbitsDiff
  have honesNe : ones ≠ [] := by
    intro hnil
    rw [hnil] at honesSum
    simp at honesSum
    omega
  have hgapsNe : gaps ≠ [] := by
    intro hnil
    rw [hnil] at hlen
    simp only [List.length_nil] at hlen
    have : ones = [] := by
      apply List.eq_nil_of_length_eq_zero
      omega
    exact honesNe this
  cases hgapsEq : gaps with
  | nil => exact (hgapsNe hgapsEq).elim
  | cons g gs =>
      have hgsPos : gs.Forall fun z => 0 < z := by
        simpa [hgapsEq] using hgaps
      have hzeroSum : ((g + 1) :: gs).sum = N - h + 1 := by
        rw [hgapsEq] at hgapsSum
        simp only [List.sum_cons] at hgapsSum ⊢
        omega
      have honesLength : ones.length = (w + 1) / 2 := by
        rw [hgapsEq] at hrunCount hlen
        simp only [List.length_cons, Nat.add_sub_cancel] at hrunCount hlen
        omega
      have hzeroLength : ((g + 1) :: gs).length = w / 2 + 1 := by
        rw [hgapsEq] at hrunCount hlen
        simp only [List.length_cons, Nat.add_sub_cancel] at hrunCount hlen ⊢
        omega
      let oneComposition : Composition h :=
        { blocks := ones
          blocks_pos := by
            rw [List.forall_iff_forall_mem] at hones
            intro i hi
            exact hones i hi
          blocks_sum := honesSum }
      let zeroComposition : Composition (N - h + 1) :=
        { blocks := (g + 1) :: gs
          blocks_pos := by
            intro z hz
            simp only [List.mem_cons] at hz
            rcases hz with rfl | hz
            · omega
            · rw [List.forall_iff_forall_mem] at hgsPos
              exact hgsPos z hz
          blocks_sum := hzeroSum }
      let d : BinaryRunCompositions N w h :=
        (⟨oneComposition, honesLength⟩,
          ⟨zeroComposition, hzeroLength⟩)
      refine ⟨d, ?_⟩
      apply Subtype.ext
      change boolVectorToBinaryWord (binaryRunCompositionVector hhN d) = y.1
      rw [← boolVectorToBinaryWord_binaryWordToBoolVector y.1]
      apply congrArg boolVectorToBinaryWord
      apply List.Vector.toList_injective
      have hlist : (binaryRunCompositionVector hhN d).toList = bits := by
        simpa [d, oneComposition, zeroComposition, binaryRunCompositionVector,
          binaryRunCompositionBits, zeroGapBlocks, hgapsEq] using hdecode
      simpa [bits] using hlist

/-- The symbolic run-decomposition bijection underlying B2. -/
noncomputable def binaryRunCompositionOutputEventEquiv {N w h : ℕ}
    (hw : 0 < w) (hh : 0 < h) (hhN : h ≤ N) :
    BinaryRunCompositions N w h ≃
      {y : BinaryWord N //
        hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧
          hammingNorm y = h} :=
  Equiv.ofBijective (binaryRunCompositionToOutputEvent hw hhN)
    ⟨binaryRunCompositionToOutputEvent_injective hw hhN,
      binaryRunCompositionToOutputEvent_surjective hw hh hhN⟩

theorem card_binaryAccumulatorOutputEvent (N w h : ℕ) :
    Fintype.card
      {y : BinaryWord N //
        hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧
          hammingNorm y = h} =
      binaryAccumulatorOutputCount N w h := by
  classical
  unfold binaryAccumulatorOutputCount
  rw [Fintype.card_of_subtype
    (Finset.univ.filter fun y : BinaryWord N =>
      hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧ hammingNorm y = h)
    (by simp)]

/-- B2, with the positive output-weight condition required by the exact
statement. -/
theorem binaryAccumulatorCount_eq_formula {N w h : ℕ}
    (hw : 0 < w) (hh : 0 < h) (hhN : h ≤ N) :
    binaryAccumulatorCount N w h = binaryAccumulatorFormula N w h := by
  rw [binaryAccumulatorCount_eq_outputCount,
    ← card_binaryAccumulatorOutputEvent,
    ← card_binaryRunCompositions N w h hw hh]
  exact Fintype.card_congr (binaryRunCompositionOutputEventEquiv hw hh hhN).symm

end BAA
