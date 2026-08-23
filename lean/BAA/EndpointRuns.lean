import BAA.BinaryBoundary
import BAA.BlockObstruction
import BAA.FiniteProbability

/-!
# Endpoint-refined run encodings

The endpoint flag is part of the finite index.  `false` means that the final
one-run ends before coordinate `N-1`; `true` means that it reaches that
coordinate.  The existing symbolic run bijection realizes each encoding as a
unique binary characteristic word, hence as a unique subset of `Fin N`.
-/

namespace BAA

def endpointBit (e : Bool) : ℕ := if e then 1 else 0

/-- Exact size after adjoining successors to a set with `r` runs and endpoint
flag `e`. -/
def runExpandedSize (t r : ℕ) (e : Bool) : ℕ :=
  t + r - endpointBit e

/-- Number of initial/change transitions for `r` one-runs with endpoint flag
`e`; the missing final transition is exactly the endpoint correction. -/
def endpointTransitionWeight (r : ℕ) (e : Bool) : ℕ :=
  if e then 2 * r - 1 else 2 * r

/-- Recover the endpoint flag from the parity of a positive transition
weight. -/
def transitionEndpointFlag (w : ℕ) : Bool := decide (w % 2 = 1)

/-- Recover the number of one-runs from a positive transition weight. -/
def transitionRunCount (w : ℕ) : ℕ := (w + 1) / 2

theorem transitionRunCount_pos {w : ℕ} (hw : 0 < w) :
    0 < transitionRunCount w := by
  unfold transitionRunCount
  omega

theorem endpointTransitionWeight_decode {w : ℕ} (hw : 0 < w) :
    endpointTransitionWeight (transitionRunCount w)
      (transitionEndpointFlag w) = w := by
  have hmod : w % 2 < 2 := Nat.mod_lt w (by omega)
  by_cases hodd : w % 2 = 1
  · simp [endpointTransitionWeight, transitionRunCount,
      transitionEndpointFlag, hodd]
    omega
  · have heven : w % 2 = 0 := by omega
    simp [endpointTransitionWeight, transitionRunCount,
      transitionEndpointFlag, hodd]
    omega

theorem endpointTransitionWeight_injective_positive
    {r₁ r₂ : ℕ} {e₁ e₂ : Bool} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂)
    (h : endpointTransitionWeight r₁ e₁ = endpointTransitionWeight r₂ e₂) :
    r₁ = r₂ ∧ e₁ = e₂ := by
  cases e₁ <;> cases e₂ <;>
    simp [endpointTransitionWeight] at h ⊢ <;> omega

/-- Positive run/gap compositions realizing a `t`-subset with `r` runs and
right-endpoint flag `e`. -/
abbrev EndpointRunEncoding (N t r : ℕ) (e : Bool) :=
  BinaryRunCompositions N (endpointTransitionWeight r e) t

/-- Realize an endpoint-refined encoding as its binary characteristic word. -/
def endpointRunWord {N t r : ℕ} {e : Bool} (htN : t ≤ N)
    (d : EndpointRunEncoding N t r e) : BinaryWord N :=
  binaryRunCompositionWord htN d

theorem endpointRunWord_injective {N t r : ℕ} {e : Bool} (htN : t ≤ N) :
    Function.Injective (@endpointRunWord N t r e htN) :=
  binaryRunCompositionWord_injective htN

/-- Endpoint-refined encodings are exactly the binary supports with the stated
weight and transition count. -/
noncomputable def endpointRunOutputEventEquiv {N t r : ℕ} {e : Bool}
    (hr : 0 < r) (ht : 0 < t) (htN : t ≤ N) :
    EndpointRunEncoding N t r e ≃
      {y : BinaryWord N //
        hammingNorm (diffLinearEquiv (ZMod 2) N y) =
          endpointTransitionWeight r e ∧ hammingNorm y = t} :=
  binaryRunCompositionOutputEventEquiv
    (by cases e <;> simp [endpointTransitionWeight] <;> omega) ht htN

theorem hammingNorm_endpointRunWord {N t r : ℕ} {e : Bool} (htN : t ≤ N)
    (d : EndpointRunEncoding N t r e) :
    hammingNorm (endpointRunWord htN d) = t :=
  hammingNorm_binaryRunCompositionWord htN d

theorem transitionWeight_endpointRunWord {N t r : ℕ} {e : Bool}
    (hr : 0 < r) (htN : t ≤ N) (d : EndpointRunEncoding N t r e) :
    hammingNorm (diffLinearEquiv (ZMod 2) N (endpointRunWord htN d)) =
      endpointTransitionWeight r e := by
  apply hammingNorm_diff_binaryRunCompositionWord
  cases e <;> simp [endpointTransitionWeight] <;> omega

/-- Exact one-step neighborhood size, including the right-endpoint
correction. -/
theorem card_forwardBoundary_endpointRunWord {N t r : ℕ} {e : Bool}
    (hr : 0 < r) (htN : t ≤ N) (d : EndpointRunEncoding N t r e) :
    (forwardBoundary (support (endpointRunWord htN d))).card =
      runExpandedSize t r e := by
  rw [card_forwardBoundary_support_binary,
    hammingNorm_endpointRunWord htN d,
    transitionWeight_endpointRunWord hr htN d]
  cases e <;> simp [endpointTransitionWeight, runExpandedSize, endpointBit] <;>
    omega

/-- Fixed-cardinality supports having the endpoint-refined transition
statistic. -/
abbrev EndpointRunSupportEvent (N t r : ℕ) (e : Bool) :=
  {T : FixedCardFinset N t //
    hammingNorm (diffLinearEquiv (ZMod 2) N
      ((binaryWordSupportEquiv N).symm T.1)) = endpointTransitionWeight r e}

/-- Reassociate the two output conditions so that the weight condition is in
the ambient subtype. -/
def binaryOutputEventNestedEquiv (N w t : ℕ) :
    {y : BinaryWord N //
      hammingNorm (diffLinearEquiv (ZMod 2) N y) = w ∧ hammingNorm y = t} ≃
      {y : BinaryWeightSlice N t //
        hammingNorm (diffLinearEquiv (ZMod 2) N y.1) = w} where
  toFun y := ⟨⟨y.1, y.2.2⟩, y.2.1⟩
  invFun y := ⟨y.1.1, y.2, y.1.2⟩
  left_inv y := by cases y; rfl
  right_inv y := by cases y with | mk y hy => cases y; rfl

/-- Restrict the binary support equivalence to one transition slice. -/
def binaryTransitionSliceSupportEquiv (N w t : ℕ) :
    {y : BinaryWeightSlice N t //
      hammingNorm (diffLinearEquiv (ZMod 2) N y.1) = w} ≃
      {T : FixedCardFinset N t //
        hammingNorm (diffLinearEquiv (ZMod 2) N
          ((binaryWordSupportEquiv N).symm T.1)) = w} :=
  Equiv.subtypeEquiv (binaryWeightSliceSupportEquiv N t) (by
    intro y
    have hy : (binaryWordSupportEquiv N).symm
        (binaryWeightSliceSupportEquiv N t y).1 = y.1 := by
      exact (binaryWordSupportEquiv N).symm_apply_apply y.1
    rw [hy])

/-- Endpoint run encodings are exactly the supports in their fixed transition
slice. -/
noncomputable def endpointRunSupportEventEquiv {N t r : ℕ} {e : Bool}
    (hr : 0 < r) (ht : 0 < t) (htN : t ≤ N) :
    EndpointRunEncoding N t r e ≃ EndpointRunSupportEvent N t r e :=
  (endpointRunOutputEventEquiv hr ht htN).trans
    ((binaryOutputEventNestedEquiv N (endpointTransitionWeight r e) t).trans
      (binaryTransitionSliceSupportEquiv N (endpointTransitionWeight r e) t))

theorem card_endpointRunSupportEvent {N t r : ℕ} {e : Bool}
    (hr : 0 < r) (ht : 0 < t) (htN : t ≤ N) :
    Fintype.card (EndpointRunSupportEvent N t r e) =
      Fintype.card (EndpointRunEncoding N t r e) := by
  exact Fintype.card_congr (endpointRunSupportEventEquiv hr ht htN).symm

/-- Every support in an endpoint transition slice has the corrected expanded
cardinality. -/
theorem card_forwardBoundary_endpointRunSupportEvent {N t r : ℕ} {e : Bool}
    (hr : 0 < r) (T : EndpointRunSupportEvent N t r e) :
    (forwardBoundary T.1.1).card = runExpandedSize t r e := by
  let x : BinaryWord N := (binaryWordSupportEquiv N).symm T.1.1
  have hs : support x = T.1.1 :=
    (binaryWordSupportEquiv N).apply_symm_apply T.1.1
  have hw : hammingNorm x = t := by
    rw [← card_support_eq_hammingNorm, hs]
    exact T.1.2
  rw [← hs, card_forwardBoundary_support_binary, hw, T.2]
  cases e <;> simp [endpointTransitionWeight, runExpandedSize, endpointBit] <;>
    omega

/-- Every nonempty fixed-cardinality support belongs to one endpoint/run
transition slice. -/
theorem exists_endpointRunSupportEvent {N t : ℕ} (ht : 0 < t)
    (T : FixedCardFinset N t) :
    ∃ r : Fin (N + 1), ∃ e : Bool,
      ∃ U : EndpointRunSupportEvent N t r.val e, U.1 = T := by
  let x : BinaryWord N := (binaryWordSupportEquiv N).symm T.1
  let w := hammingNorm (diffLinearEquiv (ZMod 2) N x)
  have hxweight : hammingNorm x = t := by
    rw [← card_support_eq_hammingNorm]
    change (support ((binaryWordSupportEquiv N).symm T.1)).card = t
    have hs : support ((binaryWordSupportEquiv N).symm T.1) = T.1 :=
      (binaryWordSupportEquiv N).apply_symm_apply T.1
    rw [hs]
    exact T.2
  have hx0 : x ≠ 0 := by
    intro hx
    rw [hx, hammingNorm_zero] at hxweight
    omega
  have hdiff0 : diffLinearEquiv (ZMod 2) N x ≠ 0 := by
    simpa using (diffLinearEquiv (ZMod 2) N).injective.ne hx0
  have hw : 0 < w := by
    exact hammingNorm_pos_iff.mpr hdiff0
  have hwN : w ≤ N := by
    simpa using hammingNorm_le_card_fintype
      (x := diffLinearEquiv (ZMod 2) N x)
  let r : Fin (N + 1) := ⟨transitionRunCount w, by
    unfold transitionRunCount
    omega⟩
  let U : EndpointRunSupportEvent N t r.val (transitionEndpointFlag w) :=
    ⟨T, by
      change w = endpointTransitionWeight r.val (transitionEndpointFlag w)
      exact (endpointTransitionWeight_decode hw).symm⟩
  exact ⟨r, transitionEndpointFlag w, U, rfl⟩

/-- Endpoint/run transition slices are disjoint for positive run counts. -/
theorem endpointRunSupportEvent_index_unique {N t : ℕ}
    {r₁ r₂ : Fin (N + 1)} {e₁ e₂ : Bool}
    (hr₁ : 0 < r₁.val) (hr₂ : 0 < r₂.val)
    (T₁ : EndpointRunSupportEvent N t r₁.val e₁)
    (T₂ : EndpointRunSupportEvent N t r₂.val e₂)
    (hT : T₁.1 = T₂.1) : r₁ = r₂ ∧ e₁ = e₂ := by
  have hw : endpointTransitionWeight r₁.val e₁ =
      endpointTransitionWeight r₂.val e₂ := by
    rw [← T₁.2, ← T₂.2]
    congr 3
    exact congrArg Subtype.val hT
  obtain ⟨hr, he⟩ := endpointTransitionWeight_injective_positive hr₁ hr₂ hw
  exact ⟨Fin.ext hr, he⟩

theorem endpointRunSupportEvent_run_pos {N t r : ℕ} {e : Bool}
    (ht : 0 < t) (T : EndpointRunSupportEvent N t r e) : 0 < r := by
  let x : BinaryWord N := (binaryWordSupportEquiv N).symm T.1.1
  have hxweight : hammingNorm x = t := by
    rw [← card_support_eq_hammingNorm]
    have hs : support x = T.1.1 :=
      (binaryWordSupportEquiv N).apply_symm_apply T.1.1
    rw [hs]
    exact T.1.2
  have hx0 : x ≠ 0 := hammingNorm_ne_zero_iff.mp (by omega)
  have hdiff0 : diffLinearEquiv (ZMod 2) N x ≠ 0 := by
    simpa using (diffLinearEquiv (ZMod 2) N).injective.ne hx0
  have hpos : 0 < hammingNorm (diffLinearEquiv (ZMod 2) N x) :=
    hammingNorm_pos_iff.mpr hdiff0
  rw [T.2] at hpos
  cases e <;> simp [endpointTransitionWeight] at hpos ⊢ <;> omega

abbrev EndpointRunSupportPartition (N t : ℕ) :=
  Σ r : Fin (N + 1), Σ e : Bool, EndpointRunSupportEvent N t r.val e

def endpointRunSupportPartitionToSupport {N t : ℕ} :
    EndpointRunSupportPartition N t → FixedCardFinset N t
  | ⟨_, _, T⟩ => T.1

theorem endpointRunSupportPartitionToSupport_bijective {N t : ℕ}
    (ht : 0 < t) :
    Function.Bijective (@endpointRunSupportPartitionToSupport N t) := by
  constructor
  · rintro ⟨r₁, e₁, T₁⟩ ⟨r₂, e₂, T₂⟩ hT
    have hr₁ := endpointRunSupportEvent_run_pos ht T₁
    have hr₂ := endpointRunSupportEvent_run_pos ht T₂
    obtain ⟨hr, he⟩ :=
      endpointRunSupportEvent_index_unique hr₁ hr₂ T₁ T₂ hT
    subst r₂
    subst e₂
    have hsub : T₁ = T₂ := Subtype.ext hT
    subst T₂
    rfl
  · intro T
    obtain ⟨r, e, U, hU⟩ := exists_endpointRunSupportEvent ht T
    exact ⟨⟨r, e, U⟩, hU⟩

/-- The positive-weight fixed support slice is partitioned exactly by the two
endpoint flags and the run count. -/
noncomputable def endpointRunSupportPartitionEquiv {N t : ℕ} (ht : 0 < t) :
    EndpointRunSupportPartition N t ≃ FixedCardFinset N t :=
  Equiv.ofBijective endpointRunSupportPartitionToSupport
    (endpointRunSupportPartitionToSupport_bijective ht)

theorem sum_card_endpointRunSupportEvent {N t : ℕ} (ht : 0 < t) :
    (∑ r : Fin (N + 1), ∑ e : Bool,
      Fintype.card (EndpointRunSupportEvent N t r.val e)) = Nat.choose N t := by
  rw [← card_fixedCardFinset N t,
    ← Fintype.card_congr (endpointRunSupportPartitionEquiv ht)]
  simp [EndpointRunSupportPartition]

theorem endpointRunEncoding_zero_isEmpty {N t : ℕ} (ht : 0 < t)
    (e : Bool) : IsEmpty (EndpointRunEncoding N t 0 e) := by
  constructor
  intro d
  have hlen : d.1.1.length = 0 := by
    simpa [EndpointRunEncoding, endpointTransitionWeight] using d.1.2
  have ht0 : t = 0 := d.1.1.length_eq_zero.mp hlen
  omega

theorem endpointRunSupportEvent_zero_isEmpty {N t : ℕ} (ht : 0 < t)
    (e : Bool) : IsEmpty (EndpointRunSupportEvent N t 0 e) := by
  constructor
  intro T
  have := endpointRunSupportEvent_run_pos ht T
  omega

theorem card_endpointRunSupportEvent_eq_encoding {N t : ℕ}
    (ht : 0 < t) (htN : t ≤ N) (r : Fin (N + 1)) (e : Bool) :
    Fintype.card (EndpointRunSupportEvent N t r.val e) =
      Fintype.card (EndpointRunEncoding N t r.val e) := by
  by_cases hr : r.val = 0
  · haveI : IsEmpty (EndpointRunSupportEvent N t r.val e) := by
      rw [hr]
      exact endpointRunSupportEvent_zero_isEmpty ht e
    haveI : IsEmpty (EndpointRunEncoding N t r.val e) := by
      rw [hr]
      exact endpointRunEncoding_zero_isEmpty ht e
    simp
  · exact card_endpointRunSupportEvent (Nat.pos_of_ne_zero hr) ht htN

/-- Exact endpoint-refined decomposition of all `t`-subsets, over the same
finite ranges used in corrected G9. -/
theorem sum_card_endpointRunEncoding {N t : ℕ} (ht : 0 < t) (htN : t ≤ N) :
    (∑ r : Fin (N + 1), ∑ e : Bool,
      Fintype.card (EndpointRunEncoding N t r.val e)) = Nat.choose N t := by
  rw [← sum_card_endpointRunSupportEvent ht]
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro e _
  exact (card_endpointRunSupportEvent_eq_encoding ht htN r e).symm

/-- Exact probability mass of one endpoint/run slice under the uniform
distribution on `t`-subsets. -/
theorem endpointRunSupportEvent_probability {N t : ℕ}
    (ht : 0 < t) (htN : t ≤ N) (r : Fin (N + 1)) (e : Bool) :
    uniformEventProbability (Omega := FixedCardFinset N t)
      (fun T => hammingNorm (diffLinearEquiv (ZMod 2) N
        ((binaryWordSupportEquiv N).symm T.1)) =
          endpointTransitionWeight r.val e) =
      (Fintype.card (EndpointRunEncoding N t r.val e) : ℚ) /
        Nat.choose N t := by
  classical
  unfold uniformEventProbability
  rw [← Fintype.card_of_subtype
      (p := fun T : FixedCardFinset N t =>
        hammingNorm (diffLinearEquiv (ZMod 2) N
          ((binaryWordSupportEquiv N).symm T.1)) =
            endpointTransitionWeight r.val e)
      _ (by simp),
    card_endpointRunSupportEvent_eq_encoding ht htN r e,
    card_fixedCardFinset]

/-- Reindex any finite sum over nonempty fixed-cardinality supports by the
exact endpoint/run partition. -/
theorem sum_fixedCardFinset_by_endpoint
    {N t : ℕ} (ht : 0 < t) {R : Type*} [AddCommMonoid R]
    (f : FixedCardFinset N t → R) :
    (∑ T, f T) =
      ∑ r : Fin (N + 1), ∑ e : Bool,
        ∑ U : EndpointRunSupportEvent N t r.val e, f U.1 := by
  rw [← (endpointRunSupportPartitionEquiv ht).sum_comp f]
  simp only [EndpointRunSupportPartition, Fintype.sum_sigma]
  rfl

/-- The endpoint-zero refinement in Section 6.4. -/
theorem card_endpointRunEncoding_false {N t r : ℕ}
    (hr : 0 < r) (ht : 0 < t) (htN : t ≤ N) :
    Fintype.card (EndpointRunEncoding N t r false) =
      Nat.choose (t - 1) (r - 1) * Nat.choose (N - t) r := by
  obtain ⟨r, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : r ≠ 0)
  rw [card_binaryRunCompositions N (endpointTransitionWeight (r + 1) false) t
    (by simp [endpointTransitionWeight]) ht]
  simp [binaryAccumulatorFormula, endpointTransitionWeight, Nat.mul_add_div]

/-- The endpoint-one refinement in Section 6.4. -/
theorem card_endpointRunEncoding_true {N t r : ℕ}
    (hr : 0 < r) (ht : 0 < t) (htN : t ≤ N) :
    Fintype.card (EndpointRunEncoding N t r true) =
      Nat.choose (t - 1) (r - 1) * Nat.choose (N - t) (r - 1) := by
  obtain ⟨r, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : r ≠ 0)
  rw [card_binaryRunCompositions N (endpointTransitionWeight (r + 1) true) t
    (by simp [endpointTransitionWeight]; omega) ht]
  simp only [binaryAccumulatorFormula, endpointTransitionWeight, if_true]
  rw [show 2 * (r + 1) - 1 = 2 * r + 1 by omega]
  rw [show 2 * r + 1 + 1 = 2 * (r + 1) by omega]
  simp [Nat.mul_add_div]

/-- G6 is the sum of the two exact endpoint refinements. -/
theorem endpointRunCount_sum {N t r : ℕ}
    (hr : 0 < r) (ht : 0 < t) (htN : t ≤ N) :
    Fintype.card (EndpointRunEncoding N t r false) +
        Fintype.card (EndpointRunEncoding N t r true) =
      Nat.choose (t - 1) (r - 1) * Nat.choose (N - t + 1) r := by
  rw [card_endpointRunEncoding_false hr ht htN,
    card_endpointRunEncoding_true hr ht htN]
  rw [← Nat.mul_add]
  have hpascal :
      Nat.choose (N - t + 1) r =
        Nat.choose (N - t) (r - 1) + Nat.choose (N - t) r := by
    obtain ⟨r, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : r ≠ 0)
    simpa [Nat.add_comm] using Nat.choose_succ_succ (N - t) r
  rw [hpascal, Nat.add_comm]

end BAA
