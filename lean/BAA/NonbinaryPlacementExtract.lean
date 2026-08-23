import BAA.NonbinaryEvent
import BAA.Projective

/-!
# Extracting equation-(1) placements from actual accumulator inputs

This module proves the converse to the placement decoder: the ordered support
and zero-return behavior of an actual nonzero accumulator input determine the
compressed trace and the two positive interval compositions counted by (1).
-/

namespace BAA

/-- Length of the accumulator-state interval beginning at ordered active
coordinate `i`; the final interval ends at coordinate `N`. -/
def orderedActiveInterval {N s : ℕ} (f : Fin s ↪o Fin N) (i : Fin s) : ℕ :=
  if hi : i.val + 1 < s then
    (f ⟨i.val + 1, hi⟩).val - (f i).val
  else
    N - (f i).val

theorem orderedActiveInterval_pos {N s : ℕ} (f : Fin s ↪o Fin N)
    (i : Fin s) : 0 < orderedActiveInterval f i := by
  unfold orderedActiveInterval
  split_ifs with hi
  · have hlt := f.strictMono
      (show i < (⟨i.val + 1, hi⟩ : Fin s) by
        change i.val < i.val + 1
        omega)
    omega
  · have := (f i).isLt
    omega

/-- Ordered gaps reconstruct each support coordinate from the first one. -/
theorem orderedActivePosition_eq_leading_add_sum_Iio {N s : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s) (i : Fin s) :
    (f ⟨0, hs⟩).val +
      ∑ j ∈ Finset.Iio i, orderedActiveInterval f j = (f i).val := by
  cases s with
  | zero => exact Fin.elim0 i
  | succ n => induction i using Fin.induction with
    | zero => simp
    | succ i ih =>
      have hset : Finset.Iio i.succ =
          insert i.castSucc (Finset.Iio i.castSucc) := by
        ext j
        simp only [Finset.mem_Iio, Finset.mem_insert]
        constructor
        · intro hj
          by_cases heq : j.val = i.val
          · left
            exact Fin.ext heq
          · right
            change j.val < i.val
            change j.val < i.val + 1 at hj
            omega
        · rintro (rfl | hj)
          · simp
          · change j.val < i.val at hj
            change j.val < i.val + 1
            omega
      have hnmem : i.castSucc ∉ Finset.Iio i.castSucc := by simp
      rw [hset, Finset.sum_insert hnmem]
      have hinterval : orderedActiveInterval f i.castSucc =
          (f i.succ).val - (f i.castSucc).val := by
        have hi : i.castSucc.val + 1 < n + 1 := by
          simpa using i.isLt
        rw [orderedActiveInterval, dif_pos hi]
        congr 2
      rw [hinterval]
      have hmono := f.strictMono i.castSucc_lt_succ
      omega

/-- The leading gap and all ordered active intervals fill all `N`
coordinates. -/
theorem leading_add_sum_orderedActiveInterval {N s : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s) :
    (f ⟨0, hs⟩).val + ∑ i : Fin s, orderedActiveInterval f i = N := by
  let last : Fin s := ⟨s - 1, by omega⟩
  have hprefix := orderedActivePosition_eq_leading_add_sum_Iio f hs last
  have huniv : Finset.univ = insert last (Finset.Iio last) := by
    ext i
    simp only [Finset.mem_univ, Finset.mem_insert, Finset.mem_Iio, true_iff]
    by_cases h : i = last
    · exact Or.inl h
    · right
      change i.val < s - 1
      have hi := i.isLt
      have hne : i.val ≠ s - 1 := fun hv => h (Fin.ext hv)
      omega
  have hnmem : last ∉ Finset.Iio last := by simp
  rw [huniv, Finset.sum_insert hnmem]
  have hlast : orderedActiveInterval f last = N - (f last).val := by
    rw [orderedActiveInterval, dif_neg]
    dsimp [last]
    omega
  rw [hlast]
  omega

/-- Total length of intervals whose active state does not return to zero. -/
def orderedNonreturnWeight {N s a : ℕ} (f : Fin s ↪o Fin N)
    (A : Finset (Fin s)) (hcard : A.card = a) : ℕ :=
  ∑ j : Fin (s - a), orderedActiveInterval f
    (Aᶜ.orderEmbOfFin (by
      rw [Finset.card_compl, Fintype.card_fin, hcard]) j)

theorem sum_orderedReturnIntervals_add_nonreturnWeight {N s a : ℕ}
    (f : Fin s ↪o Fin N) (A : Finset (Fin s)) (hcard : A.card = a) :
    (∑ j : Fin a, orderedActiveInterval f (A.orderEmbOfFin hcard j)) +
      orderedNonreturnWeight f A hcard =
        ∑ i : Fin s, orderedActiveInterval f i := by
  let hcomp : Aᶜ.card = s - a := by
    rw [Finset.card_compl, Fintype.card_fin, hcard]
  let eA := (A.orderIsoOfFin hcard).symm.toEquiv
  let eC := (Aᶜ.orderIsoOfFin hcomp).symm.toEquiv
  have hA : (∑ j : Fin a,
      orderedActiveInterval f (A.orderEmbOfFin hcard j)) =
      ∑ i ∈ A, orderedActiveInterval f i := by
    calc
      _ = ∑ i : ↥A, orderedActiveInterval f i.1 :=
        eA.symm.sum_comp (fun i : ↥A => orderedActiveInterval f i.1)
      _ = _ := by
        simpa [Finset.univ_eq_attach] using
          Finset.sum_attach A (orderedActiveInterval f)
  have hC : orderedNonreturnWeight f A hcard =
      ∑ i ∈ Aᶜ, orderedActiveInterval f i := by
    unfold orderedNonreturnWeight
    calc
      _ = ∑ i : ↥(Aᶜ), orderedActiveInterval f i.1 :=
        eC.symm.sum_comp (fun i : ↥(Aᶜ) => orderedActiveInterval f i.1)
      _ = _ := by
        simpa [Finset.univ_eq_attach] using
          Finset.sum_attach Aᶜ (orderedActiveInterval f)
  rw [hA, hC]
  exact Finset.sum_add_sum_compl A (orderedActiveInterval f)

/-- The leading zero interval, shifted by one, followed by the return
intervals. -/
def orderedZeroCompositionParts {N s a : ℕ} (f : Fin s ↪o Fin N)
    (hs : 0 < s) (A : Finset (Fin s)) (hcard : A.card = a) :
    Fin (a + 1) → ℕ :=
  Fin.cases ((f ⟨0, hs⟩).val + 1)
    (fun j => orderedActiveInterval f (A.orderEmbOfFin hcard j))

theorem orderedZeroCompositionParts_pos {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a) :
    ∀ j, 0 < orderedZeroCompositionParts f hs A hcard j := by
  intro j
  refine Fin.cases ?_ (fun i => ?_) j
  · simp [orderedZeroCompositionParts]
  · simpa [orderedZeroCompositionParts] using
      orderedActiveInterval_pos f (A.orderEmbOfFin hcard i)

theorem sum_orderedZeroCompositionParts {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a) :
    ∑ j, orderedZeroCompositionParts f hs A hcard j =
      N - orderedNonreturnWeight f A hcard + 1 := by
  rw [Fin.sum_univ_succ]
  simp only [orderedZeroCompositionParts, Fin.cases_zero, Fin.cases_succ]
  have hsplit := sum_orderedReturnIntervals_add_nonreturnWeight f A hcard
  have htotal := leading_add_sum_orderedActiveInterval f hs
  omega

/-- The positive composition of non-return interval lengths. -/
def orderedNonzeroCompositionParts {N s a : ℕ} (f : Fin s ↪o Fin N)
    (A : Finset (Fin s)) (hcard : A.card = a) :
    Fin (s - a) → ℕ :=
  fun j => orderedActiveInterval f
    (Aᶜ.orderEmbOfFin (by
      rw [Finset.card_compl, Fintype.card_fin, hcard]) j)

theorem orderedNonzeroCompositionParts_pos {N s a : ℕ}
    (f : Fin s ↪o Fin N) (A : Finset (Fin s)) (hcard : A.card = a) :
    ∀ j, 0 < orderedNonzeroCompositionParts f A hcard j := by
  intro j
  exact orderedActiveInterval_pos f _

theorem sum_orderedNonzeroCompositionParts {N s a : ℕ}
    (f : Fin s ↪o Fin N) (A : Finset (Fin s)) (hcard : A.card = a) :
    ∑ j, orderedNonzeroCompositionParts f A hcard j =
      orderedNonreturnWeight f A hcard := by
  simp only [orderedNonzeroCompositionParts, orderedNonreturnWeight]

theorem orderedNonreturnWeight_le {N s a : ℕ} (f : Fin s ↪o Fin N)
    (hs : 0 < s) (A : Finset (Fin s)) (hcard : A.card = a) :
    orderedNonreturnWeight f A hcard ≤ N := by
  have hsplit := sum_orderedReturnIntervals_add_nonreturnWeight f A hcard
  have htotal := leading_add_sum_orderedActiveInterval f hs
  omega

/-- Reassemble a valid ordered trace and its support gaps into the exact
placement object counted by equation (1). -/
def placementOfOrderedData {N s a : ℕ} (f : Fin s ↪o Fin N)
    (hs : 0 < s) (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    NonbinaryAccumulatorPlacements N s
      (orderedNonreturnWeight f A hcard) a :=
  (compressedTraceOfValid A hcard hzero hnonadj,
    fixedLengthCompositionOfFn
      (orderedZeroCompositionParts f hs A hcard)
      (orderedZeroCompositionParts_pos f hs A hcard)
      (sum_orderedZeroCompositionParts f hs A hcard),
    fixedLengthCompositionOfFn
      (orderedNonzeroCompositionParts f A hcard)
      (orderedNonzeroCompositionParts_pos f A hcard)
      (sum_orderedNonzeroCompositionParts f A hcard))

theorem decodedTrace_placementOfOrderedData {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    decodedZeroReturnTrace
      (placementOfOrderedData f hs A hcard hzero hnonadj).1 = A := by
  exact decodedZeroReturnTrace_compressedTraceOfValid A hcard hzero hnonadj

theorem scheduleLeadingZero_placementOfOrderedData {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    scheduleLeadingZero (placementOfOrderedData f hs A hcard hzero hnonadj) =
      (f ⟨0, hs⟩).val := by
  unfold scheduleLeadingZero placementOfOrderedData
  rw [fixedCompositionPart_fixedLengthCompositionOfFn]
  simp [orderedZeroCompositionParts]

theorem scheduleActiveInterval_placementOfOrderedData {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A))
    (i : Fin s) :
    scheduleActiveInterval (placementOfOrderedData f hs A hcard hzero hnonadj) i =
      orderedActiveInterval f i := by
  let p := placementOfOrderedData f hs A hcard hzero hnonadj
  have htrace : decodedZeroReturnTrace p.1 = A :=
    decodedTrace_placementOfOrderedData f hs A hcard hzero hnonadj
  have horder :
      (decodedZeroReturnTrace p.1).orderEmbOfFin
          (card_decodedZeroReturnTrace p.1) =
        A.orderEmbOfFin hcard := by
    ext j
    exact congrArg Fin.val (congrFun (Finset.orderEmbOfFin_unique hcard
      (fun j => by
        have hm := Finset.orderEmbOfFin_mem
          (decodedZeroReturnTrace p.1)
          (card_decodedZeroReturnTrace p.1) j
        simpa only [htrace] using hm)
      ((decodedZeroReturnTrace p.1).orderEmbOfFin
        (card_decodedZeroReturnTrace p.1)).strictMono) j)
  unfold scheduleActiveInterval
  split_ifs with hi
  · have hiA : i ∈ A := by
      change i ∈ decodedZeroReturnTrace p.1 at hi
      rwa [htrace] at hi
    have hrank :
        (decodedZeroReturnTrace p.1).orderEmbOfFin
          (card_decodedZeroReturnTrace p.1)
          (decodedReturnRank p.1 ⟨i, hi⟩) = i := by
      have hinv := (decodedZeroReturnTrace p.1).orderIsoOfFin
        (card_decodedZeroReturnTrace p.1) |>.apply_symm_apply ⟨i, hi⟩
      exact congrArg Subtype.val hinv
    rw [horder] at hrank
    change fixedCompositionPart p.2.1
      (Fin.succ (decodedReturnRank p.1 ⟨i, hi⟩)) = _
    change fixedCompositionPart
      (fixedLengthCompositionOfFn
        (orderedZeroCompositionParts f hs A hcard)
        (orderedZeroCompositionParts_pos f hs A hcard)
        (sum_orderedZeroCompositionParts f hs A hcard))
      (Fin.succ (decodedReturnRank p.1 ⟨i, hi⟩)) = _
    rw [fixedCompositionPart_fixedLengthCompositionOfFn]
    simp only [orderedZeroCompositionParts, Fin.cases_succ]
    rw [hrank]
  · have hiA : i ∉ A := by
      change i ∉ decodedZeroReturnTrace p.1 at hi
      rwa [htrace] at hi
    have hcomp : Aᶜ.card = s - a := by
      rw [Finset.card_compl, Fintype.card_fin, hcard]
    have hdecodedComp : (decodedZeroReturnTrace p.1)ᶜ.card = s - a := by
      rw [Finset.card_compl, Fintype.card_fin,
        card_decodedZeroReturnTrace]
    have horderComp :
        (decodedZeroReturnTrace p.1)ᶜ.orderEmbOfFin hdecodedComp =
          Aᶜ.orderEmbOfFin hcomp := by
      have htraceComp : (decodedZeroReturnTrace p.1)ᶜ = Aᶜ :=
        congrArg (·ᶜ) htrace
      ext j
      exact congrArg Fin.val (congrFun (Finset.orderEmbOfFin_unique hcomp
        (fun j =>
          htraceComp ▸ Finset.orderEmbOfFin_mem
            (decodedZeroReturnTrace p.1)ᶜ hdecodedComp j)
        ((decodedZeroReturnTrace p.1)ᶜ.orderEmbOfFin
          hdecodedComp).strictMono) j)
    have hrank :
        (decodedZeroReturnTrace p.1)ᶜ.orderEmbOfFin hdecodedComp
          (decodedNonreturnRank p.1 ⟨i, hi⟩) = i := by
      let e₀ : {j : Fin s // j ∉ decodedZeroReturnTrace p.1} ≃
          ↥((decodedZeroReturnTrace p.1)ᶜ) :=
        { toFun j := ⟨j.1, Finset.mem_compl.mpr j.2⟩
          invFun j := ⟨j.1, Finset.mem_compl.mp j.2⟩
          left_inv _ := rfl
          right_inv _ := rfl }
      have hinv := ((decodedZeroReturnTrace p.1)ᶜ.orderIsoOfFin
        hdecodedComp).apply_symm_apply (e₀ ⟨i, hi⟩)
      exact congrArg Subtype.val hinv
    rw [horderComp] at hrank
    change fixedCompositionPart p.2.2
      (decodedNonreturnRank p.1 ⟨i, hi⟩) = _
    change fixedCompositionPart
      (fixedLengthCompositionOfFn
        (orderedNonzeroCompositionParts f A hcard)
        (orderedNonzeroCompositionParts_pos f A hcard)
        (sum_orderedNonzeroCompositionParts f A hcard))
      (decodedNonreturnRank p.1 ⟨i, hi⟩) = _
    rw [fixedCompositionPart_fixedLengthCompositionOfFn]
    unfold orderedNonzeroCompositionParts
    rw [hrank]

theorem scheduleActivePositionNat_placementOfOrderedData {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A))
    (i : Fin s) :
    scheduleActivePositionNat
      (placementOfOrderedData f hs A hcard hzero hnonadj) i = (f i).val := by
  unfold scheduleActivePositionNat
  rw [scheduleLeadingZero_placementOfOrderedData]
  have hsum :
      (∑ j ∈ Finset.Iio i,
        scheduleActiveInterval
          (placementOfOrderedData f hs A hcard hzero hnonadj) j) =
      ∑ j ∈ Finset.Iio i, orderedActiveInterval f j := by
    apply Finset.sum_congr rfl
    intro j _
    exact scheduleActiveInterval_placementOfOrderedData
      f hs A hcard hzero hnonadj j
  rw [hsum]
  exact orderedActivePosition_eq_leading_add_sum_Iio f hs i

theorem scheduleActivePosition_placementOfOrderedData {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A))
    (i : Fin s) :
    scheduleActivePosition
      (placementOfOrderedData f hs A hcard hzero hnonadj)
      (orderedNonreturnWeight_le f hs A hcard) i = f i := by
  apply Fin.ext
  exact scheduleActivePositionNat_placementOfOrderedData
    f hs A hcard hzero hnonadj i

theorem scheduleInputSupport_placementOfOrderedData {N s a : ℕ}
    (f : Fin s ↪o Fin N) (hs : 0 < s)
    (A : Finset (Fin s)) (hcard : A.card = a)
    (hzero : ∀ hs' : 0 < s, (⟨0, hs'⟩ : Fin s) ∉ A)
    (hnonadj : ∀ i : Fin (s - 1),
      ¬ (⟨i.val, by omega⟩ ∈ A ∧ ⟨i.val + 1, by omega⟩ ∈ A)) :
    scheduleInputSupport (placementOfOrderedData f hs A hcard hzero hnonadj)
      (orderedNonreturnWeight_le f hs A hcard) =
        Finset.univ.image f := by
  unfold scheduleInputSupport
  apply Finset.image_congr
  intro i _
  exact scheduleActivePosition_placementOfOrderedData
    f hs A hcard hzero hnonadj i

section ActualWord

variable {F : Type*} [Field F] [DecidableEq F]

/-- Active values of a word, read in increasing support order. -/
def orderedSupportValues {N s : ℕ} (x : Word F N)
    (hcard : (support x).card = s) : Fin s → F :=
  fun i => x ((support x).orderEmbOfFin hcard i)

theorem orderedSupportValues_ne_zero {N s : ℕ} (x : Word F N)
    (hcard : (support x).card = s) :
    ∀ i, orderedSupportValues x hcard i ≠ 0 := by
  intro i
  exact (mem_support x _).mp (Finset.orderEmbOfFin_mem _ _ _)

/-- Every actual nonzero accumulator input is realized by the placement
decoder.  The target weight is the actual accumulator-output weight, so this
is the converse bridge needed to apply equation (1) to concrete words. -/
theorem exists_placementOfWord {N : ℕ} (x : Word F N) (hx : x ≠ 0) :
    let S := support x
    let s := S.card
    let u := orderedSupportValues x rfl
    ∃ h a, ∃ hh : h ≤ N,
      ∃ p : NonbinaryAccumulatorPlacements N s h a,
        h = hammingNorm (accumLinearEquiv F N x) ∧
        a = (accumulatorZeroReturnTrace u).card ∧
        MatchesDecodedReturnTrace p u ∧
        scheduleInputWord p hh u = x ∧
        scheduleInputSupport p hh = S := by
  dsimp only
  let S := support x
  let s := S.card
  have hs : 0 < s := Finset.card_pos.mpr (support_nonempty_of_ne_zero x hx)
  let f : Fin s ↪o Fin N := S.orderEmbOfFin rfl
  let u : Fin s → F := orderedSupportValues x rfl
  have hu : ∀ i, u i ≠ 0 := orderedSupportValues_ne_zero x rfl
  let A := accumulatorZeroReturnTrace u
  let a := A.card
  let h' := orderedNonreturnWeight f A rfl
  let p₀ : NonbinaryAccumulatorPlacements N s h' a :=
    placementOfOrderedData f hs A rfl
      (fun hs' => first_not_mem_zeroReturnTrace u hs' hu)
      (zeroReturnTrace_nonadjacent u hu)
  have hh' : h' ≤ N := orderedNonreturnWeight_le f hs A rfl
  have htrace : MatchesDecodedReturnTrace p₀ u := by
    unfold MatchesDecodedReturnTrace
    exact (decodedTrace_placementOfOrderedData f hs A rfl
      (fun hs' => first_not_mem_zeroReturnTrace u hs' hu)
      (zeroReturnTrace_nonadjacent u hu)).symm
  have hsupport : scheduleInputSupport p₀ hh' = S := by
    calc
      scheduleInputSupport p₀ hh' = Finset.univ.image f := by
        exact scheduleInputSupport_placementOfOrderedData f hs A rfl
          (fun hs' => first_not_mem_zeroReturnTrace u hs' hu)
          (zeroReturnTrace_nonadjacent u hu)
      _ = S := S.image_orderEmbOfFin_univ rfl
  have hword : scheduleInputWord p₀ hh' u = x := by
    funext k
    by_cases hk : k ∈ S
    · rw [← S.image_orderEmbOfFin_univ rfl] at hk
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp hk
      change scheduleInputWord p₀ hh' u (f i) = u i
      rw [← scheduleActivePosition_placementOfOrderedData f hs A rfl
        (fun hs' => first_not_mem_zeroReturnTrace u hs' hu)
        (zeroReturnTrace_nonadjacent u hu) i]
      exact scheduleInputWord_active p₀ hh' u i
    · have hxk : x k = 0 := by
        rw [← not_ne_iff]
        intro hne
        exact hk ((mem_support x k).mpr hne)
      have hscheduled : scheduleInputWord p₀ hh' u k = 0 := by
        by_contra hne
        have hmem : k ∈ support (scheduleInputWord p₀ hh' u) :=
          (mem_support _ _).mpr hne
        rw [support_scheduleInputWord p₀ hh' u hu, hsupport] at hmem
        exact hk hmem
      rw [hscheduled, hxk]
  have hweight : hammingNorm (accumLinearEquiv F N x) = h' := by
    rw [← hword]
    exact hammingNorm_accum_scheduleInputWord p₀ hh' u htrace
  exact ⟨h', a, hh', p₀, hweight.symm, rfl, htrace, hword, hsupport⟩

/-- Exact-index form of `exists_placementOfWord`.  Supplying the three
observed statistics produces a placement in the corresponding fixed finite
type, avoiding transports in later counting arguments. -/
theorem exists_placementOfWord_at_stats {N s h a : ℕ}
    (x : Word F N) (hx : x ≠ 0)
    (hsupportCard : (support x).card = s)
    (hweight : hammingNorm (accumLinearEquiv F N x) = h)
    (hreturns :
      (accumulatorZeroReturnTrace
        (orderedSupportValues x hsupportCard)).card = a) :
    ∃ hh : h ≤ N, ∃ p : NonbinaryAccumulatorPlacements N s h a,
      MatchesDecodedReturnTrace p
        (orderedSupportValues x hsupportCard) ∧
      scheduleInputWord p hh (orderedSupportValues x hsupportCard) = x ∧
      scheduleInputSupport p hh = support x := by
  subst s
  obtain ⟨h', a', hle, p, hpweight, ha', htrace, hword, hsupport⟩ :=
    exists_placementOfWord x hx
  have eh : h' = h := hpweight.trans hweight
  have ea : a' = a := ha'.trans (by simpa using hreturns)
  cases eh
  cases ea
  exact ⟨hle, p, by simpa using htrace, by simpa using hword, hsupport⟩

end ActualWord

end BAA
