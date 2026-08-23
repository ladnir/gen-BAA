import BAA.NonbinaryStageBound
import BAA.NonbinaryEnsemble
import BAA.SupportPropagation
import Mathlib.Logic.Equiv.Fintype

/-!
# Uniformity of finite monomial images

Permutation-plus-unit-diagonal samples act transitively on nonzero words of a
fixed Hamming weight.  Left composition is an explicit equivalence of the
sample space, so all output fibers have equal cardinality.
-/

namespace BAA

section

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

noncomputable local instance monomialClassicalDecidablePred
    {α : Type*} (p : α → Prop) : DecidablePred p :=
  Classical.decPred p

/-- One monomial sample at length `N`. -/
abbrev MonomialSample (F : Type*) [Monoid F] (N : ℕ) :=
  Equiv.Perm (Fin N) × (Fin N → Fˣ)

/-- The sampled monomial action on a word. -/
def monomialSampleOutput {N : ℕ} (x : Word F N)
    (z : MonomialSample F N) : Word F N :=
  monomialLinearEquiv F z.1 z.2 x

/-- Left composition by a fixed monomial map, written directly on permutation
and diagonal coordinates. -/
def monomialSampleLeftEquiv {N : ℕ} (tau : Equiv.Perm (Fin N))
    (e : Fin N → Fˣ) : MonomialSample F N ≃ MonomialSample F N where
  toFun z :=
    (z.1.trans tau, fun j => e (z.1 j) * z.2 j)
  invFun z :=
    (z.1.trans tau.symm,
      fun j => (e ((z.1.trans tau.symm) j))⁻¹ * z.2 j)
  left_inv z := by
    apply Prod.ext
    · ext j
      simp
    · funext j
      simp [mul_assoc]
  right_inv z := by
    apply Prod.ext
    · ext j
      simp
    · funext j
      simp [mul_assoc]

theorem monomialSampleOutput_leftEquiv {N : ℕ}
    (tau : Equiv.Perm (Fin N)) (e : Fin N → Fˣ)
    (x : Word F N) (z : MonomialSample F N) :
    monomialSampleOutput x (monomialSampleLeftEquiv tau e z) =
      monomialLinearEquiv F tau e (monomialSampleOutput x z) := by
  funext i
  simp [monomialSampleOutput, monomialSampleLeftEquiv,
    monomialLinearEquiv, LinearEquiv.trans_apply, mul_assoc]

/-- Canonical ambient permutation taking the support of `x` to the support of
`y`, when their support cardinalities agree. -/
noncomputable def supportTransportPerm {N : ℕ} (x y : Word F N)
    (hcard : (support x).card = (support y).card) :
    Equiv.Perm (Fin N) :=
  let e : ↥(support x) ≃ ↥(support y) :=
    (support x).orderIsoOfFin rfl |>.symm.toEquiv.trans
      ((support y).orderIsoOfFin hcard.symm).toEquiv
  e.extendSubtype

theorem supportTransportPerm_mem_iff {N : ℕ} (x y : Word F N)
    (hcard : (support x).card = (support y).card) (i : Fin N) :
    supportTransportPerm x y hcard i ∈ support y ↔ i ∈ support x := by
  classical
  unfold supportTransportPerm
  let e : ↥(support x) ≃ ↥(support y) :=
    (support x).orderIsoOfFin rfl |>.symm.toEquiv.trans
      ((support y).orderIsoOfFin hcard.symm).toEquiv
  by_cases hi : i ∈ support x
  · exact ⟨fun _ => hi, fun _ => e.extendSubtype_mem i hi⟩
  · exact ⟨fun h => (e.extendSubtype_not_mem i hi h).elim,
      fun h => (hi h).elim⟩

theorem supportTransportPerm_nonzero_iff {N : ℕ} (x y : Word F N)
    (hcard : (support x).card = (support y).card) (i : Fin N) :
    y (supportTransportPerm x y hcard i) ≠ 0 ↔ x i ≠ 0 := by
  rw [← mem_support, ← mem_support]
  exact supportTransportPerm_mem_iff x y hcard i

/-- Unit diagonal which, after the support transport permutation, changes the
nonzero coordinate values of `x` into those of `y`. -/
noncomputable def supportTransportDiagonal {N : ℕ} (x y : Word F N)
    (hcard : (support x).card = (support y).card) : Fin N → Fˣ :=
  fun i => if hi : x i ≠ 0 then
    Units.mk0
      (y (supportTransportPerm x y hcard i) / x i)
      (div_ne_zero
        ((supportTransportPerm_nonzero_iff x y hcard i).mpr hi) hi)
  else 1

theorem supportTransport_monomial_apply {N : ℕ} (x y : Word F N)
    (hcard : (support x).card = (support y).card) :
    monomialLinearEquiv F (supportTransportPerm x y hcard)
      (supportTransportDiagonal x y hcard) x = y := by
  funext k
  let tau := supportTransportPerm x y hcard
  let i := tau.symm k
  have htau : tau i = k := tau.apply_symm_apply k
  by_cases hi : x i ≠ 0
  · simp only [monomialLinearEquiv, LinearEquiv.trans_apply,
      permLinearEquiv_apply, diagLinearEquiv_apply]
    change ((supportTransportDiagonal x y hcard i : F) * x i) = y k
    rw [supportTransportDiagonal, dif_pos hi]
    simp only [Units.val_mk0]
    rw [div_mul_cancel₀ _ hi, htau]
  · have hy : y k = 0 := by
      rw [← htau]
      exact not_ne_iff.mp
        ((not_congr (supportTransportPerm_nonzero_iff x y hcard i)).mpr hi)
    simp only [monomialLinearEquiv, LinearEquiv.trans_apply,
      permLinearEquiv_apply, diagLinearEquiv_apply]
    change ((supportTransportDiagonal x y hcard i : F) * x i) = y k
    rw [not_ne_iff.mp hi, hy, mul_zero]

/-- Fibers over any two words of the same Hamming weight are equivalent. -/
noncomputable def monomialOutputFiberEquiv {N : ℕ} (x y z : Word F N)
    (hweight : hammingNorm y = hammingNorm z) :
    {omega : MonomialSample F N // monomialSampleOutput x omega = y} ≃
      {omega : MonomialSample F N // monomialSampleOutput x omega = z} := by
  have hcard : (support y).card = (support z).card := by
    simpa only [card_support_eq_hammingNorm] using hweight
  let tau := supportTransportPerm y z hcard
  let e := supportTransportDiagonal y z hcard
  exact Equiv.subtypeEquiv (monomialSampleLeftEquiv tau e) (by
    intro omega
    rw [monomialSampleOutput_leftEquiv]
    constructor <;> intro h
    · rw [h]
      exact supportTransport_monomial_apply y z hcard
    · apply (monomialLinearEquiv F tau e).injective
      rw [h]
      exact (supportTransport_monomial_apply y z hcard).symm)

/-- Words in the length-`N`, weight-`s` nonbinary Hamming slice. -/
abbrev NonbinaryWeightSlice (F : Type*) [Zero F] [DecidableEq F]
    (N s : ℕ) :=
  {y : Word F N // hammingNorm y = s}

theorem hammingNorm_monomialSampleOutput {N : ℕ} (x : Word F N)
    (omega : MonomialSample F N) :
    hammingNorm (monomialSampleOutput x omega) = hammingNorm x := by
  rw [← card_support_eq_hammingNorm, monomialSampleOutput,
    support_monomialLinearEquiv, permuteCoordinateSet,
    Finset.card_image_of_injective _ omega.1.injective,
    card_support_eq_hammingNorm]

/-- Monomial output bundled into the Hamming slice of the input. -/
def monomialImageInSlice {N : ℕ} (x : Word F N) :
    MonomialSample F N → NonbinaryWeightSlice F N (hammingNorm x) :=
  fun omega => ⟨monomialSampleOutput x omega,
    hammingNorm_monomialSampleOutput x omega⟩

theorem card_monomialImageInSlice_fiber_eq {N : ℕ} (x : Word F N)
    (y z : NonbinaryWeightSlice F N (hammingNorm x)) :
    (Finset.univ.filter fun omega : MonomialSample F N =>
      monomialImageInSlice x omega = y).card =
    (Finset.univ.filter fun omega : MonomialSample F N =>
      monomialImageInSlice x omega = z).card := by
  classical
  let e₁ : {omega : MonomialSample F N // monomialImageInSlice x omega = y} ≃
      {omega : MonomialSample F N // monomialSampleOutput x omega = y.1} :=
    Equiv.subtypeEquiv (Equiv.refl _) (by
      intro omega
      exact Subtype.ext_iff)
  let e₂ : {omega : MonomialSample F N // monomialImageInSlice x omega = z} ≃
      {omega : MonomialSample F N // monomialSampleOutput x omega = z.1} :=
    Equiv.subtypeEquiv (Equiv.refl _) (by
      intro omega
      exact Subtype.ext_iff)
  rw [← Fintype.card_subtype, ← Fintype.card_subtype,
    Fintype.card_congr e₁, Fintype.card_congr e₂]
  exact Fintype.card_congr
    (monomialOutputFiberEquiv x y.1 z.1 (y.2.trans z.2.symm))

theorem card_monomialSample_eq_slice_mul_fiber {N : ℕ} (x : Word F N)
    (y : NonbinaryWeightSlice F N (hammingNorm x)) :
    Fintype.card (MonomialSample F N) =
      Fintype.card (NonbinaryWeightSlice F N (hammingNorm x)) *
        (Finset.univ.filter fun omega : MonomialSample F N =>
          monomialImageInSlice x omega = y).card := by
  classical
  rw [Fintype.card, Finset.card_eq_sum_card_fiberwise
    (s := Finset.univ) (t := Finset.univ)
    (f := monomialImageInSlice x) (by simp)]
  have hfiber := fun z : NonbinaryWeightSlice F N (hammingNorm x) =>
    card_monomialImageInSlice_fiber_eq x z y
  simp_rw [hfiber]
  simp

/-- A uniform monomial image of a fixed word is uniform on its nonbinary
Hamming slice. -/
theorem uniform_monomial_event_probability {N : ℕ} (x : Word F N)
    (p : Word F N → Prop) [DecidablePred p] :
    uniformEventProbability (Omega := MonomialSample F N)
      (fun omega => p (monomialSampleOutput x omega)) =
    uniformEventProbability
      (Omega := NonbinaryWeightSlice F N (hammingNorm x))
      (fun y => p y.1) := by
  classical
  let xs : NonbinaryWeightSlice F N (hammingNorm x) := ⟨x, rfl⟩
  let k := (Finset.univ.filter fun omega : MonomialSample F N =>
    monomialImageInSlice x omega = xs).card
  have hgroup := card_monomialSample_eq_slice_mul_fiber x xs
  change Fintype.card (MonomialSample F N) =
    Fintype.card (NonbinaryWeightSlice F N (hammingNorm x)) * k at hgroup
  have hevent :
      (Finset.univ.filter fun omega : MonomialSample F N =>
        p (monomialSampleOutput x omega)).card =
      (Finset.univ.filter fun y : NonbinaryWeightSlice F N (hammingNorm x) =>
        p y.1).card * k := by
    rw [Finset.card_eq_sum_card_fiberwise
      (s := Finset.univ.filter fun omega : MonomialSample F N =>
        p (monomialSampleOutput x omega))
      (t := Finset.univ.filter fun y :
        NonbinaryWeightSlice F N (hammingNorm x) => p y.1)
      (f := monomialImageInSlice x)]
    · have hfiber : ∀ y ∈ (Finset.univ.filter fun y :
          NonbinaryWeightSlice F N (hammingNorm x) => p y.1),
          (Finset.univ.filter fun omega : MonomialSample F N =>
            p (monomialSampleOutput x omega) ∧
              monomialImageInSlice x omega = y).card = k := by
        intro y hy
        have hcard := card_monomialImageInSlice_fiber_eq x y xs
        change (Finset.univ.filter fun omega : MonomialSample F N =>
          p (monomialSampleOutput x omega) ∧
            monomialImageInSlice x omega = y).card =
          (Finset.univ.filter fun omega : MonomialSample F N =>
            monomialImageInSlice x omega = xs).card
        calc
          _ = (Finset.univ.filter fun omega : MonomialSample F N =>
              monomialImageInSlice x omega = y).card := by
            congr 1
            ext omega
            simp only [Finset.mem_filter, Finset.mem_univ, true_and]
            constructor
            · exact And.right
            · intro homega
              refine ⟨?_, homega⟩
              have hyp : p y.1 := (Finset.mem_filter.mp hy).2
              have hval := congrArg Subtype.val homega
              change monomialSampleOutput x omega = y.1 at hval
              exact hval.symm ▸ hyp
          _ = _ := hcard
      calc
        _ = ∑ y ∈ (Finset.univ.filter fun y :
            NonbinaryWeightSlice F N (hammingNorm x) => p y.1), k := by
          apply Finset.sum_congr rfl
          intro y hy
          simpa [Finset.filter_filter, and_assoc, and_comm, and_left_comm] using
            hfiber y hy
        _ = _ := by simp
    · intro omega homega
      have hp := (Finset.mem_filter.mp homega).2
      exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, by
        simpa [monomialImageInSlice] using hp⟩
  have hk : k ≠ 0 := by
    change (Finset.univ.filter fun omega : MonomialSample F N =>
      monomialImageInSlice x omega = xs).card ≠ 0
    rw [Finset.card_ne_zero]
    let one : Fin N → Fˣ := fun _ => 1
    refine ⟨(Equiv.refl _, one), Finset.mem_filter.mpr
      ⟨Finset.mem_univ _, ?_⟩⟩
    apply Subtype.ext
    change monomialSampleOutput x (Equiv.refl _, one) = x
    funext i
    simp [monomialImageInSlice, monomialSampleOutput, one,
      monomialLinearEquiv, LinearEquiv.trans_apply]
  unfold uniformEventProbability
  rw [hevent, hgroup]
  push_cast
  have hkQ : (k : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hk
  field_simp

/-- Slice-valued version of monomial uniformity, allowing predicates whose
types use the slice's weight certificate. -/
theorem uniform_monomial_slice_event_probability {N : ℕ} (x : Word F N)
    (p : NonbinaryWeightSlice F N (hammingNorm x) → Prop)
    [DecidablePred p] :
    uniformEventProbability (Omega := MonomialSample F N)
      (fun omega => p (monomialImageInSlice x omega)) =
    uniformEventProbability p := by
  classical
  let q : Word F N → Prop := fun y =>
    ∃ hy : hammingNorm y = hammingNorm x, p ⟨y, hy⟩
  calc
    uniformEventProbability
        (fun omega : MonomialSample F N => p (monomialImageInSlice x omega)) =
        uniformEventProbability
          (fun omega : MonomialSample F N => q (monomialSampleOutput x omega)) := by
      apply uniformEventProbability_congr
      intro omega
      constructor
      · intro hp
        refine ⟨hammingNorm_monomialSampleOutput x omega, ?_⟩
        have heq :
            (⟨monomialSampleOutput x omega,
              hammingNorm_monomialSampleOutput x omega⟩ :
                NonbinaryWeightSlice F N (hammingNorm x)) =
              monomialImageInSlice x omega := Subtype.ext rfl
        rw [heq]
        exact hp
      · rintro ⟨hy, hp⟩
        have heq :
            (⟨monomialSampleOutput x omega, hy⟩ :
                NonbinaryWeightSlice F N (hammingNorm x)) =
              monomialImageInSlice x omega := Subtype.ext rfl
        rw [← heq]
        exact hp
    _ = uniformEventProbability
        (Omega := NonbinaryWeightSlice F N (hammingNorm x))
        (fun y => q y.1) := uniform_monomial_event_probability x q
    _ = uniformEventProbability p := by
      apply uniformEventProbability_congr
      intro y
      constructor
      · rintro ⟨hy, hp⟩
        simpa using hp
      · intro hp
        exact ⟨y.2, by simpa using hp⟩

/-- Ordered nonzero values of a word, bundled as units. -/
noncomputable def orderedSupportUnits {N s : ℕ} (x : Word F N)
    (hcard : (support x).card = s) : Fin s → Fˣ :=
  fun i => Units.mk0 (orderedSupportValues x hcard i)
    (orderedSupportValues_ne_zero x hcard i)

/-- The reduced support/active-unit sample is exactly the nonbinary Hamming
slice. -/
noncomputable def reducedStageSampleEquivSlice {N s : ℕ} :
    ReducedNonbinaryStageSample F N s ≃ NonbinaryWeightSlice F N s where
  toFun z := ⟨wordOnOrderedSupport z.1 (fun i => (z.2 i : F)), by
    rw [← card_support_eq_hammingNorm,
      support_wordOnOrderedSupport z.1 (fun i => (z.2 i : F))
        (fun i => Units.ne_zero (z.2 i))]
    exact z.1.2⟩
  invFun y :=
    (⟨support y.1, by simpa only [card_support_eq_hammingNorm] using y.2⟩,
      orderedSupportUnits y.1
        (by simpa only [card_support_eq_hammingNorm] using y.2))
  left_inv z := by
    apply Prod.ext
    · apply Subtype.ext
      exact support_wordOnOrderedSupport z.1 (fun i => (z.2 i : F))
        (fun i => Units.ne_zero (z.2 i))
    · funext i
      apply Units.ext
      have hvalues := orderedSupportValues_wordOnOrderedSupport z.1
        (fun i => (z.2 i : F)) (fun i => Units.ne_zero (z.2 i))
      exact congrFun hvalues i
  right_inv y := by
    apply Subtype.ext
    exact wordOnOrderedSupport_orderedSupportValues y.1
      (by simpa only [card_support_eq_hammingNorm] using y.2)

theorem reducedStageSampleEquivSlice_apply {N s : ℕ}
    (z : ReducedNonbinaryStageSample F N s) :
    (reducedStageSampleEquivSlice (F := F) (N := N) (s := s) z).1 =
      reducedStageInput (fun _ : Fin s => (1 : F)) z := by
  unfold reducedStageInput
  change wordOnOrderedSupport z.1 (fun i => (z.2 i : F)) =
    wordOnOrderedSupport z.1 (weightedActiveValues (fun _ : Fin s => (1 : F)) z.2)
  apply congrArg (wordOnOrderedSupport z.1)
  funext i
  change (z.2 i : F) = 1 * (z.2 i : F)
  rw [one_mul]

/-- Statistic event for a full monomial sample followed by one accumulator. -/
def monomialAccumulatorStatEvent {N s : ℕ} (x : Word F N)
    (hsupport : (support x).card = s) (h a : ℕ)
    (omega : MonomialSample F N) : Prop :=
  hammingNorm (accumLinearEquiv F N (monomialSampleOutput x omega)) = h ∧
    accumulatorReturnCount (monomialSampleOutput x omega) = a

/-- The full permutation-plus-diagonal stage has the same statistic law as
the reduced support/unit model. -/
theorem monomialAccumulatorStatEvent_probability_eq_reduced
    {N s h a : ℕ} (x : Word F N)
    (hsupport : (support x).card = s) :
    uniformEventProbability (monomialAccumulatorStatEvent x hsupport h a) =
      uniformEventProbability
        (reducedStageStatEvent (N := N) (fun _ : Fin s => (1 : F)) h a) := by
  classical
  have hsweight : hammingNorm x = s :=
    (card_support_eq_hammingNorm x).symm.trans hsupport
  let p : NonbinaryWeightSlice F N (hammingNorm x) → Prop := fun y =>
    hammingNorm (accumLinearEquiv F N y) = h ∧
      accumulatorReturnCount y.1 = a
  calc
    uniformEventProbability (monomialAccumulatorStatEvent x hsupport h a) =
        uniformEventProbability (Omega := MonomialSample F N)
          (fun omega => p (monomialImageInSlice x omega)) := by
      apply uniformEventProbability_congr
      intro omega
      rfl
    _ = uniformEventProbability p :=
      uniform_monomial_slice_event_probability x p
    _ = uniformEventProbability
        (Omega := NonbinaryWeightSlice F N s)
        (fun y => p ⟨y.1, by simpa only [hsweight] using y.2⟩) := by
      cases hsweight
      rfl
    _ = uniformEventProbability
        (fun z : ReducedNonbinaryStageSample F N s =>
          p ⟨(reducedStageSampleEquivSlice (F := F) z).1,
            by simpa only [hsweight] using
              (reducedStageSampleEquivSlice (F := F) z).2⟩) :=
      (uniformEventProbability_equiv
        (reducedStageSampleEquivSlice (F := F) (N := N) (s := s))
        (fun y => p ⟨y.1, by simpa only [hsweight] using y.2⟩)).symm
    _ = uniformEventProbability
        (reducedStageStatEvent (N := N) (fun _ : Fin s => (1 : F)) h a) := by
      apply uniformEventProbability_congr
      intro z
      let yz : NonbinaryWeightSlice F N (hammingNorm x) :=
        ⟨(reducedStageSampleEquivSlice (F := F) z).1,
          by simpa only [hsweight] using
            (reducedStageSampleEquivSlice (F := F) z).2⟩
      let rz : NonbinaryWeightSlice F N (hammingNorm x) :=
        ⟨reducedStageInput (fun _ : Fin s => (1 : F)) z, by
          calc
            hammingNorm (reducedStageInput (fun _ : Fin s => (1 : F)) z) = s := by
              rw [← card_support_eq_hammingNorm,
                support_reducedStageInput (fun _ : Fin s => (1 : F))
                  (fun _ => one_ne_zero) z]
              exact z.1.2
            _ = hammingNorm x := hsweight.symm⟩
      have hyr : yz = rz := by
        apply Subtype.ext
        exact reducedStageSampleEquivSlice_apply z
      change p yz ↔ reducedStageStatEvent
        (fun _ : Fin s => (1 : F)) h a z
      rw [hyr]
      simp only [p, rz, reducedStageStatEvent]
      have hcard :
          (support (reducedStageInput (fun _ : Fin s => (1 : F)) z)).card = s := by
        rw [support_reducedStageInput (fun _ : Fin s => (1 : F))
          (fun _ => one_ne_zero) z]
        exact z.1.2
      have hreturn := accumulatorReturnCount_eq_activeTraceCard_of_card
        (reducedStageInput (fun _ : Fin s => (1 : F)) z) hcard
      have hordered := orderedSupportValues_reducedStageInput
        (fun _ : Fin s => (1 : F)) (fun _ => one_ne_zero) z
      rw [hordered] at hreturn
      rw [hreturn]

/-- Equation (1), including the forced-cancellation factor, for the actual
permutation-plus-diagonal monomial ensemble. -/
theorem monomialAccumulatorStatEvent_probability_le
    {N s h a : ℕ} (x : Word F N) (hx : x ≠ 0)
    (hsupport : (support x).card = s) (hsN : s ≤ N)
    (hhs : a < s) (hh : 0 < h) (hhN : h ≤ N) :
    uniformEventProbability (monomialAccumulatorStatEvent x hsupport h a) ≤
      nonbinaryAccumulatorKernel N s h a *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
  rw [monomialAccumulatorStatEvent_probability_eq_reduced x hsupport]
  apply reducedStageStatEvent_probability_le
  · intro i
    exact one_ne_zero
  · rw [← hsupport]
    exact Finset.card_pos.mpr (support_nonempty_of_ne_zero x hx)
  · exact hsN
  · exact hhs
  · exact hh
  · exact hhN

/-- Proof-independent form of the stage statistic, indexed by the actual
support cardinality of the input word. -/
def monomialAccumulatorCanonicalStatEvent {N : ℕ} (x : Word F N) (h a : ℕ)
    (omega : MonomialSample F N) : Prop :=
  hammingNorm (accumLinearEquiv F N (monomialSampleOutput x omega)) = h ∧
    accumulatorReturnCount (monomialSampleOutput x omega) = a

theorem monomialAccumulatorCanonicalStatEvent_probability_le_of_support_eq
    {N s h a : ℕ} (x : Word F N) (hx : x ≠ 0)
    (hsupport : (support x).card = s) (hsN : s ≤ N)
    (hhs : a < s) (hh : 0 < h) (hhN : h ≤ N) :
    uniformEventProbability (monomialAccumulatorCanonicalStatEvent x h a) ≤
      nonbinaryAccumulatorKernel N s h a *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
  calc
    uniformEventProbability (monomialAccumulatorCanonicalStatEvent x h a) =
        uniformEventProbability
          (monomialAccumulatorStatEvent x hsupport h a) := by
      apply uniformEventProbability_congr
      intro omega
      rfl
    _ ≤ _ := monomialAccumulatorStatEvent_probability_le
      x hx hsupport hsN hhs hh hhN

theorem monomialAccumulatorCanonicalStatEvent_probability_le
    {N h a : ℕ} (x : Word F N) (hx : x ≠ 0)
    (hsN : (support x).card ≤ N)
    (hhs : a < (support x).card) (hh : 0 < h) (hhN : h ≤ N) :
    uniformEventProbability (monomialAccumulatorCanonicalStatEvent x h a) ≤
      nonbinaryAccumulatorKernel N (support x).card h a *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a) := by
  exact monomialAccumulatorCanonicalStatEvent_probability_le_of_support_eq
    x hx rfl hsN hhs hh hhN

/-- Joint statistics for two independent monomial-accumulator stages. -/
def twoStageMonomialStatEvent {N : ℕ} (x : Word F N)
    (h a j b : ℕ) (z : MonomialSample F N × MonomialSample F N) : Prop :=
  monomialAccumulatorCanonicalStatEvent x h a z.1 ∧
    monomialAccumulatorCanonicalStatEvent
      (accumLinearEquiv F N (monomialSampleOutput x z.1)) j b z.2

/- Two-stage exact finite product bound. -/
theorem twoStageMonomialStatEvent_probability_le
    {N s h a j b : ℕ} (x : Word F N) (hx : x ≠ 0)
    (hsupport : (support x).card = s) (hsN : s ≤ N)
    (ha : a < s) (hh : 0 < h) (hhN : h ≤ N)
    (hb : b < h) (hj : 0 < j) (hjN : j ≤ N) :
    uniformEventProbability (twoStageMonomialStatEvent x h a j b) ≤
      (nonbinaryAccumulatorKernel N s h a *
        nonbinaryAccumulatorKernel N h j b) *
      (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ (a + b)) := by
  classical
  let c : ℚ := nonbinaryAccumulatorKernel N h j b *
    (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ b)
  have hc : 0 ≤ c := by
    unfold c nonbinaryAccumulatorKernel
    positivity
  have hcond : ∀ omega, monomialAccumulatorCanonicalStatEvent x h a omega →
      uniformEventProbability (fun eta =>
        monomialAccumulatorCanonicalStatEvent
          (accumLinearEquiv F N (monomialSampleOutput x omega)) j b eta) ≤ c := by
    intro omega homega
    let y : Word F N :=
      accumLinearEquiv F N (monomialSampleOutput x omega)
    change uniformEventProbability
      (monomialAccumulatorCanonicalStatEvent y j b) ≤
        nonbinaryAccumulatorKernel N h j b *
          (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ b)
    have hinput : monomialSampleOutput x omega ≠ 0 := by
      simpa only [monomialSampleOutput, map_zero] using
        (monomialLinearEquiv F omega.1 omega.2).injective.ne hx
    have hy : y ≠ 0 := by
      unfold y
      simpa only [map_zero] using (accumLinearEquiv F N).injective.ne hinput
    have hsupportY : (support y).card = h := by
      rw [card_support_eq_hammingNorm]
      unfold y
      exact homega.1
    exact monomialAccumulatorCanonicalStatEvent_probability_le_of_support_eq
      y hy hsupportY hhN hb hj hjN
  calc
    uniformEventProbability (twoStageMonomialStatEvent x h a j b) =
        uniformEventProbability (fun z : MonomialSample F N × MonomialSample F N =>
          monomialAccumulatorCanonicalStatEvent x h a z.1 ∧
            monomialAccumulatorCanonicalStatEvent
              (accumLinearEquiv F N (monomialSampleOutput x z.1)) j b z.2) := by
      apply uniformEventProbability_congr
      intro z
      rfl
    _ ≤
        uniformEventProbability
          (monomialAccumulatorCanonicalStatEvent x h a) * c := by
      exact uniformEventProbability_and_prod_le
        (monomialAccumulatorCanonicalStatEvent x h a)
        (fun omega eta => monomialAccumulatorCanonicalStatEvent
          (accumLinearEquiv F N (monomialSampleOutput x omega)) j b eta)
        c hc hcond
    _ ≤ (nonbinaryAccumulatorKernel N s h a *
          (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ a)) * c := by
      apply mul_le_mul_of_nonneg_right
      · exact monomialAccumulatorCanonicalStatEvent_probability_le_of_support_eq
          x hx hsupport hsN ha hh hhN
      · exact hc
    _ = (nonbinaryAccumulatorKernel N s h a *
          nonbinaryAccumulatorKernel N h j b) *
        (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ (a + b)) := by
      unfold c
      rw [pow_add]
      ring

/-- The four-coordinate concrete ensemble is exactly a pair of independent
monomial samples. -/
def nonbinaryBAAEnsembleEquivMonomialProduct (N : ℕ) :
    NonbinaryBAAEnsemble F N ≃ MonomialSample F N × MonomialSample F N where
  toFun omega :=
    ((omega.firstPermutation, omega.firstDiagonal),
      (omega.secondPermutation, omega.secondDiagonal))
  invFun z :=
    { firstPermutation := z.1.1
      firstDiagonal := z.1.2
      secondPermutation := z.2.1
      secondDiagonal := z.2.2 }
  left_inv omega := by cases omega; rfl
  right_inv z := by rcases z with ⟨⟨p₁, d₁⟩, p₂, d₂⟩; rfl

/-- Fixed-word version of the concrete trace statistics, before the outer-code
and exact-support quantifiers are introduced. -/
def nonbinaryBAAFixedWordStatEvent {N : ℕ} (x : Word F N)
    (h a j b : ℕ) (omega : NonbinaryBAAEnsemble F N) : Prop :=
  accumulatorReturnCount (nonbinaryFirstInput omega x) = a ∧
    hammingNorm (nonbinaryFirstOutput omega x) = h ∧
    accumulatorReturnCount (nonbinarySecondInput omega x) = b ∧
    hammingNorm (nonbinaryFinalOutput omega x) = j

theorem twoStageMonomialStatEvent_ensembleEquiv {N : ℕ} (x : Word F N)
    (h a j b : ℕ) (omega : NonbinaryBAAEnsemble F N) :
    twoStageMonomialStatEvent x h a j b
        (nonbinaryBAAEnsembleEquivMonomialProduct (F := F) N omega) ↔
      nonbinaryBAAFixedWordStatEvent x h a j b omega := by
  unfold twoStageMonomialStatEvent monomialAccumulatorCanonicalStatEvent
    nonbinaryBAAFixedWordStatEvent nonbinaryBAAEnsembleEquivMonomialProduct
    nonbinaryFirstInput nonbinaryFirstOutput nonbinarySecondInput
    nonbinaryFinalOutput monomialSampleOutput
  aesop

/-- The concrete four-coordinate ensemble inherits the exact two-stage
fixed-word bound from the product monomial sample. -/
theorem nonbinaryBAAFixedWordStatEvent_probability_le
    {N s h a j b : ℕ} (x : Word F N) (hx : x ≠ 0)
    (hsupport : (support x).card = s) (hsN : s ≤ N)
    (ha : a < s) (hh : 0 < h) (hhN : h ≤ N)
    (hb : b < h) (hj : 0 < j) (hjN : j ≤ N) :
    uniformEventProbability (nonbinaryBAAFixedWordStatEvent x h a j b) ≤
      (nonbinaryAccumulatorKernel N s h a *
        nonbinaryAccumulatorKernel N h j b) *
      (1 / ((Fintype.card F - 1 : ℕ) : ℚ) ^ (a + b)) := by
  classical
  calc
    uniformEventProbability (nonbinaryBAAFixedWordStatEvent x h a j b) =
        uniformEventProbability (fun omega : NonbinaryBAAEnsemble F N =>
          twoStageMonomialStatEvent x h a j b
            (nonbinaryBAAEnsembleEquivMonomialProduct (F := F) N omega)) := by
      apply uniformEventProbability_congr
      intro omega
      exact (twoStageMonomialStatEvent_ensembleEquiv x h a j b omega).symm
    _ = uniformEventProbability (twoStageMonomialStatEvent x h a j b) :=
      uniformEventProbability_equiv
        (nonbinaryBAAEnsembleEquivMonomialProduct (F := F) N)
        (twoStageMonomialStatEvent x h a j b)
    _ ≤ _ := twoStageMonomialStatEvent_probability_le
      x hx hsupport hsN ha hh hhN hb hj hjN

theorem accumulatorReturnCount_smul (c : F) (hc : c ≠ 0)
    {N : ℕ} (x : Word F N) :
    accumulatorReturnCount (c • x) = accumulatorReturnCount x := by
  unfold accumulatorReturnCount
  rw [support_smul_ne_zero c hc x]
  apply congrArg Finset.card
  ext i
  simp only [Finset.mem_filter]
  rw [LinearEquiv.map_smul]
  simp only [Pi.smul_apply, smul_eq_mul]
  simp [hc]

/-- All four fixed-word trace statistics depend only on the projective class
of the nonzero input word. -/
theorem nonbinaryBAAFixedWordStatEvent_smul (c : F) (hc : c ≠ 0)
    {N : ℕ} (x : Word F N) (h a j b : ℕ)
    (omega : NonbinaryBAAEnsemble F N) :
    nonbinaryBAAFixedWordStatEvent (c • x) h a j b omega ↔
      nonbinaryBAAFixedWordStatEvent x h a j b omega := by
  have hfirstInput :
      nonbinaryFirstInput omega (c • x) =
        c • nonbinaryFirstInput omega x := by
    exact LinearEquiv.map_smul _ c x
  have hfirstOutput :
      nonbinaryFirstOutput omega (c • x) =
        c • nonbinaryFirstOutput omega x := by
    unfold nonbinaryFirstOutput
    rw [hfirstInput, LinearEquiv.map_smul]
  have hsecondInput :
      nonbinarySecondInput omega (c • x) =
        c • nonbinarySecondInput omega x := by
    unfold nonbinarySecondInput
    rw [hfirstOutput, LinearEquiv.map_smul]
  have hfinalOutput :
      nonbinaryFinalOutput omega (c • x) =
        c • nonbinaryFinalOutput omega x := by
    unfold nonbinaryFinalOutput
    rw [hsecondInput, LinearEquiv.map_smul]
  unfold nonbinaryBAAFixedWordStatEvent
  rw [hfirstInput, accumulatorReturnCount_smul c hc,
    hfirstOutput, hammingNorm_smul (fun _ => IsSMulRegular.of_ne_zero hc),
    hsecondInput, accumulatorReturnCount_smul c hc,
    hfinalOutput, hammingNorm_smul (fun _ => IsSMulRegular.of_ne_zero hc)]

end

end BAA
