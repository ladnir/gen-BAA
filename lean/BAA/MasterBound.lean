import BAA.NonbinaryTrace
import BAA.Projective

/-!
# Exact finite support/trace master bound
-/

namespace BAA

/-- Finite index set for equation (3).  All trace and weight parameters range
over their exact finite bounds; the final weight is definitionally at most
`L`. -/
structure MasterTraceIndex (N L : ℕ) where
  support : Finset (Fin N)
  firstReturns : Fin (N + 1)
  intermediateWeight : Fin (N + 1)
  secondReturns : Fin (N + 1)
  finalWeight : Fin (L + 1)
deriving DecidableEq

abbrev MasterTraceData (N L : ℕ) :=
  Finset (Fin N) × Fin (N + 1) × Fin (N + 1) × Fin (N + 1) × Fin (L + 1)

def masterTraceIndexEquiv (N L : ℕ) :
    MasterTraceIndex N L ≃ MasterTraceData N L where
  toFun i := (i.support, i.firstReturns, i.intermediateWeight,
    i.secondReturns, i.finalWeight)
  invFun i := ⟨i.1, i.2.1, i.2.2.1, i.2.2.2.1, i.2.2.2.2⟩
  left_inv i := by cases i; rfl
  right_inv i := by rcases i with ⟨support, firstReturns, intermediateWeight,
    secondReturns, finalWeight⟩; rfl

noncomputable instance masterTraceIndexFintype (N L : ℕ) :
    Fintype (MasterTraceIndex N L) :=
  Fintype.ofEquiv (MasterTraceData N L) (masterTraceIndexEquiv N L).symm

noncomputable local instance classicalDecidablePred {α : Type*} (p : α → Prop) :
    DecidablePred p :=
  Classical.decPred p

section

variable (F : Type*) [Field F] [Fintype F] [DecidableEq F]

/-- One summand of the exact finite master bound (3). -/
noncomputable def masterTraceTerm {N L : ℕ}
    (shorteningRank : Finset (Fin N) → ℕ) (i : MasterTraceIndex N L) : ℚ :=
  nonbinaryAccumulatorKernel N i.support.card i.intermediateWeight.val
      i.firstReturns.val *
    nonbinaryAccumulatorKernel N i.intermediateWeight.val i.finalWeight.val
      i.secondReturns.val *
    projectiveCancellationBound F (shorteningRank i.support)
      (i.firstReturns.val + i.secondReturns.val)

/-- The right side of equation (3), with every finite range explicit. -/
noncomputable def masterTraceBound {N L : ℕ}
    (shorteningRank : Finset (Fin N) → ℕ) : ℚ :=
  ∑ i : MasterTraceIndex N L, masterTraceTerm F shorteningRank i

/-- Abstract finite union theorem behind equation (3).  A concrete ensemble
instantiation supplies the indexed trace events, their cover of the bad-code
event, and the per-index estimate from (1) and (2). -/
theorem probability_badCode_le_masterTraceBound
    {Omega : Type*} [Fintype Omega] [Nonempty Omega] {N L : ℕ}
    (shorteningRank : Finset (Fin N) → ℕ)
    (bad : Omega → Prop)
    (traceEvent : MasterTraceIndex N L → Omega → Prop)
    (hcover : ∀ omega, bad omega ↔ ∃ i, traceEvent i omega)
    (hterm : ∀ i, uniformEventProbability (traceEvent i) ≤
      masterTraceTerm F shorteningRank i) :
    uniformEventProbability bad ≤ masterTraceBound F (L := L) shorteningRank := by
  classical
  calc
    uniformEventProbability bad =
        uniformEventProbability (fun omega => ∃ i, traceEvent i omega) :=
      uniformEventProbability_congr hcover
    _ ≤ ∑ i, uniformEventProbability (traceEvent i) :=
      uniformEventProbability_exists_le_sum traceEvent
    _ ≤ ∑ i, masterTraceTerm F shorteningRank i := by
      exact Finset.sum_le_sum fun i _ => hterm i
    _ = masterTraceBound F shorteningRank := rfl

theorem masterTraceTerm_formula {N L : ℕ}
    (shorteningRank : Finset (Fin N) → ℕ) (i : MasterTraceIndex N L)
    (hs : 0 < i.support.card)
    (ha : i.firstReturns.val < i.support.card)
    (hh : 0 < i.intermediateWeight.val)
    (hhN : i.intermediateWeight.val ≤ N)
    (hah : i.secondReturns.val < i.intermediateWeight.val)
    (hj : 0 < i.finalWeight.val)
    (hjN : i.finalWeight.val ≤ N) :
    masterTraceTerm F shorteningRank i =
      (((Nat.choose (i.support.card - i.firstReturns.val) i.firstReturns.val : ℕ) : ℚ) *
          Nat.choose (N - i.intermediateWeight.val) i.firstReturns.val *
          Nat.choose (i.intermediateWeight.val - 1)
            (i.support.card - i.firstReturns.val - 1) /
          Nat.choose N i.support.card) *
      (((Nat.choose (i.intermediateWeight.val - i.secondReturns.val)
            i.secondReturns.val : ℕ) : ℚ) *
          Nat.choose (N - i.finalWeight.val) i.secondReturns.val *
          Nat.choose (i.finalWeight.val - 1)
            (i.intermediateWeight.val - i.secondReturns.val - 1) /
          Nat.choose N i.intermediateWeight.val) *
      projectiveCancellationBound F (shorteningRank i.support)
        (i.firstReturns.val + i.secondReturns.val) := by
  unfold masterTraceTerm
  rw [nonbinaryAccumulatorKernel_formula hs ha hh hhN,
    nonbinaryAccumulatorKernel_formula hh hah hj hjN]

end

end BAA
