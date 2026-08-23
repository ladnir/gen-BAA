import BAA.NonbinaryOutput

/-!
# Concrete realizable placement events for equation (1)

The decoded data records an actual return set, the ordered zero/nonzero state
interval lengths, and the resulting input support.  The concrete event type is
the range of this decoder, so it contains exactly the realizable data.  Decoder
injectivity gives the cardinality bridge to the compressed parameter space.
-/

namespace BAA

structure DecodedAccumulatorPlacementData (N s h a : ℕ) where
  returnTrace : Finset (Fin s)
  zeroIntervals : Fin (a + 1) → ℕ
  nonzeroIntervals : Fin (s - a) → ℕ
  inputSupport : Finset (Fin N)

def decodeAccumulatorPlacement {N s h a : ℕ} (hhN : h ≤ N)
    (p : NonbinaryAccumulatorPlacements N s h a) :
    DecodedAccumulatorPlacementData N s h a where
  returnTrace := decodedZeroReturnTrace p.1
  zeroIntervals := fixedCompositionPart p.2.1
  nonzeroIntervals := fixedCompositionPart p.2.2
  inputSupport := scheduleInputSupport p hhN

theorem decodeAccumulatorPlacement_injective {N s h a : ℕ} (hhN : h ≤ N) :
    Function.Injective (@decodeAccumulatorPlacement N s h a hhN) := by
  intro p q hpq
  have htrace : decodedZeroReturnTrace p.1 = decodedZeroReturnTrace q.1 :=
    congrArg DecodedAccumulatorPlacementData.returnTrace hpq
  have hp1 : p.1 = q.1 := decodedZeroReturnTrace_injective htrace
  have hzero : fixedCompositionPart p.2.1 = fixedCompositionPart q.2.1 :=
    congrArg DecodedAccumulatorPlacementData.zeroIntervals hpq
  have hpzero : p.2.1 = q.2.1 := fixedCompositionPart_injective hzero
  have hnonzero : fixedCompositionPart p.2.2 = fixedCompositionPart q.2.2 :=
    congrArg DecodedAccumulatorPlacementData.nonzeroIntervals hpq
  have hpnonzero : p.2.2 = q.2.2 := fixedCompositionPart_injective hnonzero
  apply Prod.ext hp1
  exact Prod.ext hpzero hpnonzero

/-- Concrete realizable support/interval events decoded from equation (1)'s
compressed parameters. -/
abbrev ConcreteAccumulatorPlacementEvent (N s h a : ℕ) (hhN : h ≤ N) :=
  Set.range (@decodeAccumulatorPlacement N s h a hhN)

noncomputable def accumulatorPlacementEquivConcrete {N s h a : ℕ}
    (hhN : h ≤ N) :
    NonbinaryAccumulatorPlacements N s h a ≃
      ConcreteAccumulatorPlacementEvent N s h a hhN :=
  Equiv.ofInjective (decodeAccumulatorPlacement hhN)
    (decodeAccumulatorPlacement_injective hhN)

noncomputable instance concreteAccumulatorPlacementEventFintype
    {N s h a : ℕ} (hhN : h ≤ N) :
    Fintype (ConcreteAccumulatorPlacementEvent N s h a hhN) :=
  Fintype.ofEquiv (NonbinaryAccumulatorPlacements N s h a)
    (accumulatorPlacementEquivConcrete hhN)

theorem card_concreteAccumulatorPlacementEvent {N s h a : ℕ}
    (hhN : h ≤ N) :
    Fintype.card (ConcreteAccumulatorPlacementEvent N s h a hhN) =
      Fintype.card (NonbinaryAccumulatorPlacements N s h a) := by
  exact Fintype.card_congr (accumulatorPlacementEquivConcrete hhN).symm

/-- The numerator of equation (1) counts the concrete realizable decoded
placement events exactly. -/
theorem card_concreteAccumulatorPlacementEvent_formula {N s h a : ℕ}
    (hs : 0 < s) (hhs : a < s) (hh : 0 < h) (hhN : h ≤ N) :
    Fintype.card (ConcreteAccumulatorPlacementEvent N s h a hhN) =
      Nat.choose (s - a) a * Nat.choose (N - h) a *
        Nat.choose (h - 1) (s - a - 1) := by
  rw [card_concreteAccumulatorPlacementEvent (hhN := hhN)]
  exact card_nonbinaryAccumulatorPlacements hs hhs hh hhN

end BAA
