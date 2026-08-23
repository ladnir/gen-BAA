import BAA.BackwardRank
import BAA.OuterCode

/-!
# Two-step neighborhoods and block-rank obstruction

G3 and G4 are definitions.  The final theorem isolates G5's direct logical
step from the separate network/minor theorem needed to prove the rank cap for
the concrete BAA parity-check map.
-/

namespace BAA

/-- Apply a coordinate permutation to a finite coordinate set. -/
def permuteCoordinateSet {N : ℕ} (sigma : Equiv.Perm (Fin N))
    (S : Finset (Fin N)) : Finset (Fin N) :=
  S.image sigma

/-- `S ∪ (S+1)`, with the right endpoint omitted rather than wrapped. -/
def forwardBoundary {N : ℕ} (S : Finset (Fin N)) : Finset (Fin N) :=
  Finset.univ.filter fun j =>
    j ∈ S ∨ ∃ i ∈ S, i.val + 1 = j.val

@[simp]
theorem mem_forwardBoundary {N : ℕ} {S : Finset (Fin N)} (j : Fin N) :
    j ∈ forwardBoundary S ↔
      j ∈ S ∨ ∃ i ∈ S, i.val + 1 = j.val := by
  simp [forwardBoundary]

/-- G3: the two-step backward neighborhood under the two permutation
orderings. -/
def twoStepBackwardNeighborhood {N : ℕ}
    (sigma₁ sigma₂ : Equiv.Perm (Fin N)) (T : Finset (Fin N)) :
    Finset (Fin N) :=
  permuteCoordinateSet sigma₁
    (forwardBoundary (permuteCoordinateSet sigma₂ (forwardBoundary T)))

/-- G4: the sum of local block ranks, capped at `h` per block. -/
def localBlockRank {B m : ℕ} (h : ℕ) (U : Finset (Fin (B * m))) : ℕ :=
  ∑ b : Fin B, min (localCoordinateSet U b).card h

/-- The rank-cap property supplied by the direct matrix/path argument for the
concrete BAA parity-check map. -/
def SatisfiesTwoStepBlockRankBound {F : Type*} [Field F] {B m r : ℕ}
    (q : Word F (B * m) →ₗ[F] Word F r)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m))) (h : ℕ) : Prop :=
  ∀ T, columnRank q T ≤
    localBlockRank h (twoStepBackwardNeighborhood sigma₁ sigma₂ T)

/-- G5 once the coefficient-independent two-step rank cap has been
established: a strict block-cap defect forces a nonzero supported kernel word. -/
theorem coefficientIndependentObstruction
    {F : Type*} [Field F] {B m r : ℕ}
    (q : Word F (B * m) →ₗ[F] Word F r)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m))) (h : ℕ)
    (hrank : SatisfiesTwoStepBlockRankBound q sigma₁ sigma₂ h)
    (T : Finset (Fin (B * m)))
    (hdefect : localBlockRank h
      (twoStepBackwardNeighborhood sigma₁ sigma₂ T) < T.card) :
    ∃ x : Word F (B * m),
      x ≠ 0 ∧ x ∈ supportedCode (F := F) T ∧ q x = 0 := by
  rw [exists_nonzero_kernel_supported_iff_columnRank_lt]
  exact lt_of_le_of_lt (hrank T) hdefect

end BAA
