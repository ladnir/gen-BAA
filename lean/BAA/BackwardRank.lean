import BAA.Construction
import BAA.Shortening
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

/-!
# Backward column-rank criterion

This module isolates the coefficient-independent linear-algebra content of G2.
Restricting a parity-check map to words supported in `T` is the coordinate-free
version of taking the columns indexed by `T`.
-/

namespace BAA

section TransportedParityCheck

variable {F : Type*} [Field F] {n r : ℕ}

/-- Pull a parity-check map backward through an invertible encoder. -/
def transportedParityCheck (H : Word F n →ₗ[F] Word F r)
    (E : Word F n ≃ₗ[F] Word F n) : Word F n →ₗ[F] Word F r :=
  H.comp E.symm.toLinearMap

@[simp]
theorem transportedParityCheck_apply (H : Word F n →ₗ[F] Word F r)
    (E : Word F n ≃ₗ[F] Word F n) (x : Word F n) :
    transportedParityCheck H E x = H (E.symm x) :=
  rfl

theorem ker_transportedParityCheck (H : Word F n →ₗ[F] Word F r)
    (E : Word F n ≃ₗ[F] Word F n) :
    LinearMap.ker (transportedParityCheck H E) =
      (LinearMap.ker H).map E.toLinearMap := by
  ext x
  constructor
  · intro hx
    have hxH : E.symm x ∈ LinearMap.ker H := by
      exact LinearMap.mem_ker.mpr (LinearMap.mem_ker.mp hx)
    exact ⟨E.symm x, hxH, E.apply_symm_apply x⟩
  · rintro ⟨y, hyH, rfl⟩
    exact LinearMap.mem_ker.mpr (by
      change H (E.symm (E y)) = 0
      rw [E.symm_apply_apply]
      exact LinearMap.mem_ker.mp hyH)

/-- G1 in the repository's fixed convention: the final-word parity checker is
the outer checker pulled backward through the two monomial/accumulator layers.
The displayed matrix product in the paper is the matrix representation of this
composition under column-vector convention. -/
def baaParityCheck {m B r : ℕ}
    (H : Word F (B * m) →ₗ[F] Word F r)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) :
    Word F (B * m) →ₗ[F] Word F r :=
  transportedParityCheck H (baaTransform F sigma₁ sigma₂ d₁ d₂)

theorem ker_baaParityCheck {m B r : ℕ} (C0 : Code F m)
    (H : Word F (B * m) →ₗ[F] Word F r)
    (hH : LinearMap.ker H = outerCode C0 B)
    (sigma₁ sigma₂ : Equiv.Perm (Fin (B * m)))
    (d₁ d₂ : Fin (B * m) → Fˣ) :
    LinearMap.ker (baaParityCheck H sigma₁ sigma₂ d₁ d₂) =
      baaCode F C0 sigma₁ sigma₂ d₁ d₂ := by
  rw [baaParityCheck, ker_transportedParityCheck, hH]
  rfl

end TransportedParityCheck

section SupportedCoordinates

variable (F : Type*) [Field F] {n : ℕ}

/-- Restrict a supported word to its active coordinates. -/
def supportedCodeRestriction (T : Finset (Fin n)) :
    supportedCode (F := F) T →ₗ[F] (T → F) where
  toFun x i := x.1 i.1
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- Extend a function on `T` by zero outside `T`. -/
def supportedCodeExtension (T : Finset (Fin n)) :
    (T → F) →ₗ[F] supportedCode (F := F) T where
  toFun y := ⟨fun i => if hi : i ∈ T then y ⟨i, hi⟩ else 0, by
    intro i hi
    simp [hi]⟩
  map_add' x y := by
    apply Subtype.ext
    funext i
    by_cases hi : i ∈ T <;> simp [hi]
  map_smul' c x := by
    apply Subtype.ext
    funext i
    by_cases hi : i ∈ T <;> simp [hi]

/-- Supported words are exactly functions on the chosen coordinate subtype. -/
def supportedCodeLinearEquiv (T : Finset (Fin n)) :
    supportedCode (F := F) T ≃ₗ[F] (T → F) where
  toLinearMap := supportedCodeRestriction F T
  invFun := supportedCodeExtension F T
  left_inv x := by
    apply Subtype.ext
    funext i
    by_cases hi : i ∈ T
    · simp [supportedCodeRestriction, supportedCodeExtension, hi]
    · simp [supportedCodeRestriction, supportedCodeExtension, hi, x.2 i hi]
  right_inv y := by
    funext i
    simp [supportedCodeRestriction, supportedCodeExtension, i.2]

theorem finrank_supportedCode (T : Finset (Fin n)) :
    Module.finrank F (supportedCode (F := F) T) = T.card := by
  rw [LinearEquiv.finrank_eq (supportedCodeLinearEquiv F T)]
  simp

end SupportedCoordinates

section ColumnRank

variable {F : Type*} [Field F] {n r : ℕ}

/-- Coordinate-free restriction of a parity-check map to columns in `T`. -/
def columnRestriction (q : Word F n →ₗ[F] Word F r)
    (T : Finset (Fin n)) : supportedCode (F := F) T →ₗ[F] Word F r :=
  q.domRestrict (supportedCode (F := F) T)

/-- Rank of the columns of `q` selected by `T`. -/
noncomputable def columnRank (q : Word F n →ₗ[F] Word F r)
    (T : Finset (Fin n)) : ℕ :=
  Module.finrank F (LinearMap.range (columnRestriction q T))

theorem columnRank_lt_card_iff
    (q : Word F n →ₗ[F] Word F r) (T : Finset (Fin n)) :
    columnRank q T < T.card ↔
      ∃ x : supportedCode (F := F) T,
        x ≠ 0 ∧ columnRestriction q T x = 0 := by
  have hrankNullity :=
    LinearMap.finrank_range_add_finrank_ker (columnRestriction q T)
  rw [← finrank_supportedCode F T]
  constructor
  · intro hlt
    have hkerPos : 0 < Module.finrank F (columnRestriction q T).ker := by
      unfold columnRank at hlt
      omega
    have hker : (columnRestriction q T).ker ≠ ⊥ :=
      Submodule.one_le_finrank_iff.mp (by omega)
    obtain ⟨x, hxker, hx0⟩ :=
      Submodule.exists_mem_ne_zero_of_ne_bot hker
    exact ⟨x, hx0, LinearMap.mem_ker.mp hxker⟩
  · rintro ⟨x, hx0, hxker⟩
    have hker : (columnRestriction q T).ker ≠ ⊥ := by
      intro hbot
      have : x = 0 := by
        have hxmem : x ∈ (columnRestriction q T).ker :=
          LinearMap.mem_ker.mpr hxker
        rw [hbot] at hxmem
        simpa using hxmem
      exact hx0 this
    have hkerPos : 0 < Module.finrank F (columnRestriction q T).ker := by
      have := Submodule.one_le_finrank_iff.mpr hker
      omega
    unfold columnRank
    omega

/-- G2: a nonzero kernel word supported in `T` exists exactly when the
selected-column rank is smaller than `|T|`. -/
theorem exists_nonzero_kernel_supported_iff_columnRank_lt
    (q : Word F n →ₗ[F] Word F r) (T : Finset (Fin n)) :
    (∃ x : Word F n, x ≠ 0 ∧ x ∈ supportedCode (F := F) T ∧ q x = 0) ↔
      columnRank q T < T.card := by
  rw [columnRank_lt_card_iff]
  constructor
  · rintro ⟨x, hx0, hxT, hxq⟩
    exact ⟨⟨x, hxT⟩, by simpa using hx0, hxq⟩
  · rintro ⟨x, hx0, hxq⟩
    exact ⟨x.1, by simpa using hx0, x.2, hxq⟩

end ColumnRank

end BAA
