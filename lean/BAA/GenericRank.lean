import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.RingTheory.MvPolynomial.Basic
import Mathlib.RingTheory.Localization.FractionRing

open scoped BigOperators

/-!
# Generic matrix rank over rational-function fields

The coefficient variables live in an `MvPolynomial` ring and rank is taken
after embedding that domain into its fraction field.  This is the algebraic
interface required by the later Cauchy--Binet/path-packing proof.
-/

namespace BAA

abbrev GenericCoefficientRing (Var : Type*) (F : Type*) [CommRing F] :=
  MvPolynomial Var F

abbrev GenericCoefficientField (Var : Type*) (F : Type*) [Field F] :=
  FractionRing (GenericCoefficientRing Var F)

/-- Embed a polynomial matrix into the rational-function field in which its
generic rank is computed. -/
noncomputable def genericMatrix
    {Var F Rows Cols : Type*} [Field F]
    (A : Matrix Rows Cols (GenericCoefficientRing Var F)) :
    Matrix Rows Cols (GenericCoefficientField Var F) :=
  A.map (algebraMap (GenericCoefficientRing Var F)
    (GenericCoefficientField Var F))

/-- Generic rank of a polynomial matrix. -/
noncomputable def genericMatrixRank
    {Var F Rows Cols : Type*} [Field F] [Fintype Rows] [Fintype Cols]
    (A : Matrix Rows Cols (GenericCoefficientRing Var F)) : ℕ :=
  Matrix.rank (genericMatrix A)

/-- Independent diagonal variables before passage to the fraction field. -/
noncomputable def genericDiagonal
    {Var F : Type*} [DecidableEq Var] [CommRing F] :
    Matrix Var Var (GenericCoefficientRing Var F) :=
  Matrix.diagonal fun i => MvPolynomial.X i

@[simp]
theorem genericDiagonal_apply
    {Var F : Type*} [DecidableEq Var] [CommRing F] (i j : Var) :
    genericDiagonal (F := F) i j =
      if i = j then MvPolynomial.X i else 0 := by
  by_cases h : i = j
  · subst j
    simp [genericDiagonal]
  · simp [genericDiagonal, h, Ne.symm h]

/-- First tagged diagonal layer for a two-layer generic matrix product. -/
noncomputable def firstLayerGenericDiagonal
    {I J F : Type*} [DecidableEq I] [CommRing F] :
    Matrix I I (GenericCoefficientRing (Sum I J) F) :=
  Matrix.diagonal fun i => MvPolynomial.X (Sum.inl i)

/-- Second tagged diagonal layer for a two-layer generic matrix product. -/
noncomputable def secondLayerGenericDiagonal
    {I J F : Type*} [DecidableEq J] [CommRing F] :
    Matrix J J (GenericCoefficientRing (Sum I J) F) :=
  Matrix.diagonal fun j => MvPolynomial.X (Sum.inr j)

@[simp]
theorem firstLayerGenericDiagonal_apply
    {I J F : Type*} [DecidableEq I] [CommRing F] (i i' : I) :
    firstLayerGenericDiagonal (I := I) (J := J) (F := F) i i' =
      if i = i' then MvPolynomial.X (Sum.inl i) else 0 := by
  by_cases h : i = i'
  · subst i'
    simp [firstLayerGenericDiagonal]
  · simp [firstLayerGenericDiagonal, h]

@[simp]
theorem secondLayerGenericDiagonal_apply
    {I J F : Type*} [DecidableEq J] [CommRing F] (j j' : J) :
    secondLayerGenericDiagonal (I := I) (J := J) (F := F) j j' =
      if j = j' then MvPolynomial.X (Sum.inr j) else 0 := by
  by_cases h : j = j'
  · subst j'
    simp [secondLayerGenericDiagonal]
  · simp [secondLayerGenericDiagonal, h]

/-- A three-link layered matrix with independent diagonal variables on its two
intermediate coordinate layers. -/
noncomputable def genericLayeredMatrix
    {F Rows I J Cols : Type*} [CommRing F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J]
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F) :
    Matrix Rows Cols (GenericCoefficientRing (Sum I J) F) :=
  A.map MvPolynomial.C * firstLayerGenericDiagonal * B.map MvPolynomial.C *
    secondLayerGenericDiagonal * C.map MvPolynomial.C

/-- Generic rank of a two-diagonal layered product. -/
noncomputable def genericLayeredMatrixRank
    {F Rows I J Cols : Type*} [Field F]
    [Fintype Rows] [Fintype I] [Fintype J] [Fintype Cols]
    [DecidableEq I] [DecidableEq J]
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F) : ℕ :=
  genericMatrixRank (genericLayeredMatrix A B C)

/-- Entrywise path expansion of the generic layered product. -/
theorem genericLayeredMatrix_apply
    {F Rows I J Cols : Type*} [CommRing F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J]
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (row : Rows) (col : Cols) :
    genericLayeredMatrix A B C row col =
      ∑ i, ∑ j,
        MvPolynomial.C (A row i * B i j * C j col) *
          MvPolynomial.X (Sum.inl i) * MvPolynomial.X (Sum.inr j) := by
  classical
  simp [genericLayeredMatrix, Matrix.mul_apply]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.sum_mul, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-! ## Tagged layer-selection monomials

The two diagonal layers use disjoint variable namespaces.  The exponent of a
minor term is therefore the incidence vector of an ordered pair of coordinate
sets.  The following lemmas make the paper's "distinct selections cannot
cancel" assertion exact, including over fields of positive characteristic. -/

/-- The squarefree exponent vector belonging to two intermediate coordinate
selections.  `Sum.inl` tags the first diagonal layer and `Sum.inr` the second. -/
noncomputable def layerSelectionExponent {I J : Type*}
    (S : Finset I) (T : Finset J) : Sum I J →₀ ℕ :=
  S.sum (fun i => Finsupp.single (Sum.inl i) 1) +
    T.sum (fun j => Finsupp.single (Sum.inr j) 1)

@[simp]
theorem layerSelectionExponent_inl {I J : Type*} [DecidableEq I]
    (S : Finset I) (T : Finset J) (i : I) :
    layerSelectionExponent S T (Sum.inl i) = if i ∈ S then 1 else 0 := by
  classical
  simp [layerSelectionExponent, Finsupp.single_apply]

@[simp]
theorem layerSelectionExponent_inr {I J : Type*} [DecidableEq J]
    (S : Finset I) (T : Finset J) (j : J) :
    layerSelectionExponent S T (Sum.inr j) = if j ∈ T then 1 else 0 := by
  classical
  simp [layerSelectionExponent, Finsupp.single_apply]

/-- The tagged exponent vector uniquely determines both coordinate sets. -/
theorem layerSelectionExponent_injective {I J : Type*} :
    Function.Injective (fun p : Finset I × Finset J =>
      layerSelectionExponent p.1 p.2) := by
  classical
  rintro ⟨S, T⟩ ⟨S', T'⟩ h
  have hS : S = S' := by
    ext i
    have hi := DFunLike.congr_fun h (Sum.inl i)
    simp only [layerSelectionExponent_inl] at hi
    by_cases hiS : i ∈ S <;> by_cases hiS' : i ∈ S' <;> simp_all
  have hT : T = T' := by
    ext j
    have hj := DFunLike.congr_fun h (Sum.inr j)
    simp only [layerSelectionExponent_inr] at hj
    by_cases hjT : j ∈ T <;> by_cases hjT' : j ∈ T' <;> simp_all
  simp [hS, hT]

/-- The monomial attached to a pair of intermediate coordinate sets. -/
noncomputable def layerSelectionMonomial {I J F : Type*} [CommSemiring F]
    (S : Finset I) (T : Finset J) : MvPolynomial (Sum I J) F :=
  MvPolynomial.monomial (layerSelectionExponent S T) 1

/-- Different pairs of selections give different tagged monomials. -/
theorem layerSelectionMonomial_injective {I J F : Type*} [CommSemiring F]
    [Nontrivial F] :
    Function.Injective (fun p : Finset I × Finset J =>
      layerSelectionMonomial (F := F) p.1 p.2) := by
  intro p q h
  apply layerSelectionExponent_injective
  exact ((MvPolynomial.monomial_eq_monomial_iff _ _ _ _).mp h).resolve_right
    (by simp) |>.1

/-- Exact noncancellation theorem for a finite minor expansion indexed by two
layer selections.  The polynomial sum is nonzero precisely when one scalar
minor coefficient is nonzero. -/
theorem layerSelectionMonomial_sum_ne_zero_iff
    {I J F : Type*} [CommRing F] [Nontrivial F]
    (U : Finset (Finset I × Finset J))
    (c : Finset I × Finset J → F) :
    (∑ p ∈ U,
        MvPolynomial.monomial (layerSelectionExponent p.1 p.2) (c p)) ≠ 0 ↔
      ∃ p ∈ U, c p ≠ 0 := by
  classical
  constructor
  · contrapose!
    intro h
    apply Finset.sum_eq_zero
    intro p hp
    simp [h p hp]
  · rintro ⟨p, hpU, hcp⟩ hzero
    have hc := congr_arg
      (MvPolynomial.coeff (layerSelectionExponent p.1 p.2)) hzero
    rw [MvPolynomial.coeff_sum] at hc
    simp only [MvPolynomial.coeff_monomial] at hc
    rw [Finset.sum_eq_single p] at hc
    · have : c p = 0 := by simpa using hc
      exact hcp this
    · intro q hqU hqp
      rw [if_neg]
      exact fun he => hqp (layerSelectionExponent_injective he)
    · exact fun hp => (hp hpU).elim

/-! ## Layered minor expansions

For fixed row and column selections of size `r`, Cauchy--Binet contributes one
coefficient for every pair of size-`r` intermediate selections.  We record
that coefficient and the resulting polynomial explicitly.  The arbitrary
enumeration supplied by `Finset.equivFinOfCardEq` can change determinant signs,
but never whether the coefficient is zero. -/

/-- Product of the three square minors selected by a layered routing choice. -/
noncomputable def layeredMinorCoefficient
    {F Rows I J Cols : Type*} [CommRing F] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols)
    (S : Finset I) (T : Finset J) (hS : S.card = r) (hT : T.card = r) : F :=
  let s : Fin r → I := fun i => ((S.equivFinOfCardEq hS).symm i).1
  let t : Fin r → J := fun i => ((T.equivFinOfCardEq hT).symm i).1
  (A.submatrix rows s).det * (B.submatrix s t).det * (C.submatrix t cols).det

/-- Extend the layered minor coefficient by zero to selections of the wrong
cardinality. -/
noncomputable def layeredMinorTermCoefficient
    {F Rows I J Cols : Type*} [CommRing F] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols)
    (p : Finset I × Finset J) : F :=
  if hS : p.1.card = r then
    if hT : p.2.card = r then
      layeredMinorCoefficient A B C rows cols p.1 p.2 hS hT
    else 0
  else 0

/-- The exact two-layer Cauchy--Binet polynomial associated with a chosen
`r × r` outer minor. -/
noncomputable def layeredMinorExpansion
    {F Rows I J Cols : Type*} [CommRing F] [Fintype I] [Fintype J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    MvPolynomial (Sum I J) F :=
  ∑ p : Finset I × Finset J,
    MvPolynomial.monomial (layerSelectionExponent p.1 p.2)
      (layeredMinorTermCoefficient A B C rows cols p)

/-- A size-`r` algebraic layered packing: both intermediate layers select
`r` coordinates and all three connecting minors are simultaneously nonzero. -/
def HasLayeredMinorPacking
    {F Rows I J Cols : Type*} [CommRing F] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) : Prop :=
  ∃ (S : Finset I) (T : Finset J) (hS : S.card = r) (hT : T.card = r),
    layeredMinorCoefficient A B C rows cols S T hS hT ≠ 0

/-- The layered minor expansion is nonzero exactly when it contains an
algebraic layered packing.  In particular, no two distinct routing choices can
cancel, in any characteristic. -/
theorem layeredMinorExpansion_ne_zero_iff
    {F Rows I J Cols : Type*} [CommRing F] [Nontrivial F]
    [Fintype I] [Fintype J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    layeredMinorExpansion A B C rows cols ≠ 0 ↔
      HasLayeredMinorPacking A B C rows cols := by
  classical
  rw [layeredMinorExpansion, layerSelectionMonomial_sum_ne_zero_iff]
  constructor
  · rintro ⟨p, -, hp⟩
    simp only [layeredMinorTermCoefficient] at hp
    split at hp <;> rename_i hS
    · split at hp <;> rename_i hT
      · exact ⟨p.1, p.2, hS, hT, hp⟩
      · exact (hp rfl).elim
    · exact (hp rfl).elim
  · rintro ⟨S, T, hS, hT, hcoef⟩
    refine ⟨(S, T), Finset.mem_univ _, ?_⟩
    simp [layeredMinorTermCoefficient, hS, hT, hcoef]

/-! ## Specialization bridge to actual generic matrix rank -/

/-- Diagonal projection onto a finite coordinate set. -/
def coordinateProjection {I F : Type*} [DecidableEq I] [Semiring F]
    (S : Finset I) : Matrix I I F :=
  Matrix.diagonal fun i => if i ∈ S then 1 else 0

theorem mul_coordinateProjection_mul_apply
    {F Rows I Cols : Type*} [CommRing F] [Fintype I] [DecidableEq I]
    (A : Matrix Rows I F) (B : Matrix I Cols F) (S : Finset I)
    (row : Rows) (col : Cols) :
    (A * coordinateProjection (F := F) S * B) row col =
      ∑ i ∈ S, A row i * B i col := by
  classical
  unfold coordinateProjection
  rw [Matrix.mul_apply]
  simp_rw [Matrix.mul_diagonal]
  simp

/-- Restricting a product through a coordinate projection is the product of
the corresponding two restricted matrices. -/
theorem mul_coordinateProjection_mul_submatrix
    {F Rows I Cols : Type*} [CommRing F] [Fintype I] [DecidableEq I]
    {r : ℕ} (A : Matrix Rows I F) (B : Matrix I Cols F)
    (S : Finset I) (hS : S.card = r)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    (A * coordinateProjection (F := F) S * B).submatrix rows cols =
      A.submatrix rows (fun i => ((S.equivFinOfCardEq hS).symm i).1) *
        B.submatrix (fun i => ((S.equivFinOfCardEq hS).symm i).1) cols := by
  classical
  ext row col
  rw [Matrix.submatrix_apply, mul_coordinateProjection_mul_apply]
  simp only [Matrix.mul_apply, Matrix.submatrix_apply]
  let e := (S.equivFinOfCardEq hS).symm
  rw [← Finset.sum_attach]
  exact (Fintype.sum_equiv e
    (fun i : Fin r => A (rows row) (e i).1 * B (e i).1 (cols col))
    (fun i : S => A (rows row) i.1 * B i.1 (cols col)) (fun _ => rfl)).symm

/-- Evaluation that retains precisely one pair of intermediate selections. -/
noncomputable def layerSelectionEval
    {I J F : Type*} [CommRing F] [DecidableEq I] [DecidableEq J]
    (S : Finset I) (T : Finset J) :
    MvPolynomial (Sum I J) F →+* F :=
  MvPolynomial.eval fun v => Sum.elim
    (fun i => if i ∈ S then 1 else 0)
    (fun j => if j ∈ T then 1 else 0) v

/-- Under the selection evaluation, the independent generic diagonals become
ordinary coordinate projections. -/
theorem map_genericLayeredMatrix_layerSelectionEval
    {F Rows I J Cols : Type*} [CommRing F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J]
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (S : Finset I) (T : Finset J) :
    (genericLayeredMatrix A B C).map (layerSelectionEval S T) =
      A * coordinateProjection (F := F) S * B *
        coordinateProjection (F := F) T * C := by
  classical
  ext row col
  rw [Matrix.map_apply, genericLayeredMatrix_apply]
  rw [mul_coordinateProjection_mul_apply
    (A := A * coordinateProjection (F := F) S * B) (B := C) T row col]
  simp [layerSelectionEval]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  rw [mul_coordinateProjection_mul_apply A B S row j]
  rw [Finset.sum_mul]

/-- Both coordinate projections reduce a selected square submatrix to the
three square minors in its layered packing coefficient. -/
theorem layeredProjection_submatrix
    {F Rows I J Cols : Type*} [CommRing F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols)
    (S : Finset I) (T : Finset J) (hS : S.card = r) (hT : T.card = r) :
    (A * coordinateProjection (F := F) S * B *
      coordinateProjection (F := F) T * C).submatrix rows cols =
      A.submatrix rows (fun i => ((S.equivFinOfCardEq hS).symm i).1) *
        B.submatrix (fun i => ((S.equivFinOfCardEq hS).symm i).1)
          (fun i => ((T.equivFinOfCardEq hT).symm i).1) *
        C.submatrix (fun i => ((T.equivFinOfCardEq hT).symm i).1) cols := by
  classical
  rw [mul_coordinateProjection_mul_submatrix
    (A := A * coordinateProjection (F := F) S * B) (B := C) T hT rows cols]
  rw [mul_coordinateProjection_mul_submatrix A B S hS rows]

/-- A layered packing gives a genuinely nonzero minor of the polynomial
generic matrix.  The proof specializes all unused variables to zero and the
selected ones to one. -/
theorem HasLayeredMinorPacking.genericMinor_ne_zero
    {F Rows I J Cols : Type*} [Field F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols)
    (hpack : HasLayeredMinorPacking A B C rows cols) :
    ((genericLayeredMatrix A B C).submatrix rows cols).det ≠ 0 := by
  classical
  rintro hzero
  obtain ⟨S, T, hS, hT, hcoef⟩ := hpack
  apply hcoef
  have heval := congr_arg (layerSelectionEval S T) hzero
  rw [map_zero] at heval
  rw [(layerSelectionEval S T).map_det] at heval
  change (((genericLayeredMatrix A B C).submatrix rows cols).map
    (layerSelectionEval S T)).det = 0 at heval
  rw [← Matrix.submatrix_map] at heval
  rw [map_genericLayeredMatrix_layerSelectionEval] at heval
  rw [layeredProjection_submatrix A B C rows cols S T hS hT] at heval
  simp only [Matrix.det_mul] at heval
  exact heval

private theorem matrixRank_eq_card_of_det_ne_zero {K n : Type*} [Field K]
    [Fintype n] [DecidableEq n] (M : Matrix n n K) (h : M.det ≠ 0) :
    M.rank = Fintype.card n := by
  have hu : IsUnit M.det := isUnit_iff_ne_zero.mpr h
  simpa using Matrix.rank_mul_eq_right_of_isUnit_det M (1 : Matrix n n K) hu

/-- Every size-`r` algebraic layered packing contributes `r` independent
columns to the actual generic matrix over the fraction field. -/
theorem HasLayeredMinorPacking.card_le_genericLayeredMatrixRank
    {F Rows I J Cols : Type*} [Field F]
    [Fintype Rows] [Fintype I] [Fintype J] [Fintype Cols]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols)
    (hpack : HasLayeredMinorPacking A B C rows cols) :
    r ≤ genericLayeredMatrixRank A B C := by
  classical
  let P := genericLayeredMatrix A B C
  let K := GenericCoefficientField (Sum I J) F
  have hp : (P.submatrix rows cols).det ≠ 0 :=
    hpack.genericMinor_ne_zero A B C rows cols
  have hK : ((genericMatrix P).submatrix rows cols).det ≠ 0 := by
    change ((P.map (algebraMap (GenericCoefficientRing (Sum I J) F) K)).submatrix
      rows cols).det ≠ 0
    rw [Matrix.submatrix_map]
    change ((algebraMap (GenericCoefficientRing (Sum I J) F) K).mapMatrix
      (P.submatrix rows cols)).det ≠ 0
    rw [← (algebraMap (GenericCoefficientRing (Sum I J) F) K).map_det]
    exact IsFractionRing.to_map_eq_zero_iff.not.mpr hp
  have hrank : ((genericMatrix P).submatrix rows cols).rank = r := by
    rw [matrixRank_eq_card_of_det_ne_zero _ hK]
    simp
  calc
    r = ((genericMatrix P).submatrix rows cols).rank := hrank.symm
    _ ≤ (genericMatrix P).rank := Matrix.rank_submatrix_le _ _ _
    _ = genericLayeredMatrixRank A B C := rfl

end BAA
