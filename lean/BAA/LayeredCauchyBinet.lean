import BAA.CauchyBinet
import BAA.GenericRank

open scoped BigOperators

namespace BAA

@[simp]
theorem selectionEmbedding_apply
    {I : Type*} [Fintype I] [DecidableEq I] {r : ℕ}
    (S : SizedSelection I r) (i : Fin r) :
    selectionEmbedding S i = ((S.1.equivFinOfCardEq S.2).symm i).1 :=
  rfl

theorem submatrix_mul_diagonal
    {R Rows I : Type*} [CommRing R] [Fintype I] [DecidableEq I]
    {r : ℕ} (A : Matrix Rows I R) (d : I → R)
    (rows : Fin r → Rows) (cols : Fin r ↪ I) :
    (A * Matrix.diagonal d).submatrix rows cols =
      A.submatrix rows cols * Matrix.diagonal (d ∘ cols) := by
  ext i j
  simp [Matrix.submatrix_apply, Matrix.mul_diagonal]

theorem submatrix_diagonal_mul
    {R I Cols : Type*} [CommRing R] [Fintype I] [DecidableEq I]
    {r : ℕ} (d : I → R) (A : Matrix I Cols R)
    (rows : Fin r ↪ I) (cols : Fin r → Cols) :
    (Matrix.diagonal d * A).submatrix rows cols =
      Matrix.diagonal (d ∘ rows) * A.submatrix rows cols := by
  ext i j
  simp [Matrix.submatrix_apply, Matrix.diagonal_mul]

theorem submatrix_rows_mul
    {R Rows I Cols : Type*} [CommRing R] [Fintype I]
    {r : ℕ} (A : Matrix Rows I R) (B : Matrix I Cols R)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    (A * B).submatrix rows cols = A.submatrix rows id * B.submatrix id cols := by
  ext i j
  rfl

theorem det_threeLink_diagonal
    {R I J : Type*} [CommRing R]
    [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J]
    {r : ℕ} (A : Matrix (Fin r) I R) (B : Matrix I J R)
    (C : Matrix J (Fin r) R) (x : I → R) (y : J → R) :
    (A * Matrix.diagonal x * B * Matrix.diagonal y * C).det =
      ∑ S : SizedSelection I r, ∑ T : SizedSelection J r,
        (A.submatrix id (selectionEmbedding S)).det *
          (B.submatrix (selectionEmbedding S) (selectionEmbedding T)).det *
          (C.submatrix (selectionEmbedding T) id).det *
          (∏ i, x (selectionEmbedding S i)) *
          (∏ j, y (selectionEmbedding T j)) := by
  classical
  rw [show A * Matrix.diagonal x * B * Matrix.diagonal y * C =
      (A * Matrix.diagonal x) * (B * Matrix.diagonal y * C) by
        simp only [Matrix.mul_assoc]]
  rw [rectangular_cauchyBinet]
  apply Fintype.sum_congr
  intro S
  rw [submatrix_mul_diagonal]
  rw [Matrix.det_mul, Matrix.det_diagonal]
  rw [show (B * Matrix.diagonal y * C).submatrix (selectionEmbedding S) id =
      (B.submatrix (selectionEmbedding S) id * Matrix.diagonal y) * C by
        ext i j
        rfl]
  rw [rectangular_cauchyBinet]
  rw [Finset.mul_sum]
  apply Fintype.sum_congr
  intro T
  rw [submatrix_mul_diagonal]
  rw [Matrix.det_mul, Matrix.det_diagonal]
  simp [Matrix.submatrix_submatrix]
  ring

theorem prod_X_finset_eq_monomial_sum_single
    {K R : Type*} [CommSemiring R] (S : Finset K) :
    (∏ k ∈ S, MvPolynomial.X k : MvPolynomial K R) =
      MvPolynomial.monomial (S.sum fun k => Finsupp.single k 1) 1 := by
  classical
  induction S using Finset.induction_on with
  | empty => simp
  | @insert a S ha ih =>
      rw [Finset.prod_insert ha, Finset.sum_insert ha, ih]
      rw [MvPolynomial.X, MvPolynomial.monomial_mul]
      simp

theorem prod_selectionEmbedding_X_eq
    {I J R : Type*} [CommSemiring R]
    [Fintype I] [DecidableEq I] {r : ℕ} (S : SizedSelection I r) :
    (∏ i : Fin r, MvPolynomial.X (Sum.inl (selectionEmbedding S i)) :
      MvPolynomial (Sum I J) R) =
      MvPolynomial.monomial
        (S.1.sum fun i => Finsupp.single (Sum.inl i) 1) 1 := by
  classical
  let emb : I ↪ Sum I J := ⟨Sum.inl, Sum.inl_injective⟩
  have hfin :
      (∏ i ∈ S.1, MvPolynomial.X (Sum.inl i) : MvPolynomial (Sum I J) R) =
        MvPolynomial.monomial
          (S.1.sum fun i => Finsupp.single (Sum.inl i) 1) 1 := by
    simpa [emb] using
      (prod_X_finset_eq_monomial_sum_single (R := R) (S.1.map emb))
  rw [← hfin]
  calc
    _ = ∏ i : S.1,
        (MvPolynomial.X (Sum.inl i.1) : MvPolynomial (Sum I J) R) :=
      Fintype.prod_equiv (selectionEquiv S)
        (fun i : Fin r =>
          (MvPolynomial.X (Sum.inl (selectionEmbedding S i)) :
            MvPolynomial (Sum I J) R))
        (fun i : S.1 =>
          (MvPolynomial.X (Sum.inl i.1) : MvPolynomial (Sum I J) R))
        (fun _ => rfl)
    _ = _ := Finset.prod_coe_sort S.1
      (fun i => (MvPolynomial.X (Sum.inl i) : MvPolynomial (Sum I J) R))

theorem prod_selectionEmbedding_X_inr_eq
    {I J R : Type*} [CommSemiring R]
    [Fintype J] [DecidableEq J] {r : ℕ} (T : SizedSelection J r) :
    (∏ j : Fin r, MvPolynomial.X (Sum.inr (selectionEmbedding T j)) :
      MvPolynomial (Sum I J) R) =
      MvPolynomial.monomial
        (T.1.sum fun j => Finsupp.single (Sum.inr j) 1) 1 := by
  classical
  let emb : J ↪ Sum I J := ⟨Sum.inr, Sum.inr_injective⟩
  have hfin :
      (∏ j ∈ T.1, MvPolynomial.X (Sum.inr j) : MvPolynomial (Sum I J) R) =
        MvPolynomial.monomial
          (T.1.sum fun j => Finsupp.single (Sum.inr j) 1) 1 := by
    simpa [emb] using
      (prod_X_finset_eq_monomial_sum_single (R := R) (T.1.map emb))
  rw [← hfin]
  calc
    _ = ∏ j : T.1,
        (MvPolynomial.X (Sum.inr j.1) : MvPolynomial (Sum I J) R) :=
      Fintype.prod_equiv (selectionEquiv T)
        (fun j : Fin r =>
          (MvPolynomial.X (Sum.inr (selectionEmbedding T j)) :
            MvPolynomial (Sum I J) R))
        (fun j : T.1 =>
          (MvPolynomial.X (Sum.inr j.1) : MvPolynomial (Sum I J) R))
        (fun _ => rfl)
    _ = _ := Finset.prod_coe_sort T.1
      (fun j => (MvPolynomial.X (Sum.inr j) : MvPolynomial (Sum I J) R))

theorem genericLayeredMatrix_submatrix
    {F Rows I J Cols : Type*} [CommRing F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    (genericLayeredMatrix A B C).submatrix rows cols =
      (A.map MvPolynomial.C).submatrix rows id * firstLayerGenericDiagonal *
        B.map MvPolynomial.C * secondLayerGenericDiagonal *
        (C.map MvPolynomial.C).submatrix id cols := by
  rfl

theorem det_map_C
    {F K n : Type*} [CommRing F] [Fintype n] [DecidableEq n]
    (M : Matrix n n F) :
    (M.map (MvPolynomial.C : F → MvPolynomial K F)).det = MvPolynomial.C M.det := by
  change (MvPolynomial.C.mapMatrix M).det = MvPolynomial.C M.det
  exact (MvPolynomial.C.map_det M).symm

theorem C_mul_monomial
    {F K : Type*} [CommRing F] (a b : F) (e : K →₀ ℕ) :
    MvPolynomial.C a * MvPolynomial.monomial e b =
      MvPolynomial.monomial e (a * b) := by
  rw [MvPolynomial.C_apply, MvPolynomial.monomial_mul, zero_add]

theorem det_genericLayeredMatrix_submatrix_eq_sized_sum
    {F Rows I J Cols : Type*} [Field F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    ((genericLayeredMatrix A B C).submatrix rows cols).det =
      ∑ S : SizedSelection I r, ∑ T : SizedSelection J r,
        MvPolynomial.monomial (layerSelectionExponent S.1 T.1)
          (layeredMinorCoefficient A B C rows cols S.1 T.1 S.2 T.2) := by
  classical
  rw [genericLayeredMatrix_submatrix]
  unfold firstLayerGenericDiagonal secondLayerGenericDiagonal
  rw [det_threeLink_diagonal]
  apply Fintype.sum_congr
  intro S
  apply Fintype.sum_congr
  intro T
  rw [prod_selectionEmbedding_X_eq S, prod_selectionEmbedding_X_inr_eq T]
  simp only [Matrix.submatrix_submatrix, Function.comp_def, id_eq,
    Matrix.submatrix_map]
  rw [det_map_C, det_map_C, det_map_C]
  unfold layeredMinorCoefficient layerSelectionExponent
  rw [show MvPolynomial.C
      (A.submatrix rows (selectionEmbedding S)).det *
        MvPolynomial.C
          (B.submatrix (selectionEmbedding S) (selectionEmbedding T)).det =
      MvPolynomial.C
        ((A.submatrix rows (selectionEmbedding S)).det *
          (B.submatrix (selectionEmbedding S) (selectionEmbedding T)).det) by
        rw [map_mul]]
  rw [show MvPolynomial.C
        ((A.submatrix rows (selectionEmbedding S)).det *
          (B.submatrix (selectionEmbedding S) (selectionEmbedding T)).det) *
        MvPolynomial.C (C.submatrix (selectionEmbedding T) cols).det *
        MvPolynomial.monomial
          (S.1.sum fun i => Finsupp.single (Sum.inl i) 1) 1 *
        MvPolynomial.monomial
          (T.1.sum fun j => Finsupp.single (Sum.inr j) 1) 1 =
      (MvPolynomial.C
          ((A.submatrix rows (selectionEmbedding S)).det *
            (B.submatrix (selectionEmbedding S) (selectionEmbedding T)).det) *
        MvPolynomial.monomial
          (S.1.sum fun i => Finsupp.single (Sum.inl i) 1) 1) *
      (MvPolynomial.C (C.submatrix (selectionEmbedding T) cols).det *
        MvPolynomial.monomial
          (T.1.sum fun j => Finsupp.single (Sum.inr j) 1) 1) by ring]
  rw [C_mul_monomial, C_mul_monomial]
  have heS : (selectionEmbedding S : Fin r → I) =
      fun i => ((S.1.equivFinOfCardEq S.2).symm i).1 :=
    funext (selectionEmbedding_apply S)
  have heT : (selectionEmbedding T : Fin r → J) =
      fun i => ((T.1.equivFinOfCardEq T.2).symm i).1 :=
    funext (selectionEmbedding_apply T)
  rw [heS, heT]
  rw [MvPolynomial.monomial_mul]
  simp

theorem sizedLayeredMinorSum_eq_layeredMinorExpansion
    {F Rows I J Cols : Type*} [Field F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    (∑ S : SizedSelection I r, ∑ T : SizedSelection J r,
        MvPolynomial.monomial (layerSelectionExponent S.1 T.1)
          (layeredMinorCoefficient A B C rows cols S.1 T.1 S.2 T.2)) =
      layeredMinorExpansion A B C rows cols := by
  classical
  let good : Finset I × Finset J → Prop :=
    fun p => p.1.card = r ∧ p.2.card = r
  let term : Finset I × Finset J → MvPolynomial (Sum I J) F :=
    fun p => MvPolynomial.monomial (layerSelectionExponent p.1 p.2)
      (layeredMinorTermCoefficient A B C rows cols p)
  let e : (SizedSelection I r × SizedSelection J r) ≃
      {p : Finset I × Finset J // good p} :=
    { toFun := fun p => ⟨(p.1.1, p.2.1), p.1.2, p.2.2⟩
      invFun := fun p => (⟨p.1.1, p.2.1⟩, ⟨p.1.2, p.2.2⟩)
      left_inv := fun p => rfl
      right_inv := fun p => rfl }
  have hterm (p : SizedSelection I r × SizedSelection J r) :
      MvPolynomial.monomial (layerSelectionExponent p.1.1 p.2.1)
          (layeredMinorCoefficient A B C rows cols
            p.1.1 p.2.1 p.1.2 p.2.2) = term (e p).1 := by
    simp [term, e, layeredMinorTermCoefficient, p.1.2, p.2.2]
  have hzero (p : Finset I × Finset J) (hp : p ∈ Finset.univ)
      (hpg : p ∉ Finset.univ.filter good) : term p = 0 := by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hpg
    simp only [good] at hpg
    rcases not_and_or.mp hpg with hS | hT
    · simp [term, layeredMinorTermCoefficient, hS]
    · simp [term, layeredMinorTermCoefficient, hT]
  calc
    (∑ S : SizedSelection I r, ∑ T : SizedSelection J r,
        MvPolynomial.monomial (layerSelectionExponent S.1 T.1)
          (layeredMinorCoefficient A B C rows cols S.1 T.1 S.2 T.2)) =
        ∑ p : SizedSelection I r × SizedSelection J r,
          MvPolynomial.monomial (layerSelectionExponent p.1.1 p.2.1)
            (layeredMinorCoefficient A B C rows cols
              p.1.1 p.2.1 p.1.2 p.2.2) := by
      rw [Fintype.sum_prod_type]
    _ = ∑ p : {p : Finset I × Finset J // good p}, term p.1 := by
      apply Fintype.sum_equiv e
      exact hterm
    _ = (∑ p ∈ Finset.univ.filter good, term p) := by
      symm
      apply Finset.sum_subtype
      intro p
      simp
    _ = ∑ p : Finset I × Finset J, term p := by
      exact Finset.sum_subset (Finset.filter_subset _ _) hzero
    _ = layeredMinorExpansion A B C rows cols := by
      rfl

/-- The determinant of an actual square minor of the generic layered matrix
is exactly the tagged Cauchy--Binet expansion used for noncancellation. -/
theorem det_genericLayeredMatrix_submatrix_eq_layeredMinorExpansion
    {F Rows I J Cols : Type*} [Field F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    ((genericLayeredMatrix A B C).submatrix rows cols).det =
      layeredMinorExpansion A B C rows cols :=
  (det_genericLayeredMatrix_submatrix_eq_sized_sum A B C rows cols).trans
    (sizedLayeredMinorSum_eq_layeredMinorExpansion A B C rows cols)

/-- A square minor of the generic layered matrix is nonzero exactly when its
three constant links admit a common full-size minor. -/
theorem genericLayeredMinor_ne_zero_iff_layeredPacking
    {F Rows I J Cols : Type*} [Field F] [Fintype I] [Fintype J]
    [DecidableEq I] [DecidableEq J] {r : ℕ}
    (A : Matrix Rows I F) (B : Matrix I J F) (C : Matrix J Cols F)
    (rows : Fin r → Rows) (cols : Fin r → Cols) :
    ((genericLayeredMatrix A B C).submatrix rows cols).det ≠ 0 ↔
      HasLayeredMinorPacking A B C rows cols := by
  rw [det_genericLayeredMatrix_submatrix_eq_layeredMinorExpansion]
  exact layeredMinorExpansion_ne_zero_iff A B C rows cols

end BAA
