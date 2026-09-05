from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np
from mpmath import iv

import optimize_bulk_exponent as bulk
import optimize_conditional_product_bound as product_bound
import verify_reduced_composition_bulk as reduced
from verify_conditional_product_bulk import Interval, default_hybrid_cuts
from verify_conditional_product_bulk import default_cuts
from verify_eliminated_composition_bulk import (
    Box,
    Certificate,
    ExpandedCertificate,
    ONE,
    RegionalCertificate,
    ZERO,
    box_bound,
    candidate_certificate,
    common_cut_certificates,
    expand_certificate,
    materialize_certificates,
    evaluate_box_bound,
    evaluate_affine,
    has_common_first_cut,
    reference_certificates,
    make_stationary_cut,
    pure_cut_coefficients,
    regional_relation,
    select_regional_certificates,
    trim_certificate_bank,
)


class EliminatedCompositionVerifierTests(unittest.TestCase):
    def test_stationary_cut_lower_bounds_every_stationary_coupling(self) -> None:
        iv.dps = 80
        increment_support = Fraction(7, 50)
        state_support = Fraction(9, 50)
        crossing = Fraction(3, 50)
        difference = 2
        source = 3
        increment = np.zeros(5)
        state = np.zeros(5)
        increment[0] = 1.0 - float(increment_support)
        increment[difference] = float(increment_support)
        state[0] = 1.0 - float(state_support)
        state[source] = float(state_support)
        cut = make_stationary_cut(increment, state, float(crossing))
        certified = evaluate_affine(
            pure_cut_coefficients(cut, difference, source),
            (increment_support, state_support, crossing),
        )

        reference = product_bound.conditional_product_source_target(
            increment, state, float(crossing)
        )
        stationary = np.zeros((5, 5))
        stationary[0, 0] = 1.0 - float(state_support)
        stationary[source, source] = float(state_support)
        divergence = product_bound.relative_entropy(
            stationary.reshape(-1), reference.reshape(-1)
        )
        self.assertLessEqual(float(certified), divergence + 1e-12)

    def test_reduced_stationary_cut_matches_pure_branch(self) -> None:
        iv.dps = 80
        increment_support = Fraction(7, 50)
        state_support = Fraction(9, 50)
        crossing = Fraction(3, 50)
        difference = 2
        source = 3
        increment = np.zeros(5)
        state = np.zeros(5)
        increment[0] = 1.0 - float(increment_support)
        increment[difference] = float(increment_support)
        state[0] = 1.0 - float(state_support)
        state[source] = float(state_support)
        cut = make_stationary_cut(increment, state, float(crossing))
        point = lambda value: Interval(value, value)
        increment_conditional = tuple(
            point(ONE if value == difference else ZERO)
            for value in range(1, 5)
        )
        state_conditional = tuple(
            point(ONE if value == source else ZERO)
            for value in range(1, 5)
        )
        reduced_value = reduced.cut_lower_bound(
            cut,
            increment_conditional,
            state_conditional,
            point(increment_support),
            point(state_support),
            point(crossing),
        )
        branch_value = evaluate_affine(
            pure_cut_coefficients(cut, difference, source),
            (increment_support, state_support, crossing),
        )
        self.assertEqual(reduced_value, branch_value)

    def test_reduced_quinary_hybrid_point_is_exactly_negative(self) -> None:
        iv.dps = 80
        mu = np.array(
            [
                0.9979999999888409,
                7.540815878468292e-5,
                0.0009383929678498461,
                0.0009098391574175256,
                7.635972710697824e-5,
            ]
        )
        beta = np.array(
            [
                0.9969826141939717,
                2.8693660660114944e-5,
                0.0014518537060515479,
                0.0015075258714485654,
                2.9312567868007754e-5,
            ]
        )
        zeta = np.array(
            [
                0.78,
                0.012906651964062937,
                0.09835552547215248,
                0.09568445427262884,
                0.013053368291242531,
            ]
        )
        values = [
            0.002,
            0.0030173858060282364,
            0.22,
            0.0009919938017294817,
            0.0013089763630273346,
        ]
        values += list(mu[1:4] / mu[1:].sum())
        values += list(beta[1:4] / beta[1:].sum())
        values += list(zeta[1:4] / zeta[1:].sum())
        point = tuple(Fraction(str(value)) for value in values)
        bound = reduced.box_bound(
            reduced.Box(point, point), default_hybrid_cuts()
        )
        self.assertLess(bound.value, ZERO)
        self.assertLess(bound.value, Fraction(-1, 1000))

    def test_branch_dependent_candidate_is_not_accepted(self) -> None:
        iv.dps = 80
        point = tuple(
            Fraction(value)
            for value in (
                "0.002",
                "0.002813303962336616",
                "0.1323",
                "0.0009924297599807304",
                "0.0013463064097057693",
            )
        )
        box = Box(point, point)
        bound = box_bound(
            box,
            default_cuts(),
            (candidate_certificate(),),
            Fraction(1, 500),
            Fraction(1323, 10_000),
        )
        self.assertIsNotNone(bound)
        assert bound is not None
        self.assertFalse(has_common_first_cut(candidate_certificate()))
        self.assertGreater(bound.value, ZERO)
        self.assertLess(bound.value, Fraction(1, 10_000))

    def test_trimmed_local_cut_indices_are_remapped_exactly(self) -> None:
        static_cuts = default_cuts()
        cuts = static_cuts + tuple(static_cuts[0] for _ in range(20))
        certificates = list(reference_certificates())
        for group in range(5):
            start = 32 + 4 * group
            certificates.append(
                Certificate(
                    tuple(((start + index, ONE),) for index in range(4)),
                    tuple((ZERO,) * 4 for _ in range(4)),
                )
            )
        trimmed_cuts, trimmed_certificates = trim_certificate_bank(
            cuts, tuple(certificates), local_history=3
        )
        self.assertEqual(len(trimmed_cuts), 44)
        self.assertEqual(len(trimmed_certificates), 7)
        referenced = {
            cut_index
            for certificate in trimmed_certificates[4:]
            for weights in certificate.first_weights
            for cut_index, _ in weights
        }
        self.assertEqual(referenced, set(range(32, 44)))

    def test_expanded_certificate_round_trip_preserves_weights(self) -> None:
        cuts = default_cuts()
        certificate = candidate_certificate()
        expanded = expand_certificate(certificate, cuts)
        materialized_cuts, materialized = materialize_certificates(
            cuts, (expanded,)
        )
        self.assertEqual(materialized_cuts, cuts)
        self.assertEqual(materialized, (certificate,))

    def test_regional_selection_prefers_narrow_overlapping_box(self) -> None:
        wide = Box((ZERO,) * 5, (ONE,) * 5)
        narrow = Box((ZERO,) * 5, (Fraction(1, 10),) * 5)
        target = Box((ZERO,) * 5, (Fraction(1, 20),) * 5)
        wide_certificate = ExpandedCertificate(
            ((), (), (), ()), ((ZERO,) * 4,) * 4
        )
        narrow_certificate = ExpandedCertificate(
            ((), (), (), ()), ((ONE,) * 4,) * 4
        )
        selected = select_regional_certificates(
            target,
            (
                RegionalCertificate(wide, wide_certificate),
                RegionalCertificate(narrow, narrow_certificate),
            ),
            Fraction(1, 1000),
            1,
        )
        self.assertEqual(selected, (narrow_certificate,))

    def test_regional_relation_prioritizes_containment(self) -> None:
        target = Box((Fraction(1, 4),) * 5, (Fraction(1, 2),) * 5)
        containing = Box((ZERO,) * 5, (ONE,) * 5)
        touching = Box((Fraction(1, 2),) * 5, (ONE,) * 5)
        disjoint = Box((Fraction(3, 4),) * 5, (ONE,) * 5)
        self.assertEqual(regional_relation(target, containing), 0)
        self.assertEqual(regional_relation(target, touching), 1)
        self.assertEqual(regional_relation(target, disjoint), 2)

    def test_global_cut_quantifiers_bound_feasible_full_type(self) -> None:
        iv.dps = 80
        support = tuple(
            Fraction(value)
            for value in (
                "0.0183671875",
                "0.020640625",
                "0.21",
                "0.008952578203306872",
                "0.006796874999998952",
            )
        )
        box = Box(support, support)
        cuts = default_cuts()
        bound = evaluate_box_bound(
            box, cuts, common_cut_certificates(cuts)[:1]
        )

        layout = bulk.Layout(5)
        point = np.zeros(layout.size)
        point[layout.mu] = np.array(
            [
                0.9816328125,
                0.0091680048854503,
                1.55887448078442e-05,
                1.5589384042286077e-05,
                0.009168004485704782,
            ]
        )
        point[layout.beta] = np.array(
            [
                0.9793593749999748,
                0.010253716727887743,
                6.658518989716544e-05,
                6.658670561923554e-05,
                0.01025373637662102,
            ]
        )
        point[layout.zeta] = np.array(
            [
                0.7899999999999986,
                0.0713573830478184,
                0.03364271461274673,
                0.03364267290474526,
                0.07135722943469107,
            ]
        )
        point[layout.tau1] = np.array(
            [
                0.9702364774375136, 0.004556510428853393, 4.936895231497555e-06, 4.937240723018412e-06, 0.004556512997653281,
                0.0045565101603227505, 0.005665386313172295, 2.648256368824854e-05, 3.773490784382108e-07, 4.9603416260089055e-06,
                4.937216322032143e-06, 2.648243044757163e-05, 3.277403542073007e-05, 2.014204905837079e-06, 3.773028009945202e-07,
                4.936979255658395e-06, 3.7733697322769865e-07, 2.014447158734005e-06, 3.277346078949539e-05, 2.6484481442120043e-05,
                0.004556513206560702, 4.96021844125552e-06, 3.772483979989194e-07, 2.6484450122446436e-05, 0.005665401253098617,
            ]
        )
        point[layout.tau2] = np.array(
            [
                0.7832031250000007, 0.0033815119589334685, 1.692491628048944e-05, 1.692593113628897e-05, 0.0033815121936477156,
                0.0033815171592154495, 0.0666533185466824, 0.0012992825300624233, 9.470874804137815e-06, 1.3793937053978679e-05,
                1.692363592117438e-05, 0.0012992872304727363, 0.03142489945328306, 0.0008921333262099291, 9.470966859830504e-06,
                1.69251110115204e-05, 9.471990788597687e-06, 0.0008921365024015616, 0.03142485948171141, 0.0012992798188321665,
                0.003381509093849755, 1.3793320941187288e-05, 9.471210719195824e-06, 0.0012992832908835577, 0.06665317251829739,
            ]
        )
        equality, equality_target, _, _ = bulk.constraints(layout, 0.002, 0.21)
        self.assertLess(
            float(np.max(np.abs(equality @ point - equality_target))), 1e-10
        )
        feasible_value = bulk.exponent(point, layout, 3)
        self.assertGreaterEqual(float(bound.value) + 1e-10, feasible_value)


if __name__ == "__main__":
    unittest.main()
