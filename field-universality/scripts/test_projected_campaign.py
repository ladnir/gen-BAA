#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_projected_kl_campaign as campaign


class ProjectedCampaignTests(unittest.TestCase):
    def test_frontier_state_uses_the_worst_priority(self) -> None:
        entries = []
        for priority, bound, marker in ((-0.1, 0.1, 1.0), (-0.2, 0.2, 2.0)):
            entries.append(
                {
                    "priority": priority,
                    "evaluated": {"bound": bound, "point": [marker] * 15},
                }
            )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "frontier.json"
            checkpoint.write_text(
                json.dumps(
                    {"splits_completed": 17, "queue": entries}
                ),
                encoding="utf-8",
            )
            with patch.object(
                campaign.projected,
                "composition_bank_upper",
                return_value=-0.03,
            ) as upper:
                state = campaign.frontier_state(checkpoint, [], [])
        self.assertEqual(state["splits"], 17)
        self.assertEqual(state["leaves"], 2)
        self.assertEqual(state["upper"], 0.2)
        self.assertEqual(upper.call_args.args[0][0], 2.0)

    def test_campaign_enriches_only_above_margin(self) -> None:
        arguments = argparse.Namespace(
            batches=1,
            batch_splits=25,
            enrich_target_upper=-0.005,
            checkpoint=Path("unused.json"),
            target_upper=0.0,
            max_boxes=0,
        )
        before = {
            "splits": 100,
            "leaves": 80,
            "upper": 0.02,
            "worst_exact_bank_upper": 0.001,
        }
        after = {
            "splits": 125,
            "leaves": 90,
            "upper": 0.01,
            "worst_exact_bank_upper": -0.01,
        }
        refinement = {"added": True}
        with (
            patch.object(
                campaign.projected,
                "load_or_generate_bank",
                return_value=({}, [object()], [object()]),
            ),
            patch.object(
                campaign,
                "frontier_state",
                side_effect=(before, after, after),
            ),
            patch.object(
                campaign.projected,
                "enrich_bank_from_frontier",
                return_value=[refinement],
            ) as enrich,
            patch.object(
                campaign.projected,
                "cover",
                return_value={"status": "floating LP box cover incomplete"},
            ),
            patch("builtins.print"),
        ):
            result = campaign.campaign(arguments)
        enrich.assert_called_once()
        self.assertEqual(arguments.max_boxes, 125)
        self.assertEqual(result["cycles"][0]["refinements_added"], 1)


if __name__ == "__main__":
    unittest.main()
