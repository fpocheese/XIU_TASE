#!/usr/bin/env python3
"""Identify which assigned group is incomplete in screened Case-1 episodes."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/a2rl/reviewer_failure_cases_20260730")
SCREEN = ROOT / "results/partial_failure_screen_nominal_n100"
CANDIDATES = SCREEN / "partial_failure_candidates.csv"
ASSIGNMENT = [
    20, 21, 22, 23, 24, 25, 26, 27, 20, 21,
    22, 23, 24, 25, 26, 27, 20, 21, 22, 23,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    output: list[dict] = []
    for candidate in read_csv(CANDIDATES):
        if candidate["case"] != "case1":
            continue
        episode = int(candidate["episode"])
        summary_path = Path(candidate["source_summary"])
        event_path = summary_path.with_name("case1_hit_events.csv")
        events = [
            row for row in read_csv(event_path)
            if int(row["episode"]) == episode
        ]
        hit_ids = {int(row["defender_id"]) for row in events}
        missed = sorted(set(range(20)) - hit_ids)
        missed_targets = sorted({ASSIGNMENT[idx] for idx in missed})
        output.append(
            {
                "eval_seed": int(candidate["eval_seed"]),
                "hit_count": int(candidate["hit_count"]),
                "complete_groups": int(candidate["target_sync_count"]),
                "missed_defender_ids": ";".join(map(str, missed)),
                "missed_paper_labels": ";".join(
                    f"D{idx + 1}" for idx in missed
                ),
                "incomplete_internal_target_ids": ";".join(
                    map(str, missed_targets)
                ),
                "incomplete_group_labels": ";".join(
                    f"A{target - 19}" for target in missed_targets
                ),
                "source_events": str(event_path),
                "source_episode": episode,
            }
        )
    out_path = SCREEN / "case1_candidate_failure_groups.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(row)


if __name__ == "__main__":
    main()
