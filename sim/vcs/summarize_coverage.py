"""Export compact DUT coverage and FSM gaps from an accepted VCS/URG run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    record = json.loads((args.run / "verification.json").read_text())
    if record.get("status") != "PASS":
        raise ValueError("Run has not passed")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    review = []
    for profile in record["profiles"]:
        directory = args.run / profile["coverage_directory"]
        dashboard = (directory / "dashboard.txt").read_text()
        match = re.search(
            r"Total Coverage Summary\s+SCORE\s+LINE\s+COND\s+TOGGLE\s+FSM\s+BRANCH\s+GROUP\s*\n\s*([\d. ]+)",
            dashboard,
        )
        if not match:
            raise ValueError("Cannot parse dashboard " + str(directory))
        values = list(map(float, match[1].split()))
        if len(values) != 7:
            raise ValueError("Wrong metrics")
        row = dict(
            lanes=profile["lanes"],
            dim=profile["dim"],
            **dict(
                zip(
                    (
                        "score",
                        "line",
                        "condition",
                        "toggle",
                        "fsm_transition",
                        "branch",
                        "functional_group",
                    ),
                    values,
                )
            )
        )
        row["report_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("dashboard.txt", "modinfo.txt", "groups.txt")
        }
        rows.append(row)
        info = (directory / "modinfo.txt").read_text()
        review.append("PROFILE l%d_d%d" % (profile["lanes"], profile["dim"]))
        for name in (
            "tile_controller_banked_prefetch",
            "flash_attn_top_dma_banked_prefetch",
            "dma_engine_vec",
            "array_controller",
            "banked_tile_loader",
        ):
            m = re.search(
                r"FSM Coverage for Module : "
                + name
                + r"(.*?)(?:Branch Coverage|Module : )",
                info,
                re.S,
            )
            if m:
                review.append("Module: " + name + "\n" + m[1].strip())
        # All 16 softmax instances have separate reports. Preserve one typical
        # instance and the exact state/transition totals for each, without waiver.
        for m in re.finditer(
            r"FSM Coverage for Instance : (\S*gen_softmax\S+)(.*?)(?:Branch Coverage|FSM Coverage for Instance|Module : )",
            info,
            re.S,
        ):
            summary = m[2].split("State, Transition")[0].strip()
            review.append("Instance: " + m[1] + "\n" + summary)
            if "gen_softmax[0]" in m[1]:
                review.append(m[2].split("State, Transition")[1].strip())
    summary = dict(
        status="PASS",
        vcs_version="V-2023.12-SP2",
        urg_version="V-2023.12-SP2",
        scope="tb_attention.dut.u_dut",
        metrics="line+cond+tgl+fsm+branch+assert",
        invocations=record["invocations"],
        exact_transactions=record["exact_transactions"],
        rows=rows,
    )
    (args.output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "fsm_review.txt").write_text(
        "\n".join(line.rstrip() for line in "\n\n".join(review).splitlines()) + "\n"
    )
    fields = [
        "lanes",
        "dim",
        "score",
        "line",
        "condition",
        "toggle",
        "fsm_transition",
        "branch",
        "functional_group",
    ]
    with (args.output / "coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Exported %d profiles" % len(rows))


if __name__ == "__main__":
    main()
