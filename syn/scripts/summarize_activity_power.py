"""Export matched workload estimates and their explicit annotation limits.

Does not run licensed tools; requires accepted input summaries. The source and
flow hashes must match this checkout, and the full 3-lane / 4-workload grid must
be present. Run once with a 12-case power.json or combine disjoint lane summaries.
"""

import csv
import hashlib
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description="Validate and collect a complete matched 12-case activity-power comparison."
)
parser.add_argument(
    "inputs",
    nargs="+",
    type=Path,
    help="Accepted power.json files from one or more disjoint lane runs",
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--flow-commit",
    required=True,
    help="Revision containing the measured bench and power scripts",
)
parser.add_argument(
    "--pt-version", required=True, help="Version verified in raw PrimeTime reports"
)
parser.add_argument(
    "--vcs-version", required=True, help="Version verified in the simulation logs"
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
parts = [json.loads(p.read_text()) for p in args.inputs]
if not (all(x["status"] == "PASS" for x in parts)):
    raise ValueError(
        "Invalid matched comparison: all(x['status']=='PASS' for x in parts)"
    )
rows = [row for part in parts for row in part["rows"]]
keys = {(r["lanes"], r["case"], r["causal"]) for r in rows}
expected = {
    (lane, case, mask)
    for lane in (256, 32, 16)
    for case, mask in [
        ("canonical", 0),
        ("canonical", 1),
        ("random_seed7_amp1", 0),
        ("saturated_distinct", 0),
    ]
}
if not (keys == expected and len(rows) == 12):
    raise ValueError("Invalid matched comparison: keys==expected and len(rows)==12")
for key in (
    "source_sha256",
    "script_sha256",
    "measured_commit",
    "clock_period_ns",
    "mode",
):
    if not (all(part[key] == parts[0][key] for part in parts)):
        raise ValueError(
            "Invalid matched comparison: all(part[key]==parts[0][key] for part in parts)"
        )
for name, sha in parts[0]["source_sha256"].items():
    if not (hashlib.sha256((root / name).read_bytes()).hexdigest() == sha):
        raise ValueError(
            "Invalid matched comparison: hashlib.sha256((root/name).read_bytes()).hexdigest()==sha"
        )
for name, sha in parts[0]["script_sha256"].items():
    if not (hashlib.sha256((root / name).read_bytes()).hexdigest() == sha):
        raise ValueError(
            "Invalid matched comparison: hashlib.sha256((root/name).read_bytes()).hexdigest()==sha"
        )
rows.sort(key=lambda r: ((256, 32, 16).index(r["lanes"]), r["case"], r["causal"]))
base = {(r["case"], r["causal"]): r for r in rows if r["lanes"] == 256}
mapping = {}
clock = 10
if parts[0]["clock_period_ns"] != clock:
    raise ValueError("This collector requires a 10 ns matched comparison")
for row in rows:
    ref = base[row["case"], row["causal"]]
    if row["library_sha256"] != ref["library_sha256"]:
        raise ValueError("Unmatched target library")
    if row["end_ns"] - row["begin_ns"] != row["duration_ns"]:
        raise ValueError("Inconsistent activity window")
    if not (row["fixture_sha256"] == ref["fixture_sha256"]):
        raise ValueError(
            "Invalid matched comparison: row['fixture_sha256']==ref['fixture_sha256']"
        )
    if not (
        row["duration_ns"]
        == clock * row["cycles_including_start"]
        == clock * (row["tb_cycles"] + 1)
    ):
        raise ValueError(
            "Invalid matched comparison: row['duration_ns']==clock*row['cycles_including_start']==clock*(row['tb_cycles']+1)"
        )
    if not (row["energy_nj"] == row["total_w"] * row["duration_ns"]):
        raise ValueError(
            "Invalid matched comparison: row['energy_nj']==row['total_w']*row['duration_ns']"
        )
    row["energy_reduction_percent"] = 100 * (1 - row["energy_nj"] / ref["energy_nj"])
    row["power_reduction_percent"] = 100 * (1 - row["total_w"] / ref["total_w"])
    lane = str(row["lanes"])
    scope = row.pop("scope_mapping")
    if lane in mapping:
        if not (mapping[lane] == scope):
            raise ValueError("Invalid matched comparison: mapping[lane]==scope")
    else:
        mapping[lane] = scope
    a = row["annotation"]
    if a["Primary Input"]["file"] != a["Primary Input"]["total"]:
        raise ValueError("Primary inputs are not fully annotated")
    if not (a["Sequential"]["not_annotated"] == a["Sequential"]["default"] == 0):
        raise ValueError(
            "Invalid matched comparison: a['Sequential']['not_annotated']==a['Sequential']['default']==0"
        )
    if not (
        a["Sequential"]["file"] + a["Sequential"]["implied"]
        >= 0.95 * a["Sequential"]["total"]
    ):
        raise ValueError(
            "Invalid matched comparison: a['Sequential']['file']+a['Sequential']['implied']>=.95*a['Sequential']['total']"
        )
result = {k: v for k, v in parts[0].items() if k not in ("rows", "cases", "status")}
result.update(
    status="PASS",
    cases=12,
    rows=rows,
    scope_mapping_by_lanes=mapping,
    flow_commit=args.flow_commit,
    pt_version=args.pt_version,
    vcs_version=args.vcs_version,
    conditions=dict(
        operating_corner="typical",
        wire_load_model="none",
        voltage_unit="1 V",
        modeled_read_latency_cycles=0,
        geometry="N64/d16",
        activity_window="accepted-start enclosure through sampled done; includes accepted-start cycle",
        power_scope="mapped standard cells only; SRAM/ROM logical blackboxes; no physical parasitics or external memory",
        activity_type="RTL VCD plus implied/statistical propagated activity; no mapped simulation or power signoff",
    ),
    worker_summary_sha256={
        str(i): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for i, p in enumerate(args.inputs)
    },
)
result["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(out / "power.json").write_text(json.dumps(result, indent=2) + "\n")
fields = [
    "lanes",
    "case",
    "causal",
    "tb_cycles",
    "cycles_including_start",
    "duration_ns",
    "dynamic_w",
    "leakage_w",
    "total_w",
    "energy_nj",
    "power_reduction_percent",
    "energy_reduction_percent",
]
with (out / "power.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
fields = [
    "lanes",
    "case",
    "causal",
    "primary_input_file_percent",
    "sequential_file_percent",
    "sequential_implied_percent",
    "sequential_propagated_percent",
    "nets_default_percent",
]
with (out / "activity_annotation.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for row in rows:
        s = row["annotation"]["Sequential"]
        nets = row["annotation"]["Nets"]
        pi = row["annotation"]["Primary Input"]
        w.writerow(
            dict(
                lanes=row["lanes"],
                case=row["case"],
                causal=row["causal"],
                primary_input_file_percent=100 * pi["file"] / pi["total"],
                sequential_file_percent=100 * s["file"] / s["total"],
                sequential_implied_percent=100 * s["implied"] / s["total"],
                sequential_propagated_percent=100 * s["propagated"] / s["total"],
                nets_default_percent=100 * nets["default"] / nets["total"],
            )
        )
for row in rows:
    print(
        row["lanes"],
        row["case"],
        row["causal"],
        row["total_w"] * 1000,
        row["energy_nj"] / 1000,
        row["energy_reduction_percent"],
    )
