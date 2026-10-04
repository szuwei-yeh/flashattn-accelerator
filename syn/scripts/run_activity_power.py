"""Capture exact-checked RTL activity and analyze preserved N64/d16 netlists.

Python 3.6+, VCS coverage binaries, PrimeTime/PrimePower and target library needed.
No licensed library, waveform, netlist or simulator database is copied to git.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/vcs"))
from run_coverage import canonical, validate_fixture, run, METRICS


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def activity_counts(text):
    # First table is toggle rate; second table reports static probability.
    result = {}
    for label in ("Nets", "Primary Input", "Sequential", "Combinational"):
        m = re.search(r"^\s*" + label + r"\s+(.*)$", text, re.M)
        if not m:
            raise ValueError("Missing activity category " + label)
        row = m[1]
        counts = [int(x) for x in re.findall(r"(\d+)\([\d.]+%\)", row)]
        total = int(row.split()[-1])
        if len(counts) != 10 or sum(counts) != total:
            raise ValueError("Malformed activity row " + label)
        result[label] = dict(
            zip(
                (
                    "file",
                    "ssa",
                    "ssa_force_annotated",
                    "ssa_force_implied",
                    "sca",
                    "clock",
                    "default",
                    "propagated",
                    "implied",
                    "not_annotated",
                ),
                counts,
            )
        )
        result[label]["total"] = total
    return result


def power_values(text, units):
    if not all(
        re.search(label + r" Power Units\s*=\s*1(?:\.0*)?\s*W\b", text)
        for label in ("Dynamic", "Leakage")
    ):
        raise ValueError("Unexpected power units; require 1 W")
    values = {}
    for key, label in [
        ("switching_w", "Net Switching Power"),
        ("internal_w", "Cell Internal Power"),
        ("leakage_w", "Cell Leakage Power"),
        ("total_w", "Total Power"),
    ]:
        m = re.search(re.escape(label) + r"\s*=\s*([\d.eE+-]+)", text)
        if not m:
            raise ValueError("Missing power " + label)
        values[key] = float(m[1])
    values["dynamic_w"] = values["switching_w"] + values["internal_w"]
    if abs(values["dynamic_w"] + values["leakage_w"] - values["total_w"]) > max(
        0.0001, values["total_w"] * 0.003
    ):
        raise ValueError("Power sum inconsistency")
    return values


def normalize_scope_header(raw, target):
    # DC change_names -rules verilog flattens generate scopes. Preserve all
    # timestamps/values; rename only scope tokens in the VCD header.
    scopes = []
    with raw.open() as src, target.open("w") as dst:
        header = True
        for line in src:
            if header and line.startswith("$scope "):
                fields = line.split()
                old = fields[2]
                new = old.replace(".", "_").replace("[", "_").replace("]", "_")
                fields[2] = new
                if old != new:
                    scopes.append(dict(rtl=old, mapped=new))
                line = " ".join(fields) + "\n"
            if header and line.startswith("$var "):
                fields = line.split()
                # DC packs one-dimensional unpacked vectors into a single bus.
                m = re.fullmatch(r"(.+)\[(\d+)\]", fields[4])
                if m and len(fields) == 7 and re.fullmatch(r"\[\d+:\d+\]", fields[5]):
                    hi, lo = map(int, fields[5][1:-1].split(":"))
                    width = int(fields[2])
                    offset = int(m[2]) * width
                    fields[4] = m[1]
                    fields[5] = "[%d:%d]" % (offset + hi, offset + lo)
                fields[4] = (
                    fields[4].replace(".", "_").replace("[", "_").replace("]", "_")
                )
                line = " ".join(fields) + "\n"
            dst.write(line)
            if "$enddefinitions" in line:
                header = False
    return scopes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--vcs-run", type=Path, required=True)
    p.add_argument(
        "--mapped-root",
        type=Path,
        required=True,
        help="Parent of preserved DC run tags",
    )
    p.add_argument("--mapped-tag", default="dequant_20261003_l{lanes}_n64d16")
    p.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "docs/evidence/2026-10-03/provenance.json",
    )
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument(
        "--lanes", type=int, nargs="+", choices=(256, 32, 16), default=[256, 32, 16]
    )
    args = p.parse_args()
    base = args.run_dir.resolve()
    if base.exists():
        p.error("Use a new run directory")
    base.mkdir(parents=True)
    provenance = json.loads(args.provenance.read_text())
    verification = json.loads((args.vcs_run / "verification.json").read_text())
    if verification.get("status") != "PASS":
        raise ValueError("Unaccepted VCS run")
    for name, sha in verification["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("VCS source hash mismatch " + name)
    library = Path(os.environ["DC_TARGET_LIBRARY"]).resolve()
    library_hash = digest(library)
    extras = json.loads((args.fixtures / "manifest.json").read_text())
    workloads = [canonical(16, mask) for mask in (0, 1)]
    for case in extras:
        if (
            case["dim"] == 16
            and case["causal"] == 0
            and case["name"] in ("random_seed7_amp1", "saturated_distinct")
        ):
            case["directory"] = (args.fixtures / case["directory"]).resolve()
            workloads.append(case)
    if len(workloads) != 4:
        raise ValueError("Require two canonical and two generated workloads")
    summary = dict(
        status="RUNNING",
        measured_commit=provenance["measured_commit"],
        mode="averaged RTL VCD on mapped standard cells",
        clock_period_ns=10,
        source_sha256=verification["source_sha256"],
        rows=[],
        script_sha256={
            str(x.relative_to(ROOT)): digest(x)
            for x in (Path(__file__), ROOT / "syn/scripts/pt_activity_power.tcl")
        },
    )
    for lanes in args.lanes:
        source = provenance["runs"][str(lanes)]
        if library_hash != source["library_sha256"]:
            raise ValueError("Library hash mismatch")
        for name, sha in source["source_sha256"].items():
            if name.startswith("rtl/") and name in verification["source_sha256"]:
                if (
                    digest(ROOT / name) != sha
                    or verification["source_sha256"][name] != sha
                ):
                    raise ValueError("RTL mismatch " + name)
        mapped = args.mapped_root / args.mapped_tag.format(lanes=lanes) / "top"
        for name in ("artifacts/mapped.v", "artifacts/constraints.sdc"):
            if digest(mapped / name) != source["artifact_sha256"][name]:
                raise ValueError("Artifact hash mismatch " + name)
        profile = base / ("build_l%d" % lanes)
        profile.mkdir()
        filelist = ROOT / "syn/filelists/top_dma_banked_prefetch.f"
        sources = [
            ROOT / x.strip()
            for x in filelist.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        ]
        sources += [
            ROOT / "rtl/interface/axi_mem_model.sv",
            ROOT / "sim/verilator/tb_dma_banked_prefetch_harness.sv",
            ROOT / "sim/vcs/tb_attention.sv",
        ]
        binary = profile / "simv"
        run(
            shlex.split(os.environ.get("VCS", "vcs"))
            + [
                "-full64",
                "-sverilog",
                "-debug_acc+all",
                "-top",
                "tb_attention",
                "-pvalue+tb_attention.D=16",
                "-pvalue+tb_attention.LANES=%d" % lanes,
                "-Mdir=" + str(profile / "csrc"),
                "-o",
                str(binary),
            ]
            + [str(x) for x in sources],
            profile / "build.log",
        )
        for case in workloads:
            name = "l%d_%s_c%d" % (lanes, case["name"], case["causal"])
            folder = base / name
            folder.mkdir()
            hashes = validate_fixture(Path(case["directory"]), 16)
            raw = folder / "rtl_activity.vcd"
            waveform = folder / "activity.vcd"
            sq, sk, sv = case["scales"]
            command = [
                str(profile / "simv"),
                "+DATA=" + str(case["directory"]),
                "+SQ=%d" % sq,
                "+SK=%d" % sk,
                "+SV=%d" % sv,
                "+CAUSAL=%d" % case["causal"],
                "+LATENCY=0",
                "+RESTART=0",
                "+ACTIVITY=" + str(raw),
                "+vcs+dumparrays+65536",
            ]
            output = run(
                command,
                folder / "simulation.log",
                require_pass=True,
                cwd=ROOT / "sim/verilator",
            )
            exact = re.findall(
                r"EXACT_PASS d=16 lanes=(\d+) causal=(\d+) latency=0 zero_v=0 tb_cycles=(\d+) words=1024",
                output,
            )
            if (
                len(exact) != 1
                or int(exact[0][0]) != lanes
                or int(exact[0][1]) != case["causal"]
            ):
                raise ValueError("Wrong exact completion")
            window = re.search(
                r"ACTIVITY_WINDOW begin_ns=(\d+) end_ns=(\d+) cycles_including_start=(\d+)",
                output,
            )
            if not window:
                raise ValueError("Missing active window")
            begin, end, cycles = map(int, window.groups())
            if cycles != int(exact[0][2]) + 1 or end - begin != 10 * cycles:
                raise ValueError("Window/cycle mismatch")
            scope_mapping = normalize_scope_header(raw, waveform)
            env = os.environ.copy()
            env.update(
                PT_RUN_DIR=str(folder / "pt"),
                PT_MAPPED_V=str(mapped / "artifacts/mapped.v"),
                PT_SDC=str(mapped / "artifacts/constraints.sdc"),
                PT_TOP=source["manifest"]["effective_top"],
                PT_ACTIVITY=str(waveform),
                PT_ACTIVITY_BEGIN_NS=str(begin),
                PT_ACTIVITY_END_NS=str(end),
            )
            with (folder / "pt.log").open("w") as log:
                result = subprocess.run(
                    shlex.split(env.get("PT_SHELL_BIN", "pt_shell"))
                    + ["-f", str(ROOT / "syn/scripts/pt_activity_power.tcl")],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(folder),
                    timeout=1200,
                )
            log = (folder / "pt.log").read_text(errors="replace")
            if (
                result.returncode
                or re.search(r"^Error:", log, re.M)
                or "RESULT: PASS PrimeTime activity power" not in log.splitlines()
            ):
                raise RuntimeError("PrimeTime analysis rejected: " + name)
            annotation = activity_counts((folder / "pt/activity_after.rpt").read_text())
            # Reject vectorless fallback. 'implied' is derived without random-vector
            # propagation from annotated points (e.g. aliases/constants). Keep
            # direct, implied, propagated and default counts separate in evidence.
            for category in ("Primary Input", "Sequential"):
                item = annotation[category]
                if item["file"] + item["implied"] < 0.95 * item["total"]:
                    raise ValueError(
                        "Insufficient file/implied activity for "
                        + category
                        + " in "
                        + name
                    )
            values = power_values(
                (folder / "pt/power.rpt").read_text(),
                (folder / "pt/units.rpt").read_text(),
            )
            values["energy_nj"] = values["total_w"] * (end - begin)
            row = dict(
                lanes=lanes,
                case=case["name"],
                causal=case["causal"],
                tb_cycles=int(exact[0][2]),
                cycles_including_start=cycles,
                begin_ns=begin,
                end_ns=end,
                duration_ns=end - begin,
                fixture_sha256=hashes,
                mapped_sha256=source["artifact_sha256"],
                library_sha256=library_hash,
                activity_sha256=digest(waveform),
                raw_activity_sha256=digest(raw),
                scope_mapping=scope_mapping,
                simulation_sha256=digest(folder / "simulation.log"),
                annotation=annotation,
                **values
            )
            row["report_sha256"] = {
                x.name: digest(x) for x in (folder / "pt").iterdir() if x.is_file()
            }
            summary["rows"].append(row)
            (base / "power.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                "POWER_PASS %s total_mW=%.4f energy_nJ=%.3f"
                % (name, values["total_w"] * 1000, values["energy_nj"]),
                flush=True,
            )
    summary["status"] = "PASS"
    summary["cases"] = len(summary["rows"])
    (base / "power.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "lanes",
        "case",
        "causal",
        "tb_cycles",
        "cycles_including_start",
        "duration_ns",
        "switching_w",
        "internal_w",
        "dynamic_w",
        "leakage_w",
        "total_w",
        "energy_nj",
    ]
    with (base / "power.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["rows"])
    print("RESULT: PASS %d activity power cases" % len(summary["rows"]), flush=True)


if __name__ == "__main__":
    main()
