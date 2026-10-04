"""Run isolated VCS four-state main-path checks and produce per-profile URG reports.

Requires licensed VCS/URG. Python 3.6+; generated fixtures can be prepared on a
separate machine with prepare_cases.py. All compiler/database files stay in runs/.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
METRICS = "line+cond+tgl+fsm+branch+assert"
PASS = "RESULT: PASS VCS attention four-state"
FAIL = re.compile(
    r"^\s*(?:Fatal|Error)(?:[:\s-])|\b(?:FAIL|FAILED|MISMATCH|TIMEOUT)\b", re.I | re.M
)


def run(argv, log, timeout=600, require_pass=False, cwd=ROOT):
    with log.open("w") as stream:
        result = subprocess.run(
            argv, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
        )
    output = log.read_text(errors="replace")
    if (
        result.returncode
        or FAIL.search(output)
        or (require_pass and PASS not in output.splitlines())
    ):
        raise RuntimeError(
            "Command rejected (%s): %s\n%s"
            % (
                result.returncode,
                shlex.join(argv) if hasattr(shlex, "join") else str(argv),
                output[-5000:],
            )
        )
    return output


def validate_fixture(folder, dim):
    hashes = {}
    for name, width in [
        ("q_input.hex", 2),
        ("k_input.hex", 2),
        ("v_input.hex", 2),
        ("expected.hex", 8),
    ]:
        p = folder / name
        tokens = p.read_text().split()
        if len(tokens) != 64 * dim or any(
            not re.fullmatch("[0-9a-fA-F]{1,%d}" % width, t) for t in tokens
        ):
            raise ValueError("Malformed fixture: " + str(p))
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def canonical(dim, causal):
    folder = (
        ROOT
        / "data"
        / (("causal_N64" if causal else "N64") + ("_d64" if dim == 64 else ""))
    )
    scales = []
    text = (folder / "scales.txt").read_text()
    for label in ("q", "k", "v"):
        m = re.search(
            r"^scale_%s_q88\s*=\s*(0x[0-9a-fA-F]+|[-+]?\d+)\s*$" % label, text, re.M
        )
        if not m:
            raise ValueError("Missing scale " + label)
        value = int(m[1], 0)
        scales.append(value - 65536 if value >= 32768 else value)
    return dict(
        name="canonical", dim=dim, causal=causal, scales=scales, directory=folder
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--lanes", nargs="+", type=int, choices=(256, 32, 16), default=[256, 32, 16]
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, choices=(16, 64), default=[16, 64]
    )
    parser.add_argument(
        "--fixtures", type=Path, help="Directory prepared by prepare_cases.py"
    )
    args = parser.parse_args()
    base = args.run_dir.resolve()
    if base.exists():
        parser.error("Use a new run directory")
    base.mkdir(parents=True)
    vcs = shlex.split(os.environ.get("VCS", "vcs"))
    urg = shlex.split(os.environ.get("URG", "urg"))
    filelist = ROOT / "syn/filelists/top_dma_banked_prefetch.f"
    sources = [
        ROOT / line.strip()
        for line in filelist.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    sources += [
        ROOT / "rtl/interface/axi_mem_model.sv",
        ROOT / "sim/verilator/tb_dma_banked_prefetch_harness.sv",
        ROOT / "sim/vcs/tb_attention.sv",
    ]
    record = dict(
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        profiles=[],
        transactions=[],
    )
    extra = []
    if args.fixtures:
        extra = json.loads((args.fixtures / "manifest.json").read_text())
        for case in extra:
            case["directory"] = (args.fixtures / case["directory"]).resolve()
    for lanes in args.lanes:
        for dim in args.dims:
            profile = base / ("l%d_d%d" % (lanes, dim))
            profile.mkdir()
            hier = profile / "dut.hier"
            hier.write_text("+tree tb_attention.dut.u_dut\n")
            db = profile / "simv.vdb"
            binary = profile / "simv"
            command = (
                vcs
                + [
                    "-full64",
                    "-sverilog",
                    "-top",
                    "tb_attention",
                    "-pvalue+tb_attention.D=%d" % dim,
                    "-pvalue+tb_attention.LANES=%d" % lanes,
                    "-cm",
                    METRICS,
                    "-cm_hier",
                    str(hier),
                    "-cm_dir",
                    str(db),
                    "-Mdir=" + str(profile / "csrc"),
                    "-o",
                    str(binary),
                ]
                + [str(p) for p in sources]
            )
            run(command, profile / "build.log")
            cases = []
            for mask in (0, 1):
                for latency in (0, 20, 100):
                    cases.append((canonical(dim, mask), latency, False))
            cases += [(case, 20, False) for case in extra if case["dim"] == dim]
            cases.append((canonical(dim, 0), 20, True))
            for index, (case, latency, contract) in enumerate(cases):
                name = "%02d_%s_c%d_%s" % (
                    index,
                    case["name"],
                    case["causal"],
                    "contract" if contract else "lat%d" % latency,
                )
                folder = Path(case["directory"])
                hashes = validate_fixture(folder, dim)
                if any(not -32768 <= s <= 32767 for s in case["scales"]):
                    raise ValueError("Invalid scales")
                sq, sk, sv = case["scales"]
                command = [
                    str(binary),
                    "+DATA=" + str(folder),
                    "+SQ=%d" % sq,
                    "+SK=%d" % sk,
                    "+SV=%d" % sv,
                    "+CAUSAL=%d" % case["causal"],
                    "+LATENCY=%d" % latency,
                    "+CONTRACT=%d" % contract,
                    "-cm",
                    METRICS,
                    "-cm_dir",
                    str(db),
                    "-cm_name",
                    name,
                ]
                output = run(
                    command,
                    profile / (name + ".log"),
                    require_pass=True,
                    cwd=ROOT / "sim/verilator",
                )
                matches = re.findall(
                    r"EXACT_PASS d=(\d+) lanes=(\d+) causal=(\d+) latency=(\d+) zero_v=(\d+) tb_cycles=(\d+) words=(\d+)",
                    output,
                )
                expected_count = 3 if contract else 2
                if len(matches) != expected_count or any(
                    tuple(map(int, m[:4])) != (dim, lanes, case["causal"], latency)
                    or int(m[6]) != 64 * dim
                    for m in matches
                ):
                    raise RuntimeError("Missing or mismatched exact completions")
                if contract and "CONTRACT_PASS" not in output.splitlines():
                    raise RuntimeError("Missing contract PASS")
                for m in matches:
                    record["transactions"].append(
                        dict(
                            dim=int(m[0]),
                            lanes=int(m[1]),
                            causal=int(m[2]),
                            latency=int(m[3]),
                            zero_v=int(m[4]),
                            tb_cycles=int(m[5]),
                            words=int(m[6]),
                            case=case["name"],
                            contract=contract,
                            fixture_sha256=hashes,
                        )
                    )
                print(
                    "PASS l%d d%d %s exact=%d" % (lanes, dim, name, len(matches)),
                    flush=True,
                )
            run(
                urg
                + [
                    "-dir",
                    str(db),
                    "-report",
                    str(profile / "coverage"),
                    "-format",
                    "both",
                ],
                profile / "urg.log",
            )
            record["profiles"].append(
                dict(
                    lanes=lanes,
                    dim=dim,
                    invocations=len(cases),
                    coverage_directory=str((profile / "coverage").relative_to(base)),
                )
            )
            (base / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    record["status"] = "PASS"
    record["exact_transactions"] = len(record["transactions"])
    record["invocations"] = sum(p["invocations"] for p in record["profiles"])
    (base / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        "RESULT: PASS %d invocations / %d exact transactions"
        % (record["invocations"], record["exact_transactions"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
