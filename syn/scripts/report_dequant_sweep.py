"""Publish portable evidence for completed, matched 256/32/16-lane DC runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text,label):
    match=re.search(r'^\s*'+re.escape(label)+r'\s*:?\s*([-+0-9.eE]+)\s*$',text,re.M)
    if not match:
        raise ValueError(f'Missing metric: {label}')
    return float(match[1])


def collect(directory,lanes):
    if not (directory/'SUCCESS').exists():
        raise ValueError(f'Run not accepted: {directory}')
    raw=dict(line.split('=',1) for line in (directory/'manifest.txt').read_text().splitlines() if '=' in line)
    params=dict(item.split('=',1) for item in raw['elaboration_parameters'].split(','))
    expected=dict(SEQ_LEN='64',HEAD_DIM='16',TILE_SIZE='16',SRAM_DEPTH='4096',
                  DEQUANT_LANES=str(lanes),AXI_ADDR_W='32',AXI_DATA_W='64')
    if (params!=expected or raw['git_dirty']!='no' or raw['run_mode']!='compile'
            or raw['profile']!='top' or raw['top_module']!='flash_attn_top_dma_banked_prefetch'):
        raise ValueError('Require clean N64/d16 top, selected lanes and compile flow')
    reports=directory/'reports'
    qor=(reports/'qor.rpt').read_text()
    area=(reports/'area.rpt').read_text()
    timing=(reports/'timing_setup.rpt').read_text()
    power=(reports/'power.rpt').read_text()
    version=re.search(r'^Version:\s*(.+)$',qor,re.M)
    if not version: raise ValueError('Missing DC version')
    netlist=(directory/'artifacts/mapped.v').read_text()
    # Count actual mapped leaf instances, excluding the tile adapter itself.
    leaves=re.findall(r'^\s*dequantizer(?:_OUT_WIDTH16_FRAC_BITS8)?(?:_\d+)?\s+\S+\s*\(',netlist,re.M)
    if len(leaves)!=lanes:
        raise ValueError(f'Mapped lane count {len(leaves)} != requested {lanes}')
    hierarchy=(reports/'hierarchy.rpt').read_text()
    if not re.search(r'flash_attn_core_banked_prefetch_\S*HEAD_DIM16_\S*SEQ_LEN64_\S*DEQUANT_LANES'+str(lanes),hierarchy):
        raise ValueError('Missing mapped core parameter proof')
    mapped_core=re.search(r'^\s*module\s+(flash_attn_core_banked_prefetch_\S*HEAD_DIM16_\S*SEQ_LEN64_\S*DEQUANT_LANES'+str(lanes)+r')\s*\(',netlist,re.M)
    if not mapped_core:
        raise ValueError('Missing mapped core module declaration')
    source=json.loads((directory/'source_sha256.json').read_text())
    metrics=dict(cell_area=number(area,'Total cell area:'),
                 critical_path_ns=number(qor,'Critical Path Length:'),
                 worst_setup_slack_ns=number(qor,'Critical Path Slack:'),
                 setup_tns_ns=number(qor,'Total Negative Slack:'),
                 setup_violating_paths=int(number(qor,'No. of Violating Paths:')),
                 max_cap_violations=int(number(qor,'Max Cap Violations:')),
                 macro_blackbox_area=number(area,'Macro/Black Box area:'))
    if number(qor,'Critical Path Clk Period:') != float(raw['clock_period_ns']):
        raise ValueError('Actual clock period disagrees with manifest')
    structure=(reports/'postcheck_structure.rpt').read_text()
    postlog=(directory/'logs/postcheck.log').read_text()
    if 'MAPPED_POSTCHECK_DONE' not in structure or re.search(r'(^|\s)Error:',postlog):
        raise ValueError('Mapped postcheck incomplete or reported tool errors')
    metrics['mapped_latch_count']=int(number(structure,'MAPPED_LATCH_COUNT'))
    metrics['unused_low_product_pins']=int(number(structure,'UNUSED_LOW_PRODUCT_PINS'))
    metrics['unused_low_product_live_endpoints']=int(number(structure,'UNUSED_LOW_PRODUCT_LIVE_ENDPOINTS'))
    if metrics['mapped_latch_count']!=0 or metrics['unused_low_product_live_endpoints']!=0:
        raise ValueError('Mapped latch/unused-product disposition not established')
    lint=(reports/'postcheck_design.rpt').read_text()
    metrics['mapped_lint_counts']={code:int(count) for code,count in
        re.findall(r'^\s*[^\n]+?\((LINT-\d+)\)\s+(\d+)\s*$',lint,re.M)}
    if any(metrics['mapped_lint_counts'].get(code)!=metrics['unused_low_product_pins']
           for code in ('LINT-3','LINT-5')):
        raise ValueError('Undriven lint count differs from audited product pins')
    check_timing=(reports/'check_timing.rpt').read_text()
    metrics['mapped_check_timing_warnings']=len(re.findall(r'^Warning:',check_timing,re.M))
    minimum=re.search(r'^\s*slack\s+\((?:MET|VIOLATED)\)\s+([-+0-9.eE]+)\s*$',
                      (reports/'timing_min.rpt').read_text(),re.M)
    if not minimum: raise ValueError('Missing min-delay report slack')
    metrics['reported_min_delay_slack_ns']=float(minimum[1])
    cap=(reports/'postcheck_constraints.rpt').read_text()
    block=re.search(r'(?ms)^\s*max_capacitance\s*\n(.*?)(?=^\s*max_\w+\s*\n|\Z)',cap)
    value=r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    caps=[] if not block else re.findall(value+r'\s+'+value+r'\s+'+value+r'\s+\(VIOLATED\)',block[1])
    if len(caps)!=metrics['max_cap_violations']:
        raise ValueError('High-precision cap violation count disagrees with QoR')
    metrics['max_cap_required_limit_min']=min((float(row[0]) for row in caps),default=None)
    metrics['max_cap_required_limit_max']=max((float(row[0]) for row in caps),default=None)
    for label,key in (('Startpoint','startpoint'),('Endpoint','endpoint')):
        match=re.search(r'^\s*'+label+r':\s*(\S+)',timing,re.M)
        if not match: raise ValueError(f'Missing {label}')
        metrics[key]=match[1]
    slack=re.search(r'^\s*slack\s+\((?:MET|VIOLATED)\)\s+([-+0-9.eE]+)\s*$',timing,re.M)
    if not slack:
        raise ValueError('Missing setup timing-report slack')
    metrics['timing_report_slack_ns']=float(slack[1])
    if abs(metrics['timing_report_slack_ns']-metrics['worst_setup_slack_ns'])>0.011:
        raise ValueError('Setup timing report and clock-group QoR disagree')
    for label,key in (('Total Dynamic Power','vectorless_dynamic_power_w'),('Cell Leakage Power','vectorless_leakage_power_w')):
        match=re.search(re.escape(label)+r'\s*=\s*([0-9.eE+-]+)\s+(mW|uW|nW|W)',power)
        if not match:raise ValueError(f'Missing {label}')
        metrics[key]=float(match[1])*dict(W=1,mW=1e-3,uW=1e-6,nW=1e-9)[match[2]]
    manifest=dict(raw)
    manifest['target_library']=Path(raw['target_library']).name
    result=dict(manifest=manifest,lanes=lanes,dc_version=version[1].strip(),mapped_dequantizer_instances=len(leaves),
                library_sha256=digest(Path(raw['target_library'])),source_sha256=source,metrics=metrics,
                mapped_core_module=mapped_core[1],
                report_sha256={str(p.relative_to(directory)):digest(p) for p in sorted(reports.glob('*.rpt'))},
                artifact_sha256={str(p.relative_to(directory)):digest(p) for p in sorted((directory/'artifacts').iterdir()) if p.is_file()})
    clock=qor[qor.index("Timing Path Group 'clk'"):qor.index('Cell Count')].strip()
    area_lines='\n'.join(line for line in area.splitlines() if re.match(r'^(Combinational area|Buf/Inv area|Noncombinational area|Macro/Black Box area|Net Interconnect area|Total cell area|Total area)',line))
    excerpt=f"Run: {raw['run_tag']}/top\nMeasured commit: {raw['git_commit']} (clean)\nElaboration: {raw['elaboration_parameters']}\nMapped dequantizer instances: {len(leaves)}\nUnits: library cell-area units; timing in ns. Logical SRAM/ROM blackboxes.\n\n[qor.rpt — clock group]\n{clock}\n\n[area.rpt — totals]\n{area_lines}\n\n[timing_setup.rpt — worst path]\nStartpoint: {metrics['startpoint']}\nEndpoint: {metrics['endpoint']}\n\n[Derived metrics]\n{json.dumps(metrics,indent=2)}\n\nPower is DC vectorless estimation with unannotated activity; not workload or system power.\nMax-cap violations are retained; no physical/electrical signoff is claimed.\n"
    return result,excerpt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path,required=True)
    parser.add_argument('--run-prefix',default='dequant_20261003')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    runs,excerpts={},{}
    for lanes in (256,32,16):
        directory=args.runs_root/f'{args.run_prefix}_l{lanes}_n64d16'/'top'
        runs[str(lanes)],excerpts[str(lanes)]=collect(directory,lanes)
    baseline=runs['256']
    for run in runs.values():
        if run['source_sha256']!=baseline['source_sha256'] or run['library_sha256']!=baseline['library_sha256']:
            raise ValueError('Runs must use identical sources and library')
        if run['dc_version']!=baseline['dc_version']:
            raise ValueError('Matched DC version mismatch')
        for key in ('git_commit','clock_period_ns','io_delay_ns','run_mode'):
            if run['manifest'][key]!=baseline['manifest'][key]:
                raise ValueError(f'Matched-run mismatch: {key}')
        run['area_reduction_percent']=100*(1-run['metrics']['cell_area']/baseline['metrics']['cell_area'])
    result=dict(schema=1,measured_commit=baseline['manifest']['git_commit'],runs=runs,
                scope='Matched N64/d16 integrated top; DC compile; logical SRAM/ROM blackboxes. No d64 PPA, physical signoff, gate-level simulation or mapped equivalence claim.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'provenance.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for lanes,excerpt in excerpts.items():
        (args.output_dir/f'top_l{lanes}.txt').write_text(excerpt)
    print(json.dumps({lanes:run['metrics'] for lanes,run in runs.items()},indent=2))


if __name__=='__main__':main()
