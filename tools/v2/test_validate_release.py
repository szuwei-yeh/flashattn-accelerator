#!/usr/bin/env python3
"""Independent corruption controls for the versioned release validator."""
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('release_checks', ROOT / 'tools/v2/validate_release.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ReleaseControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v2-release-control-')
        self.root = pathlib.Path(self.tmp.name)
        for p in checks.payload(ROOT):
            dest = self.root / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        # Register the disposable snapshot independently. The immutable v1
        # evidence lock must still reject corruption even after rehashing v2.
        self.inventory = 'release/manifest-v2.json'
        manifest = json.loads((self.root / checks.MANIFEST).read_text())
        manifest['self_excluded'] = [self.inventory]
        manifest['files'] = [dict(path=str(p.relative_to(self.root)),
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  bytes=p.stat().st_size,
                                  mode='100755' if p.stat().st_mode & 0o111 else '100644')
                             for p in checks.payload(self.root)
                             if str(p.relative_to(self.root)) != self.inventory]
        (self.root / self.inventory).write_text(json.dumps(manifest, indent=2) + '\n')

    def tearDown(self):
        self.tmp.cleanup()

    def rejected(self, key):
        result = checks.validate(self.root, inventory=self.inventory)
        self.assertEqual(result['status'], 'FAIL', result)
        self.assertTrue(any(key in x for x in result['errors']), result['errors'])
        return result

    def test_positive_and_historical_navigation(self):
        result = checks.validate(self.root, inventory=self.inventory)
        self.assertEqual(result['status'], 'PASS', result['errors'])
        self.assertEqual(result['archived_references_resolved_by_explicit_ledger'], 24)
        self.assertEqual(result['accepted_evidence_entries'], 77)
        self.assertEqual(result['figures_checked'], 8)

    def test_default_inventory_checks_current_layout(self):
        result = checks.validate(self.root)
        self.assertEqual(result['status'], 'PASS', result['errors'])
        self.assertEqual(checks.DEFAULT_INVENTORY, self.inventory)

    def test_broken_link_in_accepted_text(self):
        p = self.root / 'docs/v2/evidence/verification/formal_gap_review.txt'
        p.write_bytes(p.read_bytes() + b'\n[Injected missing evidence](absent-evidence.json)\n')
        result = self.rejected('broken_link')
        self.assertTrue(any(x.get('broken_link', {}).get('target') == 'absent-evidence.json'
                            for x in result['errors']))

    def test_missing_relocation_receipt(self):
        (self.root / 'docs/v2/evidence/results/host_formal.json').unlink()
        self.rejected('invalid_historical_relocation')

    def test_production_source_drift(self):
        p = self.root / 'rtl/v2/soc/flash_attn_soc.sv'
        p.write_bytes(p.read_bytes() + b'\n// Integrity negative control.\n')
        self.rejected('protected_source_or_evidence_hash')

    def test_rehashed_inventory_cannot_accept_source_drift(self):
        p = self.root / 'rtl/v2/soc/flash_attn_soc.sv'
        p.write_bytes(p.read_bytes() + b'\n// Integrity negative control.\n')
        manifest = self.root / self.inventory
        value = json.loads(manifest.read_text())
        row = next(x for x in value['files'] if x['path'] == str(p.relative_to(self.root)))
        row.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
        manifest.write_text(json.dumps(value))
        self.rejected('protected_source_or_evidence_hash')

    def test_fixture_drift(self):
        p = self.root / 'data/exp_lut.hex'
        p.write_bytes(b'00\n' + p.read_bytes()[3:])
        self.rejected('protected_source_or_evidence_hash')

    def test_accepted_figure_drift(self):
        p = self.root / 'docs/v2/figures/soc_architecture.svg'
        p.write_bytes(p.read_bytes() + b'\n<!-- Integrity negative control -->\n')
        self.rejected('figure_hash')

    def test_undeclared_payload(self):
        (self.root / 'unreviewed-file.txt').write_text('Unreviewed addition\n')
        self.rejected('payload_file_set')

    def test_missing_source_map(self):
        (self.root / 'docs/v2/evidence/verification/source-map.json').unlink()
        self.rejected('validation_structure_error')

    def test_changed_historical_manifest(self):
        p = self.root / 'release/manifest-v1.json'
        p.write_bytes(p.read_bytes() + b'\n')
        self.rejected('historical_manifest_hash')

    def test_inconsistent_baseline_identity(self):
        p = self.root / self.inventory
        value = json.loads(p.read_text())
        value['base_commit'] = '0' * 40
        p.write_text(json.dumps(value))
        self.rejected('evidence_lock_schema')

    def test_invalid_manifest_structure(self):
        (self.root / self.inventory).write_text('[]\n')
        self.rejected('validation_structure_error')

    def test_unsafe_path_mapping(self):
        p = self.root / 'release/path-map-v1.json'
        value = json.loads(p.read_text())
        value['files'][0]['path'] = '../escaped'
        p.write_text(json.dumps(value))
        self.rejected('invalid_or_duplicate_origin')

    def test_functional_source_cannot_be_archived(self):
        p = self.root / 'release/layout-v2.json'
        value = json.loads(p.read_text())
        value['archived_preparation'].append(dict(
            path='rtl/v2/m6b3/flash_attn_soc_m6b3.sv', sha256='0' * 64))
        p.write_text(json.dumps(value))
        self.rejected('archive_policy')

    def test_arithmetic_patch_cannot_be_disguised_as_path_relocation(self):
        p = self.root / 'release/layout-v2.json'
        value = json.loads(p.read_text())
        row = value['path_only_derivatives'][0]
        row['replacements'] = [{'from':'old_value + data_in', 'to':'data_in', 'count':1}]
        p.write_text(json.dumps(value))
        self.rejected('path_patch_policy')

    def test_cli_exits_nonzero_on_failure(self):
        (self.root / 'rtl/v2/soc/soc.f').unlink()
        # Put the report outside the checked snapshot, avoiding an undeclared-file side effect.
        with tempfile.TemporaryDirectory(prefix='v2-cli-report-') as out:
            output = pathlib.Path(out) / 'result.json'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/v2/validate_release.py'),
                                     '--snapshot', str(self.root), '--inventory', self.inventory, '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
