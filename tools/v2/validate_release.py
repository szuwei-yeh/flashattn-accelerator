#!/usr/bin/env python3
"""Strict versioned portfolio inventory, evidence, provenance and link validation."""
import argparse
import csv
import hashlib
import json
import pathlib
import posixpath
import re
import subprocess
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = 'release/manifest-v1.json'
DEFAULT_INVENTORY = 'release/manifest-v2.json'
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)]+)\)')
PRIVACY = {
    'credential': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}',
    'private_home': r'/(?:Users|home)/[A-Za-z0-9_.-]+',
    'report_account': r'^\s*(?::\s*)?(?:(?://|#)\s*)?User\s*:\s*(?!<)[A-Za-z0-9_.-]+',
    'mac_temp': r'/(?:private/)?var/folders/[A-Za-z0-9_./-]+',
    'institution_host': r'\b[A-Za-z0-9.-]+\.(?:edu|ac\.[a-z]{2})\b',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(name):
    return isinstance(name, str) and name != '' and not name.startswith('/') and all(
        part not in ('', '.', '..') for part in name.split('/'))


def payload(root):
    if (root / '.git').exists():
        raw = subprocess.check_output(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
        return sorted({root / p.decode() for p in raw.split(b'\0') if p and (root / p.decode()).is_file()})
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and '.git' not in p.parts and '__pycache__' not in p.parts)


# Exact four-file/nine-span authorization, independent of the mutable inventory.
# Changing this digest is a validator policy change, not a manifest refresh.
COMMENT_CONTRACT_SHA256 = '18cb91d9787b3d336727997b4fe4881c035a1f37560e2826aa80461d15c8f4a3'
COMMENT_DIRECTIVE = re.compile(r'\b(?:verilator|synthesis|synopsys|pragma|translate_off|translate_on|lint_off|lint_on)\b', re.I)


def reverse_comment_spans(data, row):
    """Invert exact whole-line comments; preserve every other byte and line."""
    lines = data.splitlines(keepends=True)
    previous_end = 0
    for op in row['replacements']:
        old, new = op['from'].encode(), op['to'].encode()
        old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
        start = op['line'] - 1
        if (op['count'] != 1 or not old_lines or len(old_lines) != len(new_lines)
                or not old.endswith(b'\n') or not new.endswith(b'\n')
                or start < previous_end or start + len(new_lines) > len(lines)):
            raise ValueError('comment span geometry: ' + row['to'])
        for line in old_lines + new_lines:
            if (not line.lstrip(b' \t').startswith(b'//') or b'\r' in line
                    or COMMENT_DIRECTIVE.search(line.decode())):
                raise ValueError('comment span is not an ordinary comment: ' + row['to'])
        # Position and unique occurrence are both required; never global replace.
        if (b''.join(lines[start:start + len(new_lines)]) != new
                or data.count(new) != 1):
            raise ValueError('comment span position or ambiguity: ' + row['to'])
        lines[start:start + len(new_lines)] = old_lines
        previous_end = start + len(new_lines)
    return b''.join(lines)


def comment_originals(root, records, baseline_rows, moves):
    """Only the pinned contract can reconstruct sources for historical checks."""
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != COMMENT_CONTRACT_SHA256:
        raise ValueError('unapproved comment contract')
    originals = {}
    for row in records:
        old, new = row['from'], row['to']
        if (not safe_name(old) or not safe_name(new) or new in originals
                or old not in baseline_rows or new != moves.get(old, old)
                or row['original_sha256'] != baseline_rows[old]['sha256']):
            raise ValueError('comment source identity: ' + str(new))
        data = (root / new).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['current_sha256']:
            raise ValueError('comment current source hash: ' + new)
        original = reverse_comment_spans(data, row)
        if hashlib.sha256(original).hexdigest() != row['original_sha256']:
            raise ValueError('comment original source hash: ' + new)
        originals[new] = original
    return originals


def anchors(text):
    found = set(re.findall(r'<a\s+(?:name|id)=[\"\']([^\"\']+)', text))
    seen = {}
    for line in text.splitlines():
        if re.match(r'^#{1,6}\s', line):
            value = re.sub(r'^#{1,6}\s+', '', line).strip().lower()
            value = re.sub(r'<[^>]*>', '', value)
            value = re.sub(r'[^\w\- ]', '', value).replace(' ', '-')
            number = seen.get(value, 0)
            seen[value] = number + 1
            found.add(value + (f'-{number}' if number else ''))
    return found


def validate(root, inventory=DEFAULT_INVENTORY):
    root = root.resolve()
    errors = []
    def fail(kind, value):
        errors.append({kind: value})
    def read(name):
        if not safe_name(name):
            raise ValueError('unsafe manifest path: ' + str(name))
        return json.loads((root / name).read_text())
    try:
        moves, patches, retired = {}, {}, set()
        editorial = set()
        filename_config = set()
        comments, reconstructed_comments = [], {}
        archived_verified = 0
        baseline = read(MANIFEST)
        baseline_rows = {x['path']: x for x in baseline['files']}
        layout_path = root / 'release/layout-v2.json'
        if layout_path.exists():
            layout = read('release/layout-v2.json')
            if layout.get('schema_version') != 1:
                fail('layout_schema', 'release/layout-v2.json')
            if set(layout['historical_sha256']) != {MANIFEST, 'release/path-map-v1.json',
                    'release/evidence-lock-v1.json', 'release/verification-v1.json'}:
                fail('historical_manifest_set', 'release/layout-v2.json')
            for name, digest in layout['historical_sha256'].items():
                if name not in {MANIFEST, 'release/path-map-v1.json',
                                'release/evidence-lock-v1.json', 'release/verification-v1.json'}:
                    fail('historical_manifest_path', name)
                elif sha(root / name) != digest:
                    fail('historical_manifest_hash', name)
            for row in layout['moves']:
                old, new = row['from'], row['to']
                if (not safe_name(old) or not safe_name(new) or old in moves
                        or new in moves.values() or old not in baseline_rows):
                    fail('invalid_layout_move', row)
                    continue
                if row['sha256'] != baseline_rows[old]['sha256']:
                    fail('layout_move_identity', old)
                moves[old] = new
            editorial = set(layout.get('editorial_derivatives', []))
            if not editorial <= {'.gitignore'}:
                fail('editorial_policy', sorted(editorial))
            preparation = {
                'docs/v2/evidence/m6d/PUBLIC_FILE_INVENTORY.csv',
                'docs/v2/evidence/m6d/RELEASE_IDENTITY.json',
                'docs/v2/evidence/m6d/snapshot_receipt.json',
                'docs/v2/evidence/m6d/provenance.json',
                'docs/v2/evidence/m7/CHANGES.json',
                'docs/v2/evidence/m7/CHANGES_AT_M8.json',
                'docs/v2/evidence/m7/integrity.json',
            }
            for row in layout['archived_preparation']:
                name = row['path']
                if name not in preparation or name in retired or name not in baseline_rows:
                    fail('archive_policy', name)
                    continue
                if row['sha256'] != baseline_rows[name]['sha256']:
                    fail('archive_identity', name)
                retired.add(name)
                local = root / 'private/history' / name
                if local.exists():
                    if sha(local) != row['sha256']:
                        fail('local_archive_hash', name)
                    else:
                        archived_verified += 1
            # Only declared path translations may be inverted for source checking.
            # Arbitrary RTL, arithmetic or assertion replacements are rejected.
            allowed = set(moves.items())
            for old, new in moves.items():
                for quote in ("'", '"'):
                    allowed.add((quote + str(pathlib.PurePosixPath(old).parent) + '/' + quote,
                                 quote + str(pathlib.PurePosixPath(new).parent) + '/' + quote))
                    allowed.add((quote + str(pathlib.PurePosixPath(old).parent) + quote,
                                 quote + str(pathlib.PurePosixPath(new).parent) + quote))
            allowed.update({
                ('"../m5c/axi_monitor.hpp"', '"../common/axi_monitor.hpp"'),
                ('"*flash_attn_serial_m6b3.sv"', '"*flash_attn_serial.sv"'),
                ("((16,'m2c1'),(64,'m2c2'))", "((16,'decode_d16'),(64,'decode_d64'))"),
                ('rtl/v2/m6b3/{', 'rtl/v2/soc/{'),
            })
            for row in layout['path_only_derivatives']:
                old, new = row['from'], row['to']
                if (not safe_name(old) or not safe_name(new) or new in patches
                        or new != moves.get(old, old) or old not in baseline_rows
                        or row['sha256'] != baseline_rows[old]['sha256']):
                    fail('path_patch_identity', row)
                    continue
                if any((x['from'], x['to']) not in allowed or
                       not isinstance(x['count'], int) or x['count'] <= 0
                       for x in row['replacements']):
                    fail('path_patch_policy', old)
                    continue
                patches[new] = row['replacements']
                if row.get('filename_rules'):
                    if old != 'rtl/v2/m6b3/lint.vlt' or new != 'rtl/v2/soc/lint.vlt':
                        fail('filename_config_policy', old)
                    else:
                        filename_config.add(new)
            if 'comment_only_derivatives' in layout:
                comments = layout['comment_only_derivatives']
                try:
                    reconstructed_comments = comment_originals(root, comments, baseline_rows, moves)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    fail('comment_derivative_contract', str(exc))
        def relocated(name):
            return moves.get(name, name)
        def original_bytes(path):
            data = path.read_bytes()
            rel = str(path.relative_to(root))
            if rel in reconstructed_comments:
                data = reconstructed_comments[rel]
            if rel in filename_config:
                renamed = sorted(new for old, new in moves.items()
                                 if old.endswith('.sv') and
                                 pathlib.PurePosixPath(old).name != pathlib.PurePosixPath(new).name)
                suffix = '\n// Functional filenames; verified module identifiers retain their original source identity.\n'
                suffix += ''.join('lint_off -rule DECLFILENAME -file "*' + name + '"\n' for name in renamed)
                if data.endswith(suffix.encode()):
                    data = data[:-len(suffix.encode())]
                else:
                    fail('filename_config_suffix', rel)
            for op in reversed(patches.get(rel, [])):
                old, new = op['from'].encode(), op['to'].encode()
                if data.count(new) != op['count']:
                    fail('path_patch_count', rel)
                data = data.replace(new, old)
            return data
        def preserved_sha(path):
            return hashlib.sha256(original_bytes(path)).hexdigest()
        manifest = read(inventory)
        if not isinstance(manifest, dict):
            raise ValueError('manifest must be an object')
        if manifest.get('schema_version') != 1 or manifest.get('self_excluded') != [inventory]:
            fail('manifest_schema', inventory)
        rows = manifest['files']
        declared = {}
        for item in rows:
            name = item['path']
            if not safe_name(name) or name == inventory or name in declared:
                fail('invalid_or_duplicate_inventory_path', name)
                continue
            if inventory == MANIFEST and name in retired:
                continue
            declared[relocated(name) if inventory == MANIFEST else name] = item
        files = payload(root)
        actual = {str(p.relative_to(root)) for p in files}
        for name, item in declared.items():
            p = root / name
            if not p.is_file():
                fail('missing_payload', name)
                continue
            data = original_bytes(p) if inventory == MANIFEST else p.read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
                fail('payload_hash', name)
            mode = '100755' if p.stat().st_mode & 0o111 else '100644'
            if mode != item['mode']:
                fail('payload_mode', name)
        if actual != set(declared) | {inventory}:
            fail('payload_file_set', {'extra': sorted(actual - set(declared) - {inventory}),
                                      'missing': sorted(set(declared) - actual)})
        mapping = read('release/path-map-v1.json')
        if mapping.get('schema_version') != 1:
            fail('path_map_schema', 'release/path-map-v1.json')
        origins = {}
        for item in mapping['files']:
            origin = item['origin']
            dest = item.get('path')
            retired_origin = dest in retired
            if dest is not None:
                dest = relocated(dest)
            if not safe_name(origin) or origin in origins or (dest is not None and not safe_name(dest)):
                fail('invalid_or_duplicate_origin', origin)
                continue
            origins[origin] = item
            if retired_origin or dest in editorial:
                continue
            if item['disposition'] in ('kept', 'moved'):
                if not dest or not (root / dest).is_file() or preserved_sha(root / dest) != item['sha256']:
                    fail('preserved_origin_hash', origin)
            elif item['disposition'] == 'derived':
                if not dest or dest not in declared:
                    fail('missing_documentation_derivative', origin)
            elif item['disposition'] == 'archived_locally':
                if dest is not None:
                    fail('invalid_archive_mapping', origin)
            else:
                fail('unknown_origin_disposition', origin)
        def locate(origin):
            if not safe_name(origin):
                raise ValueError('unsafe evidence origin: ' + str(origin))
            row = origins.get(origin)
            return root / relocated(row['path'] if row and row.get('path') else origin)
        lock = read('release/evidence-lock-v1.json')
        if (lock.get('schema_version') != 1
                or lock.get('base_commit') != mapping.get('base_commit')
                or manifest.get('base_commit') != mapping.get('base_commit')
                or baseline.get('base_commit') != mapping.get('base_commit')):
            fail('evidence_lock_schema', 'release/evidence-lock-v1.json')
        locked = set()
        checked_protected = 0
        for item in lock['files']:
            origin = item['origin']
            if origin in locked:
                fail('duplicate_evidence_lock', origin)
            locked.add(origin)
            if origins.get(origin, {}).get('path') in retired:
                continue
            p = locate(origin)
            checked_protected += 1
            if not p.is_file() or preserved_sha(p) != item['sha256']:
                fail('protected_source_or_evidence_hash', origin)
            if not origins.get(origin) or origins[origin]['disposition'] not in ('kept', 'moved'):
                fail('protected_origin_disposition', origin)
        # Every functional input must remain in the independent evidence lock.
        for origin, item in origins.items():
            if origin.startswith(('rtl/', 'sim/', 'golden/', 'data/', 'formal/', 'syn/')) and origin not in locked:
                fail('unlocked_functional_input', origin)
        ledger_count = 0
        for origin in ['docs/v2/m6d/evidence/source-map.json', 'docs/v2/m7/evidence/source-map.json']:
            for item in json.loads(locate(origin).read_text()):
                ledger_count += 1
                p = locate(item['public_path'])
                if not p.is_file() or sha(p) != item['public_sha256']:
                    fail('accepted_evidence_hash', item['public_path'])
        figures = json.loads(locate('docs/v2/m6d/figures.json').read_text())['figures']
        for origin, digest in figures.items():
            p = locate(origin)
            if not p.is_file() or preserved_sha(p) != digest:
                fail('figure_hash', origin)
        source_counts = {}
        for origin in ['docs/v2/m6d/evidence/soc_provenance.json', 'docs/v2/m6d/evidence/ppa_provenance.json']:
            sources = json.loads(locate(origin).read_text())['verified_production_source_sha256']
            source_counts[origin] = len(sources)
            for name, digest in sources.items():
                p = locate(name)
                if not p.is_file() or preserved_sha(p) != digest:
                    fail('measured_production_source_hash', name)
        relocations = json.loads(locate('docs/v2/m7/link_relocations.json').read_text())
        archived = {(x['file'], x['line'], x['target']): x for x in relocations}
        if len(archived) != 24:
            fail('archive_relocation_count', len(archived))
        used = set()
        # A moved immutable document retains its original relative-link context.
        original_context = {relocated(x['path']): origin for origin, x in origins.items()
                            if x.get('path') and x['disposition'] == 'moved'}
        links = 0
        mapped_links = 0
        privacy = []
        lists = 0
        for p in files:
            rel = str(p.relative_to(root))
            if p.is_symlink():
                fail('symlink', rel)
            if p.stat().st_size > 50 * 1024**2:
                fail('oversized_payload', rel)
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                if p.suffix not in {'.png', '.jpg', '.jpeg', '.pdf'}:
                    fail('unexpected_binary', rel)
                continue
            for kind, pattern in PRIVACY.items():
                if re.search(pattern, text, re.M):
                    privacy.append({'path': rel, 'kind': kind})
            if p.suffix == '.f':
                lists += 1
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith(('#', '//', '+')) and not (root / line).is_file():
                        fail('filelist_missing_dependency', {'file': rel, 'target': line})
            if p.suffix not in {'.md', '.txt'}:
                continue
            context = original_context.get(rel, rel)
            text = re.sub(r'```.*?```', lambda m: '\n' * m[0].count('\n'), text, flags=re.S)
            for number, line in enumerate(text.splitlines(), 1):
                for target in LINK.findall(line):
                    target = target.strip().split(' "')[0].strip('<>')
                    url = urllib.parse.urlsplit(target)
                    if url.scheme or target.startswith('//'):
                        continue
                    links += 1
                    path = urllib.parse.unquote(url.path)
                    dest_name = posixpath.normpath(path.lstrip('/') if path.startswith('/')
                        else posixpath.join(posixpath.dirname(context), path) if path else context)
                    if not safe_name(dest_name):
                        fail('escaping_link', {'file': rel, 'target': target})
                        continue
                    dest = locate(dest_name)
                    if not dest.is_file():
                        key = (context, number, target)
                        item = archived.get(key)
                        if item is None:
                            fail('broken_link', {'file': rel, 'line': number, 'target': target})
                            continue
                        derivative = locate(item['derivative'])
                        replacement = locate(item['public_target'])
                        if (sha(p) != item['original_sha256'] or not derivative.is_file()
                            or not replacement.is_file()
                            or item['replacement_target'] not in LINK.findall(derivative.read_text())):
                            fail('invalid_historical_relocation', key)
                        else:
                            used.add(key)
                    elif url.fragment and dest.suffix in {'.md', '.txt'} and urllib.parse.unquote(url.fragment) not in anchors(dest.read_text()):
                        fail('missing_anchor', {'file': rel, 'target': target})
                    elif dest_name in origins and origins[dest_name].get('path') != dest_name:
                        mapped_links += 1
        if used != set(archived):
            fail('unmatched_historical_relocations', sorted(set(archived) - used))
        if privacy:
            fail('privacy', privacy)
        return {'status': 'FAIL' if errors else 'PASS', 'schema_version': 1,
                'files': len(files), 'inventory_entries': len(rows), 'protected_files': len(locked), 'protected_files_checked': checked_protected,
                'accepted_evidence_entries': ledger_count, 'figures_checked': len(figures),
                'production_source_counts': source_counts, 'filelists_checked': lists,
                'local_links_checked_md_and_txt': links, 'mapped_historical_links': mapped_links,
                'archived_references_resolved_by_explicit_ledger': len(used),
                'privacy_pattern_hits': len(privacy), 'archived_preparation_records': len(retired),
                'local_archives_verified': archived_verified, 'path_only_derivatives': len(patches), 'errors': errors,
                'comment_only_derivatives': len(reconstructed_comments),
                'comment_source_hashes': {row['to']: {
                    'current_sha256': row['current_sha256'],
                    'original_sha256': hashlib.sha256(reconstructed_comments[row['to']]).hexdigest()}
                    for row in comments if row['to'] in reconstructed_comments},
                'limits': 'Historical source checks invert only pinned comment spans and declared path translations. No external URL checks, licensed EDA reruns, Linux execution or verification closure implied. Historical links use immutable original context and explicit mapping; raw text is preserved.'}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        fail('validation_structure_error', str(exc))
        return {'status': 'FAIL', 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default=str(ROOT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--inventory', default=DEFAULT_INVENTORY,
                        help='Root-relative inventory; default checks the current v2 payload against the immutable v1 evidence baseline.')
    args = parser.parse_args()
    result = validate(pathlib.Path(args.snapshot), inventory=args.inventory)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
