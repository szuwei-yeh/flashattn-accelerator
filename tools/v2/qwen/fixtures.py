"""Load hash-pinned public Qwen fixtures and check the unchanged scalar oracle.

The accepted INT8 activations and expectations are ordinary text files. Routine
RTL replay needs no model weights, PyTorch, private archive or temporary inputs.
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import struct

PUBLIC_MANIFEST_SHA256 = '1baf4e8994b0ec9d26d913536e65f5c34de6184de74fafdb4bf1bdffbcea2544'
Q_SCALES, K_SCALES, V_SCALES = [34, 27, 55, 33], [17, 21], [6, 11]
TOKENS = [32,11773,64356,57203,58457,6529,553,27572,6894,323,2750,11,8692,19556,11,323,21010,279]
PROMPT = 'A hardware accelerator computes causal attention by storing keys and values, processing queries, and updating the output as each new token arrives.'


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def words_bytes(words):
    return b''.join(struct.pack('<i', int(word)) for word in words)


def prepare(dataset, out, root):
    dataset, out, root = map(pathlib.Path, (dataset, out, root))
    if sha(dataset/'manifest.json') != PUBLIC_MANIFEST_SHA256:
        raise ValueError('frozen public fixture manifest SHA256 mismatch')
    manifest = json.loads((dataset/'manifest.json').read_text())
    frozen = {'schema_version':1, 'model':'Qwen/Qwen2.5-0.5B-Instruct',
              'model_revision':'7ae557604adf67be50417f59c2c2f167def9a775',
              'layer_zero_based':11, 'q_heads':[0,1,7,8], 'kv_mapping':[0,0,1,1],
              'head_dim':64, 'tokens':18, 'token_ids':TOKENS, 'prompt':PROMPT,
              'add_special_tokens':False,
              'encoded_scales':{'q':Q_SCALES,'k':K_SCALES,'v':V_SCALES}}
    if any(manifest.get(k) != v for k,v in frozen.items()):
        raise ValueError('frozen public workload settings mismatch')
    for name,row in manifest['files'].items():
        f = dataset/name
        if pathlib.Path(name).name != name or f.is_symlink() or not f.is_file():
            raise ValueError('invalid fixture path: '+name)
        if sha(f) != row['sha256'] or f.stat().st_size != row['bytes']:
            raise ValueError('frozen public fixture hash mismatch: '+name)
    for name,digest in manifest['oracle_lut_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('original oracle/LUT source changed: '+name)

    def read_hex(name, width, count, signed=False):
        tokens = (dataset/name).read_text().split()
        if len(tokens) != count or any(not re.fullmatch('[0-9a-f]{'+str(width)+'}',t) for t in tokens):
            raise ValueError('fixture width/count mismatch: '+name)
        values = [int(t,16) for t in tokens]
        if signed:
            sign = 1 << (width*4-1)
            values = [v-(sign<<1) if v&sign else v for v in values]
        return values

    q = [read_hex(f'pq{h}.hex',2,1024,True)+read_hex(f'dq17{h}.hex',2,64,True)+
         read_hex(f'dq18{h}.hex',2,64,True) for h in range(4)]
    k = [read_hex(f'k{g}.hex',2,1152,True) for g in range(2)]
    v = [read_hex(f'v{g}.hex',2,1152,True) for g in range(2)]
    stream = bytes(x&255 for group in (q,k,v) for head in group for x in head)
    if hashlib.sha256(stream).hexdigest() != manifest['input_byte_stream_sha256']:
        raise ValueError('frozen INT8 input stream mismatch')
    q = [[head[t*64:(t+1)*64] for t in range(18)] for head in q]
    k = [[head[t*64:(t+1)*64] for t in range(18)] for head in k]
    v = [[head[t*64:(t+1)*64] for t in range(18)] for head in v]
    spec = importlib.util.spec_from_file_location('unchanged_qwen_oracle',root/'golden/v2/stateful/decode_reference.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    lut = oracle.load_lut()
    expected = {name:[] for name in ('N16','L17','L18')}
    for h,g in enumerate((0,0,1,1)):
        scales = (Q_SCALES[h],K_SCALES[g],V_SCALES[g])
        prefill = [x for row in oracle.prefill(q[h][:16],k[g][:16],v[g][:16],scales,True,lut) for x in row]
        if prefill != read_hex(f'pref_h{h}.hex',8,1024,True):
            raise ValueError('independent oracle disagrees with frozen prefill expectations')
        expected['N16'].extend(prefill)
        for n in (17,18):
            decode = oracle.decode(q[h][n-1],k[g][:n],v[g][:n],scales,n-1,True,lut)
            if decode != read_hex(f'dec{n}_h{h}.hex',8,64,True):
                raise ValueError('independent oracle disagrees with frozen decode expectations')
            expected['L'+str(n)].extend(decode)
    d = out/'fixtures/p1'
    d.mkdir(parents=True,exist_ok=False)
    # Copy accepted expectations verbatim; never replace them with RTL outputs.
    for f in sorted(dataset.glob('*.hex')):
        shutil.copy2(f,d/f.name)
    all_words = b''
    for name,words in expected.items():
        raw = words_bytes(words)
        if hashlib.sha256(raw).hexdigest() != manifest['expected_job_sha256'][name]:
            raise ValueError('frozen oracle word hash mismatch: '+name)
        (out/('expected_'+name+'.bin')).write_bytes(raw)
        all_words += raw
    if len(all_words)!=4608*4 or hashlib.sha256(all_words).hexdigest()!=manifest['expected_word_stream_sha256']:
        raise ValueError('frozen complete output stream mismatch')
    (out/'expected_all.bin').write_bytes(all_words)
    fixtures = {str(f.relative_to(out/'fixtures')):sha(f) for f in sorted(d.iterdir())}
    record = {'public_manifest_sha256':PUBLIC_MANIFEST_SHA256, 'frozen_origin':manifest['origin'],
              'settings':frozen,'encoded_scales':manifest['encoded_scales'],
              'input_byte_stream_sha256':manifest['input_byte_stream_sha256'],
              'oracle_words_checked_against_phase2a':4608,'fixture_sha256':fixtures,
              'input_fixture_sha256':{n:h for n,h in fixtures.items() if pathlib.Path(n).name.startswith(('k','v','pq','dq'))},
              'expected_word_stream_format':manifest['expected_word_stream_order'],
              'expected_word_stream_sha256':manifest['expected_word_stream_sha256'],
              'phase2a_numerical_results_unchanged':True}
    (out/'fixture_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
