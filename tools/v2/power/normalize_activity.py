"""Server-side header-only normalization and independent clock/window gate."""
import pathlib,json,re,hashlib,sys
raw=pathlib.Path(sys.argv[1]);target=pathlib.Path(sys.argv[2]);begin=float(sys.argv[3]);end=float(sys.argv[4]);cycles=int(sys.argv[5])
scopes=[];changes=[];clock=None;events=[];timestamp=0;body_raw=hashlib.sha256();body_normalized=hashlib.sha256()
with raw.open() as src,target.open('w') as dst:
 header=True
 for line in src:
  old=line
  if header:
   if line.startswith('$scope '):
    f=line.split();scopes.append(f[2]);name=f[2].replace('.','_').replace('[','_').replace(']','_');changes.append(dict(original=f[2],mapped=name));f[2]=name;line=' '.join(f)+'\n'
   elif line.startswith('$upscope'):scopes.pop()
   elif line.startswith('$var '):
    f=line.split()
    if '.'.join(scopes+[f[4]])=='tb_m6b1.h.u_dut.clk':clock=f[3]
    q=re.fullmatch(r'(.+)\[(\d+)\]',f[4])
    if q and len(f)==7 and re.fullmatch(r'\[\d+:\d+\]',f[5]):
     hi,lo=map(int,f[5][1:-1].split(':'));offset=int(q[2])*int(f[2]);f[4]=q[1];f[5]=f'[{offset+hi}:{offset+lo}]'
    f[4]=f[4].replace('.','_').replace('[','_').replace(']','_');line=' '.join(f)+'\n'
   if '$enddefinitions' in line:header=False
  else:
   body_raw.update(old.encode());body_normalized.update(line.encode())
   if line.startswith('#'):timestamp=int(line[1:])
   if clock and line.strip() in ('0'+clock,'1'+clock):events.append((timestamp,line[0]))
  dst.write(line)
assert clock and body_raw.digest()==body_normalized.digest()
lo,hi=round(begin*1000),round(end*1000);assert hi-lo==cycles*10000
rises=[t for t,v in events if v=='1' and lo<=t<hi]
assert len(rises)==cycles and rises[0]==lo+5000 and rises[-1]==hi-5000
assert all(b-a==10000 for a,b in zip(rises,rises[1:]))
falls={t for t,v in events if v=='0'};assert all(t+5000 in falls for t in rises)
print(json.dumps(dict(status='PASS_10NS_50_PERCENT_CLOCK_HEADER_ONLY_MAPPING',begin_ns=begin,end_ns=end,duration_ns=end-begin,cycles_including_accepted=cycles,observed_clock_period_ns=10,observed_high_ns=5,body_sha256=body_raw.hexdigest(),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),normalized_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scope_changes=[x for x in changes if x['original']!=x['mapped']])))
