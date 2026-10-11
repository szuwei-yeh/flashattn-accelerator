"""Independent deterministic d16/d64 logical-row GQA fixtures; no RTL-derived outputs."""
import pathlib,random,hashlib,importlib.util
ROOT=pathlib.Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('ref',ROOT/'golden/v2/stateful/decode_reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
def fixtures(out,qh,kh,dim=16):
 assert dim in (16,64) and qh==4 and kh==2
 for pop in range(2):
  d=out/f'p{pop}';d.mkdir(parents=True);rng=random.Random(9102026+pop)
  ks=[];vs=[]
  def write(n,x,w): (d/n).write_text(''.join(f'{int(v)&((1<<(w*4))-1):0{w}x}\n' for v in x))
  for g in range(kh):
   rng=random.Random(9102026+pop+100*g)
   k=[[rng.randint(-16,16) for _ in range(dim)] for _ in range(64)];v=[[rng.randint(-25,25) for _ in range(dim)] for _ in range(64)];ks.append(k);vs.append(v)
   write(f'k{g}.hex',sum(k,[]),2);write(f'v{g}.hex',sum(v,[]),2)
  for h in range(qh):
   rng=random.Random(9102026+pop+1000+h)
   g=h//(qh//kh);sq=(-1 if h%2 else 1)*(64+32*h+16*pop);sk=(-1 if g%2 else 1)*(64+16*g+16*pop);sv=(-1 if g%2 else 1)*(256-32*g-32*pop)
   pq=[[rng.randint(-12,12) for _ in range(dim)] for _ in range(16)]
   write(f'pq{h}.hex',sum(pq,[]),2);write(f'pref_h{h}.hex',sum(ref.prefill(pq,ks[g][:16],vs[g][:16],(sq,sk,sv),bool(pop)),[]),8)
   for l in (16,17,18):
    q=[rng.randint(-12,12) for _ in range(dim)];write(f'dq{l}{h}.hex',q,2);write(f'dec{l}_h{h}.hex',ref.decode(q,ks[g][:l],vs[g][:l],(sq,sk,sv),l-1,bool(pop)),8)
    if l==17:write(f'zdec{l}_h{h}.hex',ref.decode([0]*dim,ks[g][:l],vs[g][:l],(sq,sk,sv),l-1,bool(pop)),8)
