"""Single isolated PT endpoint watchdog; no synthesis or automatic second lane."""
import json,pathlib,subprocess,os,time,signal,hashlib
R=pathlib.Path('.');m=json.loads((R/'inputs.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify():
 for n,h in m['input_hashes'].items():assert sha(n)==h,n
def mem():
 d={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':',1)
  if k in ('MemAvailable','SwapTotal','SwapFree'):d[k]=int(v.split()[0])
 return d
verify();before=mem();assert before['MemAvailable']>=12*1024**2
env=dict(os.environ,**m['environment']);status={'status':'RUNNING','lane':m['lane'],'start_memory_KiB':before}
def save():(R/'receipt.json').write_text(json.dumps(status,indent=2)+'\n')
save();start=time.monotonic();reason=None;low=0
with (R/'pt.log').open('w') as f,(R/'telemetry.jsonl').open('w') as trace:
 p=subprocess.Popen([m['pt'],'-f','pt_power.tcl'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 status['process_group']=p.pid;save()
 try:
  while p.poll() is None:
   group=[]
   for line in subprocess.check_output(['ps','-eo','pid,pgid,rss,pcpu,comm'],universal_newlines=True).splitlines()[1:]:
    x=line.split()
    if len(x)==5 and int(x[1])==p.pid:
     swap=0
     try:
      for s in pathlib.Path('/proc/'+x[0]+'/status').read_text().splitlines():
       if s.startswith('VmSwap:'):swap=int(s.split()[1])
     except FileNotFoundError:pass
     group.append(dict(pid=int(x[0]),RSS_KiB=int(x[2]),CPU_percent=float(x[3]),VmSwap_KiB=swap))
   memory=mem();rss=sum(x['RSS_KiB'] for x in group);low=low+1 if memory['MemAvailable']<2*1024**2 else 0
   row=dict(elapsed_seconds=time.monotonic()-start,memory_KiB=memory,processes=group,group_RSS_KiB=rss);trace.write(json.dumps(row)+'\n');trace.flush()
   if rss>12*1024**2:reason='RSS_above12GiB'
   elif low>=2:reason='available_below2GiB'
   elif sum(x['VmSwap_KiB'] for x in group)>.25*1024**2:reason='process_swap_pressure'
   elif memory['SwapTotal']-memory['SwapFree']>.5*1024**2:reason='host_swap_pressure'
   elif time.monotonic()-start>1200:reason='bounded20minute_timeout'
   if reason:break
   time.sleep(2)
 except BaseException as exc:reason='monitor_exception_'+type(exc).__name__
 if reason and p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 code=p.wait()
verify();text=(R/'pt.log').read_text();errors=[s for s in text.splitlines() if s.startswith('Error:')]
status.update(status='RESOURCE-LIMITED' if reason else 'PASS_TOOL_RUN' if code==0 and 'M6C_POWER_COMPLETE' in text and not errors else 'BLOCKED',exit=code,termination_reason=reason,seconds=time.monotonic()-start,source_artifacts_unchanged=True,errors=errors,end_memory_KiB=mem(),kernel_OOM_claim=False)
save()
