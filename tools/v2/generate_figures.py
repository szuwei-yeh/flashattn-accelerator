#!/usr/bin/env python3
"""Generate diagrams and matched-result plots from portable evidence extracts."""
import pathlib,json,hashlib,html,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[2];E=R/'docs/v2/evidence/results'
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--preview-dir');a=p.parse_args();O=pathlib.Path(a.output_dir).resolve();O.mkdir(parents=True,exist_ok=False);D=json.loads((E/'plot_data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'m6d-measured-v1'})
blue='#146c94';orange='#d37919'
def chart(name,title,labels,left,right,unit,caption):
 fig,ax=plt.subplots(figsize=(8.8,4.8));fig.subplots_adjust(bottom=.25,top=.80,left=.16,right=.96)
 x=list(range(len(labels)));width=.32
 for values,color,off,label in [(left,blue,-width/2,'Baseline'),(right,orange,width/2,'Changed configuration')]:
  bars=ax.bar([z+off for z in x],values,width,color=color,label=label)
  for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:,.3f}' if not float(v).is_integer() else f'{v:,.0f}',ha='center',va='bottom',fontsize=10)
 ax.set_xticks(x,labels);ax.set_ylabel(unit);ax.set_ylim(0,max(left+right)*1.30);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 fig.suptitle(title,fontsize=14,fontweight='bold',y=.96);ax.legend(loc='upper right',frameon=False,fontsize=9);fig.text(.08,.06,caption,fontsize=9,ha='left')
 fig.savefig(O/(name+'.svg'),metadata={'Date':None})
 if a.preview_dir:
  q=pathlib.Path(a.preview_dir);q.mkdir(parents=True,exist_ok=True);fig.savefig(q/(name+'.png'),dpi=130)
 plt.close(fig)
chart('decode_cycles','Measured single-head decode: output traversal',['d16','d64'],[x['baseline_cycles'] for x in D['decode']],[x['optimized_cycles'] for x in D['decode']],'RTL cycles','Warm L16, one query at position15, AXI modeled latency0,256 DQ lanes.\nBaseline full-row vs query-row-aware traversal. Excludes host/writeback; M2D.3B.')
chart('soc_area','Matched d16 SoC standard-cell area',['256 lanes vs16 lanes'],[D['ppa'][0]['standard_cell_area']],[D['ppa'][1]['standard_cell_area']],'Library area units','Same M6B.1 top, N16/d16,10ns,gscl45nm typical,DC ordinary compile; M6B.2.\n81 SRAM/16 ROM logical blackboxes. Pre-layout; no physical/electrical signoff.')
chart('gqa_traffic','Measured GQA read-payload reuse',['K/V payload','Total read payload'],[D['gqa_KV_bytes']['mha'],D['gqa']['mha_axi_bytes']],[D['gqa_KV_bytes']['gqa'],D['gqa']['gqa_axi_bytes']],'AXI payload bytes','M4 d16/N16 five commands: prefill,append,decode17,append,decode18.\nBaseline 4Q:4KV vs physical 4Q:2KV; no AXI/control overhead or writeback in this scope.')
chart('soc_energy','Matched d16 standard-cell job energy estimates',['Total energy','Dynamic energy'],[D['energy'][0]['total_nJ']/1000,D['energy'][0]['dynamic_nJ']/1000],[D['energy'][1]['total_nJ']/1000,D['energy'][1]['dynamic_nJ']/1000],'Energy (microjoules/job)','256 vs16 lanes; warm4Q:2KV/L16/compact1/AXI0;10ns accepted-job window.\nM6C activity-based pre-layout estimate; physical memory/parasitics/system excluded.')
def diagram(name,title,boxes,arrows,caption,height=450):
 from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
 fig,ax=plt.subplots(figsize=(10,height/100));fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
 ax.set_xlim(0,1000);ax.set_ylim(height,0);ax.axis('off');ax.text(30,34,title,fontsize=15,fontweight='bold',color='#163248')
 for x,y,w,h,label,color in boxes:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',facecolor=color,edgecolor='#748b99'))
  for i,line in enumerate(label.split('\n')):ax.text(x+w/2,y+25+i*21,line,ha='center',fontsize=10,color='#163248')
 for x1,y1,x2,y2,label in arrows:
  ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,color='#526573',linewidth=1.4))
  if label:ax.text((x1+x2)/2,(y1+y2)/2-8,label,ha='center',fontsize=9,color='#526573')
 for i,line in enumerate(caption.split('\n')):ax.text(30,height-45+i*19,line,fontsize=9,color='#526573')
 fig.savefig(O/(name+'.svg'),metadata={'Date':None})
 if a.preview_dir:fig.savefig(pathlib.Path(a.preview_dir)/(name+'.png'),dpi=130)
 plt.close(fig)
c='#e4f2f8';s='#fff1db';n='#eff2f4'
diagram('soc_architecture','Host-controlled 4Q:2KV attention SoC',[(30,75,190,75,'Host + memory model\nSimulation only',n),(280,75,210,75,'AXI4-Lite CSR\nCaptured jobs / IRQ',c),(550,75,180,75,'Read DMA\nAXI payload',c),(790,75,180,75,'Banked Q / K / V\nPersistent cache',c),(790,220,180,100,'2 physical KV groups\nQ0,Q1 → KV0\nQ2,Q3 → KV1',s),(530,220,200,100,'Shared 16×16 array\nQK and PV reuse\n256 or16 DQ lanes',c),(280,220,190,100,'Online softmax\nOutput update\nNormalize / pack',c),(30,220,190,100,'AXI writeback\nFinal OKAY B\nthen DONE / IRQ',c)],[(220,112,280,112,''),(490,112,550,112,''),(730,112,790,112,''),(880,150,880,220,''),(790,270,730,270,''),(530,270,470,270,''),(280,270,220,270,''),(120,220,120,150,'')],'One serial compute engine for 4 query heads; cache capacity64; d16/d64; host decodeL16/17/18.\nRTL filelist selects the DUT; C++ AXI/host responder is outside all PPA boundaries.')
diagram('stateful_sequence','One persistent sequence: no common reset between accepted jobs',[(30,90,180,115,'PREFILL N16\nPublish 16 K/V tokens\nCheck all 16 rows',c),(230,90,155,115,'APPEND token 16\nK then V complete\nPublish length 17',s),(405,90,170,115,'DECODE Q16\nPosition 16 / L17\nOne valid output row',c),(595,90,155,115,'APPEND token 17\nK then V complete\nPublish length 18',s),(770,90,200,115,'DECODE Q17\nPosition 17 / L18\nOne valid output row',c)],[(210,148,230,148,''),(385,148,405,148,''),(575,148,595,148,''),(750,148,770,148,'')],'Host observes final successful B before DONE/IRQ, then ACK releases each result owner.\nCLEAR begins a new sequence; rejected operations never publish provisional K/V. Fatal AXI faults require common reset.',300)
diagram('output_traversal','Decode output traversal: preserve arithmetic, visit only owned row',[(40,90,390,155,'Baseline per PV chunk\nTraverse 16 rows ×16 features\nUpdate 257 + normalize 257 cycles\nOne valid query row; 15 padded rows',n),(560,90,390,155,'Query-row-aware per PV chunk\nTraverse 1 row ×16 features\nUpdate 17 + normalize 17 cycles\nDrain delayed writes before DONE',c)],[(430,165,560,165,'decode only')],'Measured L16: d16 saves 480 cycles; d64 four PV chunks save 1920 cycles; M2D.3B.\nPrefill retains full traversal. QK/PV invocation counts, traffic, rounding and output ownership are unchanged.',340)
diagram('dequantizer_sharing','Selectable dequantization: complete score tile before softmax',[(40,90,390,150,'256-lane default\n256 physical dequantizers\nParallel score-tile production\nOriginal arithmetic preserved',n),(560,90,390,150,'16-lane sharing\n16 physical dequantizers\n16 ordered batches per256-score tile\nIssue / retire / stage / publish',c),(290,285,420,65,'Complete valid score tile → masked softmax',s)],[(235,240,380,285,''),(755,240,620,285,'')],'d64 retains four QK accumulation chunks and four PV output chunks; no partial tile consumption.\nLane count independently verified from elaborated hierarchy. Area/latency tradeoff measured separately.',430)
manifest={'status':'GENERATED_FROM_ACCEPTED_EXTRACTS','data_sha256':hashlib.sha256((E/'plot_data.json').read_bytes()).hexdigest(),'evidence_source_map_sha256':hashlib.sha256((E/'source-map.json').read_bytes()).hexdigest(),'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.glob('*.svg'))},'plot_tool':'matplotlib '+matplotlib.__version__,'diagrams':'native SVG; engineering interpretation of verified RTL, not new measurement'}
(O/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Generated',len(manifest['figures']),'figures')
