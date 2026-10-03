"""Regenerate public SVG figures: python3 docs/figures/render.py.

Requires Matplotlib for the measured cycle chart. No simulator or synthesis run.
Historical cycle data is in the adjacent CSV; see ../DESIGN.md for provenance.
"""
from pathlib import Path
import csv
from html import escape

HERE = Path(__file__).resolve().parent
NAVY = '#152b4b'
TEAL = '#087e87'
MUTED = '#52647a'


def architecture():
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="830" viewBox="0 0 1400 830" role="img" aria-labelledby="title desc">
<title id="title">Current single-head FlashAttention architecture</title>
<desc id="desc">64-bit AXI vector DMA fills 16-bank Q/K/V scratchpads. A tile loader stages Q and active/shadow K/V registers. One 16 by 16 array performs QK and PV. QK goes through 256 dequantizers and 16 online-softmax lanes; exponential weights return to the array for PV. Fused output update rescales and accumulates PV results, then normalizes to an output read port. Simplified logical dataflow; control wires and operand muxes are omitted.</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{TEAL}"/></marker></defs>
<rect width="1400" height="830" rx="20" fill="#ffffff"/>
<style>text {{font-family:Arial,Helvetica,sans-serif;fill:{NAVY}}} .label {{font-size:23px;fill:{MUTED}}} .edge {{fill:none;stroke:{TEAL};stroke-width:3;marker-end:url(#arrow);stroke-linejoin:round}}</style>''']

    def text(x, y, value, size=26, weight='normal', color=NAVY):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" style="fill:{color}">{escape(value)}</text>')

    def box(x, y, w, h, title, lines, accent=False):
        fill = '#eaf7f6' if accent else '#f2f5f9'
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#c6d6df" stroke-width="1.5"/>')
        text(x+20,y+37,title,26,'bold')
        for i,line in enumerate(lines):
            text(x+20,y+73+i*30,line,23,color=MUTED)

    def arrow(points):
        parts.append(f'<path class="edge" d="{points}"/>')

    text(40,55,'Tiled attention. Shared compute. Overlapped memory.',36,'bold')
    text(40,95,'Current single-head path  ·  INT8 operands / INT32 accumulation  ·  N64, d16 / d64',24,color=MUTED)
    parts.append('<rect x="40" y="125" width="1320" height="65" rx="12" fill="#152b4b"/>')
    text(60,166,'CONTROL  ·  Accepted configuration  ·  DMA residency  ·  loader interlock  ·  valid shadow promotion',25,color='#ffffff')
    text(40,237,'External memory → AXI read',23,color=TEAL)
    box(40,260,245,150,'Vector DMA',['64-bit AXI','2 beats → 16 B stripe'])
    box(330,260,270,150,'Q / K / V SRAM',['16 banks × 1 byte','4 KB per scratchpad'])
    box(645,260,290,150,'Tile loader + registers',['Q registers','K/V active + shadow'])
    box(980,260,380,150,'Shared 16 × 16 array',['One array for QKᵀ and PV','Signed INT32 partial sums'],True)
    arrow('M285 335 H330')
    arrow('M600 335 H645')
    arrow('M935 335 H980')
    box(980,490,380,105,'256 dequantizers',['Shared Q/K scale + 1/√d'])
    box(565,490,330,105,'16 online-softmax lanes',['Running max / sum + LUT'],True)
    arrow('M1270 410 V490')
    text(1283,456,'QKᵀ',23,color=TEAL)
    arrow('M980 542 H895')
    text(903,523,'scores',20,color=TEAL)
    arrow('M800 490 V450 H1050 V410')
    text(820,440,'P weights → PV',22,color=TEAL)
    box(565,690,500,100,'Fused output update + SRAM',['Rescale + accumulate; final normalization'],True)
    box(1120,690,240,100,'Output read port',['Normalized INT32'])
    arrow('M1065 740 H1120')
    arrow('M730 595 V690')
    text(749,632,'Rescale factors',22,color=TEAL)
    text(749,660,'and running sum',22,color=TEAL)
    arrow('M1360 335 H1380 V647 H1020 V690')
    text(1130,632,'PV × scale_v',23,color=TEAL)
    text(40,525,'WHY THIS ORGANIZATION',23,'bold',TEAL)
    text(40,572,'Keep score tiles on chip.',26)
    text(40,617,'Reuse the array for both products.',26)
    text(40,662,'Stage the next resident K/V tile.',26)
    text(40,707,'Fuse two output-buffer passes.',26)
    text(40,807,'Logical dataflow; operand muxes and control wires omitted. Output readout is outside measured start → done.',20,color=MUTED)
    parts.append('</svg>')
    (HERE/'architecture.svg').write_text('\n'.join(parts)+'\n')


def cycle_chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'path','svg.hashsalt':'flashattn-milestones'})
    with (HERE/'cycle-milestones.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    fig, axes=plt.subplots(1,2,figsize=(14,5.8))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.165,right=.965,top=.72,bottom=.24,wspace=.65)
    fig.text(.035,.91,'Where the cycles went',fontsize=23,weight='bold',color=NAVY)
    fig.text(.035,.84,'N64 · noncausal · zero modeled read latency · start → done · lower is better',fontsize=14,color=MUTED)
    for ax,dim in zip(axes,(16,64)):
        values=[int(row[f'd{dim}_cycles']) for row in rows]
        ax.barh(range(4),values,color=['#8091a5','#506981','#294764',TEAL],height=.56)
        ax.set_yticks(range(4),[row['stage'] for row in rows],color=NAVY)
        ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.25)
        ax.set_title(f'Head dimension {dim}',loc='left',color=NAVY,weight='bold',pad=14)
        ax.set_xlabel('Cycles (panel-specific scale)',color=MUTED,labelpad=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
        ax.tick_params(axis='both',length=0,labelsize=12)
        ax.tick_params(axis='y',pad=10)
        ax.grid(axis='x',color='#e5ebf0')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i,value in enumerate(values):
            ax.text(value+max(values)*.025,i,f'{value:,}',va='center',fontsize=14,color=NAVY,weight='bold' if i==3 else 'normal')
    fig.text(.035,.095,'Successive RTL milestones; fusion alone saves 34.52% (d16) / 36.06% (d64) vs. the prefetch stage.',fontsize=12,color=MUTED)
    fig.text(.035,.04,'Excludes host setup, output readout and writeback. Data: cycle-milestones.csv · Scope: docs/RESULTS_EVIDENCE.md',fontsize=11,color=MUTED)
    fig.savefig(HERE/'cycle-milestones.svg',metadata={'Date':None,'Title':'Measured cycle-count milestones','Description':'Two panels with different cycle-axis ranges; see adjacent CSV and docs/RESULTS_EVIDENCE.md.'})
    plt.close(fig)
    svg = HERE/'cycle-milestones.svg'
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")


if __name__=='__main__':
    architecture()
    cycle_chart()
