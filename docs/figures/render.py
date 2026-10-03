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
    # Functional groups follow the portfolio diagram; they are not module boundaries.
    navy, teal, muted = '#081d4f', '#00999c', '#445f7b'
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="950" viewBox="0 0 1680 950" role="img" aria-labelledby="title desc">
<title id="title">FlashAttention accelerator architecture and simulation boundary</title>
<desc id="desc">A simulation-only behavioral AXI memory model and testbench are outside the synthesizable DUT. Vector DMA fills 16-bank Q/K/V scratchpads; a shared loader fills active Q/K/V or prefetches the next resident K/V tile into shadow registers. Shadow is copied to active before use. Only active registers supply operands to the shared 16 by 16 array. QK scores pass through 256 dequantizers and masking into 16 online-softmax lanes; unnormalized P returns for PV on the same array. Rescale factors and running sums feed fused output update and final normalization. Output is read after done, with no AXI writeback. Data movement and tiled compute are functional groupings, not module or physical boundaries. Teal arrows are data, gray dashed arrows are control; individual ports and interlocks are summarized.</desc>
<defs>
<marker id="data-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{teal}"/></marker>
<marker id="control-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{muted}"/></marker>
</defs>
<rect width="1680" height="950" fill="white"/>
<style>text {{font-family:Arial,Helvetica,sans-serif;fill:{navy}}} .data {{fill:none;stroke:{teal};stroke-width:3.5;marker-end:url(#data-arrow);stroke-linejoin:round}} .control {{fill:none;stroke:{muted};stroke-width:2.5;stroke-dasharray:7 6;marker-end:url(#control-arrow);stroke-linejoin:round}}</style>''']

    def text(x, y, value, size=24, weight='normal', color=navy, center=False):
        anchor = ' text-anchor="middle"' if center else ''
        parts.append(f'<text x="{x}" y="{y}"{anchor} font-size="{size}" font-weight="{weight}" style="fill:{color}">{escape(value)}</text>')

    def panel(x, y, w, h, fill, stroke, width=1.7):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def box(x, y, w, h, title, subtitle=None, accent=False, control=False):
        fill = '#e7f7f6' if accent else '#e9eef3' if control else '#f5f7fc'
        panel(x, y, w, h, fill, '#138391' if accent else navy)
        text(x+w/2,y+(35 if subtitle else h/2+9),title,25,'bold',center=True)
        if subtitle:
            text(x+w/2,y+64,subtitle,21,color=muted,center=True)

    def arrow(path, control=False):
        parts.append(f'<path class="{"control" if control else "data"}" d="{path}"/>')

    text(40,58,'FlashAttention Accelerator',52,'bold')
    text(40,98,'Single-head prefill · N = 64 · d = 16 / 64',30,color=muted)
    panel(24,130,1632,115,'#f3f6fb','#8fa4bf')
    text(45,176,'Simulation',31,'bold')
    text(45,211,'environment',31,'bold')
    box(385,147,380,80,'AXI memory model','Behavioral model only')
    box(1095,147,345,80,'Testbench','Configure · start · check')

    panel(24,279,1632,605,'#ffffff',navy,2.4)
    text(45,317,'Synthesizable DUT',34,'bold')
    panel(40,334,1600,251,'#fbfcff','#a7c5d7')
    text(56,373,'Data movement',30,'bold')
    panel(40,602,1600,267,'#f0fbfa','#97d6d9')
    text(56,640,'Tiled compute',30,'bold')

    box(297,364,283,75,'Vector DMA','64-bit AXI → 16 B stripes')
    box(632,364,285,75,'Q / K / V scratchpads','16 banks per scratchpad')
    box(970,364,280,75,'Shared tile loader','Load + prefetch')
    box(1308,364,302,75,'Active Q / K / V','Current tile')
    box(1340,489,270,75,'Shadow K / V','Next resident tile')
    box(540,485,416,80,'Run control + DMA scheduler','Q first; then K/V tiles',control=True)

    arrow('M500 227 V364')
    text(515,267,'AXI read',23,'bold',teal)
    # Config/start terminates at the top interface, not at a scratchpad.
    arrow('M1268 227 V261 H893 V279',True)
    text(1280,269,'control / start',21,color=muted)
    arrow('M580 402 H632')
    arrow('M917 402 H970')
    arrow('M1250 402 H1308')
    arrow('M1195 439 V527 H1340')
    text(1210,510,'prefetch',22,color=teal)
    arrow('M1467 489 V439')
    text(1490,474,'copy',22,color=teal)
    arrow('M540 525 H439 V439',True)
    arrow('M956 525 H1057 V439',True)
    # Array operands come from ACTIVE registers; shadow has no array read path.
    arrow('M1610 402 H1625 V622 H1025 V658')
    text(1240,615,'Active operands',20,color=teal)

    box(60,658,235,85,'Output read port','Read after done',accent=True)
    box(365,658,330,85,'Output update + SRAM','Rescale · accumulate · normalize',accent=True)
    box(850,658,350,85,'Shared 16 × 16 array','QKᵀ and PV reuse the same array',accent=True)
    box(1362,658,260,85,'Dequantize + mask','256 dequantizers',accent=True)
    box(850,792,350,60,'16 online-softmax lanes',accent=True)
    arrow('M365 700 H295')
    arrow('M850 700 H695')
    text(708,683,'PV × scale_v',20,color=teal)
    arrow('M1200 700 H1362')
    text(1218,683,'QKᵀ scores',22,color=teal)
    arrow('M1492 743 V822 H1200')
    arrow('M1025 792 V743')
    text(1052,774,'P for PV',22,color=teal)
    arrow('M850 822 H545 V743')
    text(615,803,'Rescale + row sum',22,color=teal)

    arrow('M40 919 H103')
    text(120,926,'Data',22,color=teal)
    arrow('M208 919 H275',True)
    text(292,926,'Control',22,color=muted)
    text(615,925,'d = 64: four chunks on the same array',20,color=muted)
    text(1000,925,'External memory is simulation-only · Read port, no AXI writeback',20,color=muted)
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
