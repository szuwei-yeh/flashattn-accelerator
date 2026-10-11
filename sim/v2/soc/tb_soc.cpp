#include "Vtb_soc_m6b3.h"
#include "verilated.h"
#include "timing.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

static Vtb_soc_m6b3 d;
static uint64_t clocks=0,csr_checks=0,successful=0,exact_words=0;
static std::string root,scope,control;
static unsigned latency=0;
static bool read_present=false,irq_collision=false;
static uint64_t completion_edges=0,owner_edges=0,publication_edges=0,head_edges=0;
static void must(bool ok,const std::string& why){if(!ok)throw std::runtime_error("M5D_CHECK "+why);}
static std::vector<U> load(const std::string& file){std::ifstream f(file);must(bool(f),"missing fixture "+file);std::vector<U> x;std::string s;while(f>>s)x.push_back(std::stoul(s,nullptr,16));return x;}
static U qb(int h){return 0x1000+h*0x1000;}static U kb(int g){return 0x6000+g*0x1000;}static U vb(int g){return 0xa000+g*0x1000;}
struct Plan {int op=0,len=16,pop=0,fatal=0;U base=0x20ff8;};
struct Addr {U addr,beats;};
static std::vector<Addr> expected_ar(Plan p){
 std::vector<Addr> v;
 auto descriptor=[&](U base,U bytes){while(bytes){U beats=std::min<U>(16,std::min<U>(bytes/8,(4096-base%4096)/8));must(beats>0,"oracle burst shape");v.push_back({base,beats});base+=beats*8;bytes-=beats*8;}};
 if(p.op==0||p.op==1)for(int g=0;g<2;g++)for(U b:{kb(g),vb(g)})descriptor(b+(p.op==1?U(p.len*DIM):0),p.op==0?16*DIM:DIM);
 if(p.op==0||p.op==2)for(int h=0;h<4;h++)descriptor(qb(h),p.op==0?16*DIM:DIM);
 return v;
}
struct Job {
 Plan p;bool owned=false,child=false,child_finished=false,terminal=false,writing=false,last_b=false;
 uint64_t accepted=0,child_tick=0,child_done_tick=0,write_tick=0,done_tick=0,request_tick=0,irq_tick=0;
 unsigned ar=0,r=0,reads=0,writes=0,copy=0,array=0,array_cycles=0,core=0,source_done=0,wb_phase=0;
 unsigned phases[16]={},hs[4]={},he[4]={},hc[4]={};std::array<U,14> captured{};
 std::vector<Addr> ars;std::vector<U> expected;std::unique_ptr<M7Monitor> wb;
};
static Plan next_plan;static Job job;static U expected_id=0;static uint64_t first_request=0;
static std::map<Q,uint8_t> memory;
// Read endpoint has the same IDLE/LAT/DATA schedule as accepted axi_mem_model.
static int read_state=0;static U read_address=0,read_left=0,read_countdown=0;
static int inject_read_resp=0,inject_read_after=-1;static bool inject_bad_last=false;
static int write_fault_burst=32,write_fault_resp=2;
static bool stalled_aw=false,stalled_w=false,stalled_r=false;
static U held_aw=0,held_w=0,held_r=0;static unsigned held_strb=0,held_bresp=0,held_rresp=0;
static bool held_b=false,held_rr=false;
static std::array<U,14> config_bits(){std::array<U,14>x{};for(unsigned i=0;i<14;i++)x[i]=d.obs_config[i];return x;}
static void begin_job(){
 must(!job.owned||(job.terminal&&d.obs_phase==0&&!d.child_output_valid),"M7 consecutive command accepted before prior ACK/release");job=Job{};job.p=next_plan;job.owned=true;job.accepted=clocks+1;job.request_tick=first_request;job.ars=expected_ar(job.p);
 if(job.p.op==0||job.p.op==2){for(int h=0;h<4;h++){auto e=load(root+"/p"+std::to_string(job.p.pop)+"/"+(job.p.op==0?"pref":"dec"+std::to_string(job.p.len))+"_h"+std::to_string(h)+".hex");job.expected.insert(job.expected.end(),e.begin(),e.end());}
  if(control=="expected")job.expected.back()^=1;
  job.wb=std::make_unique<M7Monitor>(job.p.base,bytes_of(job.expected),3,latency);
  if(job.p.fatal==2){job.wb->fail_burst=write_fault_burst;job.wb->resp=write_fault_resp;}
 }
 expected_id++;
}
static void tick(){
 d.clk=0;d.eval();
 if(!d.rst_n){read_state=0;read_present=false;job=Job{};expected_id=0;stalled_aw=stalled_w=stalled_r=held_b=held_rr=false;d.m_arready=0;d.m_rvalid=0;d.m_awready=0;d.m_wready=0;d.m_bvalid=0;d.eval();}
 else {
  if(d.obs_command)begin_job();
  unsigned age=job.child&&!job.child_finished?unsigned(clocks-job.child_tick):0;
  d.m_arready=read_state==0 && (timing.seed ? timing.ready(2) : !(age%13==3||age%13==4));
  if(read_state!=2)read_present=false;
  d.m_rvalid=read_state==2 && (timing.seed ? (read_present||timing.ready(3)) : age%11!=7);
  read_present=d.m_rvalid;d.m_rdata=0;
  if(read_state==2)for(unsigned b=0;b<8;b++){must(memory.count(Q(read_address)+b),"read uninitialized external byte");d.m_rdata|=Q(memory.at(Q(read_address)+b))<<(8*b);}
  bool read_fault=inject_read_after<0 || job.r>=unsigned(inject_read_after);d.m_rresp=read_fault?inject_read_resp:0;d.m_rlast=read_left==1;if(inject_bad_last && read_fault)d.m_rlast=!d.m_rlast;
  if(job.writing&&job.wb)job.wb->drive(d,unsigned(clocks-job.write_tick));
  else {d.m_awready=0;d.m_wready=0;d.m_bvalid=0;d.m_bresp=0;}
  d.eval();
 }
 bool cmd=d.rst_n&&d.obs_command,child_accept=d.rst_n&&d.obs_accept;
 bool child_commit=d.rst_n&&d.obs_commit,desc_accept=d.rst_n&&d.obs_wb_accept;
 bool success=d.rst_n&&d.obs_success,fatal=d.rst_n&&d.obs_fatal;
 bool ax_ar=d.rst_n&&d.m_arvalid&&d.m_arready,ax_r=d.rst_n&&d.m_rvalid&&d.m_rready;
 unsigned pre_phase=d.obs_phase;
 if(d.rst_n){
  if(stalled_aw)must(d.s_axil_awvalid&&d.s_axil_awaddr==held_aw,"AXIL AW stalled stability");
  if(stalled_w)must(d.s_axil_wvalid&&d.s_axil_wdata==held_w&&d.s_axil_wstrb==held_strb,"AXIL W stalled stability");
  if(held_b)must(d.s_axil_bvalid&&d.s_axil_bresp==held_bresp,"AXIL B stalled stability");
  if(held_rr)must(d.s_axil_rvalid&&d.s_axil_rdata==held_r&&d.s_axil_rresp==held_rresp,"AXIL R stalled stability");
  stalled_aw=d.s_axil_awvalid&&!d.s_axil_awready;held_aw=d.s_axil_awaddr;
  stalled_w=d.s_axil_wvalid&&!d.s_axil_wready;held_w=d.s_axil_wdata;held_strb=d.s_axil_wstrb;
  held_b=d.s_axil_bvalid&&!d.s_axil_bready;held_bresp=d.s_axil_bresp;
  held_rr=d.s_axil_rvalid&&!d.s_axil_rready;held_r=d.s_axil_rdata;held_rresp=d.s_axil_rresp;
  if(job.owned&&!job.terminal&&pre_phase==3)job.wb_phase++;
  if(job.child&&!job.child_finished){
   job.phases[d.obs_state]++;if(d.obs_cache_read)job.reads++;if(d.obs_cache_write)job.writes++;
   if(d.obs_state>=9&&d.obs_state<=11)job.copy++;
   if(d.obs_array_start)job.array++;if(d.obs_array_compute)job.array_cycles++;
   if(d.obs_core_active){job.core++;job.hc[d.obs_qhead]++;}
   if(d.obs_core_start)job.hs[d.obs_qhead]=unsigned(clocks-job.child_tick);
   if(d.obs_copy_store&&d.obs_copy_word+1==(job.p.op==0?16*DIM:DIM))job.he[d.obs_qhead]=unsigned(clocks-job.child_tick);
  }
  if(ax_ar){must(job.child&&job.ar<job.ars.size(),"unexpected/duplicate read transaction");auto a=job.ars[job.ar++];must(d.m_araddr==a.addr&&d.m_arlen+1==a.beats&&d.m_arsize==3&&d.m_arburst==1,"input head/group read address/geometry");must((d.m_araddr%4096)+(d.m_arlen+1)*8<=4096,"read 4 KiB crossing");}
  if(ax_r)job.r++;
  if(job.writing&&job.wb){
   unsigned window=std::min<Q>(16,std::min<Q>((job.p.base+job.expected.size()*4-job.wb->next_address)/8,(4096-job.wb->next_address%4096)/8));
   if(d.obs_wb_busy&&job.wb->beat_data.size()<window&&!d.obs_src_valid)job.wb->srcwait++;
   auto before=job.wb->memory.size();bool last=job.wb->observe(d);if(last)job.last_b=true;
   for(Q i=before;i<job.wb->memory.size();i++)memory[job.p.base+i]=job.wb->memory.at(job.p.base+i);
  }
  if(job.owned&&!job.terminal&&job.wb&&!job.last_b){
   must(!success&&pre_phase!=4&&!d.obs_output_committed&&!d.obs_output_bytes&&!(d.obs_irq_pending&1),"M7 host success/IRQ/output published before final OKAY B");
  }
  if(job.owned&&job.child_finished&&job.wb&&!d.obs_child_ack)owner_edges++;
  if(d.obs_core_active){must(d.obs_kvhead==d.obs_qhead/2,"M7 query/KV isolation");head_edges++;}
  if(child_commit&&(job.p.op==0||job.p.op==1))publication_edges++;
  if(success&&job.wb)completion_edges++;
  if(d.obs_child_ack)must(pre_phase==5&&job.terminal,"premature/wrong-owner child ACK");
  if(job.owned&&!job.terminal&&job.child_finished&&(job.p.op==0||job.p.op==2))must(d.child_done_status&&d.child_output_valid,"lost compute output before host release");
  if(success){must(job.owned&&!job.terminal&&!job.p.fatal,"unexpected/duplicate successful completion");if(job.wb)must(job.last_b&&job.source_done==1&&job.wb->aw==job.wb->b&&job.wb->b==job.wb->okay&&job.wb->memory.size()==job.expected.size()*4,"host DONE before successful writeback/source retirement");}
  if(fatal)must(job.p.fatal!=0,"genuine unexpected integrated fatal");
 }
 d.clk=1;d.eval();clocks++;d.clk=0;d.eval();
 if(!d.rst_n){must(d.obs_phase==0&&!d.irq&&!d.child_output_valid&&!d.cache_valid&&!d.m_arvalid&&!d.m_awvalid&&!d.m_wvalid&&!d.s_axil_bvalid&&!d.s_axil_rvalid,"asynchronous common-reset quiescence");return;}
 // Update external read responder from the observed, pre-edge transaction.
 if(read_state==0&&ax_ar){read_address=d.obs_araddr;read_left=d.obs_arlen+1;read_countdown=latency;read_state=latency?1:2;}
 else if(read_state==1){if(read_countdown<=1)read_state=2;else read_countdown--;}
 else if(read_state==2&&ax_r){read_present=false;read_address+=8;if(--read_left==0)read_state=0;}
 if(cmd){job.captured=config_bits();must(d.obs_job_id==expected_id,"JOB_ID accepted command association");}
 if(child_accept){job.child=true;job.child_tick=clocks;must(d.perf_cycles==0,"child counter clear");}
 if(desc_accept){must(job.owned&&job.wb&&!job.writing,"duplicate/no-output write descriptor");job.writing=true;job.write_tick=clocks;}
 if(d.obs_pack_done&&job.writing)job.source_done++;
 if(child_commit){job.child_finished=true;job.child_done_tick=clocks;}
 if(job.owned&&d.obs_phase!=0&&d.obs_phase!=5&&d.obs_phase!=6)must(config_bits()==job.captured,"immutable active config");
 if(success||fatal){job.terminal=true;job.done_tick=clocks;if(success){successful++;must(d.obs_phase==4&&d.irq,"host completion/IRQ association");}else must(d.obs_phase==6&&!d.obs_output_bytes,"fatal output invalidity");job.irq_tick=clocks;}
 if(d.obs_wb_done)must(job.last_b,"DMA DONE before independent final B");
}
static void idle_lite(){d.s_axil_awvalid=0;d.s_axil_wvalid=0;d.s_axil_arvalid=0;d.s_axil_bready=0;d.s_axil_rready=0;}
static void reset(){idle_lite();inject_read_resp=0;inject_read_after=-1;inject_bad_last=false;d.rst_n=0;d.eval();tick();tick();d.rst_n=1;tick();must(d.obs_phase==0&&!d.cache_valid,"reset initial ownership");}
static unsigned write(U addr,U value,unsigned strb=15,int order=0,unsigned stall=2,unsigned expect=0){
 d.s_axil_awaddr=addr;d.s_axil_awprot=0;d.s_axil_wdata=value;d.s_axil_wstrb=strb;d.s_axil_bready=0;
 if(timing.seed&&!irq_collision){order=int(timing.choose(3));stall=timing.choose(9);}
 unsigned separation=timing.seed&&!irq_collision?1+timing.choose(8):3;
 bool a=false,w=false;unsigned n=0;uint64_t req=0;
 for(;n<10000;n++){d.s_axil_awvalid=!a&&(order!=2||n>=separation);d.s_axil_wvalid=!w&&(order!=1||n>=separation);d.eval();bool ah=d.s_axil_awvalid&&d.s_axil_awready,wh=d.s_axil_wvalid&&d.s_axil_wready;
  if(!req&&(ah||wh)){req=clocks+1;if(addr==8&&value!=2)first_request=req;}
  tick();a|=ah;w|=wh;if(a&&w&&d.s_axil_bvalid)break;
 }
 must(n<10000,"AXIL write timeout");d.s_axil_awvalid=0;d.s_axil_wvalid=0;
 for(unsigned i=0;i<stall;i++)tick();unsigned response=d.s_axil_bresp;must(response==expect,"AXIL write response addr="+std::to_string(addr));
 d.s_axil_bready=1;tick();d.s_axil_bready=0;csr_checks++;return response;
}
static U read(U addr,unsigned expect=0,unsigned stall=2){
 if(timing.seed)stall=timing.choose(9);
 d.s_axil_araddr=addr;d.s_axil_arprot=0;d.s_axil_arvalid=1;d.s_axil_rready=0;unsigned t=0;
 for(;t<1000;t++){d.eval();bool a=d.s_axil_arready;tick();if(a)break;}must(t<1000,"AXIL AR timeout");d.s_axil_arvalid=0;
 while(!d.s_axil_rvalid){must(++t<1000,"AXIL read timeout");tick();}for(unsigned i=0;i<stall;i++)tick();U v=d.s_axil_rdata;must(d.s_axil_rresp==expect,"AXIL read response");d.s_axil_rready=1;tick();d.s_axil_rready=0;csr_checks++;return v;
}
static void put(U base,const std::vector<U>&v){must(d.obs_phase==0,"external input mutation while owned");for(size_t i=0;i<v.size();i++){must(v[i]<=255,"input fixture byte");memory[Q(base)+i]=uint8_t(v[i]);}}
static void inputs(int pop){for(int g=0;g<2;g++){put(kb(g),load(root+"/p"+std::to_string(pop)+"/k"+std::to_string(g)+".hex"));put(vb(g),load(root+"/p"+std::to_string(pop)+"/v"+std::to_string(g)+".hex"));}for(int h=0;h<4;h++)put(qb(h),load(root+"/p"+std::to_string(pop)+"/pq"+std::to_string(h)+".hex"));}
static void query(int pop,int len){for(int h=0;h<4;h++)put(qb(h),load(root+"/p"+std::to_string(pop)+"/dq"+std::to_string(len)+std::to_string(h)+".hex"));}
static void configure(Plan p){
 next_plan=p;write(0x024,U(p.op)|U(p.pop%2)<<2|(p.op==0?8:0));write(0x028,U(p.op==0?16:p.op==2?1:0)|U(p.op==2?p.len-1:0)<<8|U(p.len)<<16);write(0x02c,p.base);
 for(int h=0;h<4;h++){write(0x040+h*4,qb(h),15,h%3);write(0x050+h*4,U((h%2?-1:1)*(64+32*h+16*p.pop))&65535,3);}
 for(int g=0;g<2;g++){write(0x060+g*4,kb(g)+(p.op==1?p.len*DIM:0));write(0x068+g*4,vb(g)+(p.op==1?p.len*DIM:0));write(0x070+g*4,U((g%2?-1:1)*(64+16*g+16*p.pop))&65535,3);write(0x078+g*4,U((g%2?-1:1)*(256-32*g-32*p.pop))&65535,3);}
}
static void finish_job(){
 must(job.terminal&&!job.p.fatal&&d.obs_phase==4,"successful terminal owner");U status=read(0x00c);must((status&0x0e)==4,"host STATUS DONE");must((read(0x014)&3)==1,"completion IRQ pending identity");must(read(0x03c)==expected_id,"CSR JOB_ID");
 must(job.ar==job.ars.size()&&d.perf_ar==job.ar&&d.perf_rbeats==job.r&&d.perf_dma_bytes==job.r*8,"read byte/transaction counters");
 must(d.perf_cycles==job.child_done_tick-job.child_tick&&d.perf_cache_reads==job.reads&&d.perf_cache_writes==job.writes&&d.perf_copy_cycles==job.copy,"independent child/cache/copy counters");
 must(d.mon_cache_checks==job.reads,"real cache-read scoreboard coverage");
 unsigned words=0,bytes=0,aw=0,w=0,b=0,okay=0,wbc=0;
 if(job.wb){
  words=job.expected.size();bytes=words*4;aw=job.wb->aw;w=job.wb->w;b=job.wb->b;okay=job.wb->okay;wbc=d.wb_cycles;
  must(d.obs_source_words==words&&d.wb_aw==aw&&d.wb_w==w&&d.wb_bytes==bytes&&d.wb_b==b&&d.wb_okay==okay,"write/source counters");
  must(d.wb_aw_stall==job.wb->awstall&&d.wb_w_stall==job.wb->wstall&&d.wb_b_wait==job.wb->bwait&&d.wb_src_wait==job.wb->srcwait,"independent write stalls");
  for(size_t i=0;i<words;i++){U value=0;for(unsigned k=0;k<4;k++)value|=U(memory.at(Q(job.p.base)+i*4+k))<<(8*k);must(value==job.expected[i],"external DRAM exact output word");}exact_words+=words;
 }
 must(read(0x0c8)==bytes&&d.obs_output_bytes==bytes,"OUTPUT_BYTES only committed job");
 std::array<U,13> expected={d.perf_cycles,job.r*8,job.ar,job.r,job.reads,job.writes,job.copy,U(job.done_tick-job.accepted),bytes,aw,b,okay,job.wb_phase};
 for(unsigned i=0;i<13;i++)must(read(0x080+4*i)==expected[i],"CSR snapshot index="+std::to_string(i));
 unsigned validlen=job.p.op==3?0:job.p.op==1?job.p.len+1:job.p.len;
 for(int g=0;g<2;g++){U value=read(0x0b8+g*4);must(value==(validlen?256+validlen:0),"public cache length/ownership");}
 std::cout<<"HOST_JOB id="<<expected_id<<" op="<<job.p.op<<" length="<<job.p.len<<" pop="<<job.p.pop<<" latency="<<latency<<" compact="<<COMPACT<<" dim="<<DIM<<" lanes="<<LANES<<" host_cycles="<<job.done_tick-job.accepted<<" request_cycles="<<job.done_tick-job.request_tick<<" compute_cycles="<<d.perf_cycles<<" dma_wb_cycles="<<wbc<<" wb_phase_cycles="<<job.wb_phase<<" read_bytes="<<job.r*8<<" write_bytes="<<bytes<<" per_head_bytes="<<bytes/4<<" ar="<<job.ar<<" r="<<job.r<<" aw="<<aw<<" w="<<w<<" b="<<b<<" okay="<<okay<<" words="<<words<<" cache_reads="<<job.reads<<" cache_writes="<<job.writes<<" copy_cycles="<<job.copy<<" array_runs="<<job.array<<" array_compute="<<job.array_cycles<<" core_active="<<job.core<<" irq_latency="<<job.irq_tick-job.done_tick;
 for(int i=0;i<16;i++)std::cout<<" phase"<<i<<"="<<job.phases[i];for(int h=0;h<4;h++)std::cout<<" h"<<h<<"_start="<<job.hs[h]<<" h"<<h<<"_end="<<job.he[h]<<" h"<<h<<"_core="<<job.hc[h];
 if(job.wb)std::cout<<" aw_stall="<<job.wb->awstall<<" w_stall="<<job.wb->wstall<<" b_wait="<<job.wb->bwait<<" source_wait="<<job.wb->srcwait;
 std::cout<<" PASS\n";
 write(0x018,1);must(!d.irq,"IRQ ACK does not create event");must(d.child_done_status&&d.obs_phase==4,"IRQ ACK must retain child owner");write(0x008,2);
 for(unsigned i=0;i<20&&d.obs_phase!=0;i++)tick();must(d.obs_phase==0&&!d.child_output_valid,"host ACK release/idle");
}
static void execute(Plan p,bool busy_tests=false){
 configure(p);write(0x008,1,15,expected_id%3,6);
 if(busy_tests&&d.obs_phase!=4){auto bits=config_bits();write(0x040,0xfffffff0,15,2,3,2);write(0x008,1,15,1,3,2);must(config_bits()==bits,"busy descriptor mutation");write(0x018,4);}
 unsigned t=0;for(;t<200000&&!job.terminal;t++)tick();must(t<200000,"host job timeout");finish_job();
}
static void rejected(Plan p,unsigned reason){
 U id=expected_id,epoch[2]={d.cache_epoch[0],d.cache_epoch[1]},len[2]={d.cache_length[0],d.cache_length[1]};
 configure(p);write(8,1,15,2,4,2);must(read(0x020)==reason&&read(0x03c)==id,"invalid command classification/identity");
 for(int g=0;g<2;g++)must(d.cache_epoch[g]==epoch[g]&&d.cache_length[g]==len[g],"rejected command cache mutation");
 must(d.obs_phase==0&&(read(0x014)&4),"nonfatal rejection remains idle");write(0x018,4);std::cout<<"REJECT op="<<p.op<<" length="<<p.len<<" reason="<<reason<<" PASS\n";
}
static void recovery(){must(read(0x034)==((COMPACT?0x07000040u:0x06000040u)|(LANES<<8)),"actual dequantizer capability");write(0x010,7);inputs(1);execute({0,16,1,0,0x80ff8},true);execute({1,16,1,0,0});query(1,17);execute({2,17,1,0,0x84ff8},true);}
static void fault_case(int kind,int response,int burst){
 reset();write(0x010,7);inputs(0);bool append_fault=kind==1 && burst>0;if(append_fault)execute({0,16,0,0,0x60ff8});Plan p{append_fault?1:0,16,0,kind,0x64ff8};
 write_fault_burst=burst;write_fault_resp=response;configure(p);
 if(kind==1){inject_read_resp=response;inject_bad_last=response==0;inject_read_after=append_fault?(burst==1?1:2*(DIM/8)-1):-1;}
 write(8,1,15,1,5);unsigned t=0;for(;t<200000&&!job.terminal;t++)tick();must(t<200000&&d.obs_phase==6&&d.obs_error==unsigned(kind),"fatal classification");
 must((read(0x00c)&0xac)==8&&(read(0x014)&3)==2,"fatal success/cache/IRQ masking");
 must(read(0x0c8)==0&&read(0x0b8)==0&&read(0x0bc)==0,"fatal public output/cache invalidity");
 if(kind==2){must(d.child_done_status&&d.child_output_valid&&d.cache_valid==3,"BRESP fault must not pretend physical rollback");must(read(0x0cc)==job.wb->failing_addr,"failed burst address association");}
 write(0x018,2);must(!d.irq&&d.obs_phase==6,"IRQ ACK cannot recover fatal");write(8,1,15,0,2,2);write(8,2,15,0,2,2);write(8,4,15,0,2,2);
 must(d.obs_phase==6&&!d.obs_child_ack,"fatal requires common reset");
 std::cout<<"FAULT kind="<<kind<<" response="<<response<<" burst="<<burst<<" accepted_write_bytes="<<(job.wb?job.wb->w*8:0)<<" successful_write_bytes=0 PASS\n";
 reset();recovery();
}
static void reset_boundary(const std::string& name){
 reset();write(0x010,7);inputs(0);bool append=name.find("append_")==0;
 if(append)execute({0,16,0,0,0x90ff8});
 Plan p{append?1:0,16,0,0,0x94ff8};configure(p);write(8,1,15,0,2);
 bool found=false;unsigned t=0;
 for(;t<200000&&!job.terminal;t++){
  bool hit=false;
  if(name=="compute")hit=job.core>=20;
  if(name=="dq_first_retire")hit=d.obs_dq_retire&&d.obs_dq_group==0;
  if(name=="dq_last_retire")hit=d.obs_dq_retire&&d.obs_dq_group==15;
  if(name=="dq_publish")hit=d.obs_dq_publish;
  if(name=="dq_consume")hit=d.obs_dq_consume;
  if(name=="stream")hit=job.wb&&job.wb->w>=8&&!job.source_done;
  if(name=="pending_b")hit=job.source_done&&!job.last_b;
  if(name=="before_final_b")hit=job.wb&&job.wb->have_aw&&job.wb->countdown==0&&job.wb->address+job.wb->length*8==Q(p.base)+job.expected.size()*4;
  if(name=="after_final_b")hit=job.last_b&&!job.terminal;
  if(name=="append_k")hit=job.writes>=1&&!job.child_finished;
  if(name=="append_v")hit=job.writes>=unsigned(DIM/16+1)&&!job.child_finished;
  if(name=="append_before_publish")hit=d.obs_commit;
  if(name=="append_after_publish")hit=job.child_finished&&!job.terminal;
  if(hit){found=true;break;}tick();
 }
 must(found,"reset checkpoint not reached "+name);
 std::cout<<"RESET_POINT name="<<name<<" phase="<<unsigned(d.obs_phase)<<" ar="<<job.ar<<" r="<<job.r<<" aw="<<(job.wb?job.wb->aw:0)<<" w="<<(job.wb?job.wb->w:0)<<" b="<<(job.wb?job.wb->b:0)<<" cache_length="<<unsigned(d.cache_length[0])<<" host_success=0 PASS\n";
 reset();for(int i=0;i<4;i++)tick();must(d.obs_phase==0&&!d.irq&&!d.cache_valid&&!d.child_output_valid,"no stale success after boundary reset");recovery();
}
static void stress(){
 reset();write(0x010,7);must(read(0x0d0,2)==0&&read(3,2)==0,"invalid read address");write(0,0,15,1,3,2);write(0x013,0,15,2,4,2);
 write(0x050,0x80,1,1,3);write(0x050,0xff00,2,2,3);must(read(0x050)==0xff80,"partial signed scale bytes");write(0x050,0,0);must(read(0x050)==0xff80,"zero strobe no-op");
 inputs(0);execute({0,16,0,0,0x40ff8},true);query(0,16);execute({2,16,0,0,0x44ff8},true);
 rejected({0,16,0,0,0xfffffff8u},4);rejected({0,16,0,0,0x40001},4);rejected({2,15,0,0,0x44000},3);
 for(int len=16;len<64;len++)execute({1,len,0,0,0});rejected({1,64,0,0,0},3);rejected({2,64,0,0,0x44000},3);
 execute({3,0,0,0,0});rejected({2,16,0,0,0x44000},3);inputs(1);
 // IRQ_ACK consumer edge coincides with genuine terminal success, after final B.
 Plan p{0,16,1,0,0x48ff8};configure(p);write(8,1);unsigned t=0;for(;t<200000&&!d.obs_wb_done;t++)tick();must(t<200000&&job.last_b&&!job.terminal,"IRQ race final-B trigger");
 irq_collision=true;write(0x018,1,15,0,2);irq_collision=false;must((read(0x014)&1)&&d.irq,"new completion event dominates same-edge IRQ ACK");finish_job();
 query(1,16);execute({2,16,1,0,0x4cff8});execute({1,16,1,0,0});query(1,17);execute({2,17,1,0,0x50ff8});execute({1,17,1,0,0});query(1,18);execute({2,18,1,0,0x54ff8});
 if(LANES==16)for(const std::string name:{"dq_first_retire","dq_last_retire","dq_publish","dq_consume"})reset_boundary(name);
 std::cout<<"STRESS capacity64=1 rejected_overflow=1 poisoned_old_tail=1 IRQ_set_dominant=1 partial_bytes=1 PASS\n";
 for(int response=1;response<=3;response++)for(int burst:{0,DIM,2*DIM})fault_case(2,response,burst);
 for(int response:{0,1,2,3})fault_case(1,response,0);
 for(const std::string name:{"compute","stream","pending_b","before_final_b","after_final_b","append_k","append_v","append_before_publish","append_after_publish"})reset_boundary(name);
}
int main(int argc,char**argv){Verilated::commandArgs(argc,argv);try{
 must(argc>=4,"args fixtures latency scope [control]");root=argv[1];latency=std::stoul(argv[2]);scope=argv[3];control=argc>4?argv[4]:"";timing.init(argc>5?std::stoul(argv[5]):0);final_b_hold=argc>6?std::stoul(argv[6]):0;reset();
 must(read(0)==0x46415432&&read(0x030)==(0x10020400u|DIM)&&read(0x038)==0x50,"geometry identity");write(0x010,7);
 if(scope=="prefill"){inputs(0);execute({0,16,0,0,0x20ff8},true);}
 else if(scope=="stateful"||scope=="warm"||scope=="focus"){
  for(int pop=0;pop<(scope=="focus"?1:2);pop++){
   if(pop){execute({3,0,pop,0,0});inputs(pop);}else inputs(pop);
   uint64_t sequence_start=clocks;execute({0,16,pop,0,U(0x20ff8+pop*0x10000)},true);
   if(scope=="warm"){query(pop,16);execute({2,16,pop,0,U(0x24ff8+pop*0x10000)});continue;}
   execute({1,16,pop,0,0});query(pop,17);execute({2,17,pop,0,U(0x28ff8+pop*0x10000)},true);
   execute({1,17,pop,0,0});query(pop,18);execute({2,18,pop,0,U(0x2cff8+pop*0x10000)},true);
   std::cout<<"SEQUENCE pop="<<pop<<" latency="<<latency<<" compact="<<COMPACT<<" cycles_with_host_service="<<clocks-sequence_start<<" commands=5 includes_CSR_programming_snapshots_IRQ_ACK_release=1 PASS\n";
  }
 }else if(scope=="stress")stress();else if(scope=="append_fault"){for(int resp:{0,1,2,3})for(int stage:{1,2})fault_case(1,resp,stage);}else must(false,"unsupported test scope");
 must(!final_b_hold||final_b_held_edges>0,"M7 final B hold was not exercised");
 std::cout<<"M7_SAFETY seed="<<timing.seed<<" final_b_hold="<<final_b_hold<<" held_edges="<<final_b_held_edges<<" completion_edges="<<completion_edges<<" owner_edges="<<owner_edges<<" publication_edges="<<publication_edges<<" head_edges="<<head_edges<<" PASS\n";
 std::cout<<"RESULT PASS M5D successful="<<successful<<" exact_words="<<exact_words<<" csr_checks="<<csr_checks<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<" clock="<<clocks<<" phase="<<unsigned(d.obs_phase)<<" job="<<d.obs_job_id<<" child_cycles="<<d.perf_cycles<<"\n";return 1;}}
