#pragma once
#include <cstdint>
#include <vector>
#include <map>
#include <stdexcept>
#include <string>
#include <algorithm>
using U=uint32_t;using Q=uint64_t;
static void check(bool ok,const std::string& why){if(!ok)throw std::runtime_error("M5C_CHECK "+why);}
// Independent AXI slave/event scoreboard. Early W data buffered until AW ownership arrives.
struct Monitor {
 Q base;std::vector<uint8_t> expected;std::map<Q,uint8_t> memory;
 int mode,delay,fail_burst=-1,resp=2,abort_tick=-1;
 unsigned aw=0,w=0,b=0,okay=0,early_w=0,awstall=0,wstall=0,bwait=0,srcwait=0;
 Q address=0,next_address=0;unsigned length=0;bool have_aw=false,responding=false;
 std::vector<Q> beat_data;std::vector<bool> lasts;int countdown=-1;Q failing_addr=0;
 bool stall_aw=false,stall_w=false;Q held_aw=0,held_w=0;unsigned held_len=0,held_strb=0;bool held_last=false;
 Monitor(Q a,std::vector<uint8_t> e,int m,int d,int fb=-1,int r=2):base(a),expected(e),mode(m),delay(d),fail_burst(fb),resp(r),next_address(a){}
 template<class D> void drive(D&d,unsigned t){
  d.m_awready=mode==1 ? bool(d.m_wvalid) : mode==2 ? t%23>=18 : mode==3 ? t%9>=3 : true;
  d.m_wready=mode==4 ? t%13>=7 : mode==3 ? t%11>=4 : true;
  d.m_bvalid=mode==5 || countdown==0;d.m_bresp=(int(b)==fail_burst ? resp : 0);
 }
 template<class D> bool observe(D&d){
  if(stall_aw)check(d.m_awvalid&&d.m_awaddr==held_aw&&d.m_awlen==held_len,"AW unstable under stall");
  if(stall_w)check(d.m_wvalid&&d.m_wdata==held_w&&d.m_wstrb==held_strb&&bool(d.m_wlast)==held_last,"W unstable under stall");
  if(d.m_awvalid&&!d.m_awready)awstall++;if(d.m_wvalid&&!d.m_wready)wstall++;
  if(d.m_bready&&!d.m_bvalid)bwait++;
  stall_aw=d.m_awvalid&&!d.m_awready;held_aw=d.m_awaddr;held_len=d.m_awlen;
  stall_w=d.m_wvalid&&!d.m_wready;held_w=d.m_wdata;held_strb=d.m_wstrb;held_last=d.m_wlast;
  if(d.m_awvalid&&d.m_awready){
   check(!have_aw&&countdown<0,"duplicate or new AW before B");
   address=d.m_awaddr;length=d.m_awlen+1;
   Q remaining=(base+expected.size()-next_address)/8;
   unsigned want=std::min<Q>(16,std::min<Q>(remaining,(4096-(next_address%4096))/8));
   check(address==next_address&&length==want,"AW address/length or head destination");
   check(d.m_awsize==3&&d.m_awburst==1&&length>=1&&length<=16,"AW geometry");
   check((address%4096)+length*8<=4096,"4 KiB boundary crossing");have_aw=true;aw++;
  }
  if(d.m_wvalid&&d.m_wready){
   check(countdown<0,"W while awaiting B");check(d.m_wstrb==255,"WSTRB not full supported beat");
   if(!have_aw)early_w++;beat_data.push_back(d.m_wdata);lasts.push_back(d.m_wlast);w++;
   check(beat_data.size()<=16,"duplicate/excess W");
  }
  if(have_aw){
   check(beat_data.size()<=length,"extra W beats");
   for(unsigned i=0;i<beat_data.size();i++)check(lasts[i]==(i+1==length),"incorrect WLAST");
   if(beat_data.size()==length&&countdown<0){
    for(unsigned i=0;i<length;i++)for(unsigned byte=0;byte<8;byte++){
     Q a=address+i*8+byte;check(a>=base&&a<base+expected.size(),"write outside expected span");
     uint8_t value=beat_data[i]>>(byte*8);check(value==expected.at(a-base),"corrupt/missing/duplicated byte or wrong head/word order");
     check(!memory.count(a),"duplicate byte address");memory[a]=value;
    }
    countdown=mode==5 ? 0 : delay+1;
   }
  }
  bool terminal=false;
  if(d.m_bvalid&&d.m_bready){
   check(have_aw&&beat_data.size()==length&&countdown==0,"early/incorrect B accounting");
   b++;if(d.m_bresp==0)okay++;else failing_addr=address;
   next_address=address+length*8;have_aw=false;beat_data.clear();lasts.clear();countdown=-1;
   terminal=d.m_bresp==0&&next_address==base+expected.size();
  }else if(countdown>0)countdown--;
  return terminal;
 }
 template<class D> void finish(D&d,bool fault){
  check(d.perf_aw==aw&&d.perf_wbeats==w&&d.perf_wbytes==w*8&&d.perf_b==b&&d.perf_okay_b==okay,"hardware traffic counters");
  check(d.perf_aw_stall==awstall&&d.perf_w_stall==wstall&&d.perf_b_wait==bwait,"hardware stall counters");
  if(fault)check(d.error&&!d.done&&d.error_kind==2&&d.failing_burst_addr==failing_addr,"BRESP error ignored or wrong classification/address");
  else check(d.done&&!d.error&&memory.size()==expected.size()&&aw==b&&b==okay,"completion/memory conservation");
 }
};
static std::vector<uint8_t> bytes_of(const std::vector<U>& words){std::vector<uint8_t>b;for(U v:words)for(unsigned i=0;i<4;i++)b.push_back(v>>(8*i));return b;}
