#pragma once
// Deterministic legal slave timing. No functional DUT inputs or goldens mutated.
#include "../common/axi_monitor.hpp"
#include <random>
struct Timing {
 unsigned seed=0;
 std::mt19937 rng;
 unsigned denied[4]={};
 void init(unsigned value){seed=value;rng.seed(value);for(auto& n:denied)n=0;}
 unsigned choose(unsigned count){return rng()%count;}
 bool ready(unsigned channel){
  bool value=denied[channel]>=7 || choose(4)!=0;
  denied[channel]=value?0:denied[channel]+1;
  return value;
 }
};
static Timing timing;
static unsigned final_b_hold=0;
static uint64_t final_b_held_edges=0;
struct M7Monitor : Monitor {
 int scheduled_b=-1;
 bool hold_started=false;
 unsigned remaining=0;
 M7Monitor(Q a,std::vector<uint8_t> e,int m,int d):Monitor(a,e,m,d){}
 template<class D> void drive(D&d,unsigned t){
  Monitor::drive(d,t);
  if(timing.seed){
   d.m_awready=timing.ready(0);d.m_wready=timing.ready(1);
   if(countdown>=0&&scheduled_b!=int(b)){
    scheduled_b=int(b);countdown+=int(timing.choose(9));
   }
   d.m_bvalid=countdown==0;
  }
  bool last=have_aw&&beat_data.size()==length&&address+length*8==base+expected.size();
  if(last&&countdown==0){
   if(!hold_started){hold_started=true;remaining=final_b_hold;}
   if(remaining){d.m_bvalid=0;remaining--;final_b_held_edges++;}
  }
 }
};
