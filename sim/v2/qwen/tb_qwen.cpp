// Frozen Qwen workload: reuse the accepted host, AXI endpoint and all checks.
// Only entry point, CSR configuration and DRAM export are added here.
#define main accepted_main
#define configure accepted_configure
#define execute accepted_execute
#include "../soc/tb_soc.cpp"
#undef execute
#undef configure
#undef main

static_assert(DIM==64 && LANES==256 && COMPACT==1, "frozen Phase 2B geometry");
static const U frozen_sq[4]={34,27,55,33},frozen_sk[2]={17,21},frozen_sv[2]={6,11};
static std::string output_root;
static unsigned cache_publications=0,exported_words=0;

static void validate_fixtures(){
 for(int g=0;g<2;g++)for(const std::string name:{"k","v"})
  must(load(root+"/p1/"+name+std::to_string(g)+".hex").size()==18*DIM,"Qwen KV fixture length");
 for(int h=0;h<4;h++){
  must(load(root+"/p1/pq"+std::to_string(h)+".hex").size()==16*DIM,"Qwen prefill Q length");
  must(load(root+"/p1/pref_h"+std::to_string(h)+".hex").size()==16*DIM,"Qwen prefill expected length");
  for(int n:{17,18}){
   must(load(root+"/p1/dq"+std::to_string(n)+std::to_string(h)+".hex").size()==DIM,"Qwen decode Q length");
   must(load(root+"/p1/dec"+std::to_string(n)+"_h"+std::to_string(h)+".hex").size()==DIM,"Qwen decode expected length");
  }
 }
}

static void configure_qwen(Plan p){
 // Existing programming and address checks remain; overwrite scales before START.
 accepted_configure(p);
 write(0x024,U(p.op)|4u|(p.op==0?8u:0u)); // explicit causal, PREFILL replacement
 for(int h=0;h<4;h++)write(0x050+h*4,frozen_sq[h],3);
 for(int g=0;g<2;g++){write(0x070+g*4,frozen_sk[g],3);write(0x078+g*4,frozen_sv[g],3);}
 for(int h=0;h<4;h++)must(read(0x050+h*4)==frozen_sq[h],"Qwen Q scale CSR");
 for(int g=0;g<2;g++)must(read(0x070+g*4)==frozen_sk[g]&&read(0x078+g*4)==frozen_sv[g],"Qwen KV scale CSR");
}

static void export_dram(){
 if(job.p.op==1)return;
 const std::string name=job.p.op==0?"N16":"L"+std::to_string(job.p.len);
 std::ofstream f(output_root+"/"+name+".bin",std::ios::binary);
 must(bool(f),"Qwen output file open");
 // Read the coherent external store filled from observed W handshakes, never goldens.
 for(size_t i=0;i<job.expected.size()*4;i++){
  auto it=memory.find(Q(job.p.base)+i);must(it!=memory.end(),"Qwen missing external DRAM output byte");
  f.put(char(it->second));
 }
 f.close();must(bool(f),"Qwen output file write");exported_words+=unsigned(job.expected.size());
 std::cout<<"QWEN_OUTPUT job="<<name<<" words="<<job.expected.size()
  <<" final_b_accepted="<<job.last_b<<" source_retirements="<<job.source_done
  <<" b="<<job.wb->b<<" okay="<<job.wb->okay<<" PASS\n";
}

static void execute_qwen(Plan p){
 configure_qwen(p);write(0x008,1,15,expected_id%3,6);
 // Keep accepted busy-descriptor/repeated-START rejection checks on output jobs.
 if(p.op!=1&&d.obs_phase!=4){auto bits=config_bits();write(0x040,0xfffffff0,15,2,3,2);
  write(0x008,1,15,1,3,2);must(config_bits()==bits,"busy descriptor mutation");write(0x018,4);}
 unsigned t=0;for(;t<200000&&!job.terminal;t++)tick();must(t<200000,"host job timeout");
 finish_job(); // Exact DRAM words, counters, status, final B, IRQ and ACK/release.
 if(p.op==0||p.op==1)cache_publications++;
 for(int g=0;g<2;g++){
  must(read(0x0c0+g*4)==cache_publications,"Qwen persistent cache epoch");
  must(read(0x070+g*4)==frozen_sk[g]&&read(0x078+g*4)==frozen_sv[g],"Qwen retained KV scales");
 }
 export_dram();
}

int main(int argc,char**argv){Verilated::commandArgs(argc,argv);try{
 must(argc==4,"args fixtures output-directory control");root=argv[1];output_root=argv[2];
 const std::string selected_control=argv[3];
 must(selected_control.empty()||selected_control=="expected","unsupported Qwen control");
 latency=0;timing.init(0);final_b_hold=0;control="";scope="qwen";
 validate_fixtures();reset();
 must(read(0)==0x46415432&&read(0x030)==(0x10020400u|DIM)&&read(0x038)==0x50,"geometry identity");
 must(read(0x034)==(0x07000040u|(LANES<<8)),"frozen dequantizer capability");write(0x010,7);inputs(1);
 uint64_t sequence_start=clocks;
 execute_qwen({0,16,1,0,0x20ff8});
 execute_qwen({1,16,1,0,0});query(1,17);execute_qwen({2,17,1,0,0x28ff8});
 execute_qwen({1,17,1,0,0});query(1,18);
 // Corrupt only the in-memory expected copy of the final word of the whole sequence.
 if(selected_control=="expected"){
  must(exact_words==4352,"Qwen control prior exact words");control="expected";
  std::cout<<"NEGATIVE_CONTROL wrong_final_word word_index=4607 prior_exact_words="<<exact_words<<"\n";
 }
 execute_qwen({2,18,1,0,0x2cff8});
 must(successful==5&&expected_id==5&&exact_words==4608&&exported_words==4608,"Qwen complete sequence counts");
 must(completion_edges==3&&publication_edges==3&&owner_edges>0&&head_edges>0,"Qwen relevant safety counts");
 std::cout<<"SEQUENCE pop=1 latency=0 compact=1 cycles_with_host_service="<<clocks-sequence_start
  <<" commands=5 includes_CSR_programming_snapshots_IRQ_ACK_release=1 PASS\n";
 std::cout<<"M7_SAFETY seed=0 final_b_hold=0 held_edges=0 completion_edges="<<completion_edges
  <<" owner_edges="<<owner_edges<<" publication_edges="<<publication_edges<<" head_edges="<<head_edges<<" PASS\n";
 std::cout<<"RESULT PASS QWEN successful="<<successful<<" exact_words="<<exact_words
  <<" mismatches=0 csr_checks="<<csr_checks<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<" clock="<<clocks<<" phase="<<unsigned(d.obs_phase)
  <<" job="<<d.obs_job_id<<" child_cycles="<<d.perf_cycles<<"\n";return 1;}}
