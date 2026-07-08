# ============================================================
#  top_dma_banked_prefetch.f — synthesis filelist for
#  TOP = flash_attn_top_dma_banked_prefetch
#
#  Paths are relative to the REPO ROOT (the dc script prepends $REPO_ROOT).
#  Synthesizable RTL only. Ordered leaf-first.
#
#  = core_banked_prefetch.f  +  the vector DMA engine  +  the DMA top.
#
#  Hierarchy:
#    flash_attn_top_dma_banked_prefetch
#      ├─ dma_engine_vec                     (AXI4 read-master, 16-byte stripe writes)
#      └─ flash_attn_core_banked_prefetch    (full banked prefetch core — see above)
#
#  IMPORTANT — sim-only, deliberately EXCLUDED from synthesis:
#    - rtl/interface/axi_mem_model.sv   (behavioral DRAM; the DMA's AXI read-master
#      ports are the synthesis boundary — the external DRAM is NOT part of the DUT).
#    - all sim/verilator/* harnesses and C++ testbenches.
#  The AXI4 read channels (m_araddr/arlen/.../rdata/rvalid/rready) are top-level
#  ports; on the server they connect to a real DRAM controller / testbench, not to
#  axi_mem_model.
# ============================================================
rtl/memory/sram_1r1w.sv
rtl/systolic/pe.sv
rtl/systolic/systolic_array.sv
rtl/systolic/array_controller.sv
rtl/quantization/dequantizer.sv
rtl/softmax/exp_lut.sv
rtl/softmax/online_softmax.sv
rtl/memory/banked_scratchpad.sv
rtl/memory/banked_tile_loader.sv
rtl/memory/output_buffer.sv
rtl/ctrl/addr_gen.sv
rtl/ctrl/tile_controller_banked_prefetch.sv
rtl/core/flash_attn_core_banked_prefetch.sv
rtl/interface/dma_engine_vec.sv
rtl/top/flash_attn_top_dma_banked_prefetch.sv
