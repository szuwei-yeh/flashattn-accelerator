# ============================================================
#  core_banked_prefetch.f — synthesis filelist for
#  TOP = flash_attn_core_banked_prefetch
#
#  Paths are relative to the REPO ROOT (the dc script prepends $REPO_ROOT).
#  Synthesizable RTL only. Ordered leaf-first so every module is analyzed
#  before it is referenced.
#
#  Hierarchy (what this top actually instantiates — verified by grep, not the
#  Verilator Makefile source list, which also carries some unused modules):
#    flash_attn_core_banked_prefetch
#      ├─ tile_controller_banked_prefetch          (FSM: prefetch + S_PF_WAIT)
#      ├─ addr_gen
#      ├─ banked_tile_loader
#      ├─ banked_scratchpad x3 (Q/K/V) ──► sram_1r1w
#      ├─ array_controller ─► systolic_array ─► pe
#      ├─ dequantizer x256
#      ├─ online_softmax x16 ─► exp_lut            (ROM via $readmemh — see syn/README.md)
#      └─ output_buffer ──► sram_1r1w
#
#  NOT included (not in this hierarchy): quantizer.sv, kv_cache.sv,
#  q_tile_buffer.sv, kv_tile_buffer.sv, axi_mem_model.sv (sim-only), any DMA.
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
