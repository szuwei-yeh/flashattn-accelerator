# ============================================================
# TOP = flash_attn_top_dma_banked_prefetch
# Recommended full DMA-top synthesis profile with logical SRAM/ROM blackboxes.
# Simulation-only axi_mem_model.sv is intentionally excluded.
# Paths are relative to the repository root and ordered leaf-first.
# ============================================================
syn/blackboxes/sram_1r1w_bb.sv
syn/blackboxes/exp_lut_bb.sv
rtl/systolic/pe.sv
rtl/systolic/systolic_array.sv
rtl/systolic/array_controller.sv
rtl/quantization/dequantizer.sv
rtl/softmax/online_softmax.sv
rtl/memory/banked_scratchpad.sv
rtl/memory/banked_tile_loader.sv
rtl/memory/output_buffer.sv
rtl/ctrl/addr_gen.sv
rtl/ctrl/tile_controller_banked_prefetch.sv
rtl/core/flash_attn_core_banked_prefetch.sv
rtl/interface/dma_engine_vec.sv
rtl/top/flash_attn_top_dma_banked_prefetch.sv
