# ============================================================
#  systolic_array.f — synthesis filelist for TOP = systolic_array
#
#  Paths are relative to the REPO ROOT (the dc script prepends $REPO_ROOT).
#  Synthesizable RTL only. Ordered leaf-first (pe before systolic_array).
#
#  Hierarchy:  systolic_array
#                └─ pe   (16x16 = 256 instances)
#
#  This is the smallest, cleanest synth target — pure combinational MAC array +
#  registered accumulators, no memories, no $readmemh. Good first sanity run.
# ============================================================
rtl/systolic/pe.sv
rtl/systolic/systolic_array.sv
