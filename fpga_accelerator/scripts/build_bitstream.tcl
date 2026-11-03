# Vivado Batch Build Script for Arty S7-25 BitGEMM Accelerator
# Part: Spartan-7 XC7S25-CSGA324-1

# Ensure working directory is the fpga_accelerator directory
set PROJ_DIR [file normalize [pwd]]
set OUTPUT_DIR [file normalize "$PROJ_DIR/build"]

file mkdir $OUTPUT_DIR

puts "=========================================================="
puts "  STARTING BITSTREAM GENERATION FOR ARTY S7-25"
puts "  Target Part: xc7s25csga324-1"
puts "  Output Dir:  $OUTPUT_DIR"
puts "=========================================================="

# 1. Read Verilog Sources using relative paths (avoids path whitespace issues)
read_verilog [list rtl/uart_rx.v rtl/uart_tx.v rtl/systolic_pe.v rtl/lif_neuron.v rtl/systolic_gemm_top.v]

# 2. Read Constraints
read_xdc [list constraints/arty_s7_25.xdc]

# 3. Synthesis
puts "--- Running Synthesis ---"
synth_design -top systolic_gemm_top -part xc7s25csga324-1 -flatten_hierarchy rebuilt
write_checkpoint -force [file join $OUTPUT_DIR post_synth.dcp]
report_utilization -file [file join $OUTPUT_DIR synth_utilization.txt]

# 4. Logic Optimization
puts "--- Running Logic Optimization (opt_design) ---"
opt_design

# 5. Placement
puts "--- Running Placement (place_design) ---"
place_design
write_checkpoint -force [file join $OUTPUT_DIR post_place.dcp]

# 6. Routing
puts "--- Running Routing (route_design) ---"
route_design
write_checkpoint -force [file join $OUTPUT_DIR post_route.dcp]
report_timing_summary -file [file join $OUTPUT_DIR route_timing_summary.txt]
report_utilization -file [file join $OUTPUT_DIR route_utilization.txt]

# 7. Generate Bitstream
puts "--- Writing Final Bitstream ---"
write_bitstream -force [file join $OUTPUT_DIR bitgemm_s7_arty.bit]

puts "=========================================================="
puts "  BUILD COMPLETE!"
puts "  Bitstream: [file join $OUTPUT_DIR bitgemm_s7_arty.bit]"
puts "=========================================================="
