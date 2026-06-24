create_clock [get_ports clk] -period 20.0 -name clk
set_false_path -from [get_ports rst_n]
set_input_delay  -clock clk 2.0 [all_inputs]
set_output_delay -clock clk 2.0 [all_outputs]
