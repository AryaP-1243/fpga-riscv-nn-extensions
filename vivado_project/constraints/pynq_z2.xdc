# PYNQ-Z2 Constraint File for RISC-V ISA Extension Project
# Xilinx Zynq-7020 FPGA on PYNQ-Z2 Board

# Clock constraints
# 125MHz system clock
create_clock -period 8.000 -name sys_clk [get_ports sys_clk]

# Clock domain crossing constraints
set_clock_groups -asynchronous -group [get_clocks sys_clk] -group [get_clocks clk_fpga_0]
set_clock_groups -asynchronous -group [get_clocks sys_clk] -group [get_clocks clk_fpga_1]

# Input/Output delay constraints
set_input_delay -clock sys_clk -max 2.0 [all_inputs]
set_input_delay -clock sys_clk -min 0.5 [all_inputs]
set_output_delay -clock sys_clk -max 2.0 [all_outputs]
set_output_delay -clock sys_clk -min 0.5 [all_outputs]

# LEDs (for status indication)
set_property PACKAGE_PIN R14 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]
set_property PACKAGE_PIN P14 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]
set_property PACKAGE_PIN N16 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]
set_property PACKAGE_PIN M14 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]

# RGB LEDs (for advanced status)
set_property PACKAGE_PIN L15 [get_ports {rgb_led0[0]}]  # Red
set_property IOSTANDARD LVCMOS33 [get_ports {rgb_led0[0]}]
set_property PACKAGE_PIN G17 [get_ports {rgb_led0[1]}]  # Green
set_property IOSTANDARD LVCMOS33 [get_ports {rgb_led0[1]}]
set_property PACKAGE_PIN N15 [get_ports {rgb_led0[2]}]  # Blue
set_property IOSTANDARD LVCMOS33 [get_ports {rgb_led0[2]}]

set_property PACKAGE_PIN G14 [get_ports {rgb_led1[0]}]  # Red
set_property IOSTANDARD LVCMOS33 [get_ports {rgb_led1[0]}]
set_property PACKAGE_PIN L14 [get_ports {rgb_led1[1]}]  # Green
set_property IOSTANDARD LVCMOS33 [get_ports {rgb_led1[1]}]
set_property PACKAGE_PIN M15 [get_ports {rgb_led1[2]}]  # Blue
set_property IOSTANDARD LVCMOS33 [get_ports {rgb_led1[2]}]

# Buttons (for manual control)
set_property PACKAGE_PIN D19 [get_ports {btn[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[0]}]
set_property PACKAGE_PIN D20 [get_ports {btn[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[1]}]
set_property PACKAGE_PIN L20 [get_ports {btn[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[2]}]
set_property PACKAGE_PIN L19 [get_ports {btn[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[3]}]

# Switches (for configuration)
set_property PACKAGE_PIN M20 [get_ports {sw[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[0]}]
set_property PACKAGE_PIN M19 [get_ports {sw[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[1]}]

# Arduino/Raspberry Pi Header pins (for external interfaces)
# Arduino Digital pins
set_property PACKAGE_PIN T14 [get_ports {arduino_d[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[0]}]
set_property PACKAGE_PIN U12 [get_ports {arduino_d[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[1]}]
set_property PACKAGE_PIN U13 [get_ports {arduino_d[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[2]}]
set_property PACKAGE_PIN V13 [get_ports {arduino_d[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[3]}]
set_property PACKAGE_PIN V15 [get_ports {arduino_d[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[4]}]
set_property PACKAGE_PIN T15 [get_ports {arduino_d[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[5]}]
set_property PACKAGE_PIN R16 [get_ports {arduino_d[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[6]}]
set_property PACKAGE_PIN U17 [get_ports {arduino_d[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_d[7]}]

# Arduino Analog pins (can be used as digital)
set_property PACKAGE_PIN F19 [get_ports {arduino_a[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_a[0]}]
set_property PACKAGE_PIN F20 [get_ports {arduino_a[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_a[1]}]
set_property PACKAGE_PIN C20 [get_ports {arduino_a[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_a[2]}]
set_property PACKAGE_PIN B20 [get_ports {arduino_a[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_a[3]}]
set_property PACKAGE_PIN B19 [get_ports {arduino_a[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_a[4]}]
set_property PACKAGE_PIN A20 [get_ports {arduino_a[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {arduino_a[5]}]

# Raspberry Pi Header pins (for additional I/O)
set_property PACKAGE_PIN W18 [get_ports {rpi_gpio[2]}]   # GPIO2 (SDA)
set_property IOSTANDARD LVCMOS33 [get_ports {rpi_gpio[2]}]
set_property PACKAGE_PIN W19 [get_ports {rpi_gpio[3]}]   # GPIO3 (SCL)
set_property IOSTANDARD LVCMOS33 [get_ports {rpi_gpio[3]}]
set_property PACKAGE_PIN Y18 [get_ports {rpi_gpio[4]}]   # GPIO4
set_property IOSTANDARD LVCMOS33 [get_ports {rpi_gpio[4]}]
set_property PACKAGE_PIN Y19 [get_ports {rpi_gpio[17]}]  # GPIO17
set_property IOSTANDARD LVCMOS33 [get_ports {rpi_gpio[17]}]
set_property PACKAGE_PIN U18 [get_ports {rpi_gpio[27]}]  # GPIO27
set_property IOSTANDARD LVCMOS33 [get_ports {rpi_gpio[27]}]
set_property PACKAGE_PIN U19 [get_ports {rpi_gpio[22]}]  # GPIO22
set_property IOSTANDARD LVCMOS33 [get_ports {rpi_gpio[22]}]

# HDMI Output (for display interface)
set_property PACKAGE_PIN H17 [get_ports {hdmi_clk_p}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_clk_p}]
set_property PACKAGE_PIN H18 [get_ports {hdmi_clk_n}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_clk_n}]

set_property PACKAGE_PIN D17 [get_ports {hdmi_d0_p}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_d0_p}]
set_property PACKAGE_PIN D18 [get_ports {hdmi_d0_n}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_d0_n}]

set_property PACKAGE_PIN C17 [get_ports {hdmi_d1_p}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_d1_p}]
set_property PACKAGE_PIN C18 [get_ports {hdmi_d1_n}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_d1_n}]

set_property PACKAGE_PIN E17 [get_ports {hdmi_d2_p}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_d2_p}]
set_property PACKAGE_PIN E18 [get_ports {hdmi_d2_n}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_d2_n}]

# Audio codec (for audio I/O)
set_property PACKAGE_PIN M17 [get_ports {ac_adc_sdata}]
set_property IOSTANDARD LVCMOS33 [get_ports {ac_adc_sdata}]
set_property PACKAGE_PIN M18 [get_ports {ac_dac_sdata}]
set_property IOSTANDARD LVCMOS33 [get_ports {ac_dac_sdata}]
set_property PACKAGE_PIN N18 [get_ports {ac_bclk}]
set_property IOSTANDARD LVCMOS33 [get_ports {ac_bclk}]
set_property PACKAGE_PIN L17 [get_ports {ac_lrclk}]
set_property IOSTANDARD LVCMOS33 [get_ports {ac_lrclk}]
set_property PACKAGE_PIN K18 [get_ports {ac_mclk}]
set_property IOSTANDARD LVCMOS33 [get_ports {ac_mclk}]

# Timing constraints for custom ISA extensions
# These constraints ensure proper timing for neural network operations

# RISC-V core timing constraints
set_max_delay -from [get_cells -hier -filter {NAME =~ "*riscv_core*"}] -to [get_cells -hier -filter {NAME =~ "*custom_isa*"}] 10.0
set_min_delay -from [get_cells -hier -filter {NAME =~ "*riscv_core*"}] -to [get_cells -hier -filter {NAME =~ "*custom_isa*"}] 1.0

# Neural accelerator timing constraints
set_max_delay -from [get_cells -hier -filter {NAME =~ "*neural_accel*"}] -to [get_cells -hier -filter {NAME =~ "*axi*"}] 8.0
set_min_delay -from [get_cells -hier -filter {NAME =~ "*neural_accel*"}] -to [get_cells -hier -filter {NAME =~ "*axi*"}] 0.5

# Memory interface timing constraints
set_max_delay -from [get_cells -hier -filter {NAME =~ "*axi_master*"}] -to [get_ports -filter {NAME =~ "*m_axi*"}] 6.0
set_min_delay -from [get_cells -hier -filter {NAME =~ "*axi_master*"}] -to [get_ports -filter {NAME =~ "*m_axi*"}] 0.2

# False path constraints for asynchronous signals
set_false_path -from [get_ports {btn[*]}]
set_false_path -from [get_ports {sw[*]}]
set_false_path -to [get_ports {led[*]}]
set_false_path -to [get_ports {rgb_led*[*]}]

# Multi-cycle path constraints for complex operations
# Convolution operations can take multiple cycles
set_multicycle_path -setup 4 -from [get_cells -hier -filter {NAME =~ "*vconv*"}] -to [get_cells -hier -filter {NAME =~ "*result*"}]
set_multicycle_path -hold 3 -from [get_cells -hier -filter {NAME =~ "*vconv*"}] -to [get_cells -hier -filter {NAME =~ "*result*"}]

# GEMM operations can take even more cycles
set_multicycle_path -setup 8 -from [get_cells -hier -filter {NAME =~ "*gemm*"}] -to [get_cells -hier -filter {NAME =~ "*result*"}]
set_multicycle_path -hold 7 -from [get_cells -hier -filter {NAME =~ "*gemm*"}] -to [get_cells -hier -filter {NAME =~ "*result*"}]

# Power optimization constraints
set_operating_conditions -ambient_temp 25.0 -board_temp 25.0 -junction_temp 85.0
set_power_opt -include_clock_gating

# Placement constraints for critical paths
# Keep neural accelerator components close together
create_pblock pblock_neural_accel
add_cells_to_pblock [get_pblocks pblock_neural_accel] [get_cells -hier -filter {NAME =~ "*neural_accel*"}]
resize_pblock [get_pblocks pblock_neural_accel] -add {SLICE_X50Y50:SLICE_X100Y100}
resize_pblock [get_pblocks pblock_neural_accel] -add {DSP48_X2Y20:DSP48_X4Y40}
resize_pblock [get_pblocks pblock_neural_accel] -add {RAMB36_X2Y10:RAMB36_X4Y20}

# Keep RISC-V core components together
create_pblock pblock_riscv_core
add_cells_to_pblock [get_pblocks pblock_riscv_core] [get_cells -hier -filter {NAME =~ "*riscv_core*"}]
resize_pblock [get_pblocks pblock_riscv_core] -add {SLICE_X0Y50:SLICE_X40Y100}
resize_pblock [get_pblocks pblock_riscv_core] -add {DSP48_X0Y20:DSP48_X1Y40}
resize_pblock [get_pblocks pblock_riscv_core] -add {RAMB36_X0Y10:RAMB36_X1Y20}

# Configuration constraints
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]

# Bitstream generation options
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE YES [current_design]

# DRC (Design Rule Check) waivers for known issues
# Waive timing violations for debug signals
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]

# Performance optimization
set_param general.maxThreads 8
set_param synth.elaboration.rodinMoreOptions {rt::set_parameter ignoreVerilogCaseEqOperators true}

# Report generation
set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs impl_1]