# RISC-V ISA Extension Tool - PYNQ-Z2 Project Creation Script
# This script creates a complete Vivado project for PYNQ-Z2 with RISC-V core and custom instructions

# Set project variables
set project_name "riscv_isa_extension"
set project_dir "./vivado_project"
set part_name "xc7z020clg400-1"
set board_part "tul.com.tw:pynq-z2:part0:1.0"

# Create project
create_project $project_name $project_dir -part $part_name -force
set_property board_part $board_part [current_project]

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property default_lib xil_defaultlib [current_project]

# Add source files
add_files -norecurse {
    ./vivado_project/hdl/riscv_core_wrapper.v
    ./vivado_project/hdl/custom_isa_extensions.v
    ./vivado_project/hdl/neural_accelerator.v
    ./vivado_project/hdl/axi_neural_accel.v
}

# Add constraint files
add_files -fileset constrs_1 -norecurse ./vivado_project/constraints/pynq_z2.xdc

# Create block design
create_bd_design "system"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Configure Zynq PS for PYNQ-Z2
set_property -dict [list \
    CONFIG.PCW_PRESET_BANK0_VOLTAGE {LVCMOS33} \
    CONFIG.PCW_PRESET_BANK1_VOLTAGE {LVCMOS18} \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50.000000} \
    CONFIG.PCW_APU_PERIPHERAL_FREQMHZ {650.000000} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
    CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {150.000000} \
    CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {200.000000} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP1 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_CLK1_PORT {1} \
    CONFIG.PCW_EN_CLK2_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_EN_RST1_PORT {1} \
] [get_bd_cells processing_system7_0]

# Apply board preset
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells processing_system7_0]

# Add RISC-V core with custom ISA extensions
create_bd_cell -type module -reference riscv_core_wrapper riscv_core_0

# Add neural network accelerator
create_bd_cell -type module -reference axi_neural_accel neural_accel_0

# Add AXI Interconnect for GP0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {2}] [get_bd_cells axi_interconnect_0]

# Add AXI Interconnect for HP0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1
set_property -dict [list CONFIG.NUM_SI {2}] [get_bd_cells axi_interconnect_1]

# Add clock wizard for multiple clock domains
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.CLKOUT3_USED {true} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100.000} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {300.000} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
    CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKFBOUT_MULT_F {12.000} \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F {12.000} \
    CONFIG.MMCM_CLKOUT1_DIVIDE {6} \
    CONFIG.MMCM_CLKOUT2_DIVIDE {4} \
    CONFIG.CLKOUT1_JITTER {130.958} \
    CONFIG.CLKOUT2_JITTER {114.829} \
    CONFIG.CLKOUT3_JITTER {109.241} \
] [get_bd_cells clk_wiz_0]

# Connect clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins clk_wiz_0/resetn]

# Connect RISC-V core
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins riscv_core_0/clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins riscv_core_0/resetn]

# Connect neural accelerator
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins neural_accel_0/s_axi_aclk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins neural_accel_0/s_axi_aresetn]

# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins riscv_core_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M01_AXI] [get_bd_intf_pins neural_accel_0/s_axi]

# Connect HP interfaces for high-performance data transfer
connect_bd_intf_net [get_bd_intf_pins neural_accel_0/m_axi] [get_bd_intf_pins axi_interconnect_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins riscv_core_0/m_axi] [get_bd_intf_pins axi_interconnect_1/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# Connect clocks to AXI interconnects
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins axi_interconnect_0/M00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M01_ACLK]

connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/S00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins axi_interconnect_1/S01_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/M00_ACLK]

# Connect resets
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_0/M00_ARESETN]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_0/M01_ARESETN]

connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_1/ARESETN]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_1/S00_ARESETN]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_1/S01_ARESETN]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins axi_interconnect_1/M00_ARESETN]

# Connect interrupts
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0
set_property -dict [list CONFIG.NUM_PORTS {4}] [get_bd_cells xlconcat_0]

connect_bd_net [get_bd_pins riscv_core_0/interrupt] [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins neural_accel_0/interrupt] [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P_INTR]

# Assign addresses
assign_bd_address [get_bd_addr_segs {riscv_core_0/s_axi/reg0 }]
assign_bd_address [get_bd_addr_segs {neural_accel_0/s_axi/reg0 }]
assign_bd_address [get_bd_addr_segs {processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM }]

# Set address ranges
set_property range 64K [get_bd_addr_segs {processing_system7_0/Data/SEG_riscv_core_0_reg0}]
set_property range 64K [get_bd_addr_segs {processing_system7_0/Data/SEG_neural_accel_0_reg0}]

# Validate design
validate_bd_design

# Save block design
save_bd_design

# Create HDL wrapper
make_wrapper -files [get_files $project_dir/$project_name.srcs/sources_1/bd/system/system.bd] -top
add_files -norecurse $project_dir/$project_name.srcs/sources_1/bd/system/hdl/system_wrapper.v
set_property top system_wrapper [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

puts "PYNQ-Z2 project created successfully!"
puts "Project location: $project_dir/$project_name"
puts "Next steps:"
puts "1. Review the block design"
puts "2. Run synthesis: launch_runs synth_1"
puts "3. Run implementation: launch_runs impl_1 -to_step write_bitstream"
puts "4. Export hardware: write_hw_platform -fixed -include_bit -force -file system_wrapper.xsa"