# ===================================================
# create_pynq_project_college.tcl
# RISC-V ISA Extension Project for PYNQ-Z2
# Modified for College System (No Board Files Required)
# ===================================================

# Project settings
set project_name "riscv_isa_extension"
set project_dir "./riscv_project"
set part_name "xc7z020clg400-1"

# Get script directory and set paths
set script_dir [file dirname [file normalize [info script]]]
set vivado_root [file dirname $script_dir]

puts "=========================================="
puts "RISC-V ISA Extension Project Setup"
puts "=========================================="
puts "Script directory: $script_dir"
puts "Vivado root: $vivado_root"
puts "Target part: $part_name"
puts "=========================================="

# Clean up any existing project
if {[file exists $project_dir]} {
    puts "Removing existing project directory..."
    file delete -force $project_dir
}

# Create project
puts "Creating Vivado project..."
create_project $project_name $project_dir -part $part_name -force

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property default_lib xil_defaultlib [current_project]

# Add source files
puts "Adding HDL source files..."
set hdl_files [list \
    "$vivado_root/hdl/riscv_core_wrapper.v" \
    "$vivado_root/hdl/custom_isa_extensions.v" \
    "$vivado_root/hdl/neural_accelerator.v" \
    "$vivado_root/hdl/axi_neural_accel.v" \
    "$vivado_root/hdl/bd_neural_accel.v" \
    "$vivado_root/hdl/axi4_lite_slave.v" \
    "$vivado_root/hdl/axi4_master.v" \
    "$vivado_root/hdl/axi4_master_neural.v" \
]

foreach hdl_file $hdl_files {
    if {[file exists $hdl_file]} {
        add_files -norecurse $hdl_file
        puts "  Added: [file tail $hdl_file]"
    } else {
        puts "  WARNING: File not found: $hdl_file"
    }
}

# Add constraint files
puts "Adding constraint files..."
set constraint_file "$vivado_root/constraints/pynq_z2.xdc"
if {[file exists $constraint_file]} {
    add_files -fileset constrs_1 -norecurse $constraint_file
    puts "  Added: [file tail $constraint_file]"
} else {
    puts "  WARNING: Constraint file not found: $constraint_file"
}

# Enable SystemVerilog support for files that use SystemVerilog constructs
puts "Configuring SystemVerilog support..."
set_property file_type SystemVerilog [get_files *custom_isa_extensions.v]
set_property file_type SystemVerilog [get_files *neural_accelerator.v]
puts "  SystemVerilog enabled for custom_isa_extensions.v and neural_accelerator.v"

# Set top module
set_property top riscv_core_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "=========================================="
puts "Creating Block Design..."
puts "=========================================="

# Create block design
create_bd_design "system"

# Add Zynq Processing System
puts "Adding Zynq Processing System..."
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Configure Zynq PS for PYNQ-Z2 (without board files)
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
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_CLK1_PORT {1} \
    CONFIG.PCW_EN_CLK2_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
] [get_bd_cells processing_system7_0]

# Make external ports for DDR and FIXED_IO
puts "Creating external ports..."
make_bd_intf_pins_external [get_bd_intf_pins processing_system7_0/DDR]
set_property name DDR [get_bd_intf_ports DDR_0]
make_bd_intf_pins_external [get_bd_intf_pins processing_system7_0/FIXED_IO]
set_property name FIXED_IO [get_bd_intf_ports FIXED_IO_0]

# Add RISC-V core wrapper
puts "Adding RISC-V core wrapper..."
create_bd_cell -type module -reference riscv_core_wrapper riscv_core_0

# Add neural network accelerator
puts "Adding neural network accelerator..."
# Create the cell using the module name directly
create_bd_cell -type module -reference bd_neural_accel neural_accel_0

# Set frequency for m_axi interface to match interconnect
set_property -dict [list CONFIG.FREQ_HZ {200000000}] [get_bd_intf_pins neural_accel_0/m_axi]

# Add AXI Interconnect for GP0
puts "Adding AXI interconnects..."
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {2}] [get_bd_cells axi_interconnect_0]

# Add AXI Interconnect for HP0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_1]

# Add clock wizard for multiple clock domains
puts "Adding clock wizard..."
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.CLKOUT3_USED {true} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100.000} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {300.000} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
] [get_bd_cells clk_wiz_0]

puts "Connecting clock and reset signals..."
# Connect clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins clk_wiz_0/resetn]

# Connect RISC-V core - use 200MHz for everything to avoid clock domain issues
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins riscv_core_0/clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins riscv_core_0/resetn]

# Connect neural accelerator
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins neural_accel_0/s_axi_aclk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins neural_accel_0/s_axi_aresetn]

puts "Connecting AXI interfaces..."
# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins riscv_core_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M01_AXI] [get_bd_intf_pins neural_accel_0/s_axi]

# Connect HP interfaces for high-performance data transfer
connect_bd_intf_net [get_bd_intf_pins neural_accel_0/m_axi] [get_bd_intf_pins axi_interconnect_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins riscv_core_0/m_axi] [get_bd_intf_pins axi_interconnect_1/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# Connect clocks to AXI interconnects - all at 200MHz
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M01_ACLK]

connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/S00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/S01_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_1/M00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

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
puts "Connecting interrupts..."
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0
set_property -dict [list CONFIG.NUM_PORTS {4}] [get_bd_cells xlconcat_0]

connect_bd_net [get_bd_pins riscv_core_0/interrupt] [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins neural_accel_0/interrupt] [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins processing_system7_0/IRQ_F2P]

puts "Assigning addresses..."
# Assign addresses
assign_bd_address [get_bd_addr_segs {riscv_core_0/s_axi/reg0 }]
assign_bd_address [get_bd_addr_segs {neural_accel_0/s_axi/reg0 }]
assign_bd_address [get_bd_addr_segs {processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM }]

# Set address ranges
set_property range 64K [get_bd_addr_segs {processing_system7_0/Data/SEG_riscv_core_0_reg0}]
set_property range 64K [get_bd_addr_segs {processing_system7_0/Data/SEG_neural_accel_0_reg0}]

puts "Validating block design..."
# Validate design
validate_bd_design

# Save block design
save_bd_design

puts "Creating HDL wrapper..."
# Create HDL wrapper
make_wrapper -files [get_files $project_dir/$project_name.srcs/sources_1/bd/system/system.bd] -top
add_files -norecurse $project_dir/$project_name.srcs/sources_1/bd/system/hdl/system_wrapper.v
set_property top system_wrapper [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

# Generate output products
puts "Generating output products..."
generate_target all [get_files $project_dir/$project_name.srcs/sources_1/bd/system/system.bd]

puts "=========================================="
puts "Project created successfully!"
puts "=========================================="
puts "Project location: [file normalize $project_dir]"
puts ""
puts "Next steps:"
puts "1. Open project: vivado $project_dir/$project_name.xpr"
puts "2. Run synthesis: launch_runs synth_1 -jobs 8"
puts "3. Run implementation: launch_runs impl_1 -to_step write_bitstream -jobs 8"
puts "4. Export hardware: write_hw_platform -fixed -include_bit -force -file system.xsa"
puts "=========================================="
