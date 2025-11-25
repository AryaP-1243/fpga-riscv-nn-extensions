# RISC-V ISA Extension Tool - Automated Build Script
# This script performs complete synthesis, implementation, and bitstream generation

# Set build configuration
set build_config "release"
set enable_debug false
set num_jobs 8

puts "Starting RISC-V ISA Extension Tool build process..."
puts "Build configuration: $build_config"
puts "Debug enabled: $enable_debug"
puts "Parallel jobs: $num_jobs"

# Set project variables
set project_name "riscv_isa_extension"
set project_dir "./riscv_project"

# Open existing project
open_project $project_dir/$project_name.xpr

# Update compile order
update_compile_order -fileset sources_1

# Set synthesis options - simplified for compatibility
set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]

# Set implementation options
set_property strategy "Vivado Implementation Defaults" [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

# Enable parallel processing
set_param general.maxThreads $num_jobs

# Run synthesis
puts "Starting synthesis..."
reset_run synth_1
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1

# Check synthesis results
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

puts "Synthesis completed successfully!"

# Generate synthesis reports
open_run synth_1 -name synth_1
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1
report_utilization -name utilization_1
report_power -name power_1

# Run implementation
puts "Starting implementation..."
reset_run impl_1
launch_runs impl_1 -jobs $num_jobs
wait_on_run impl_1

# Check implementation results
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

puts "Implementation completed successfully!"

# Generate implementation reports
open_run impl_1
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_impl
report_utilization -name utilization_impl
report_power -name power_impl
report_route_status -name route_status
report_drc -name drc_impl
report_methodology -name methodology_impl

# Check timing
set timing_met [get_property STATS.WNS [get_runs impl_1]]
if {$timing_met < 0} {
    puts "WARNING: Timing not met! WNS = $timing_met ns"
} else {
    puts "Timing constraints met! WNS = $timing_met ns"
}

# Generate bitstream
puts "Generating bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
wait_on_run impl_1

# Check bitstream generation
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Bitstream generation failed!"
    exit 1
}

puts "Bitstream generated successfully!"

# Export hardware platform
puts "Exporting hardware platform..."
write_hw_platform -fixed -include_bit -force -file $project_dir/system_wrapper.xsa

# Generate final reports
puts "Generating final reports..."

# Resource utilization summary
set util_file [open "$project_dir/utilization_summary.txt" w]
puts $util_file "RISC-V ISA Extension Tool - Resource Utilization Summary"
puts $util_file "============================================================"
puts $util_file ""

# Get utilization data
set slice_luts [get_property USED [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ LUT*}]]
set slice_regs [get_property USED [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ FD*}]]
set bram_tiles [get_property USED [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ RAMB*}]]
set dsp_blocks [get_property USED [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP*}]]

puts $util_file "LUTs Used: $slice_luts"
puts $util_file "Registers Used: $slice_regs"
puts $util_file "BRAM Tiles Used: $bram_tiles"
puts $util_file "DSP Blocks Used: $dsp_blocks"
puts $util_file ""

# Performance summary
puts $util_file "Performance Summary"
puts $util_file "==================="
puts $util_file "Worst Negative Slack (WNS): $timing_met ns"

set max_freq [expr 1000.0 / [get_property REQUESTED_PERIOD [get_clocks]]]
puts $util_file "Maximum Frequency: $max_freq MHz"
puts $util_file ""

# Power summary
set total_power [get_property TOTAL_POWER [get_power_result]]
puts $util_file "Total Power: $total_power W"

close $util_file

# Create deployment package
puts "Creating deployment package..."
file mkdir $project_dir/deployment
file copy -force $project_dir/system_wrapper.xsa $project_dir/deployment/
file copy -force $project_dir/$project_name.runs/impl_1/system_wrapper.bit $project_dir/deployment/
file copy -force $project_dir/utilization_summary.txt $project_dir/deployment/

# Generate programming files for different boot modes
puts "Generating programming files..."

# Generate .bin file for SD card boot
write_cfgmem -format bin -interface spix4 -size 16 -loadbit "up 0x0 $project_dir/$project_name.runs/impl_1/system_wrapper.bit" -file "$project_dir/deployment/boot.bin"

# Generate .mcs file for QSPI flash
write_cfgmem -format mcs -interface spix4 -size 16 -loadbit "up 0x0 $project_dir/$project_name.runs/impl_1/system_wrapper.bit" -file "$project_dir/deployment/system_wrapper.mcs"

# Create build summary
set summary_file [open "$project_dir/deployment/build_summary.txt" w]
puts $summary_file "RISC-V ISA Extension Tool - Build Summary"
puts $summary_file "========================================"
puts $summary_file ""
puts $summary_file "Build Date: [clock format [clock seconds]]"
puts $summary_file "Vivado Version: [version -short]"
puts $summary_file "Project: $project_name"
puts $summary_file "Target Device: [get_property PART [current_project]]"
puts $summary_file "Board: [get_property BOARD_PART [current_project]]"
puts $summary_file ""
puts $summary_file "Build Results:"
puts $summary_file "- Synthesis: PASSED"
puts $summary_file "- Implementation: PASSED"
puts $summary_file "- Bitstream Generation: PASSED"
puts $summary_file "- Timing: [expr {$timing_met >= 0 ? "MET" : "NOT MET"}]"
puts $summary_file ""
puts $summary_file "Generated Files:"
puts $summary_file "- system_wrapper.bit (FPGA bitstream)"
puts $summary_file "- system_wrapper.xsa (Hardware platform)"
puts $summary_file "- boot.bin (SD card boot file)"
puts $summary_file "- system_wrapper.mcs (QSPI flash file)"
puts $summary_file ""
puts $summary_file "Resource Utilization:"
puts $summary_file "- LUTs: $slice_luts"
puts $summary_file "- Registers: $slice_regs"
puts $summary_file "- BRAM: $bram_tiles"
puts $summary_file "- DSP: $dsp_blocks"
puts $summary_file ""
puts $summary_file "Performance:"
puts $summary_file "- WNS: $timing_met ns"
puts $summary_file "- Max Frequency: $max_freq MHz"
puts $summary_file "- Total Power: $total_power W"

close $summary_file

# Generate Verilog netlist for verification
puts "Generating post-implementation netlist..."
write_verilog -force -mode timesim -sdf_anno true $project_dir/deployment/system_wrapper_timesim.v
write_sdf -force $project_dir/deployment/system_wrapper_timesim.sdf

# Create PYNQ overlay files
puts "Creating PYNQ overlay files..."
file mkdir $project_dir/deployment/pynq_overlay

# Copy bitstream with .bit extension for PYNQ
file copy -force $project_dir/$project_name.runs/impl_1/system_wrapper.bit $project_dir/deployment/pynq_overlay/riscv_isa_extension.bit

# Copy hardware handoff file
file copy -force $project_dir/system_wrapper.xsa $project_dir/deployment/pynq_overlay/riscv_isa_extension.hwh

# Create PYNQ overlay Python template
set pynq_py_file [open "$project_dir/deployment/pynq_overlay/riscv_isa_extension.py" w]
puts $pynq_py_file "# RISC-V ISA Extension Tool - PYNQ Overlay"
puts $pynq_py_file "# Auto-generated overlay interface"
puts $pynq_py_file ""
puts $pynq_py_file "from pynq import Overlay, allocate"
puts $pynq_py_file "import numpy as np"
puts $pynq_py_file ""
puts $pynq_py_file "class RiscvIsaExtensionOverlay(Overlay):"
puts $pynq_py_file "    def __init__(self, bitfile_name='riscv_isa_extension.bit', **kwargs):"
puts $pynq_py_file "        super().__init__(bitfile_name, **kwargs)"
puts $pynq_py_file "        "
puts $pynq_py_file "        # Initialize accelerator components"
puts $pynq_py_file "        self.neural_accel = self.axi_neural_accel_0"
puts $pynq_py_file "        self.riscv_core = self.riscv_core_wrapper_0"
puts $pynq_py_file "        "
puts $pynq_py_file "    def run_neural_inference(self, input_data, weights, operation_type='conv2d'):"
puts $pynq_py_file "        \"\"\"Run neural network inference on FPGA\"\"\""
puts $pynq_py_file "        # Allocate buffers"
puts $pynq_py_file "        input_buffer = allocate(shape=input_data.shape, dtype=np.float32)"
puts $pynq_py_file "        output_buffer = allocate(shape=input_data.shape, dtype=np.float32)"
puts $pynq_py_file "        "
puts $pynq_py_file "        # Copy input data"
puts $pynq_py_file "        input_buffer[:] = input_data"
puts $pynq_py_file "        "
puts $pynq_py_file "        # Configure accelerator"
puts $pynq_py_file "        self.neural_accel.write(0x08, 0x01)  # Operation type: CONV2D"
puts $pynq_py_file "        self.neural_accel.write(0x0C, input_data.shape[0])  # Height"
puts $pynq_py_file "        self.neural_accel.write(0x10, input_data.shape[1])  # Width"
puts $pynq_py_file "        self.neural_accel.write(0x14, input_data.shape[2])  # Channels"
puts $pynq_py_file "        "
puts $pynq_py_file "        # Start processing"
puts $pynq_py_file "        self.neural_accel.write(0x00, 0x01)  # Start"
puts $pynq_py_file "        "
puts $pynq_py_file "        # Wait for completion"
puts $pynq_py_file "        while not (self.neural_accel.read(0x04) & 0x01):"
puts $pynq_py_file "            pass"
puts $pynq_py_file "        "
puts $pynq_py_file "        return np.array(output_buffer)"
puts $pynq_py_file "        "
puts $pynq_py_file "    def get_performance_counters(self):"
puts $pynq_py_file "        \"\"\"Get performance monitoring data\"\"\""
puts $pynq_py_file "        cycle_count = self.neural_accel.read(0x34)"
puts $pynq_py_file "        op_count = self.neural_accel.read(0x38)"
puts $pynq_py_file "        return {'cycles': cycle_count, 'operations': op_count}"

close $pynq_py_file

puts ""
puts "=============================================="
puts "Build completed successfully!"
puts "=============================================="
puts ""
puts "Generated files in $project_dir/deployment/:"
puts "- system_wrapper.bit (FPGA bitstream)"
puts "- system_wrapper.xsa (Hardware platform)"
puts "- boot.bin (SD card boot file)"
puts "- system_wrapper.mcs (QSPI flash file)"
puts "- build_summary.txt (Build report)"
puts "- utilization_summary.txt (Resource usage)"
puts ""
puts "PYNQ overlay files in $project_dir/deployment/pynq_overlay/:"
puts "- riscv_isa_extension.bit (PYNQ bitstream)"
puts "- riscv_isa_extension.hwh (Hardware handoff)"
puts "- riscv_isa_extension.py (Python overlay class)"
puts ""
puts "Next steps:"
puts "1. Copy deployment files to PYNQ-Z2 board"
puts "2. Install PYNQ overlay: overlay = RiscvIsaExtensionOverlay()"
puts "3. Run neural network inference with custom ISA extensions"
puts ""
puts "For detailed deployment instructions, see:"
puts "- PYNQ_STARTUP_GUIDE.md"
puts "- pynq_deployment_guide.py"