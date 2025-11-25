# Simple Build Script for RISC-V ISA Extension Project
puts "=========================================="
puts "Starting Build Process"
puts "=========================================="

# Set project variables
set project_name "riscv_isa_extension"
set project_dir "./riscv_project"
set num_jobs 8

# Open project
puts "Opening project..."
open_project $project_dir/$project_name.xpr

# Update compile order
update_compile_order -fileset sources_1

# Run synthesis
puts "Starting synthesis (this will take 30-90 minutes)..."
reset_run synth_1
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1

# Check synthesis results
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}
puts "Synthesis completed successfully!"

# Run implementation
puts "Starting implementation (this will take 45-150 minutes)..."
reset_run impl_1
launch_runs impl_1 -jobs $num_jobs
wait_on_run impl_1

# Check implementation results
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}
puts "Implementation completed successfully!"

# Generate bitstream
puts "Generating bitstream (this will take 5-15 minutes)..."
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
open_run impl_1
write_hw_platform -fixed -include_bit -force -file system_wrapper.xsa

puts "=========================================="
puts "Build completed successfully!"
puts "=========================================="
puts "Output files:"
puts "  Bitstream: $project_dir/$project_name.runs/impl_1/system_wrapper.bit"
puts "  Hardware handoff: $project_dir/$project_name.runs/impl_1/system_wrapper.hwh"
puts "  Hardware platform: system_wrapper.xsa"
puts "=========================================="
