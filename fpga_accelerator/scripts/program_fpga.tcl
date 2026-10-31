# Vivado JTAG Flashing Script for Arty S7-25
# Connects to local hw_server and programs the Spartan-7 device

set BIT_FILE [file normalize "build/bitgemm_s7_arty.bit"]

if {![file exists $BIT_FILE]} {
    puts "ERROR: Bitstream file $BIT_FILE not found! Run build.bat first."
    exit 1
}

puts "=========================================================="
puts "  CONNECTING TO ARTY S7-25 VIA JTAG"
puts "  Bitstream: $BIT_FILE"
puts "=========================================================="

open_hw_manager
connect_hw_server -allow_non_jtag

# Refresh and find targets
refresh_hw_server
set hw_targets [get_hw_targets]

if {[llength $hw_targets] == 0} {
    puts "ERROR: No JTAG targets found. Please make sure:"
    puts "  1. Arty S7 is plugged into USB."
    puts "  2. Power jumper (JP3) is on USB."
    puts "  3. Digilent Adept cable drivers are installed."
    exit 1
}

set target [lindex $hw_targets 0]
puts "Found target: $target"
open_hw_target $target

# Get device (Spartan-7)
set device [lindex [get_hw_devices] 0]
puts "Programming device: $device"

set_property PROGRAM.FILE $BIT_FILE $device
program_hw_devices $device
refresh_hw_device $device

puts "=========================================================="
puts "  SUCCESS: ARTY S7-25 PROGRAMMED SUCCESSFULLY!"
puts "  Check the board: LED 0 should now be blinking (heartbeat)."
puts "=========================================================="
close_hw_target
disconnect_hw_server
close_hw_manager
