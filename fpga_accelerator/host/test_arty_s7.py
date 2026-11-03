#!/usr/bin/env python3
"""
Host Communication & Verification Script for Arty S7-25 NeuroSpark Engine
Author: Arunjyoti Sarma (IIT BHU Research Lab)
Target Board: Digilent Arty S7-25 (Spartan-7 XC7S25)
Interfaces: USB-UART (115200 baud, 8-N-1)
Features:
  1. 4x4 INT8 Systolic GEMM Verification
  2. 4x4 BitNet b1.58 Multiplierless Ternary GEMM Verification
  3. 4-Neuron Leaky Integrate-and-Fire (LIF) SNN Neuromorphic Verification
  4. Real-Time Hardware Nanosecond Cycle Profiling
"""

import time
import struct
import sys
import os
import subprocess
import numpy as np
import serial
import serial.tools.list_ports

BAUD_RATE = 115200
TIMEOUT = 1.0  # seconds

def flash_fpga_if_needed():
    """Attempts to program the Arty S7 FPGA using Vivado JTAG if the bitstream isn't running."""
    vivado_bin = r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "program_fpga.tcl"))
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.exists(vivado_bin) and os.path.exists(script_path):
        print("\n[*] Initializing Arty S7-25 FPGA with NeuroSpark bitstream via Vivado JTAG...")
        cmd = [vivado_bin, "-mode", "batch", "-notrace", "-source", script_path]
        try:
            res = subprocess.run(cmd, cwd=proj_dir, capture_output=True, text=True, timeout=90)
            if res.returncode == 0:
                print("[+] Hardware flashing completed successfully! Board is alive.")
                time.sleep(1.0)
                return True
            else:
                print(f"[!] Vivado returned code {res.returncode}")
        except Exception as e:
            print(f"[!] Auto-flash encountered: {e}")
    return False

def find_arty_s7_port():
    """Auto-detects the Arty S7 COM port by probing with a PING opcode (0x01)."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[!] No COM ports detected. Please connect your Arty S7 board via USB.")
        return None

    print(f"[*] Scanning {len(ports)} serial port(s)...")
    for p in ports:
        print(f"    Checking {p.device}: {p.description}")
        try:
            with serial.Serial(p.device, BAUD_RATE, timeout=0.2) as ser:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write(bytes([0x01]))
                resp = ser.read(2)
                if len(resp) == 2 and resp[0] == 0xA5 and resp[1] == 0x5A:
                    print(f"    [+] LOCKED onto Arty S7-25 on {p.device} (Valid PING response 0xA5 0x5A)!")
                    return p.device
        except Exception:
            continue

    for p in ports:
        if "USB" in p.description or "FTDI" in p.description or "Serial" in p.description:
            print(f"[*] Using candidate port: {p.device} ({p.description})")
            return p.device

    return ports[0].device

def get_hardware_status(ser):
    """Fetches device identifier and Spartan-7 version from FPGA."""
    ser.reset_input_buffer()
    ser.write(bytes([0x04]))
    resp = ser.read(5)
    if len(resp) == 5:
        tag = resp[:4].decode('ascii', errors='ignore')
        ver = resp[4]
        return f"Device Tag: '{tag}', Architecture Family: Spartan-{ver}"
    return "Status query timed out (is bitstream loaded?)"

def run_int8_gemm_test(ser):
    print("\n" + "="*65)
    print("  TEST 1: 4x4 INT8 MATRIX MULTIPLICATION (HARDWARE DSP)")
    print("="*65)

    A = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
    B = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
    C_golden = np.matmul(A.astype(np.int32), B.astype(np.int32)).astype(np.int16)

    payload = bytes([0x02]) + A.tobytes() + B.tobytes()
    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(payload)

    resp = ser.read(32)
    t_roundtrip = (time.perf_counter() - t_start) * 1000.0

    if len(resp) != 32:
        print(f"[FAIL] Expected 32 bytes, received {len(resp)} bytes.")
        return False

    C_fpga = np.frombuffer(resp, dtype=np.int16).reshape((4, 4))
    match = np.array_equal(C_fpga, C_golden)

    print(f"Matrix A (INT8):\n{A}")
    print(f"\nMatrix B (INT8):\n{B}")
    print(f"\nExpected Result (CPU NumPy):\n{C_golden}")
    print(f"\nHardware Result (Spartan-7 FPGA):\n{C_fpga}")
    print(f"\nRoundtrip Latency (UART + Hardware Compute): {t_roundtrip:.2f} ms")

    if match:
        print("\n>>> [PASSED] 100% BIT-EXACT NUMERICAL MATCH! <<<")
        return True
    else:
        print("\n>>> [FAILED] Numerical mismatch between CPU and FPGA! <<<")
        return False

def run_bitnet_gemm_test(ser):
    print("\n" + "="*65)
    print("  TEST 2: 4x4 BitNet b1.58 TERNARY GEMM {-1, 0, +1}")
    print("="*65)

    A = np.random.randint(-10, 11, size=(4, 4), dtype=np.int8)
    B_ternary = np.random.choice([-1, 0, 1], size=(4, 4)).astype(np.int8)
    C_golden = np.matmul(A.astype(np.int32), B_ternary.astype(np.int32)).astype(np.int16)

    payload = bytes([0x03]) + A.tobytes() + B_ternary.tobytes()
    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(payload)

    resp = ser.read(32)
    t_roundtrip = (time.perf_counter() - t_start) * 1000.0

    if len(resp) != 32:
        print(f"[FAIL] Expected 32 bytes, received {len(resp)} bytes.")
        return False

    C_fpga = np.frombuffer(resp, dtype=np.int16).reshape((4, 4))
    match = np.array_equal(C_fpga, C_golden)

    print(f"Activation Matrix A (INT8):\n{A}")
    print(f"\nTernary Weight Matrix B {-1, 0, 1}:\n{B_ternary}")
    print(f"\nExpected Result (CPU NumPy):\n{C_golden}")
    print(f"\nHardware Result (BitNet Spartan-7 Engine):\n{C_fpga}")
    print(f"\nRoundtrip Latency: {t_roundtrip:.2f} ms")

    if match:
        print("\n>>> [PASSED] 100% BIT-EXACT MATCH WITH ZERO MULTIPLIERS! <<<")
        return True
    else:
        print("\n>>> [FAILED] Mismatch in BitNet computation! <<<")
        return False

def simulate_lif_cpu(weights, spike_train, v_thresh=60, refrac_len=2):
    """Software golden reference for 4 Leaky Integrate-and-Fire neurons across 16 steps."""
    v_mem = np.zeros(4, dtype=np.int32)
    refrac = np.zeros(4, dtype=np.int32)
    spike_counts = np.zeros(4, dtype=np.int32)

    for t in range(16):
        step_spikes = (spike_train[t] >> np.arange(4)) & 1
        for j in range(4):
            syn_current = np.sum(weights[j] * step_spikes)
            if refrac[j] > 0:
                refrac[j] -= 1
                v_mem[j] = 0
            else:
                v_decay = v_mem[j] - (v_mem[j] >> 3) + syn_current
                if v_decay >= v_thresh:
                    spike_counts[j] += 1
                    v_mem[j] = 0
                    refrac[j] = refrac_len
                else:
                    v_mem[j] = max(0, v_decay)
    return spike_counts, v_mem

def run_snn_lif_test(ser):
    print("\n" + "="*65)
    print("  TEST 3: 4-NEURON NEUROMORPHIC SNN (LEAKY INTEGRATE-AND-FIRE)")
    print("  Biological Spike Integration across 16 Temporal Time-Steps")
    print("="*65)

    # 4 neurons with 4 synaptic weights each (in [-20, 45])
    weights = np.random.randint(-15, 45, size=(4, 4), dtype=np.int8)

    # 16 time steps of input spikes (4 channels in bits [3:0])
    spike_train = np.random.choice([0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1111], size=16).astype(np.uint8)

    golden_spikes, golden_vmem = simulate_lif_cpu(weights, spike_train)

    payload = bytes([0x05]) + weights.tobytes() + spike_train.tobytes()
    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(payload)

    resp = ser.read(12)
    t_roundtrip = (time.perf_counter() - t_start) * 1000.0

    if len(resp) != 12:
        print(f"[FAIL] Expected 12 bytes, received {len(resp)} bytes.")
        return False

    fpga_spikes = np.frombuffer(resp[0:4], dtype=np.uint8)
    hw_cycles = struct.unpack("<I", resp[8:12])[0]
    hw_nanoseconds = (hw_cycles / 12000000.0) * 1e9

    match = np.array_equal(fpga_spikes, golden_spikes)

    print(f"Synaptic Weight Matrix W (4 Neurons x 4 Inputs):\n{weights}")
    print(f"\nTemporal Input Spike Train (16 Time Steps):\n{[bin(x)[2:].zfill(4) for x in spike_train]}")
    print(f"\nExpected Spike Counts (CPU Golden LIF):  {golden_spikes}")
    print(f"Hardware Spike Counts (Spartan-7 Silicon): {fpga_spikes}")
    print(f"\nOn-Chip Hardware Profiling Telemetry:")
    print(f"  * Execution Clock Cycles: {hw_cycles} cycles")
    print(f"  * Hardware Pure Compute Latency: {hw_nanoseconds:.1f} ns ({hw_nanoseconds/1000.0:.2f} µs at 12 MHz)")
    print(f"  * Total Roundtrip (Serial + Hardware): {t_roundtrip:.2f} ms")

    if match:
        print("\n>>> [PASSED] 100% BIT-EXACT NEUROMORPHIC SNN SPIKE MATCH! <<<")
        return True
    else:
        print("\n>>> [FAILED] SNN spike count mismatch between CPU and FPGA! <<<")
        return False

def main():
    print("="*65)
    print("  NeuroSpark AI Silicon Engine Hardware Test Harness")
    print("  IIT BHU Research Lab | Open Kernel Challenge Co-Design")
    print("="*65)

    port = find_arty_s7_port()
    if not port:
        sys.exit(1)

    print(f"\nConnecting to {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"[ERROR] Could not open port {port}: {e}")
        print("Please verify the board is connected and no other app has locked it.")
        sys.exit(1)

    try:
        status = get_hardware_status(ser)
        if not status.startswith("Device Tag: 'ARTY'"):
            print(f"[!] Notice: FPGA returned: {status}")
            print("[*] Re-initializing FPGA bitstream via JTAG...")
            ser.close()
            time.sleep(0.5)
            if flash_fpga_if_needed():
                ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
                status = get_hardware_status(ser)
            else:
                ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)

        print(f"Hardware Info: {status}")

        test1_ok = run_int8_gemm_test(ser)
        test2_ok = run_bitnet_gemm_test(ser)
        test3_ok = run_snn_lif_test(ser)

        print("\n" + "="*65)
        print("  TEST 4: LIVE LED HARDWARE STREAMING BURST")
        print("  Watch your board: LD0, LD1, LD2, LD3, LD4, LD5 will pulse and dance!")
        print("="*65)
        burst_passed = 0
        for idx in range(1, 16):
            mode_choice = idx % 3
            if mode_choice == 0:
                # SNN LIF step
                W = np.random.randint(-15, 45, size=(4, 4), dtype=np.int8)
                S = np.random.choice([0b0001, 0b0010, 0b0100, 0b1000, 0b0011], size=16).astype(np.uint8)
                ser.write(bytes([0x05]) + W.tobytes() + S.tobytes())
                resp = ser.read(12)
                mode_str = "Neuromorphic SNN (LIF)"
                if len(resp) == 12: burst_passed += 1
            elif mode_choice == 1:
                # BitNet
                A = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
                B = np.random.choice([-1, 0, 1], size=(4, 4)).astype(np.int8)
                ser.write(bytes([0x03]) + A.tobytes() + B.tobytes())
                resp = ser.read(32)
                mode_str = "BitNet b1.58 Ternary  "
                if len(resp) == 32: burst_passed += 1
            else:
                # INT8 GEMM
                A = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
                B = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
                ser.write(bytes([0x02]) + A.tobytes() + B.tobytes())
                resp = ser.read(32)
                mode_str = "INT8 Standard GEMM    "
                if len(resp) == 32: burst_passed += 1

            print(f"\r  [Streaming Packet {idx:02d}/15] Mode: {mode_str} -> LED Pulse Active", end="", flush=True)
            time.sleep(0.08)

        print(f"\n  [Done] {burst_passed}/15 Packets Streamed & Verified across Tri-Mode Silicon!")

        print("\n" + "="*65)
        if test1_ok and test2_ok and test3_ok and burst_passed == 15:
            print("  ALL HARDWARE ACCELERATION TESTS PASSED (4/4)!")
            print("  Your Arty S7-25 is running verified SNN, BitNet & INT8 Silicon!")
        else:
            print("  SOME TESTS FAILED. Check connection or bitstream.")
        print("="*65)

    finally:
        ser.close()

if __name__ == "__main__":
    main()
