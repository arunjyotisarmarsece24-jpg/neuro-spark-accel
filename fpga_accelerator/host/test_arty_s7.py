#!/usr/bin/env python3
"""
Host Communication & Verification Script for Arty S7-25 BitGEMM Accelerator
Author: Arunjyoti Sarma (IIT BHU Research Lab)
Target Board: Digilent Arty S7-25 (Spartan-7 XC7S25)
Interfaces: USB-UART (115200 baud, 8-N-1)
"""

import time
import struct
import sys
import numpy as np
import serial
import serial.tools.list_ports

BAUD_RATE = 115200
TIMEOUT = 1.0  # seconds

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
                # Flush buffers
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                # Send PING opcode
                ser.write(bytes([0x01]))
                resp = ser.read(2)
                if len(resp) == 2 and resp[0] == 0xA5 and resp[1] == 0x5A:
                    print(f"    [+] LOCKED onto Arty S7-25 on {p.device} (Valid PING response 0xA5 0x5A)!")
                    return p.device
        except Exception:
            continue

    # Fallback to first USB serial port if probe didn't answer yet (e.g. unprogrammed bitstream)
    for p in ports:
        if "USB" in p.description or "FTDI" in p.description or "Serial" in p.description:
            print(f"[*] Using candidate port: {p.device} ({p.description})")
            return p.device

    return ports[0].device

def get_hardware_status(ser):
    """Fetches device identifier and Spartan-7 version from FPGA."""
    ser.reset_input_buffer()
    ser.write(bytes([0x04]))  # Opcode 0x04: STATUS
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

    # Generate random 4x4 matrices with values in [-8, 7]
    A = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
    B = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
    
    # Golden reference computed on CPU (NumPy)
    C_golden = np.matmul(A.astype(np.int32), B.astype(np.int32)).astype(np.int16)

    # Pack 16 bytes A + 16 bytes B
    payload = bytes([0x02]) + A.tobytes() + B.tobytes()
    
    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(payload)

    # Expect 32 bytes back (16 signed 16-bit integers, Little-Endian)
    resp = ser.read(32)
    t_roundtrip = (time.perf_counter() - t_start) * 1000.0  # ms

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

    # Activations in [-10, 10]
    A = np.random.randint(-10, 11, size=(4, 4), dtype=np.int8)
    # Weights strictly ternary in {-1, 0, 1}
    B_ternary = np.random.choice([-1, 0, 1], size=(4, 4)).astype(np.int8)

    C_golden = np.matmul(A.astype(np.int32), B_ternary.astype(np.int32)).astype(np.int16)

    # Opcode 0x03: BitNet mode
    payload = bytes([0x03]) + A.tobytes() + B_ternary.tobytes()

    ser.reset_input_buffer()
    t_start = time.perf_counter()
    ser.write(payload)

    resp = ser.read(32)
    t_roundtrip = (time.perf_counter() - t_start) * 1000.0  # ms

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

def main():
    print("="*65)
    print("  Arty S7-25 BitGEMM Accelerator Hardware Test Harness")
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
        print("Please verify the board is connected and no other app (like PuTTY or Vivado) has locked it.")
        sys.exit(1)

    try:
        # Check status
        status = get_hardware_status(ser)
        print(f"Hardware Info: {status}")

        # Run tests
        test1_ok = run_int8_gemm_test(ser)
        test2_ok = run_bitnet_gemm_test(ser)

        print("\n" + "="*65)
        print("  TEST 3: LIVE LED HARDWARE STREAMING (25 Matrix Burst)")
        print("  Watch your board: LD0, LD1, LD3, LD4, LD5 will pulse and dance!")
        print("="*65)
        burst_passed = 0
        for idx in range(1, 26):
            is_bitnet = (idx % 2 == 0)
            opcode = 0x03 if is_bitnet else 0x02
            A = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
            if is_bitnet:
                B = np.random.choice([-1, 0, 1], size=(4, 4)).astype(np.int8)
            else:
                B = np.random.randint(-8, 8, size=(4, 4), dtype=np.int8)
            C_golden = np.matmul(A.astype(np.int32), B.astype(np.int32)).astype(np.int16)

            ser.write(bytes([opcode]) + A.tobytes() + B.tobytes())
            resp = ser.read(32)
            if len(resp) == 32:
                C_fpga = np.frombuffer(resp, dtype=np.int16).reshape((4, 4))
                if np.array_equal(C_fpga, C_golden):
                    burst_passed += 1

            mode_str = "BitNet Ternary" if is_bitnet else "INT8 Standard "
            print(f"\r  [Streaming Packet {idx:02d}/25] Mode: {mode_str} -> LED Pulse Active", end="", flush=True)
            time.sleep(0.08)  # 80ms delay so human eye clearly sees the LED pulses

        print(f"\n  [Done] {burst_passed}/25 Streaming Packets Verified 100% Bit-Exact!")

        print("\n" + "="*65)
        if test1_ok and test2_ok and burst_passed == 25:
            print("  ALL HARDWARE ACCELERATION TESTS PASSED (3/3)!")
            print("  Your Arty S7-25 is running verified BitNet & INT8 GEMM.")
        else:
            print("  SOME TESTS FAILED. Check connection or bitstream.")
        print("="*65)

    finally:
        ser.close()

if __name__ == "__main__":
    main()
