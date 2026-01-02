import chipwhisperer as cw
import time
import serial

scope = cw.scope()
scope.default_setup()

print("=== Testing Bootloader ===")

# IMPORTANT!!! Note: Created by AI with the research purpose...

# Method 1: Using ChipWhisperer's bootloader mode
print("1. Trying ChipWhisperer bootloader...")
try:
    # Some CW-Lite versions need special bootloader sequence
    scope.io.pdic = 'high'  # BOOT0
    time.sleep(0.05)
    scope.io.nrst = 'low'
    time.sleep(0.05)
    scope.io.nrst = 'high'
    time.sleep(0.2)
    scope.io.pdic = 'low'
    
    # Now try to connect
    target = cw.target(scope)
    
    # Bootloader expects 0x7F as first byte
    target._ser.write(b'\x7F')
    time.sleep(0.1)
    
    response = target._ser.read(1)
    if response == b'\x79':  # ACK
        print("✅ Bootloader responds with ACK (0x79)")
    elif response:
        print(f"Bootloader response: {response.hex()}")
    else:
        print("❌ No bootloader response")
        
except Exception as e:
    print(f"Bootloader test failed: {e}")

print("\n=== Method 2: Direct Serial Test ===")

# Method 2: Direct serial connection
try:
    # Connect to serial port
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    
    print("Sending 0x7F (bootloader sync)...")
    ser.write(b'\x7F')
    time.sleep(0.1)
    
    response = ser.read(10)
    if response:
        print(f"Response: {response.hex()}")
        if response[0] == 0x79:
            print("✅ Bootloader ACK received!")
            
            # Try get version command (0x00 0xFF)
            print("\nGetting bootloader version...")
            ser.write(b'\x00\xFF')
            time.sleep(0.1)
            version = ser.read(10)
            if version:
                print(f"Version response: {version.hex()}")
    else:
        print("❌ No response")
        
    ser.close()
    
except Exception as e:
    print(f"Direct serial failed: {e}")

print("\n=== Method 3: Check Current Firmware ===")

# Method 3: Check what's actually running
scope.io.nrst = 'low'
time.sleep(0.1)
scope.io.nrst = 'high'
time.sleep(0.5)

target = cw.target(scope)

# Try different baud rates
baud_rates = [115200, 38400, 9600, 57600]

for baud in baud_rates:
    print(f"\nTrying baud rate: {baud}")
    try:
        target._ser.baudrate = baud
        
        # Send simple command
        target.simpleserial_write('v', b'')
        time.sleep(0.1)
        
        # Try to read
        target._ser.timeout = 0.5
        raw = target._ser.read(100)
        
        if raw:
            print(f"✅ Response at {baud}: {raw.hex()}")
            if b'version' in raw.lower() or b'v' in raw:
                print(f"   ⚡ CORRECT BAUD RATE FOUND: {baud}")
                break
        else:
            print(f"❌ No response at {baud}")
            
    except Exception as e:
        print(f"Failed at {baud}: {e}")

scope.dis()