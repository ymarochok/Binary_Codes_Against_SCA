import chipwhisperer as cw
import time
import binascii

scope = cw.scope()
scope.default_setup()

# Reset
scope.io.nrst = 'low'
time.sleep(0.1)
scope.io.nrst = 'high'
time.sleep(0.5)

target = cw.target(scope)

print("1. Testing 'v' command...")
target.simpleserial_write('v', b'')
time.sleep(0.1)

try:
    resp = target.simpleserial_read('r', 100, timeout=1000)
    if resp:
        print(f"Response to 'v': {binascii.hexlify(resp).decode()}")
        print(f"As text: {resp}")
    else:
        print("No response to 'v'")
except:
    print("Read failed")

print("\n2. Testing 'p' command...")
target.simpleserial_write('p', b'\x00\x00\x00\x00')
time.sleep(0.1)

try:
    resp = target.simpleserial_read('r', 100, timeout=1000)
    if resp:
        print(f"Response to 'p': {binascii.hexlify(resp).decode()}")
        if len(resp) == 4:
            import struct
            try:
                result = struct.unpack('<f', resp)[0]
                print(f"As float: {result}")
            except:
                pass
    else:
        print("No response to 'p'")
except:
    print("Read failed")

scope.dis()