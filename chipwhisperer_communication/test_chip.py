import chipwhisperer as cw
import time
import matplotlib.pyplot as plt
import numpy as np

# Connect
scope = cw.scope()
scope.default_setup()

# CW-Lite ARM uses TIO3 for trigger (PA12)
scope.trigger.triggers = "tio3"  # Changed from "tio4"
scope.io.tio3 = "high_z"         # Set as input for trigger
scope.trigger.module = "basic"

print(f"Trigger source: {scope.trigger.triggers}")
print(f"TIO3 state: {scope.io.tio3}")

# Reset
scope.io.nrst = 'low'
time.sleep(0.1)
scope.io.nrst = 'high'
time.sleep(0.5)

# Test manual trigger first
print("\n1. Testing trigger output from chip...")
target = cw.target(scope)

# Send command multiple times
for i in range(3):
    print(f"  Attempt {i+1}: Sending 'p' command")
    target.simpleserial_write('p', b'')
    time.sleep(0.1)

# Capture trace
print("\n2. Capturing trace...")
scope.arm()
target.simpleserial_write('p', b'')  # Send one more time

if scope.capture():
    trace = scope.get_last_trace()
    print(f"Capture successful: {len(trace)} samples")
    
    # Analyze trace
    mean_power = np.mean(trace)
    max_power = np.max(trace)
    min_power = np.min(trace)
    
    print(f"   Mean: {mean_power:.3f}, Max: {max_power:.3f}, Min: {min_power:.3f}")
    
    # Look for trigger signal (PA12 going high)
    diff = np.diff(trace)
    trigger_points = np.where(np.abs(diff) > 0.05)[0]  # Threshold
    
    if len(trigger_points) > 0:
        print(f"Possible trigger at samples: {trigger_points[:5]}...")
    else:
        print("No clear trigger detected")
        print("Try increasing scope.gain.gain to 40-45")
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(trace, linewidth=0.5)
    plt.title(f"Power Trace (Trigger on PA12/TIO3)")
    plt.xlabel("Sample")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    
    # Mark potential trigger areas
    if len(trigger_points) > 0:
        plt.axvspan(max(0, trigger_points[0]-10), 
                   min(len(trace), trigger_points[0]+100), 
                   alpha=0.2, color='red', label='Trigger Area')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("trigger_pa12_test.png", dpi=100)
    print("Saved as 'trigger_pa12_test.png'")
    
else:
    print("Capture failed")

scope.dis()
target.dis()