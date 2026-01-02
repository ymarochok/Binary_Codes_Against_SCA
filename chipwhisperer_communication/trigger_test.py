import chipwhisperer as cw
import time
import numpy as np
import matplotlib.pyplot as plt

scope = cw.scope()
target = cw.target(scope)

# Setup scope for trigger on TIO3 (PA12)
scope.default_setup()

# trigger setup
scope.trigger.triggers = "tio3" 
scope.io.tio3 = "high_z"        

# Increase gain to see trigger better
scope.gain.gain = 45
scope.adc.samples = 10000  # More samples
scope.adc.basic_mode = "rising_edge"  # Trigger on rising edge

## Simple Trigger test
print(f"Trigger: {scope.trigger.triggers}")
print(f"TIO3 state: {scope.io.tio3}")

# Reset
scope.io.nrst = 'low'
time.sleep(0.1)
scope.io.nrst = 'high'
time.sleep(0.5)

# Capture trace
print("Arming scope...")
scope.arm()

print("Sending 'p' command to start neural network...")
target.simpleserial_write('p', b'')

print("Waiting for trigger...")
if scope.capture():
    trace = scope.get_last_trace()
    print(f"Capture successful: {len(trace)} samples")
    
    # Basic statistics
    mean_val = np.mean(trace)
    std_val = np.std(trace)
    max_val = np.max(trace)
    min_val = np.min(trace)
    
    print(f"   Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"   Max: {max_val:.4f}, Min: {min_val:.4f}")
    
    # Method 1: Look for significant changes
    diff = np.diff(trace)
    large_changes = np.where(np.abs(diff) > 2 * std_val)[0]
    
    # Method 2: Look for sustained high/low periods
    threshold_high = mean_val + 2 * std_val
    threshold_low = mean_val - 2 * std_val
    
    high_indices = np.where(trace > threshold_high)[0]
    low_indices = np.where(trace < threshold_low)[0]
    
    print(f"\nAnalysis:")
    print(f"   Large changes: {len(large_changes)} points")
    print(f"   High points (> mean+2σ): {len(high_indices)}")
    print(f"   Low points (< mean-2σ): {len(low_indices)}")
    
    if len(high_indices) > 10 or len(low_indices) > 10:
        # Find continuous segments
        def find_continuous(indices, max_gap=10):
            if len(indices) == 0:
                return []
            segments = []
            start = indices[0]
            end = indices[0]
            
            for i in range(1, len(indices)):
                if indices[i] - indices[i-1] <= max_gap:
                    end = indices[i]
                else:
                    segments.append((start, end))
                    start = indices[i]
                    end = indices[i]
            segments.append((start, end))
            return segments
        
        high_segments = find_continuous(high_indices)
        low_segments = find_continuous(low_indices)
        
        print(f"\nPossible trigger segments:")
        for seg in high_segments[:3]: 
            print(f"   High: samples {seg[0]}-{seg[1]} (duration: {seg[1]-seg[0]+1})")
        for seg in low_segments[:3]:
            print(f"   Low: samples {seg[0]}-{seg[1]} (duration: {seg[1]-seg[0]+1})")
        
        if high_segments or low_segments:
            print("Trigger activity detected!")
        else:
            print("No clear trigger pattern")
    else:
        print("No significant trigger detected")
    
    # Plot trace
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(trace, linewidth=0.5, label='Power Trace')
    plt.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.4f}')
    plt.axhline(y=threshold_high, color='red', linestyle=':', alpha=0.5, label=f'+2σ: {threshold_high:.4f}')
    plt.axhline(y=threshold_low, color='blue', linestyle=':', alpha=0.5, label=f'-2σ: {threshold_low:.4f}')
    
    # Mark potential trigger areas
    if len(high_indices) > 0:
        plt.axvspan(np.min(high_indices), np.max(high_indices), 
                   alpha=0.1, color='red', label='High Activity')
    
    plt.title(f"Power Trace (Trigger on {scope.trigger.triggers.upper()})")
    plt.xlabel("Sample")
    plt.ylabel("Power")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot histogram
    plt.subplot(2, 1, 2)
    plt.hist(trace, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.axvline(x=threshold_high, color='orange', linestyle=':', label=f'+2σ: {threshold_high:.4f}')
    plt.axvline(x=threshold_low, color='green', linestyle=':', label=f'-2σ: {threshold_low:.4f}')
    plt.title("Power Distribution")
    plt.xlabel("Power")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trigger_analysis.png", dpi=100)
    print("\nSaved detailed analysis as 'trigger_analysis.png'")
    plt.show()
    
else:
    print("Capture failed - no trigger detected")

scope.dis()
target.dis()