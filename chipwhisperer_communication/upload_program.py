import chipwhisperer as cw
import time

scope = cw.scope()
target = cw.target(scope)

scope.default_setup()

scope.io.target_pwr = True
time.sleep(0.2)

prog = cw.programmers.STM32FProgrammer

cw.program_target(
    scope,
    prog,
    "simpleserial-target-CWLITEARM.hex"
)

print("Flashed OK")

scope.io.nrst = 'low'
time.sleep(0.05)
scope.io.nrst = 'high'
time.sleep(0.1)

target.flush()
target.write(b'v\n')
time.sleep(0.1)
print(target.read())
