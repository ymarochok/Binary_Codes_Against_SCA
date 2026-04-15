#include "hal.h"
#include "simpleserial.h"
#include "network.h"
#include "network_config.h"
#include <string.h>  // For memcpy

// Optional: you can remove #include "input_fixed.h" since it's not used

#define printf(...)
#define puts(...)

uint8_t handle(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    trigger_high();
    
    // FIX #1: Use memcpy instead of direct pointer casting (more robust)
    float input[10];
    memcpy(input, buf, 40);  // Copy 40 bytes (10 floats) from buffer
    
    // FIX #2: Run network with the actual received random inputs
    float output = network_forward(input);
    
    trigger_low();

    // Send dummy response (4 bytes)
    uint8_t response[4] = {0x01, 0x02, 0x03, 0x04};
    simpleserial_put('r', 4, response);
    
    // Send acknowledgment
    uint8_t ack[1] = {0x01};
    simpleserial_put('e', 1, ack);

    return 0;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    simpleserial_init();
    
    // FIX #3: Accept 40 bytes (10 floats × 4 bytes each)
    simpleserial_addcmd('p', 40, handle);

    while (1)
        simpleserial_get();
}