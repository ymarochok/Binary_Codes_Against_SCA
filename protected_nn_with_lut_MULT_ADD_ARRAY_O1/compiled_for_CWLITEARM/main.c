#include "hal.h"
#include "simpleserial.h"
#include "network.h"          
#include "network_config.h"   
#include "code_ops.h"

// Helper macro for a NOP sled - adjust '10' to make the gap wider or narrower
// A bigger sled for better visual separation
#define NOP_SLED_50() do { \
    for(int i=0; i<800; i++) { __asm__ __volatile__ ("nop"); } \
} while(0)

uint8_t handle(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    trigger_high();
    
    NOP_SLED_50(); // Visual gap before math

    int32_t result = network_forward((uint16_t*)buf);
    
    NOP_SLED_50(); // Visual gap after math
    
    trigger_low();
    simpleserial_put('r', sizeof(int32_t), (uint8_t*)&result);
    return 0;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();
    simpleserial_addcmd('p', 10, handle);
    while(1) simpleserial_get();
}