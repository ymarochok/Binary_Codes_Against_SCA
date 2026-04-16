#include "hal.h"
#include "simpleserial.h"
#include "network.h"          
#include "network_config.h"   

uint8_t handle(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    trigger_high();
    // NOP
    // decimte value check out in capture traces script
    


    // 1. Cast buf to (int8_t*) to satisfy network_forward
    // 2. Store the result in a variable
    int8_t result = network_forward((int8_t*)buf);
    
    // 3. Pass the ADDRESS of the result (&result) to simpleserial_put
    // Note: len should match the size of what you are sending (e.g., 4 bytes for int32_t)
    
    trigger_low();
    
    simpleserial_put('r', sizeof(result), (uint8_t*)&result);
    simpleserial_put('e', 1, (uint8_t[]){0x01});
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