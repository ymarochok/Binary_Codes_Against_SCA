
#include "hal.h"
#include "simpleserial.h"
#include "network.h"
#include "network_config.h"
#include "input_fixed.h"
#define printf(...)
#define puts(...)

uint8_t handle(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf)
{
    trigger_high();

    float output = network_forward(input_fixed);

    trigger_low();

    uint8_t response[4];
    response[0] = ((uint8_t*)&output)[0];
    response[1] = ((uint8_t*)&output)[1];
    response[2] = ((uint8_t*)&output)[2];
    response[3] = ((uint8_t*)&output)[3];
    
    simpleserial_put('r', 4, response);

    return 0;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    simpleserial_init();

    simpleserial_addcmd('p', 0, handle);

    while (1)
        simpleserial_get();
}