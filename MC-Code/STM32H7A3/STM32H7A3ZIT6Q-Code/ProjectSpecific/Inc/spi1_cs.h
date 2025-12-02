
#ifndef _SPI1_CS_H_
#define _SPI1_CS_H_

#include "stm32h7xx_hal.h"

void WriteCS(GPIO_PinState state);
GPIO_PinState ReadCS(void);

#endif