
#include "spi1_cs.h"

// Write CS of SPI1
// state is the new value of the output pin
// state = GPIO_PIN_RESET or GPIO_PIN_SET
void WriteCS(GPIO_PinState state)
{
	HAL_GPIO_WritePin(GPIOD,GPIO_PIN_14,state);
}

// Read the current CS value of the SPI1
GPIO_PinState ReadCS(void)
{
	return HAL_GPIO_ReadPin(GPIOD,GPIO_PIN_14);
}