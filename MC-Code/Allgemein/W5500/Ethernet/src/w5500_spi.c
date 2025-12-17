//****************************************************************************
//
//! \file w5500_spi.c
//! \brief WIZnet W5500 EVB spi interface file
//!
//! Copyright (c)  2014, WIZnet Co., LTD.
//! All rights reserved.
//
//****************************************************************************

#include "../inc/w5500_spi.h"
#include "spi.h"
#include "spi1_cs.h"

extern SPI_HandleTypeDef hspi1;

/**
  * @brief  Initializes the peripherals used by the W5500 driver.
  * @param  None
  * @retval None
  */


void  wizchip_select(void)
{
	// The CS of the W5500 is a nCS
	WriteCS(0);
	
}

void  wizchip_deselect(void)
{
	// The CS of the W5500 is a nCS
	WriteCS(1);
}

uint8_t wizchip_read()
{
	uint8_t rb;
	HAL_SPI_Receive(&hspi1, &rb,1,5000);
	return rb;
}

void  wizchip_write(uint8_t wb)
{
	HAL_SPI_Transmit(&hspi1,&wb,1,5000);
}

void W5500_Init()
{
	uint8_t memsize[2][8] = { { 2, 2, 2, 2, 2, 2, 2, 2 }, { 2, 2, 2, 2, 2, 2, 2, 2 } };

	reg_wizchip_cs_cbfunc(wizchip_select, wizchip_deselect);
	reg_wizchip_spi_cbfunc(wizchip_read, wizchip_write);

	/* wizchip initialize*/
	if (ctlwizchip(CW_INIT_WIZCHIP, (void*) memsize) == -1) {
		//printf("WIZCHIP Initialized fail.\r\n");
		while (1);
	}
/*
	do {
		if (ctlwizchip(CW_GET_PHYLINK, (void*) &tmp) == -1)
			printf("Unknown PHY Link stauts.\r\n");
	} while (tmp == PHY_LINK_OFF);
*/
}

